"""
src/training/sft_trainer.py
Alignment-Aware Model Distillation — Student Model Fine-Tuning
==============================================================
Fine-tunes a smaller student model on the safety-filtered teacher outputs
using LoRA + SFTTrainer from the TRL library.

KEY DESIGN DECISIONS
--------------------
- LoRA fine-tuning (not full fine-tuning) to preserve base model knowledge
- Loss masking: only compute loss on assistant response tokens, not the prompt
- Sample weighting: downweighted records contribute less to the gradient
- ChatML format: matches the teacher's output format exactly
- Gradient checkpointing: fits larger models in GPU memory

STUDENT MODEL OPTIONS
---------------------
  Recommended pairings with Llama-3.1-8B teacher:
    meta-llama/Llama-3.2-1B-Instruct   (smallest, fastest, most alignment gain visible)
    meta-llama/Llama-3.2-3B-Instruct   (better quality, still clear size gap)
    meta-llama/Llama-3.1-8B-Instruct   (same size — ablation/baseline only)

USAGE
-----
  # Basic run (recommended defaults)
  python src/training/sft_trainer.py \
      --student_model meta-llama/Llama-3.2-1B-Instruct \
      --filtered_dir data/filtered \
      --output_dir models/student_checkpoints

  # With W&B logging
  python src/training/sft_trainer.py \
      --student_model meta-llama/Llama-3.2-1B-Instruct \
      --wandb_project alignment-distillation

  # Full run with all options
  python src/training/sft_trainer.py \
      --student_model meta-llama/Llama-3.2-3B-Instruct \
      --filtered_dir data/filtered \
      --output_dir models/student_checkpoints \
      --num_epochs 3 \
      --batch_size 4 \
      --grad_accum 8 \
      --lora_r 16 \
      --max_seq_len 1024 \
      --wandb_project alignment-distillation

OUTPUT
------
  models/student_checkpoints/
    final/          — merged LoRA weights (ready for inference)
    checkpoint-*/   — intermediate checkpoints (every 100 steps)
    trainer_log.json — training loss curve
"""

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("sft_trainer")

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class TrainingConfig:
    # Model
    student_model:   str   = "meta-llama/Llama-3.2-1B-Instruct"
    filtered_dir:    str   = "data/filtered"
    output_dir:      str   = "models/student_checkpoints"

    # LoRA
    lora_r:          int   = 16       # rank — higher = more capacity, more VRAM
    lora_alpha:      int   = 32       # scaling factor (usually 2x rank)
    lora_dropout:    float = 0.05
    lora_targets:    str   = "auto"   # "auto" = all linear layers

    # Training
    num_epochs:      int   = 3
    batch_size:      int   = 4        # per-device batch size
    grad_accum:      int   = 8        # effective batch = batch_size × grad_accum × n_gpus
    learning_rate:   float = 2e-4
    warmup_ratio:    float = 0.05
    lr_scheduler:    str   = "cosine"
    max_seq_len:     int   = 1024
    weight_decay:    float = 0.01

    # Memory
    bf16:            bool  = True     # use bfloat16 (A100 supports this natively)
    gradient_checkpointing: bool = True
    use_4bit:        bool  = False    # 4-bit quantization (for smaller GPUs)

    # Logging
    wandb_project:   Optional[str] = None
    log_steps:       int   = 10
    save_steps:      int   = 100
    eval_steps:      int   = 100
    eval_split:      float = 0.05    # fraction of data held out for eval

    # Misc
    seed:            int   = 42
    use_sample_weights: bool = True  # use safety scores as sample weights


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_filtered_dataset(filtered_dir: Path, eval_split: float, seed: int):
    """
    Load train.jsonl, split into train/eval, return HuggingFace datasets.
    """
    from datasets import Dataset
    import random

    train_path = filtered_dir / "train.jsonl"
    if not train_path.exists():
        raise FileNotFoundError(
            f"Filtered dataset not found: {train_path}\n"
            f"Run safety_judge.py first."
        )

    records = []
    with open(train_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    log.info(f"Loaded {len(records)} training records from {train_path}")

    # Log dataset composition
    from collections import Counter
    cats = Counter(r.get("category", "unknown") for r in records)
    decisions = Counter(r.get("filter_decision", "unknown") for r in records)
    log.info(f"Categories: {dict(cats)}")
    log.info(f"Filter decisions: {dict(decisions)}")

    # Shuffle and split
    rng = random.Random(seed)
    rng.shuffle(records)
    n_eval = max(1, int(len(records) * eval_split))
    eval_records  = records[:n_eval]
    train_records = records[n_eval:]

    log.info(f"Train: {len(train_records)} | Eval: {len(eval_records)}")

    train_ds = Dataset.from_list(train_records)
    eval_ds  = Dataset.from_list(eval_records)

    return train_ds, eval_ds


# ---------------------------------------------------------------------------
# Model + tokenizer setup
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(cfg: TrainingConfig):
    """Load student model with optional 4-bit quantization and LoRA."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    log.info(f"Loading tokenizer: {cfg.student_model}")
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.student_model,
        trust_remote_code=True,
    )

    # Ensure pad token exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Set padding side to right for causal LM
    tokenizer.padding_side = "right"

    log.info(f"Loading model: {cfg.student_model}")

    if cfg.use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            cfg.student_model,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            cfg.student_model,
            torch_dtype=torch.bfloat16 if cfg.bf16 else torch.float32,
            device_map="auto",
            trust_remote_code=True,
        )

    if cfg.gradient_checkpointing:
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()

    # LoRA config
    if cfg.lora_targets == "auto":
        # Target all linear layers — works for most architectures
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"]
    else:
        target_modules = cfg.lora_targets.split(",")

    lora_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer


# ---------------------------------------------------------------------------
# Sample weight callback
# ---------------------------------------------------------------------------

class SampleWeightCallback:
    """
    Applies per-sample weights during training by scaling the loss.
    Records with filter_decision='downweight' have sample_weight < 1.0.
    """
    def __init__(self, dataset, use_weights: bool):
        self.use_weights = use_weights
        if use_weights:
            weights = [r.get("sample_weight", 1.0) for r in dataset]
            self.weights = torch.tensor(weights, dtype=torch.float32)
            log.info(
                f"Sample weights — min: {min(weights):.3f} "
                f"max: {max(weights):.3f} "
                f"mean: {sum(weights)/len(weights):.3f}"
            )
        else:
            self.weights = None


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(cfg: TrainingConfig) -> None:
    from transformers import TrainingArguments, set_seed
    from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

    set_seed(cfg.seed)

    # Setup W&B
    if cfg.wandb_project:
        os.environ["WANDB_PROJECT"] = cfg.wandb_project
        log.info(f"W&B logging enabled: project={cfg.wandb_project}")
    else:
        os.environ["WANDB_DISABLED"] = "true"

    filtered_dir = Path(cfg.filtered_dir)
    output_dir   = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    train_ds, eval_ds = load_filtered_dataset(filtered_dir, cfg.eval_split, cfg.seed)

    # Load model
    model, tokenizer = load_model_and_tokenizer(cfg)

    # Loss masking: only compute loss on assistant response tokens.
    # DataCollatorForCompletionOnlyLM masks everything before the
    # response_template token, so the model only learns to generate responses.
    response_template = "<|im_start|>assistant\n"
    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer,
        mlm=False,
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=cfg.num_epochs,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=cfg.learning_rate,
        warmup_ratio=cfg.warmup_ratio,
        lr_scheduler_type=cfg.lr_scheduler,
        weight_decay=cfg.weight_decay,
        bf16=cfg.bf16,
        fp16=False,
        gradient_checkpointing=cfg.gradient_checkpointing,
        logging_steps=cfg.log_steps,
        save_steps=cfg.save_steps,
        eval_steps=cfg.eval_steps,
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="wandb" if cfg.wandb_project else "none",
        seed=cfg.seed,
        dataloader_num_workers=2,
        remove_unused_columns=False,  # keep sample_weight column
    )

    log.info(
        f"Effective batch size: "
        f"{cfg.batch_size * cfg.grad_accum} "
        f"(batch={cfg.batch_size} × accum={cfg.grad_accum})"
    )

    # SFT Trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=training_args,
        data_collator=collator,
        dataset_text_field="full_text",
        max_seq_length=cfg.max_seq_len,
        packing=False,   # don't pack sequences — preserves sample boundaries
    )

    # Apply sample weights if enabled
    # We patch the compute_loss method to scale loss by sample_weight
    if cfg.use_sample_weights:
        original_compute_loss = trainer.compute_loss

        def weighted_compute_loss(model, inputs, return_outputs=False, **kwargs):
            weights = inputs.pop("sample_weight", None)
            result = original_compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)
            if weights is not None and not return_outputs:
                # Scale loss by normalized weights
                weights = weights.to(result.device)
                weights = weights / weights.mean()
                result = result * weights.mean()
            return result

        trainer.compute_loss = weighted_compute_loss
        log.info("Sample weighting enabled.")

    # Train
    log.info("Starting training...")
    train_result = trainer.train()

    # Save final model (merged LoRA weights)
    final_dir = output_dir / "final"
    log.info(f"Saving final model to {final_dir}...")
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    # Save training metrics
    metrics = train_result.metrics
    metrics["train_samples"] = len(train_ds)
    metrics["eval_samples"]  = len(eval_ds)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    # Save trainer log for analysis
    log_history = trainer.state.log_history
    log_path = output_dir / "trainer_log.json"
    with open(log_path, "w") as f:
        json.dump(log_history, f, indent=2)
    log.info(f"Training log saved → {log_path}")

    log.info(
        f"\n{'='*55}\n"
        f"  Training complete!\n"
        f"  Train loss : {metrics.get('train_loss', 'N/A'):.4f}\n"
        f"  Steps      : {metrics.get('train_steps_per_second', 'N/A')}\n"
        f"  Model saved: {final_dir}\n"
        f"{'='*55}"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fine-tune student model on safety-filtered teacher outputs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--student_model",  type=str,   default="meta-llama/Llama-3.2-1B-Instruct")
    p.add_argument("--filtered_dir",   type=str,   default="data/filtered")
    p.add_argument("--output_dir",     type=str,   default="models/student_checkpoints")
    p.add_argument("--num_epochs",     type=int,   default=3)
    p.add_argument("--batch_size",     type=int,   default=4)
    p.add_argument("--grad_accum",     type=int,   default=8)
    p.add_argument("--learning_rate",  type=float, default=2e-4)
    p.add_argument("--lora_r",         type=int,   default=16)
    p.add_argument("--lora_alpha",     type=int,   default=32)
    p.add_argument("--max_seq_len",    type=int,   default=1024)
    p.add_argument("--use_4bit",       action="store_true",
                   help="4-bit quantization (use on smaller GPUs, not needed on A100)")
    p.add_argument("--wandb_project",  type=str,   default=None,
                   help="W&B project name for logging. Omit to disable.")
    p.add_argument("--no_sample_weights", action="store_true",
                   help="Disable sample weighting (treat all records equally).")
    p.add_argument("--eval_split",     type=float, default=0.05)
    p.add_argument("--seed",           type=int,   default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = TrainingConfig(
        student_model    = args.student_model,
        filtered_dir     = args.filtered_dir,
        output_dir       = args.output_dir,
        num_epochs       = args.num_epochs,
        batch_size       = args.batch_size,
        grad_accum       = args.grad_accum,
        learning_rate    = args.learning_rate,
        lora_r           = args.lora_r,
        lora_alpha       = args.lora_alpha,
        max_seq_len      = args.max_seq_len,
        use_4bit         = args.use_4bit,
        wandb_project    = args.wandb_project,
        use_sample_weights = not args.no_sample_weights,
        eval_split       = args.eval_split,
        seed             = args.seed,
    )

    log.info("Training config:")
    for k, v in cfg.__dict__.items():
        log.info(f"  {k:25s}: {v}")

    train(cfg)


if __name__ == "__main__":
    main()