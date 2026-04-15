"""
src/utils/formatting.py
Prompt and response formatting helpers for alignment distillation pipeline.
Centralizes all template logic so vllm_runner, sft_trainer, and the frontend
all produce identical formats.
"""

from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TEACHER_SYSTEM_PROMPT = (
    "You are a helpful, accurate, and thoughtful assistant. "
    "Answer the user's question directly and completely."
)

CHATML_SYSTEM = "<|im_start|>system\n{system}<|im_end|>\n"
CHATML_USER   = "<|im_start|>user\n{user}<|im_end|>\n"
CHATML_ASST   = "<|im_start|>assistant\n{assistant}<|im_end|>\n"
CHATML_ASST_OPEN = "<|im_start|>assistant\n"   # for inference (no closing token)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ChatMessage:
    role: str    # system | user | assistant
    content: str


@dataclass
class FormattedSample:
    """A training sample ready for the SFT trainer."""
    prompt: str          # full ChatML string up to and including assistant open tag
    response: str        # assistant response text only (no special tokens)
    full_text: str       # prompt + response + closing token (for SFTTrainer)


# ---------------------------------------------------------------------------
# ChatML formatting
# ---------------------------------------------------------------------------

def to_chatml(messages: list[ChatMessage], add_generation_prompt: bool = False) -> str:
    """
    Convert a list of ChatMessage objects to a ChatML string.

    Args:
        messages: Ordered list of chat turns.
        add_generation_prompt: If True, append the open assistant tag at the end
                               (used during inference to prompt the model to respond).

    Returns:
        A single ChatML-formatted string.
    """
    parts = []
    for msg in messages:
        if msg.role == "system":
            parts.append(CHATML_SYSTEM.format(system=msg.content.strip()))
        elif msg.role == "user":
            parts.append(CHATML_USER.format(user=msg.content.strip()))
        elif msg.role == "assistant":
            parts.append(CHATML_ASST.format(assistant=msg.content.strip()))
        else:
            raise ValueError(f"Unknown role: {msg.role!r}")

    if add_generation_prompt:
        parts.append(CHATML_ASST_OPEN)

    return "".join(parts)


def prompt_only_chatml(user_text: str, system_text: str = TEACHER_SYSTEM_PROMPT) -> str:
    """
    Build the inference-time ChatML string: system + user + open assistant tag.
    This is what you'd send to a tokenizer for generation.
    """
    messages = [
        ChatMessage(role="system", content=system_text),
        ChatMessage(role="user", content=user_text),
    ]
    return to_chatml(messages, add_generation_prompt=True)


def format_training_sample(
    user_text: str,
    assistant_text: str,
    system_text: str = TEACHER_SYSTEM_PROMPT,
) -> FormattedSample:
    """
    Format a (prompt, response) pair into a full training sample with ChatML.
    The SFT trainer should train only on the response portion (loss masking
    handled separately in sft_trainer.py via DataCollatorForCompletionOnlyLM).
    """
    prompt_messages = [
        ChatMessage(role="system", content=system_text),
        ChatMessage(role="user", content=user_text),
    ]
    prompt_str = to_chatml(prompt_messages, add_generation_prompt=True)
    full_text = prompt_str + assistant_text.strip() + "<|im_end|>\n"

    return FormattedSample(
        prompt=prompt_str,
        response=assistant_text.strip(),
        full_text=full_text,
    )


# ---------------------------------------------------------------------------
# vLLM / OpenAI API message formatting
# ---------------------------------------------------------------------------

def to_openai_messages(
    user_text: str,
    system_text: str = TEACHER_SYSTEM_PROMPT,
) -> list[dict]:
    """
    Format a prompt as OpenAI-style messages list for the vLLM chat endpoint.
    """
    return [
        {"role": "system", "content": system_text},
        {"role": "user",   "content": user_text},
    ]


# ---------------------------------------------------------------------------
# Dataset record formatting
# ---------------------------------------------------------------------------

def format_teacher_record(raw_record: dict) -> dict:
    """
    Given a raw teacher output record (from vllm_runner.py), return a
    standardized dict suitable for the safety_judge.py input.

    Expected keys in raw_record:
        prompt_id, prompt_text, category, subcategory, risk_tier, response, ...
    """
    return {
        "prompt_id":   raw_record["prompt_id"],
        "prompt_text": raw_record["prompt_text"].strip(),
        "response":    raw_record["response"].strip(),
        "category":    raw_record["category"],
        "subcategory": raw_record["subcategory"],
        "risk_tier":   raw_record["risk_tier"],
        # pass-through metadata
        "model":            raw_record.get("model", ""),
        "finish_reason":    raw_record.get("finish_reason", ""),
        "prompt_tokens":    raw_record.get("prompt_tokens", -1),
        "completion_tokens":raw_record.get("completion_tokens", -1),
        "latency_ms":       raw_record.get("latency_ms", -1.0),
    }


def format_filtered_record(teacher_record: dict, safety_result: dict) -> dict:
    """
    Merge a teacher record with its safety scores into the final filtered-dataset
    format that sft_trainer.py consumes.

    Args:
        teacher_record: Output of format_teacher_record().
        safety_result:  Dict with keys: safety_score, safety_label, flagged_spans,
                        llm_judge_score, toxicity_score, filter_decision.
    Returns:
        A flat dict ready to be written to data/filtered/train.jsonl.
    """
    sample = format_training_sample(
        user_text=teacher_record["prompt_text"],
        assistant_text=teacher_record["response"],
    )
    return {
        # identifiers
        "prompt_id":        teacher_record["prompt_id"],
        "category":         teacher_record["category"],
        "subcategory":      teacher_record["subcategory"],
        "risk_tier":        teacher_record["risk_tier"],
        # raw text (for the trainer)
        "prompt_text":      teacher_record["prompt_text"],
        "response":         teacher_record["response"],
        # formatted training text
        "full_text":        sample.full_text,
        # safety metadata
        "safety_score":     safety_result["safety_score"],
        "safety_label":     safety_result["safety_label"],
        "flagged_spans":    safety_result.get("flagged_spans", []),
        "filter_decision":  safety_result["filter_decision"],   # keep | downweight | exclude
        "sample_weight":    safety_result.get("sample_weight", 1.0),
        # teacher metadata
        "teacher_model":    teacher_record.get("model", ""),
        "finish_reason":    teacher_record.get("finish_reason", ""),
    }


# ---------------------------------------------------------------------------
# Convenience display
# ---------------------------------------------------------------------------

def truncate(text: str, max_chars: int = 120) -> str:
    """Truncate text for logging/display."""
    return text if len(text) <= max_chars else text[:max_chars] + "…"