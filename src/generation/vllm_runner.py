"""
src/generation/vllm_runner.py
Alignment-Aware Model Distillation — Teacher Inference Runner
=============================================================
Reads raw prompt JSONL files from data/raw/, runs batched async inference
against a vLLM server, and writes teacher outputs to data/generated/.

USAGE
-----
# 1. Start the teacher on Sol (separate terminal / sbatch job):
#
#    python -m vllm.entrypoints.openai.api_server \\
#        --model meta-llama/Llama-3.1-8B-Instruct \\
#        --max-model-len 2048 --dtype bfloat16 --port 8000
#
# 2. Run inference:
#
#    python src/generation/vllm_runner.py                        # all defaults
#    python src/generation/vllm_runner.py --resume               # skip completed
#    python src/generation/vllm_runner.py --categories safe      # single category
#    python src/generation/vllm_runner.py --dry_run              # validate without calling API

OUTPUT (data/generated/)
------------------------
    teacher_outputs_safe.jsonl        — one record per safe prompt
    teacher_outputs_borderline.jsonl  — one record per borderline prompt
    teacher_outputs_unsafe.jsonl      — one record per unsafe prompt
    run_metadata.json                 — config + run stats for reproducibility

Each output record schema:
{
    "prompt_id":         "safe_001",
    "prompt_text":       "...",
    "category":          "safe",
    "subcategory":       "coding",
    "risk_tier":         0,
    "response":          "...",
    "model":             "meta-llama/Llama-3.1-8B-Instruct",
    "finish_reason":     "stop",
    "prompt_tokens":     42,
    "completion_tokens": 180,
    "latency_ms":        312.5,
    "inference_timestamp": "2025-..."
}
"""

import argparse
import asyncio
import json
import logging
import time
import uuid
from datetime import datetime
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import aiohttp
from tqdm.asyncio import tqdm as atqdm

# Project-local import (run from repo root: python src/generation/vllm_runner.py)
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.formatting import to_openai_messages, TEACHER_SYSTEM_PROMPT

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("vllm_runner")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CATEGORY_META = {
    "safe":       {"file": "safe_prompts.jsonl",       "risk_tier": 0, "out": "teacher_outputs_safe.jsonl"},
    "borderline": {"file": "borderline_prompts.jsonl", "risk_tier": 1, "out": "teacher_outputs_borderline.jsonl"},
    "unsafe":     {"file": "unsafe_prompts.jsonl",     "risk_tier": 2, "out": "teacher_outputs_unsafe.jsonl"},
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TeacherOutput:
    prompt_id: str
    prompt_text: str
    category: str
    subcategory: str
    risk_tier: int
    response: str
    model: str
    finish_reason: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    inference_timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class FailedInference:
    prompt_id: str
    prompt_text: str
    category: str
    error: str
    attempts: int
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------

def load_prompts(raw_dir: Path, category: str) -> list[dict]:
    """Load and validate prompts from a raw JSONL file."""
    meta = CATEGORY_META[category]
    path = raw_dir / meta["file"]
    if not path.exists():
        raise FileNotFoundError(
            f"Prompt file not found: {path}\n"
            f"Expected location: {path.resolve()}"
        )
    prompts = []
    with open(path) as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                log.warning(f"{path}:{i} — skipping malformed JSON: {e}")
                continue
            # Validate required fields
            for key in ("id", "text", "subcategory"):
                if key not in record:
                    log.warning(f"{path}:{i} — missing field '{key}', skipping.")
                    continue
            record["category"] = category
            record["risk_tier"] = meta["risk_tier"]
            prompts.append(record)
    log.info(f"Loaded {len(prompts)} {category} prompts from {path.name}")
    return prompts


def get_completed_ids(output_path: Path) -> set[str]:
    """Return the set of prompt IDs already written to an output file."""
    if not output_path.exists():
        return set()
    ids = set()
    with open(output_path) as f:
        for line in f:
            try:
                ids.add(json.loads(line)["prompt_id"])
            except Exception:
                pass
    return ids


# ---------------------------------------------------------------------------
# Async inference
# ---------------------------------------------------------------------------

async def query_teacher(
    session: aiohttp.ClientSession,
    prompt: dict,
    teacher_url: str,
    model_name: str,
    temperature: float,
    max_tokens: int,
    semaphore: asyncio.Semaphore,
    max_retries: int = 3,
) -> tuple[Optional[TeacherOutput], Optional[FailedInference]]:
    """
    Send a single prompt to the vLLM OpenAI-compatible endpoint.
    Returns (TeacherOutput, None) on success, (None, FailedInference) on failure.
    """
    payload = {
        "model": model_name,
        "messages": to_openai_messages(prompt["text"]),
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    last_error = ""
    async with semaphore:
        for attempt in range(max_retries):
            try:
                t0 = time.monotonic()
                async with session.post(
                    f"{teacher_url}/v1/chat/completions",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=120),
                ) as resp:
                    if resp.status != 200:
                        body = await resp.text()
                        raise aiohttp.ClientResponseError(
                            resp.request_info, resp.history,
                            status=resp.status, message=body[:200],
                        )
                    data = await resp.json()
                latency_ms = (time.monotonic() - t0) * 1000

                choice = data["choices"][0]
                usage  = data.get("usage", {})
                return TeacherOutput(
                    prompt_id        = prompt["id"],
                    prompt_text      = prompt["text"],
                    category         = prompt["category"],
                    subcategory      = prompt["subcategory"],
                    risk_tier        = prompt["risk_tier"],
                    response         = choice["message"]["content"],
                    model            = data.get("model", model_name),
                    finish_reason    = choice.get("finish_reason", "unknown"),
                    prompt_tokens    = usage.get("prompt_tokens", -1),
                    completion_tokens= usage.get("completion_tokens", -1),
                    latency_ms       = round(latency_ms, 1),
                ), None

            except asyncio.TimeoutError:
                last_error = "Request timed out after 120s"
            except aiohttp.ClientResponseError as e:
                last_error = f"HTTP {e.status}: {e.message}"
            except aiohttp.ClientConnectorError:
                last_error = f"Cannot connect to {teacher_url} — is vLLM running?"
            except Exception as e:
                last_error = f"{type(e).__name__}: {e}"

            if attempt < max_retries - 1:
                backoff = 2 ** attempt
                log.debug(
                    f"[{prompt['id']}] attempt {attempt+1}/{max_retries} failed "
                    f"({last_error}), retrying in {backoff}s…"
                )
                await asyncio.sleep(backoff)

    return None, FailedInference(
        prompt_id=prompt["id"],
        prompt_text=prompt["text"],
        category=prompt["category"],
        error=last_error,
        attempts=max_retries,
    )


async def run_category(
    prompts: list[dict],
    category: str,
    teacher_url: str,
    model_name: str,
    generated_dir: Path,
    batch_size: int,
    temperature: float,
    max_tokens: int,
    resume: bool,
) -> dict:
    """
    Run inference for all prompts in one category.
    Returns per-category stats dict.
    """
    meta = CATEGORY_META[category]
    out_path  = generated_dir / meta["out"]
    fail_path = generated_dir / f"failed_{category}.jsonl"

    # Resume: skip already-done IDs
    completed_ids = get_completed_ids(out_path) if resume else set()
    if completed_ids:
        log.info(f"[{category}] Resuming — {len(completed_ids)} already done, skipping.")
    remaining = [p for p in prompts if p["id"] not in completed_ids]

    if not remaining:
        log.info(f"[{category}] All {len(prompts)} prompts already complete.")
        return {"category": category, "total": len(prompts), "success": len(prompts), "failures": 0, "skipped": len(completed_ids)}

    log.info(f"[{category}] Running inference on {len(remaining)} prompts (batch_size={batch_size})…")

    semaphore  = asyncio.Semaphore(batch_size)
    success_n  = len(completed_ids)
    failure_n  = 0
    latencies  = []

    connector = aiohttp.TCPConnector(limit=batch_size + 10, ttl_dns_cache=300)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [
            query_teacher(
                session, p, teacher_url, model_name,
                temperature, max_tokens, semaphore,
            )
            for p in remaining
        ]

        with open(out_path, "a") as out_f, open(fail_path, "a") as fail_f:
            pbar = atqdm(
                asyncio.as_completed(tasks),
                total=len(tasks),
                desc=f"  [{category}]",
                unit="prompt",
                colour="green" if category == "safe" else ("yellow" if category == "borderline" else "red"),
            )
            async for coro in pbar:
                result, failure = await coro
                if result is not None:
                    out_f.write(json.dumps(asdict(result)) + "\n")
                    out_f.flush()
                    latencies.append(result.latency_ms)
                    success_n += 1
                    pbar.set_postfix(success=success_n, fail=failure_n, avg_ms=f"{sum(latencies)/len(latencies):.0f}")
                else:
                    fail_f.write(json.dumps(asdict(failure)) + "\n")
                    fail_f.flush()
                    failure_n += 1
                    log.warning(f"[{category}] FAILED {failure.prompt_id}: {failure.error}")

    return {
        "category":      category,
        "total":         len(prompts),
        "success":       success_n,
        "failures":      failure_n,
        "skipped":       len(completed_ids),
        "avg_latency_ms": round(sum(latencies) / len(latencies), 1) if latencies else None,
    }


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

async def check_server(teacher_url: str, model_name: str) -> bool:
    """Ping the vLLM server and confirm the model is loaded."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{teacher_url}/v1/models",
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                data = await resp.json()
                models = [m["id"] for m in data.get("data", [])]
                if not models:
                    log.error("vLLM server returned no models.")
                    return False
                if model_name not in models:
                    log.warning(
                        f"Requested model '{model_name}' not found. "
                        f"Available: {models}. Proceeding anyway — "
                        f"vLLM may accept the first loaded model."
                    )
                else:
                    log.info(f"Server healthy. Model '{model_name}' confirmed.")
                return True
    except Exception as e:
        log.error(f"Cannot reach vLLM server at {teacher_url}: {e}")
        return False


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

def save_run_metadata(
    args: argparse.Namespace,
    category_stats: list[dict],
    elapsed_s: float,
    generated_dir: Path,
) -> None:
    total     = sum(s["total"]   for s in category_stats)
    success   = sum(s["success"] for s in category_stats)
    failures  = sum(s["failures"]for s in category_stats)
    meta = {
        "run_id":    str(uuid.uuid4()),
        "timestamp": datetime.utcnow().isoformat(),
        "config": vars(args),
        "summary": {
            "total_prompts":   total,
            "total_success":   success,
            "total_failures":  failures,
            "elapsed_seconds": round(elapsed_s, 1),
            "throughput_per_min": round(success / (elapsed_s / 60), 1) if elapsed_s else 0,
        },
        "per_category": category_stats,
    }
    path = generated_dir / "run_metadata.json"
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)
    log.info(f"Run metadata → {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Batch teacher inference for alignment-aware distillation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Paths
    p.add_argument("--raw_dir",       type=str, default="data/raw",
                   help="Directory containing raw prompt JSONL files.")
    p.add_argument("--generated_dir", type=str, default="data/generated",
                   help="Directory to write teacher output JSONL files.")
    # Server
    p.add_argument("--teacher_url",   type=str, default="http://localhost:8000",
                   help="Base URL of the running vLLM server.")
    p.add_argument("--model_name",    type=str,
                   default="meta-llama/Llama-3.1-8B-Instruct",
                   help="Model name as registered in vLLM.")
    # Generation
    p.add_argument("--temperature",   type=float, default=0.7)
    p.add_argument("--max_tokens",    type=int,   default=512)
    p.add_argument("--batch_size",    type=int,   default=32,
                   help="Max concurrent requests to vLLM.")
    # Control
    p.add_argument("--categories",    type=str,   nargs="+",
                   choices=["safe", "borderline", "unsafe"],
                   default=["safe", "borderline", "unsafe"],
                   help="Which prompt categories to process.")
    p.add_argument("--resume",        action="store_true",
                   help="Skip prompts already present in output files.")
    p.add_argument("--dry_run",       action="store_true",
                   help="Load prompts and check server, but do not run inference.")
    p.add_argument("--skip_health_check", action="store_true",
                   help="Skip the vLLM server health check (useful if model list endpoint is disabled).")
    return p.parse_args()


async def main() -> None:
    args = parse_args()

    raw_dir       = Path(args.raw_dir)
    generated_dir = Path(args.generated_dir)
    generated_dir.mkdir(parents=True, exist_ok=True)

    # ── Health check ─────────────────────────────────────────────────────────
    if not args.dry_run and not args.skip_health_check:
        ok = await check_server(args.teacher_url, args.model_name)
        if not ok:
            log.error(
                "Server health check failed. Start vLLM first:\n\n"
                "  python -m vllm.entrypoints.openai.api_server \\\n"
                f"      --model {args.model_name} \\\n"
                "      --max-model-len 2048 --dtype bfloat16 --port 8000\n"
            )
            raise SystemExit(1)

    # ── Load prompts ──────────────────────────────────────────────────────────
    all_prompts: dict[str, list[dict]] = {}
    for category in args.categories:
        all_prompts[category] = load_prompts(raw_dir, category)

    total_prompts = sum(len(v) for v in all_prompts.values())
    log.info(f"Total prompts to process: {total_prompts} across {args.categories}")

    if args.dry_run:
        log.info("Dry run complete — no inference was run.")
        for cat, prompts in all_prompts.items():
            log.info(f"  {cat}: {len(prompts)} prompts")
        return

    # ── Run inference ─────────────────────────────────────────────────────────
    t_start = time.monotonic()
    category_stats = []

    for category in args.categories:
        stats = await run_category(
            prompts      = all_prompts[category],
            category     = category,
            teacher_url  = args.teacher_url,
            model_name   = args.model_name,
            generated_dir= generated_dir,
            batch_size   = args.batch_size,
            temperature  = args.temperature,
            max_tokens   = args.max_tokens,
            resume       = args.resume,
        )
        category_stats.append(stats)
        log.info(
            f"[{category}] Done — {stats['success']}/{stats['total']} succeeded, "
            f"{stats['failures']} failed."
        )
        if stats["failures"] > 0:
            fail_path = generated_dir / f"failed_{category}.jsonl"
            log.warning(f"  Failed prompts → {fail_path}  (re-run with --resume to retry)")

    elapsed = time.monotonic() - t_start

    # ── Summary ───────────────────────────────────────────────────────────────
    total_ok   = sum(s["success"]  for s in category_stats)
    total_fail = sum(s["failures"] for s in category_stats)
    log.info(
        f"\n{'='*55}\n"
        f"  Inference complete in {elapsed:.1f}s\n"
        f"  Success : {total_ok}/{total_prompts}\n"
        f"  Failures: {total_fail}\n"
        f"  Output  : {generated_dir.resolve()}\n"
        f"{'='*55}"
    )

    save_run_metadata(args, category_stats, elapsed, generated_dir)


if __name__ == "__main__":
    asyncio.run(main())