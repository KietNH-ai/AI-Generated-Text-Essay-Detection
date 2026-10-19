#!/usr/bin/env python3
"""RQ3 external benchmark generator.

Generate AI-written essays from multiple providers and keep an audit trail suitable
for an unseen-LLM generalization experiment.

Features
--------
- JSON-driven provider/model/prompt configuration
- Smoke-test default: 5 essays per enabled model
- OpenAI Responses API, Anthropic Messages API, Google GenAI SDK
- Retry with exponential backoff + jitter
- Per-provider requests-per-minute throttling
- Error handling without aborting the whole run
- Checkpoint CSV after every job; safe resume of successful jobs
- Raw provider response stored both in CSV and JSONL
- Token usage, latency, request metadata, prompt metadata, generation settings
- Dry-run mode that performs zero API calls

Environment variables
---------------------
OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY

Example
-------
python generate_rq3_essays.py --config rq3_generation_config.json --dry-run
python generate_rq3_essays.py --config rq3_generation_config.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    from dotenv import load_dotenv
except ImportError:  # .env support is convenient, not mandatory
    def load_dotenv(*args: Any, **kwargs: Any) -> bool:  # type: ignore
        return False


CSV_FIELDS = [
    "run_id",
    "job_key",
    "sample_id",
    "replicate_idx",
    "provider",
    "model_family",
    "model",
    "prompt_id",
    "prompt_name",
    "prompt_text",
    "generation_prompt",
    "label",
    "temperature_requested",
    "max_output_tokens_requested",
    "requests_per_minute",
    "created_at_utc",
    "latency_sec",
    "attempt_count",
    "success",
    "response_id",
    "finish_reason",
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "word_count",
    "char_count",
    "response_text",
    "raw_response_json",
    "error_type",
    "error_message",
    "http_status",
]


@dataclass(frozen=True)
class Job:
    provider: str
    model_family: str
    model: str
    model_cfg: Dict[str, Any]
    prompt: Dict[str, Any]
    replicate_idx: int
    job_key: str
    sample_id: str


class RateLimiter:
    """Simple sequential per-provider minimum-interval limiter."""

    def __init__(self, requests_per_minute: float) -> None:
        rpm = max(float(requests_per_minute), 0.01)
        self.min_interval = 60.0 / rpm
        self.last_request_at: Optional[float] = None

    def wait(self) -> None:
        if self.last_request_at is not None:
            elapsed = time.monotonic() - self.last_request_at
            remaining = self.min_interval - elapsed
            if remaining > 0:
                time.sleep(remaining)
        self.last_request_at = time.monotonic()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_short_hash(value: str, length: int = 12) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:length]


def detect_project_root(explicit_root: Optional[str] = None) -> Path:
    """Find the repository root when called from a notebook or a terminal.

    Typical supported layouts:
      repo/
        notebooks/08_RQ3_generate_api.ipynb
        generate_rq3_essays.py
        rq3_generation_config.json
        data/

    or notebooks stored directly in repo/.
    """
    if explicit_root:
        return Path(explicit_root).expanduser().resolve()

    cwd = Path.cwd().resolve()

    # Common case: notebook is inside repo/notebooks/.
    if cwd.name.lower() in {"notebook", "notebooks"}:
        return cwd.parent

    # If current directory already looks like the project root, keep it.
    project_markers = (
        "rq3_generation_config.json",
        "requirements.txt",
        "src",
        "data",
    )
    if any((cwd / marker).exists() for marker in project_markers):
        return cwd

    # Walk upward and choose the nearest directory with project markers.
    for parent in cwd.parents:
        if any((parent / marker).exists() for marker in project_markers):
            return parent

    # Fallback: current directory.
    return cwd


def resolve_from_project(path_value: str | Path, project_root: Path) -> Path:
    """Resolve a relative path against the repository root."""
    path = Path(path_value).expanduser()
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def safe_json(value: Any) -> str:
    """Best-effort serialization for SDK response objects."""
    if value is None:
        return "null"
    try:
        if hasattr(value, "model_dump_json"):
            dumped = value.model_dump_json(exclude_none=True)
            if isinstance(dumped, str):
                return dumped
    except TypeError:
        try:
            dumped = value.model_dump_json()
            if isinstance(dumped, str):
                return dumped
        except Exception:
            pass
    except Exception:
        pass

    try:
        if hasattr(value, "model_dump"):
            value = value.model_dump(exclude_none=True)
        elif hasattr(value, "to_dict"):
            value = value.to_dict()
        return json.dumps(value, ensure_ascii=False, default=str)
    except Exception:
        return json.dumps({"repr": repr(value)}, ensure_ascii=False)


def get_nested(obj: Any, *path: str, default: Any = None) -> Any:
    current = obj
    for key in path:
        if current is None:
            return default
        if isinstance(current, dict):
            current = current.get(key, default)
        else:
            current = getattr(current, key, default)
    return current


def extract_http_status(exc: Exception) -> Optional[int]:
    for attr in ("status_code", "status"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value
    response = getattr(exc, "response", None)
    value = getattr(response, "status_code", None)
    return value if isinstance(value, int) else None


def retry_after_seconds(exc: Exception) -> Optional[float]:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers:
        value = headers.get("retry-after") or headers.get("Retry-After")
        if value:
            try:
                return max(float(value), 0.0)
            except (TypeError, ValueError):
                return None
    return None


def permanent_limit_reason(exc: Exception) -> Optional[str]:
    """Return a reason when retrying now cannot fix the request.

    Failed jobs are still written to CSV, so --resume can retry them on a later run
    after billing/quota has been reset or changed.
    """
    message = str(exc).lower()

    billing_markers = (
        "credit_balance_exhausted",
        "organization_usage_limit_exceeded",
        "organization_spend_limit_exceeded",
        "project_spend_limit_exceeded",
        "insufficient_quota",
    )
    if any(marker in message for marker in billing_markers):
        return "billing_or_spend_limit"

    # Strong signals for a daily quota. Do not classify a generic 429 as daily:
    # minute-level rate limits should still use exponential backoff/retry.
    daily_markers = (
        "requests per day",
        "request per day",
        "per-day",
        "per day",
        "daily quota",
        "rpd",
    )
    if any(marker in message for marker in daily_markers):
        return "daily_quota_exhausted"

    status = extract_http_status(exc)
    if status in {401, 403}:
        return "authentication_or_permission"

    return None


def is_retryable(exc: Exception) -> bool:
    # Billing exhaustion and daily quota exhaustion will not recover after a few
    # seconds, so avoid wasting all retry attempts on them.
    if permanent_limit_reason(exc) is not None:
        return False

    status = extract_http_status(exc)
    if status is not None:
        if status in {408, 409, 425, 429} or status >= 500:
            return True
        if status in {400, 401, 403, 404, 405, 422}:
            return False

    name = exc.__class__.__name__.lower()
    message = str(exc).lower()
    transient_markers = (
        "timeout", "timed out", "connection", "rate limit", "ratelimit",
        "temporarily", "overloaded", "unavailable", "internal server",
    )
    return any(marker in name or marker in message for marker in transient_markers)


def normalize_finish_reason(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def call_openai(model_cfg: Dict[str, Any], prompt: str) -> Dict[str, Any]:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("Missing package 'openai'. Install with: pip install openai") from exc

    client = OpenAI(api_key=os.getenv(model_cfg.get("api_key_env", "OPENAI_API_KEY")))
    kwargs: Dict[str, Any] = {
        "model": model_cfg["model"],
        "input": prompt,
        "max_output_tokens": int(model_cfg.get("max_output_tokens", 1200)),
    }
    # Some model families do not expose temperature. Only pass it when explicitly set.
    if model_cfg.get("temperature") is not None:
        kwargs["temperature"] = float(model_cfg["temperature"])

    response = client.responses.create(**kwargs)
    text = (getattr(response, "output_text", "") or "").strip()

    usage = getattr(response, "usage", None)
    input_tokens = get_nested(usage, "input_tokens")
    output_tokens = get_nested(usage, "output_tokens")
    total_tokens = get_nested(usage, "total_tokens")
    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens

    finish_reason = ""
    output = getattr(response, "output", None) or []
    if output:
        # Responses API details can vary by SDK version; keep this best-effort.
        finish_reason = normalize_finish_reason(getattr(output[-1], "status", None))

    return {
        "text": text,
        "raw": response,
        "response_id": str(getattr(response, "id", "") or ""),
        "finish_reason": finish_reason,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }


def call_anthropic(model_cfg: Dict[str, Any], prompt: str) -> Dict[str, Any]:
    try:
        import anthropic
    except ImportError as exc:
        raise RuntimeError("Missing package 'anthropic'. Install with: pip install anthropic") from exc

    client = anthropic.Anthropic(
        api_key=os.getenv(model_cfg.get("api_key_env", "ANTHROPIC_API_KEY"))
    )
    kwargs: Dict[str, Any] = {
        "model": model_cfg["model"],
        "max_tokens": int(model_cfg.get("max_output_tokens", 1200)),
        "messages": [{"role": "user", "content": prompt}],
    }
    # Current Anthropic SDK/model generations may reject legacy sampling
    # parameters such as temperature/top_p/top_k. Omit them and control style
    # through the generation prompt instead.
    response = client.messages.create(**kwargs)
    text_parts: List[str] = []
    for block in getattr(response, "content", []) or []:
        if getattr(block, "type", None) == "text":
            text_parts.append(getattr(block, "text", ""))
    text = "\n".join(part for part in text_parts if part).strip()

    usage = getattr(response, "usage", None)
    input_tokens = get_nested(usage, "input_tokens")
    output_tokens = get_nested(usage, "output_tokens")
    total_tokens = None
    if input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens

    return {
        "text": text,
        "raw": response,
        "response_id": str(getattr(response, "id", "") or ""),
        "finish_reason": normalize_finish_reason(getattr(response, "stop_reason", None)),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }


def call_google(model_cfg: Dict[str, Any], prompt: str) -> Dict[str, Any]:
    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise RuntimeError("Missing package 'google-genai'. Install with: pip install google-genai") from exc

    api_key = os.getenv(model_cfg.get("api_key_env", "GEMINI_API_KEY"))
    client = genai.Client(api_key=api_key)

    config_kwargs: Dict[str, Any] = {
        "max_output_tokens": int(model_cfg.get("max_output_tokens", 1200)),
    }
    if model_cfg.get("temperature") is not None:
        config_kwargs["temperature"] = float(model_cfg["temperature"])

    response = client.models.generate_content(
        model=model_cfg["model"],
        contents=prompt,
        config=types.GenerateContentConfig(**config_kwargs),
    )
    text = (getattr(response, "text", "") or "").strip()

    usage = getattr(response, "usage_metadata", None)
    input_tokens = get_nested(usage, "prompt_token_count")
    output_tokens = get_nested(usage, "candidates_token_count")
    total_tokens = get_nested(usage, "total_token_count")

    finish_reason = ""
    candidates = getattr(response, "candidates", None) or []
    if candidates:
        finish_reason = normalize_finish_reason(getattr(candidates[0], "finish_reason", None))

    response_id = (
        getattr(response, "response_id", None)
        or getattr(response, "id", None)
        or ""
    )
    return {
        "text": text,
        "raw": response,
        "response_id": str(response_id),
        "finish_reason": finish_reason,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }


PROVIDER_CALLS = {
    "openai": call_openai,
    "anthropic": call_anthropic,
    "google": call_google,
}


def validate_api_key(model_cfg: Dict[str, Any]) -> None:
    env_name = model_cfg.get("api_key_env")
    if not env_name:
        defaults = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "google": "GEMINI_API_KEY",
        }
        env_name = defaults.get(model_cfg["provider"])
    if not env_name or not os.getenv(env_name):
        raise RuntimeError(f"Missing API key environment variable: {env_name}")


def generate_with_retry(
    model_cfg: Dict[str, Any],
    prompt: str,
    limiter: RateLimiter,
    retry_cfg: Dict[str, Any],
) -> Tuple[Dict[str, Any], int, float]:
    provider = model_cfg["provider"].lower()
    caller = PROVIDER_CALLS.get(provider)
    if caller is None:
        raise ValueError(f"Unsupported provider: {provider}")

    max_attempts = max(int(retry_cfg.get("max_attempts", 5)), 1)
    base_delay = max(float(retry_cfg.get("base_delay_seconds", 2.0)), 0.0)
    max_delay = max(float(retry_cfg.get("max_delay_seconds", 30.0)), base_delay)
    jitter = max(float(retry_cfg.get("jitter_seconds", 1.0)), 0.0)

    last_exc: Optional[Exception] = None
    total_started = time.monotonic()

    for attempt in range(1, max_attempts + 1):
        try:
            limiter.wait()
            result = caller(model_cfg, prompt)
            if not result.get("text", "").strip():
                raise RuntimeError("Provider returned an empty text response")
            return result, attempt, time.monotonic() - total_started
        except Exception as exc:
            last_exc = exc
            retryable = is_retryable(exc)
            if attempt >= max_attempts or not retryable:
                break

            retry_after = retry_after_seconds(exc)
            if retry_after is not None:
                delay = min(max(retry_after, base_delay), max_delay)
            else:
                delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
                delay += random.uniform(0.0, jitter)
            print(
                f"    retry {attempt}/{max_attempts - 1} after {delay:.1f}s "
                f"({exc.__class__.__name__}: {str(exc)[:160]})",
                file=sys.stderr,
            )
            time.sleep(delay)

    assert last_exc is not None
    setattr(last_exc, "_rq3_attempts", max_attempts)
    setattr(last_exc, "_rq3_elapsed", time.monotonic() - total_started)
    raise last_exc


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    if "models" not in cfg or not isinstance(cfg["models"], list):
        raise ValueError("Config must contain a 'models' list")
    if "prompts" not in cfg or not isinstance(cfg["prompts"], list):
        raise ValueError("Config must contain a 'prompts' list")
    return cfg


def validate_config(cfg: Dict[str, Any]) -> None:
    enabled_models = [m for m in cfg["models"] if m.get("enabled", True)]
    if not enabled_models:
        raise ValueError("No enabled models in config")
    if not cfg["prompts"]:
        raise ValueError("No prompts in config")

    prompt_ids = set()
    for p in cfg["prompts"]:
        if not p.get("prompt_id"):
            raise ValueError("Every prompt needs a non-empty prompt_id")
        if p["prompt_id"] in prompt_ids:
            raise ValueError(f"Duplicate prompt_id: {p['prompt_id']}")
        prompt_ids.add(p["prompt_id"])
        if not (p.get("prompt_text") or p.get("prompt_name")):
            raise ValueError(f"Prompt {p['prompt_id']} needs prompt_text or prompt_name")

    for m in enabled_models:
        provider = str(m.get("provider", "")).lower()
        if provider not in PROVIDER_CALLS:
            raise ValueError(f"Unsupported provider in config: {provider}")
        for key in ("model", "model_family"):
            if not m.get(key):
                raise ValueError(f"Model config needs '{key}': {m}")


def render_generation_prompt(cfg: Dict[str, Any], prompt: Dict[str, Any]) -> str:
    template = cfg.get(
        "generation_prompt_template",
        "Write an essay responding to the following task:\n\n{prompt_text}\n\nReturn only the essay.",
    )
    prompt_text = prompt.get("prompt_text") or prompt.get("prompt_name") or ""
    values = {
        "prompt_id": prompt.get("prompt_id", ""),
        "prompt_name": prompt.get("prompt_name", ""),
        "prompt_text": prompt_text,
        "min_words": cfg.get("min_words", 400),
        "max_words": cfg.get("max_words", 600),
    }
    try:
        return template.format(**values)
    except KeyError as exc:
        raise ValueError(f"Unknown placeholder in generation_prompt_template: {exc}") from exc


def build_jobs(cfg: Dict[str, Any], samples_override: Optional[int] = None) -> List[Job]:
    # Full RQ3 design: every enabled model generates N essays for EVERY prompt.
    # Preferred config key: samples_per_prompt_per_model.
    # samples_per_model is retained as a backward-compatible fallback.
    samples_per_prompt = int(
        samples_override
        if samples_override is not None
        else cfg.get("samples_per_prompt_per_model", cfg.get("samples_per_model", 5))
    )
    if samples_per_prompt < 1:
        raise ValueError("samples_per_prompt_per_model must be >= 1")

    prompts = list(cfg["prompts"])
    jobs: List[Job] = []

    for model_cfg in cfg["models"]:
        if not model_cfg.get("enabled", True):
            continue
        provider = str(model_cfg["provider"]).lower()
        model = str(model_cfg["model"])
        family = str(model_cfg["model_family"])

        # Iterate over every configured prompt. replicate_idx makes each generation
        # independently resumable via a stable (provider, model, prompt, replicate) key.
        for p in prompts:
            for replicate_idx in range(1, samples_per_prompt + 1):
                identity = f"{provider}|{model}|{p['prompt_id']}|{replicate_idx}"
                job_key = stable_short_hash(identity, 20)
                sample_id = f"rq3_{provider}_{stable_short_hash(model, 6)}_{job_key[:10]}"
                jobs.append(
                    Job(
                        provider=provider,
                        model_family=family,
                        model=model,
                        model_cfg=model_cfg,
                        prompt=p,
                        replicate_idx=replicate_idx,
                        job_key=job_key,
                        sample_id=sample_id,
                    )
                )
    return jobs


def read_successful_job_keys(csv_path: Path) -> set[str]:
    keys: set[str] = set()
    if not csv_path.exists():
        return keys
    try:
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                if str(row.get("success", "")).lower() == "true" and row.get("job_key"):
                    keys.add(row["job_key"])
    except Exception as exc:
        print(f"Warning: could not read resume CSV: {exc}", file=sys.stderr)
    return keys


def append_csv_row(csv_path: Path, row: Dict[str, Any]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with csv_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})
        f.flush()
        os.fsync(f.fileno())


def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        f.flush()
        os.fsync(f.fileno())


def make_base_row(run_id: str, job: Job, cfg: Dict[str, Any], generation_prompt: str) -> Dict[str, Any]:
    p = job.prompt
    m = job.model_cfg
    return {
        "run_id": run_id,
        "job_key": job.job_key,
        "sample_id": job.sample_id,
        "replicate_idx": job.replicate_idx,
        "provider": job.provider,
        "model_family": job.model_family,
        "model": job.model,
        "prompt_id": p.get("prompt_id", ""),
        "prompt_name": p.get("prompt_name", ""),
        "prompt_text": p.get("prompt_text") or p.get("prompt_name") or "",
        "generation_prompt": generation_prompt,
        "label": 1,
        "temperature_requested": "" if m.get("temperature") is None else m.get("temperature"),
        "max_output_tokens_requested": m.get("max_output_tokens", 1200),
        "requests_per_minute": m.get("requests_per_minute", cfg.get("default_requests_per_minute", 10)),
        "created_at_utc": utc_now_iso(),
    }


def run(cfg: Dict[str, Any], args: argparse.Namespace, project_root: Path) -> int:
    validate_config(cfg)
    jobs = build_jobs(cfg, args.samples_per_model)
    output_cfg = cfg.get("output", {})
    csv_path = resolve_from_project(
        args.output_csv or output_cfg.get("csv", "data/rq3/generated_essays_smoketest.csv"),
        project_root,
    )
    jsonl_path = resolve_from_project(
        output_cfg.get("raw_jsonl", "data/rq3/generated_essays_raw.jsonl"),
        project_root,
    )

    run_id = args.run_id or f"rq3_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"

    print(f"Project root: {project_root}")
    print(f"Run ID: {run_id}")
    print(f"Models: {len([m for m in cfg['models'] if m.get('enabled', True)])}")
    print(f"Jobs: {len(jobs)}")
    print(f"CSV: {csv_path}")
    print(f"Raw JSONL: {jsonl_path}")

    if args.dry_run:
        print("\nDRY RUN — no API calls will be made.")
        enabled_models = [m for m in cfg["models"] if m.get("enabled", True)]
        samples_per_prompt = int(
            args.samples_per_model
            if args.samples_per_model is not None
            else cfg.get("samples_per_prompt_per_model", cfg.get("samples_per_model", 5))
        )
        print(f"Prompts: {len(cfg['prompts'])}")
        print(f"Enabled models: {len(enabled_models)}")
        print(f"Essays per prompt per model: {samples_per_prompt}")
        print(f"Total planned jobs: {len(jobs)}")
        print("\nPlanned jobs (first 30):")
        for i, job in enumerate(jobs[:30], 1):
            print(
                f"{i:03d}. {job.provider:9s} | {job.model:24s} | "
                f"{job.prompt['prompt_id']} | rep={job.replicate_idx} | "
                f"{job.prompt.get('prompt_name', '')}"
            )
        if len(jobs) > 30:
            print(f"... {len(jobs) - 30} additional jobs omitted from display.")
        return 0

    # Load .env only for a real run. Never write secrets to outputs.
    env_path = resolve_from_project(args.env_file, project_root) if args.env_file else None
    load_dotenv(env_path)

    successful_keys = read_successful_job_keys(csv_path) if args.resume else set()
    limiters: Dict[str, RateLimiter] = {}
    retry_cfg = cfg.get("retry", {})

    success_count = 0
    fail_count = 0
    skip_count = 0
    blocked_targets: Dict[str, str] = {}

    for idx, job in enumerate(jobs, 1):
        if job.job_key in successful_keys:
            skip_count += 1
            print(f"[{idx}/{len(jobs)}] SKIP success exists: {job.provider}/{job.model}/{job.prompt['prompt_id']}")
            continue

        target_key = f"{job.provider}:{job.model}"
        if target_key in blocked_targets:
            skip_count += 1
            print(
                f"[{idx}/{len(jobs)}] SKIP blocked for this run: "
                f"{job.provider}/{job.model} ({blocked_targets[target_key]})"
            )
            continue

        generation_prompt = render_generation_prompt(cfg, job.prompt)
        row = make_base_row(run_id, job, cfg, generation_prompt)
        print(f"[{idx}/{len(jobs)}] {job.provider}/{job.model} <- {job.prompt['prompt_id']}")

        rpm = float(row["requests_per_minute"])
        limiter_key = f"{job.provider}:{rpm}"
        limiter = limiters.setdefault(limiter_key, RateLimiter(rpm))

        raw_request = {
            "provider": job.provider,
            "model": job.model,
            "model_family": job.model_family,
            "prompt_id": job.prompt.get("prompt_id"),
            "generation_prompt": generation_prompt,
            "temperature_requested": row["temperature_requested"],
            "max_output_tokens_requested": row["max_output_tokens_requested"],
        }

        started = time.monotonic()
        try:
            validate_api_key(job.model_cfg)
            result, attempts, elapsed = generate_with_retry(
                job.model_cfg, generation_prompt, limiter, retry_cfg
            )
            response_text = result["text"].strip()
            raw_response_json = safe_json(result.get("raw"))

            row.update(
                {
                    "latency_sec": round(elapsed, 4),
                    "attempt_count": attempts,
                    "success": True,
                    "response_id": result.get("response_id", ""),
                    "finish_reason": result.get("finish_reason", ""),
                    "input_tokens": result.get("input_tokens", ""),
                    "output_tokens": result.get("output_tokens", ""),
                    "total_tokens": result.get("total_tokens", ""),
                    "word_count": len(response_text.split()),
                    "char_count": len(response_text),
                    "response_text": response_text,
                    "raw_response_json": raw_response_json,
                    "error_type": "",
                    "error_message": "",
                    "http_status": "",
                }
            )
            append_csv_row(csv_path, row)
            append_jsonl(
                jsonl_path,
                {
                    "run_id": run_id,
                    "job_key": job.job_key,
                    "sample_id": job.sample_id,
                    "created_at_utc": row["created_at_utc"],
                    "request": raw_request,
                    "success": True,
                    "response": json.loads(raw_response_json),
                },
            )
            success_count += 1
            print(f"    OK: {row['word_count']} words, {row['latency_sec']}s")

        except Exception as exc:
            attempts = int(getattr(exc, "_rq3_attempts", 1))
            elapsed = float(getattr(exc, "_rq3_elapsed", time.monotonic() - started))
            status = extract_http_status(exc)
            error_message = str(exc)
            row.update(
                {
                    "latency_sec": round(elapsed, 4),
                    "attempt_count": attempts,
                    "success": False,
                    "response_id": "",
                    "finish_reason": "",
                    "input_tokens": "",
                    "output_tokens": "",
                    "total_tokens": "",
                    "word_count": 0,
                    "char_count": 0,
                    "response_text": "",
                    "raw_response_json": "",
                    "error_type": exc.__class__.__name__,
                    "error_message": error_message[:2000],
                    "http_status": "" if status is None else status,
                }
            )
            append_csv_row(csv_path, row)
            append_jsonl(
                jsonl_path,
                {
                    "run_id": run_id,
                    "job_key": job.job_key,
                    "sample_id": job.sample_id,
                    "created_at_utc": row["created_at_utc"],
                    "request": raw_request,
                    "success": False,
                    "error": {
                        "type": exc.__class__.__name__,
                        "message": error_message,
                        "http_status": status,
                        "traceback": traceback.format_exc(limit=8),
                    },
                },
            )
            fail_count += 1
            print(f"    ERROR: {exc.__class__.__name__}: {error_message[:240]}", file=sys.stderr)

            permanent_reason = permanent_limit_reason(exc)
            if permanent_reason is not None:
                blocked_targets[target_key] = permanent_reason
                print(
                    f"    BLOCK {job.provider}/{job.model} for the remainder of this run: "
                    f"{permanent_reason}. Failed jobs remain eligible for --resume later.",
                    file=sys.stderr,
                )

            if args.fail_fast:
                print("Fail-fast enabled; stopping.", file=sys.stderr)
                return 2

    print("\nSummary")
    print(f"  success: {success_count}")
    print(f"  failed : {fail_count}")
    print(f"  skipped: {skip_count}")
    print(f"  CSV    : {csv_path}")
    print(f"  JSONL  : {jsonl_path}")
    return 0 if fail_count == 0 else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate RQ3 external AI-essay benchmark")
    parser.add_argument("--config", default="rq3_generation_config.json", help="Path to JSON config, relative to project root")
    parser.add_argument(
        "--project-root",
        default=None,
        help="Project root. If omitted, auto-detects repo root from the current notebook/working directory",
    )
    parser.add_argument("--env-file", default=".env", help="Optional .env file")
    parser.add_argument("--output-csv", default=None, help="Override CSV path from config")
    parser.add_argument("--samples-per-model", "--samples-per-prompt", dest="samples_per_model", type=int, default=None, help="Override essays per prompt per model (backward-compatible option name: --samples-per-model)")
    parser.add_argument("--run-id", default=None, help="Optional fixed run ID")
    parser.add_argument("--dry-run", action="store_true", help="Plan jobs only; make zero API calls")
    parser.add_argument("--resume", dest="resume", action="store_true", help="Skip prior successful jobs in the output CSV (default)")
    parser.add_argument("--no-resume", dest="resume", action="store_false", help="Do not skip prior successful jobs")
    parser.set_defaults(resume=True)
    parser.add_argument("--fail-fast", action="store_true", help="Stop after first failed job")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = detect_project_root(args.project_root)
    cfg_path = resolve_from_project(args.config, project_root)
    if not cfg_path.exists():
        print(f"Config not found: {cfg_path}", file=sys.stderr)
        print(f"Detected project root: {project_root}", file=sys.stderr)
        return 2
    try:
        cfg = load_config(cfg_path)
        return run(cfg, args, project_root)
    except KeyboardInterrupt:
        print("Interrupted by user.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"Fatal error: {exc.__class__.__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
