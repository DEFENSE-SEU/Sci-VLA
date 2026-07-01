import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from openai import OpenAI

from transition_generation import (
    _normalize_backend_mode,
    _normalize_thinking_mode,
    _request_json_object,
    file_to_data_url,
)


PROMPT_ACTIONS = {"advance", "retry", "fail_episode"}


def _coerce_bool(value: Any, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "1", "success", "succeeded"}:
            return True
        if normalized in {"false", "no", "0", "failure", "failed"}:
            return False
    if isinstance(value, (int, float)) and value in {0, 1}:
        return bool(value)
    raise ValueError(f"{field_name} is missing or not coercible to bool: {value!r}")


def _coerce_confidence(value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError(f"confidence must be numeric, got {value!r}")
    try:
        confidence = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"confidence must be numeric, got {value!r}") from exc
    if not 0.0 <= confidence <= 1.0:
        raise ValueError(f"confidence must be within [0, 1], got {confidence}")
    return confidence


def normalize_judgement_result(raw: dict, threshold: float = 0.6) -> dict:
    if not isinstance(raw, dict):
        raise ValueError(f"judge result must be a JSON object, got {type(raw).__name__}")
    if "success" not in raw:
        raise ValueError("judge result missing success")
    if "confidence" not in raw:
        raise ValueError("judge result missing confidence")

    success = _coerce_bool(raw.get("success"), "success")
    confidence = _coerce_confidence(raw.get("confidence"))
    reason = str(raw.get("reason") or "").strip()
    failure_mode = raw.get("failure_mode")
    if failure_mode is not None:
        failure_mode = str(failure_mode).strip() or None

    return {
        "success": success,
        "confidence": confidence,
        "reason": reason,
        "failure_mode": failure_mode,
        "prompt_success": bool(success and confidence >= float(threshold)),
        "raw": raw,
    }


def decide_prompt_action(
    *,
    prompt_success: bool,
    attempt_index: int,
    max_prompt_retries: int,
    is_final_prompt: bool,
) -> str:
    if prompt_success:
        return "advance"
    if attempt_index < max(0, int(max_prompt_retries)):
        return "retry"
    return "fail_episode"


def resolve_judge_error_success(judge_on_error: str) -> bool:
    mode = (judge_on_error or "fail").strip().lower()
    if mode == "fail":
        return False
    if mode == "pass":
        return True
    raise ValueError(f"judge_on_error must be 'fail' or 'pass', got {judge_on_error!r}")


def select_transition_target_prompt(
    prompts: list[str],
    prompt_index: int,
    action: str,
) -> str | None:
    if action not in PROMPT_ACTIONS:
        raise ValueError(f"Unsupported prompt action: {action}")
    if action == "retry":
        return prompts[prompt_index]
    if action == "advance" and prompt_index + 1 < len(prompts):
        return prompts[prompt_index + 1]
    return None


def append_judgement_log(record: dict, log_path: str | Path = "logs/task_judgements.jsonl") -> None:
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = dict(record)
    out.setdefault("timestamp", datetime.now().isoformat(timespec="seconds"))
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(out, ensure_ascii=False, default=str) + "\n")


def _build_judge_prompt(prompt: str) -> str:
    return f"""
You are a strict visual task-success judge for a robot laboratory manipulation task.

Task prompt:
{prompt}

Use the two end-state images to decide whether this prompt has been successfully completed.
Judge only the current prompt, not the whole long-horizon task.

Return strictly one JSON object with this exact schema:
{{
  "success": true,
  "confidence": 0.0,
  "reason": "short visual evidence",
  "failure_mode": null
}}

Rules:
1. success must be true only if the visible final state satisfies the task prompt.
2. confidence must be a number from 0 to 1.
3. failure_mode must be null on success, otherwise a short phrase.
4. Do not output markdown or extra text.
""".strip()


def _llm_config_value(llm_config: dict | None, key: str, default=None, env_key: str | None = None):
    value = None
    if llm_config:
        value = llm_config.get(key)
    if value is None or value == "":
        if env_key:
            value = os.environ.get(env_key)
    if value is None or value == "":
        return default
    return value


def judge_task_success(
    *,
    prompt: str,
    front_image_path: str | Path,
    side_image_path: str | Path,
    llm_config: dict | None = None,
    threshold: float = 0.6,
    client: OpenAI | None = None,
) -> dict:
    model_name = _llm_config_value(llm_config, "model_name", env_key="MODEL_NAME")
    if not model_name:
        raise ValueError(
            "llm_config['model_name'] or MODEL_NAME is required when task judge is enabled"
        )

    base_url = _llm_config_value(llm_config, "base_url", env_key="BASE_URL")
    api_key = _llm_config_value(llm_config, "api_key", env_key="API_KEY") or "EMPTY"
    temperature = _llm_config_value(llm_config, "temperature")
    top_p = _llm_config_value(llm_config, "top_p")
    max_tokens = _llm_config_value(llm_config, "max_tokens")
    max_attempts = int(_llm_config_value(llm_config, "max_attempts", 3) or 3)
    timeout = _llm_config_value(llm_config, "timeout")
    backend_mode = _normalize_backend_mode(_llm_config_value(llm_config, "backend_mode", "auto"))
    thinking_mode = _normalize_thinking_mode(_llm_config_value(llm_config, "thinking", "auto"))

    if client is None:
        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        client = OpenAI(**client_kwargs)

    request_input = [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": _build_judge_prompt(prompt)},
                {"type": "input_image", "image_url": file_to_data_url(str(front_image_path))},
                {"type": "input_image", "image_url": file_to_data_url(str(side_image_path))},
            ],
        }
    ]
    raw = _request_json_object(
        client=client,
        model_name=model_name,
        request_input=request_input,
        stage_name="task-judge",
        max_attempts=max_attempts,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        timeout=timeout,
        backend_mode=backend_mode,
        thinking_mode=thinking_mode,
    )
    return normalize_judgement_result(raw, threshold=threshold)
