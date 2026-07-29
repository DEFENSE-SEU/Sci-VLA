import argparse
import base64
import io
import json
import mimetypes
import os
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np


def _scalar_to_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return None
        return int(value.reshape(-1)[0])
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return None
        return int(value[0])
    return int(value)


def _extract_prompt(sample: dict[str, Any], tasks_map: dict[int, str]) -> str:
    for key in ("prompt", "task"):
        if key in sample and sample[key] is not None:
            return str(sample[key])

    if "task_index" in sample and sample["task_index"] is not None:
        task_index = _scalar_to_int(sample["task_index"])
        if task_index is not None and task_index in tasks_map:
            return tasks_map[task_index]

    raise ValueError("Cannot resolve task prompt from sample; expected prompt/task/task_index")


def _extract_state(sample: dict[str, Any]) -> list[float]:
    if "state" not in sample or sample["state"] is None:
        raise ValueError('Sample does not contain "state" field')
    return np.asarray(sample["state"], dtype=np.float64).reshape(-1).tolist()


def _extract_front_image(sample: dict[str, Any], front_image_key: str) -> Any | None:
    for key in (front_image_key, "observation/image", "image"):
        if key in sample and sample[key] is not None:
            return sample[key]
    return None


def _as_uint8_image(image: Any) -> np.ndarray:
    if hasattr(image, "detach") and hasattr(image, "cpu"):
        image = image.detach().cpu().numpy()
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = arr[..., None]
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
        arr = np.moveaxis(arr, 0, -1)
    if arr.ndim != 3 or arr.shape[-1] not in (1, 3, 4):
        raise ValueError(f"Unsupported front image shape: {arr.shape}")
    if np.issubdtype(arr.dtype, np.floating):
        max_value = float(np.nanmax(arr)) if arr.size else 0.0
        if max_value <= 1.0:
            arr = arr * 255.0
    return np.clip(arr, 0, 255).astype(np.uint8)


def _to_int_list(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        return np.asarray(value).reshape(-1).astype(np.int64).tolist()
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        return value.detach().cpu().numpy().reshape(-1).astype(np.int64).tolist()
    if isinstance(value, (list, tuple)):
        return [int(v) for v in value]
    return [int(value)]


def file_to_data_url(
    path: str,
    *,
    max_image_side: int | None = None,
    image_quality: int = 80,
) -> str:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Image not found: {p.resolve()}")

    if max_image_side is not None and int(max_image_side) > 0:
        try:
            from PIL import Image

            with Image.open(p) as image:
                image = image.convert("RGB")
                image.thumbnail((int(max_image_side), int(max_image_side)))
                buffer = io.BytesIO()
                quality = min(95, max(1, int(image_quality)))
                image.save(buffer, format="JPEG", quality=quality, optimize=True)
            b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
            return f"data:image/jpeg;base64,{b64}"
        except ImportError:
            pass

    mime, _ = mimetypes.guess_type(str(p))
    if mime is None:
        mime = "image/png"
    b64 = base64.b64encode(p.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


@dataclass
class ReadyPairJudgement:
    b_is_target_state: bool
    a_exceeded_ready: bool
    b_exceeded_ready: bool
    confidence: float
    reason: str
    raw: dict[str, Any]
    target_object: str | None = None
    operation_region: str | None = None
    expected_pre_contact_relation: str | None = None
    a_analysis: dict[str, Any] | None = None
    b_analysis: dict[str, Any] | None = None


@dataclass
class WindowSearchResult:
    selected_index: int
    judgement: ReadyPairJudgement | None
    history: list[dict[str, Any]]
    fallback_to_initial_frame: bool
    fallback_reason: str | None = None


def _coerce_bool(value: Any, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "1", "target", "ready"}:
            return True
        if normalized in {"false", "no", "0", "not_target", "not ready"}:
            return False
    if isinstance(value, (int, float)) and value in {0, 1}:
        return bool(value)
    raise ValueError(f"{field_name} is not coercible to bool: {value!r}")


def _optional_bool(value: Any, field_name: str) -> bool | None:
    if value is None:
        return None
    return _coerce_bool(value, field_name)


def _infer_exceeded_from_analysis(analysis: Any) -> bool | None:
    if not isinstance(analysis, dict):
        return None
    gripper_state = str(analysis.get("gripper_state") or "").strip().lower()
    target_state = str(analysis.get("target_object_state") or "").strip().lower()
    relation = str(analysis.get("gripper_target_relation") or "").strip().lower()

    exceeded_gripper = {
        "not_fully_open",
        "partially_open",
        "closing",
        "closed",
        "holding_object",
    }
    exceeded_target = {"contacted", "grasped_or_moved", "task_state_changed"}
    exceeded_relation = {"too_close", "contact_risk", "in_contact", "past_contact"}
    if (
        gripper_state in exceeded_gripper
        or target_state in exceeded_target
        or relation in exceeded_relation
    ):
        return True
    not_exceeded_gripper = {"fully_open", "open", "unclear", ""}
    not_exceeded_target = {"stationary", "unclear", ""}
    not_exceeded_relation = {
        "far",
        "approaching",
        "safe_pre_contact_gap",
        "near_pre_contact",
        "unclear",
        "",
    }
    if (
        gripper_state in not_exceeded_gripper
        and target_state in not_exceeded_target
        and relation in not_exceeded_relation
    ):
        return False
    return None


def _coerce_confidence(value: Any) -> float:
    if value is None:
        return 0.5
    if isinstance(value, bool):
        raise ValueError(f"confidence must be numeric, got {value!r}")
    confidence = float(value)
    if not 0.0 <= confidence <= 1.0:
        raise ValueError(f"confidence must be within [0, 1], got {confidence}")
    return confidence


def normalize_ready_pair_judgement(raw: dict[str, Any]) -> ReadyPairJudgement:
    if not isinstance(raw, dict):
        raise ValueError(f"ready judgement must be a JSON object, got {type(raw).__name__}")

    a_analysis = raw.get("A_analysis", raw.get("a_analysis"))
    b_analysis = raw.get("B_analysis", raw.get("b_analysis"))
    if not isinstance(a_analysis, dict):
        a_analysis = None
    if not isinstance(b_analysis, dict):
        b_analysis = None

    a_exceeded = _optional_bool(
        raw.get("A_exceeded_ready", raw.get("a_exceeded_ready")),
        "A_exceeded_ready",
    )
    if a_exceeded is None:
        a_exceeded = _infer_exceeded_from_analysis(a_analysis)
    if a_exceeded is None:
        a_exceeded = True

    b_exceeded = _optional_bool(
        raw.get("B_exceeded_ready", raw.get("b_exceeded_ready")),
        "B_exceeded_ready",
    )
    if b_exceeded is None:
        b_exceeded = _infer_exceeded_from_analysis(b_analysis)

    if b_exceeded is None:
        # Missing visual decisions must never cause a candidate to be accepted.
        b_exceeded = True
    # Python owns acceptance. The visual agent only judges the two frames.
    b_is_target = not b_exceeded
    confidence = _coerce_confidence(raw.get("confidence", 0.5))
    reason = str(raw.get("reason") or "").strip()
    target_object = str(raw.get("target_object") or "").strip() or None
    operation_region = str(raw.get("operation_region") or "").strip() or None
    expected_pre_contact_relation = str(raw.get("expected_pre_contact_relation") or "").strip() or None
    return ReadyPairJudgement(
        b_is_target_state=b_is_target,
        a_exceeded_ready=a_exceeded,
        b_exceeded_ready=b_exceeded,
        confidence=confidence,
        reason=reason,
        raw=raw,
        target_object=target_object,
        operation_region=operation_region,
        expected_pre_contact_relation=expected_pre_contact_relation,
        a_analysis=a_analysis,
        b_analysis=b_analysis,
    )


def _build_ready_pair_prompt(task_prompt: str) -> str:
    return f"""
You are ReadyStateAgent, a visual classifier used while searching one already-matched robot memory trajectory.

Target atomic task description:
{task_prompt}

Before judging the two images, first infer from the target atomic task description:
- target_object: the main object or instrument part the robot is preparing to manipulate.
- operation_region: the likely contact/manipulation region, such as plate edge, tube, lid handle, button, knob, rack slot, instrument opening, or receptacle.
- expected_pre_contact_relation: what the gripper/end-effector should roughly look like near the ready state.

Your only task:
- Judge independently whether Frame A has exceeded the ready state.
- Judge independently whether Frame B has exceeded the ready state.
- Do not decide how the search window moves, whether B is selected, or which robot state is exported. Python code makes those decisions from your two booleans.

Meaning of the two images:
- Frame A is the earlier frame in the current search window.
- Frame B is the later frame in the current search window.
- A and B are from the same demonstration trajectory and the same target atomic task memory.
- A and B are evaluated independently. Their temporal order helps you understand task progress, but it must not replace visual evidence.

Definitions:
1. "Ready state" is a safe pre-contact preparation state for the target atomic task. The gripper/end-effector is approaching the target object or operation region, remains fully released/open, and a visible safety gap remains between gripper and target.
2. A frame has "exceeded ready state" if any one of these conditions is visible:
   - the gripper or end-effector touches the target object or operation region;
   - the gripper is not fully released/open, including partially open, closing, closed, or already holding the object;
   - the gripper is too close to the target object, meaning no clear safe pre-contact gap is visible or contact is imminent;
   - the object has been grasped, pushed, pressed, inserted, moved, or otherwise changed by the robot.
3. "Not exceeded" includes early approach and usable pre-contact frames only when the gripper is fully open and a visible clearance remains. It does not mean the frame is the final target; Python decides that.

Your judgement standard:
- Set A_exceeded_ready and B_exceeded_ready independently from the images.
- A frame is false only when it is still safely before contact: the gripper is fully open and a visible pre-contact gap remains.
- A frame is true when contact, incomplete gripper release, or excessive closeness is visible.
- If image occlusion makes the gap impossible to verify and the gripper appears adjacent to the target, classify the frame as too_close and set exceeded=true.

Required visual analysis before the final boolean:
1. Identify the target object or operation region implied by the task description.
2. For Frame A, analyze:
   - gripper_state: fully_open, not_fully_open, closing, closed, holding_object, unclear
   - target_object_state: stationary, contacted, grasped_or_moved, task_state_changed, unclear
   - gripper_target_relation: far, approaching, safe_pre_contact_gap, too_close, in_contact, past_contact, unclear
3. For Frame B, analyze the same three fields.
4. Decide each exceeded boolean from that frame's gripper state, target-object state, and gripper-target relation.

Return strictly one JSON object with this exact schema:
{{
  "A_exceeded_ready": false,
  "B_exceeded_ready": false,
  "target_object": "short object or instrument part name",
  "operation_region": "short region name or null",
  "expected_pre_contact_relation": "short description",
  "A_analysis": {{
    "gripper_state": "fully_open | not_fully_open | closing | closed | holding_object | unclear",
    "target_object_state": "stationary | contacted | grasped_or_moved | task_state_changed | unclear",
    "gripper_target_relation": "far | approaching | safe_pre_contact_gap | too_close | in_contact | past_contact | unclear",
    "visual_evidence": "short evidence"
  }},
  "B_analysis": {{
    "gripper_state": "fully_open | not_fully_open | closing | closed | holding_object | unclear",
    "target_object_state": "stationary | contacted | grasped_or_moved | task_state_changed | unclear",
    "gripper_target_relation": "far | approaching | safe_pre_contact_gap | too_close | in_contact | past_contact | unclear",
    "visual_evidence": "short evidence"
  }},
  "confidence": 0.0,
  "reason": "short visual evidence"
}}

Rules:
- A_exceeded_ready and B_exceeded_ready must be true or false.
- target_object must be inferred from the task description, not from generic scene background.
- confidence must be a number from 0 to 1.
- reason must briefly mention the visual evidence for B.
- Do not output markdown or extra text.
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


def _find_local_prompt_match(
    target_prompt: str,
    candidates: list[str],
    *,
    cutoff: float = 0.5,
) -> tuple[str, float] | None:
    target = " ".join(str(target_prompt).strip().lower().split())
    best: tuple[str, float] | None = None
    for candidate in candidates:
        normalized = " ".join(str(candidate).strip().lower().split())
        score = SequenceMatcher(None, target, normalized).ratio()
        if target and normalized and (target in normalized or normalized in target):
            score = max(score, 0.95)
        if best is None or score > best[1]:
            best = (candidate, score)
    if best is None or best[1] < cutoff:
        return None
    return best


def judge_ready_pair(
    *,
    task_prompt: str,
    frame_a_path: str | Path,
    frame_b_path: str | Path,
    llm_config: dict | None = None,
    client: Any | None = None,
) -> ReadyPairJudgement:
    from openai import OpenAI
    from transition_generation import (
        _normalize_backend_mode,
        _normalize_thinking_mode,
        _request_json_object,
    )

    model_name = _llm_config_value(llm_config, "model_name", env_key="MODEL_NAME")
    if not model_name:
        raise ValueError("llm_config['model_name'] or MODEL_NAME is required")

    base_url = _llm_config_value(llm_config, "base_url", env_key="BASE_URL")
    api_key = _llm_config_value(llm_config, "api_key", env_key="API_KEY") or "EMPTY"
    temperature = _llm_config_value(llm_config, "temperature")
    top_p = _llm_config_value(llm_config, "top_p")
    max_tokens = _llm_config_value(llm_config, "max_tokens")
    max_attempts = int(_llm_config_value(llm_config, "max_attempts", 3) or 3)
    timeout = _llm_config_value(llm_config, "timeout")
    backend_mode = _normalize_backend_mode(_llm_config_value(llm_config, "backend_mode", "auto"))
    thinking_mode = _normalize_thinking_mode(_llm_config_value(llm_config, "thinking", "auto"))
    max_image_side = _llm_config_value(llm_config, "max_image_side", 768)

    if client is None:
        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        client = OpenAI(**client_kwargs)

    prompt = _build_ready_pair_prompt(task_prompt)
    request_input = [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": prompt + "\n\nFrame A:"},
                {
                    "type": "input_image",
                    "image_url": file_to_data_url(str(frame_a_path), max_image_side=int(max_image_side)),
                },
                {"type": "input_text", "text": "Frame B:"},
                {
                    "type": "input_image",
                    "image_url": file_to_data_url(str(frame_b_path), max_image_side=int(max_image_side)),
                },
            ],
        }
    ]
    raw = _request_json_object(
        client=client,
        model_name=model_name,
        request_input=request_input,
        stage_name="ready-memory-retrieval",
        max_attempts=max_attempts,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        timeout=timeout,
        backend_mode=backend_mode,
        thinking_mode=thinking_mode,
    )
    return normalize_ready_pair_judgement(raw)


def _get_episode_bounds(dataset) -> list[tuple[int, int]]:
    episode_data_index = getattr(dataset, "episode_data_index", None)
    if isinstance(episode_data_index, dict):
        starts = None
        ends = None
        for key in ("from", "start", "starts"):
            if key in episode_data_index:
                starts = _to_int_list(episode_data_index[key])
                break
        for key in ("to", "end", "ends"):
            if key in episode_data_index:
                ends = _to_int_list(episode_data_index[key])
                break
        if starts and ends and len(starts) == len(ends):
            return [(int(s), int(e)) for s, e in zip(starts, ends) if int(e) > int(s)]

    starts = _to_int_list(episode_data_index)
    if starts:
        sorted_starts = sorted(set(starts))
        bounds = []
        for i, start in enumerate(sorted_starts):
            end = sorted_starts[i + 1] if i + 1 < len(sorted_starts) else len(dataset)
            if end > start:
                bounds.append((start, end))
        if bounds:
            return bounds

    raise ValueError("Could not resolve episode bounds from dataset.episode_data_index")


def _write_sample_front_image(
    sample: dict[str, Any],
    *,
    path: Path,
    front_image_key: str,
) -> Path:
    image = _extract_front_image(sample, front_image_key)
    if image is None:
        raise ValueError(f"Sample has no front image under {front_image_key!r}, observation/image, or image")
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(path, _as_uint8_image(image))
    return path


def _load_memory_index(memory_db_path: str | Path) -> list[dict[str, Any]]:
    path = Path(memory_db_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        memories = payload
    elif isinstance(payload, dict):
        memories = payload.get("memories", payload.get("tasks", []))
    else:
        raise ValueError(f"Invalid ready memory index format in {path}")
    if not isinstance(memories, list) or not memories:
        raise ValueError(f"No memory entries found in {path}")
    return [memory for memory in memories if isinstance(memory, dict)]


def _memory_task_prompt(memory: dict[str, Any]) -> str:
    return str(memory.get("task", memory.get("task_prompt", ""))).strip()


def _memory_match_text(memory: dict[str, Any]) -> str:
    parts = []
    for key in (
        "task",
        "task_prompt",
        "description",
        "memory_description",
        "atomic_task_description",
        "instruction",
    ):
        value = memory.get(key)
        if value:
            parts.append(str(value))
    aliases = memory.get("aliases")
    if isinstance(aliases, list):
        parts.extend(str(alias) for alias in aliases if alias)
    return " ".join(parts).strip()


def _prompt_similarity(target_prompt: str, candidate_text: str) -> float:
    target = " ".join(str(target_prompt).strip().lower().split())
    candidate = " ".join(str(candidate_text).strip().lower().split())
    if not target or not candidate:
        return 0.0
    score = SequenceMatcher(None, target, candidate).ratio()
    if target in candidate or candidate in target:
        score = max(score, 0.95)
    target_tokens = set(target.split())
    candidate_tokens = set(candidate.split())
    if target_tokens and candidate_tokens:
        jaccard = len(target_tokens & candidate_tokens) / len(target_tokens | candidate_tokens)
        score = max(score, 0.5 * score + 0.5 * jaccard)
    return float(score)


def _resolve_memory_entry(
    memories: list[dict[str, Any]],
    task_prompt: str,
    *,
    match_cutoff: float = 0.5,
) -> tuple[dict[str, Any], str, float]:
    best: tuple[dict[str, Any], str, float] | None = None
    for memory in memories:
        match_text = _memory_match_text(memory)
        if not match_text:
            continue
        score = _prompt_similarity(task_prompt, match_text)
        if best is None or score > best[2]:
            prompt = _memory_task_prompt(memory) or match_text
            best = (memory, prompt, score)
    if best is None:
        raise ValueError("Ready memory index contains no usable task description fields")
    if best[2] < match_cutoff:
        raise ValueError(f"No ready memory matched task prompt: {task_prompt!r}")
    return best


def _build_memory_task_match_prompt(task_prompt: str, candidate_tasks: list[str]) -> str:
    task_lines = "\n".join(f"- {task}" for task in candidate_tasks)
    return f"""
You match a pending robot atomic task to one task label from a ready-memory index.

Pending atomic task:
{task_prompt}

Available ready-memory atomic task labels:
{task_lines}

Select the single label that semantically describes the same atomic task. You must
return a label copied exactly from the available list. Do not select a related
but different operation, object, direction, or instrument state.

Return strictly one JSON object:
{{
  "matched_task": "one exact label from the available list",
  "reason": "brief semantic justification"
}}
""".strip()


def _match_memory_task_with_llm(
    *,
    task_prompt: str,
    candidate_tasks: list[str],
    llm_config: dict | None,
    client: Any | None = None,
) -> tuple[str, dict[str, Any]] | None:
    """Use the LLM to choose one indexed atomic-task label, when configured."""
    model_name = _llm_config_value(llm_config, "model_name", env_key="MODEL_NAME")
    if not model_name:
        return None
    if not candidate_tasks:
        raise ValueError("Cannot match ready memory: no candidate task labels")

    from openai import OpenAI
    from transition_generation import (
        _normalize_backend_mode,
        _normalize_thinking_mode,
        _request_json_object,
    )

    base_url = _llm_config_value(llm_config, "base_url", env_key="BASE_URL")
    api_key = _llm_config_value(llm_config, "api_key", env_key="API_KEY") or "EMPTY"
    if client is None:
        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        client = OpenAI(**client_kwargs)

    raw = _request_json_object(
        client=client,
        model_name=model_name,
        request_input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": _build_memory_task_match_prompt(task_prompt, candidate_tasks),
                    }
                ],
            }
        ],
        stage_name="ready-memory-task-match",
        max_attempts=int(_llm_config_value(llm_config, "max_attempts", 3) or 3),
        temperature=_llm_config_value(llm_config, "temperature"),
        top_p=_llm_config_value(llm_config, "top_p"),
        max_tokens=_llm_config_value(llm_config, "max_tokens"),
        timeout=_llm_config_value(llm_config, "timeout"),
        backend_mode=_normalize_backend_mode(
            _llm_config_value(llm_config, "backend_mode", "auto")
        ),
        thinking_mode=_normalize_thinking_mode(
            _llm_config_value(llm_config, "thinking", "auto")
        ),
    )
    selected = str(raw.get("matched_task") or "").strip()
    if selected not in candidate_tasks:
        raise ValueError(
            "LLM ready-memory match must return an exact indexed task label; "
            f"got {selected!r}"
        )
    return selected, raw


def _abspath_from_base(path_value: Any, *, base_dir: Path) -> str | None:
    if path_value is None:
        return None
    path = Path(str(path_value))
    if not path.is_absolute():
        path = base_dir / path
    return str(path)


def _frames_from_memory_entry(memory: dict[str, Any], *, base_dir: Path) -> list[dict[str, Any]]:
    frames = memory.get("frames")
    if isinstance(frames, list) and frames:
        normalized = []
        for index, frame in enumerate(frames):
            if not isinstance(frame, dict):
                continue
            image_path = _abspath_from_base(
                frame.get("image_path", frame.get("front_image_path")),
                base_dir=base_dir,
            )
            state = frame.get("state", frame.get("qpos", frame.get("target_state")))
            if image_path is None or state is None:
                continue
            normalized.append(
                {
                    "frame_index": int(frame.get("frame_index", index)),
                    "image_path": image_path,
                    "state": np.asarray(state, dtype=np.float64).reshape(-1).tolist(),
                }
            )
        if normalized:
            return normalized

    image_paths = memory.get("frame_image_paths", memory.get("front_image_paths"))
    states = memory.get("states", memory.get("qpos", memory.get("trajectory_states")))
    frame_indices = memory.get("frame_indices")
    if not isinstance(image_paths, list) or not isinstance(states, list):
        raise ValueError(
            "Ready memory entry must contain either frames[] with image_path/state "
            "or parallel frame_image_paths[] and states[]"
        )
    if len(image_paths) != len(states):
        raise ValueError(
            f"frame_image_paths and states length mismatch: {len(image_paths)} != {len(states)}"
        )

    normalized = []
    for index, (image_path_value, state) in enumerate(zip(image_paths, states)):
        image_path = _abspath_from_base(image_path_value, base_dir=base_dir)
        if image_path is None or state is None:
            continue
        frame_index = int(frame_indices[index]) if isinstance(frame_indices, list) and index < len(frame_indices) else index
        normalized.append(
            {
                "frame_index": frame_index,
                "image_path": image_path,
                "state": np.asarray(state, dtype=np.float64).reshape(-1).tolist(),
            }
        )
    if not normalized:
        raise ValueError("Ready memory entry has no usable frames")
    return normalized


def _shrink_window_to_left_half(a: int, b: int, *, min_index: int) -> tuple[int, int, int]:
    """Keep the left half of the current [A, B] search window."""
    if b <= a:
        new_b = max(min_index, a)
    else:
        new_b = max(min_index, a + max(1, (b - a) // 2))
    new_a = a
    new_n = max(1, new_b - new_a)
    return new_a, new_b, new_n


def _advance_window_from_judgement(
    a: int,
    b: int,
    n: int,
    *,
    min_index: int,
    a_exceeded_ready: bool,
    b_exceeded_ready: bool,
) -> tuple[str, int, int, int]:
    """Apply the script-owned window state machine to one A/B judgement."""
    if not b_exceeded_ready:
        return "accept_b", a, b, n
    if not a_exceeded_ready and b_exceeded_ready:
        new_a, new_b, new_n = _shrink_window_to_left_half(a, b, min_index=min_index)
        return "shrink_left_half", new_a, new_b, new_n
    new_b = max(min_index, a)
    new_a = max(0, new_b - n)
    return "shift_left", new_a, new_b, n


def _window_percent_to_frame_count(window_percent: float, trajectory_length: int) -> int:
    try:
        percent = float(window_percent)
    except Exception as exc:
        raise ValueError(f"window_size must be a percentage number, got {window_percent!r}") from exc
    if not np.isfinite(percent) or percent <= 0.0 or percent > 100.0:
        raise ValueError(f"window_size percentage must be in (0, 100], got {window_percent!r}")
    return max(1, int(np.ceil((percent / 100.0) * max(1, int(trajectory_length)))))


def _run_window_search_on_frames(
    *,
    frames: list[dict[str, Any]],
    task_prompt: str,
    window_size: float,
    min_frame_ratio: float,
    max_iterations: int,
    llm_config: dict | None,
    client: Any | None = None,
) -> WindowSearchResult:
    length = len(frames)
    if length <= 1:
        raise ValueError(f"Ready memory trajectory is too short: length={length}")

    min_index = min(length - 1, max(1, int(round(float(min_frame_ratio) * length))))
    window_percent = float(window_size)
    n = _window_percent_to_frame_count(window_percent, length)
    b = max(min_index, int(0.5 * (length - 1)))
    a = max(0, b - n)
    history = []
    last_judgement = None
    accepted = False

    for iteration in range(max(1, int(max_iterations))):
        frame_a = frames[a]
        frame_b = frames[b]
        judgement = judge_ready_pair(
            task_prompt=task_prompt,
            frame_a_path=frame_a["image_path"],
            frame_b_path=frame_b["image_path"],
            llm_config=llm_config,
            client=client,
        )
        last_judgement = judgement
        action, next_a, next_b, next_n = _advance_window_from_judgement(
            a,
            b,
            n,
            min_index=min_index,
            a_exceeded_ready=judgement.a_exceeded_ready,
            b_exceeded_ready=judgement.b_exceeded_ready,
        )
        history.append(
            {
                "iteration": iteration,
                "A_index": a,
                "B_index": b,
                "A_frame_index": frame_a["frame_index"],
                "B_frame_index": frame_b["frame_index"],
                "window_size_percent": window_percent,
                "window_size_frames": n,
                "B_is_target_state": judgement.b_is_target_state,
                "A_exceeded_ready": judgement.a_exceeded_ready,
                "B_exceeded_ready": judgement.b_exceeded_ready,
                "target_object": judgement.target_object,
                "operation_region": judgement.operation_region,
                "expected_pre_contact_relation": judgement.expected_pre_contact_relation,
                "A_analysis": judgement.a_analysis,
                "B_analysis": judgement.b_analysis,
                "confidence": judgement.confidence,
                "reason": judgement.reason,
                "window_action": action,
                "frame_a_path": frame_a["image_path"],
                "frame_b_path": frame_b["image_path"],
            }
        )

        if action == "accept_b":
            accepted = True
            break
        a, b, n = next_a, next_b, next_n

    if accepted:
        return WindowSearchResult(
            selected_index=b,
            judgement=last_judgement,
            history=history,
            fallback_to_initial_frame=False,
        )

    print(
        "[ReadyStateAgent] fallback to initial frame 0: "
        f"no acceptable pre-contact B after {len(history)} judgements"
    )
    return WindowSearchResult(
        selected_index=0,
        judgement=last_judgement,
        history=history,
        fallback_to_initial_frame=True,
        fallback_reason="max_iterations_exhausted",
    )


def retrieve_ready_memory_from_index(
    *,
    memory_db_path: str | Path,
    task_prompt: str,
    window_size: float,
    output_path: Path,
    min_frame_ratio: float,
    max_iterations: int,
    llm_config: dict | None = None,
    match_cutoff: float = 0.5,
    client: Any | None = None,
) -> dict[str, Any]:
    db_path = Path(memory_db_path)
    memories = _load_memory_index(db_path)
    candidate_tasks = list(dict.fromkeys(
        task for task in (_memory_task_prompt(memory) for memory in memories) if task
    ))
    llm_match = _match_memory_task_with_llm(
        task_prompt=task_prompt,
        candidate_tasks=candidate_tasks,
        llm_config=llm_config,
        client=client,
    )
    if llm_match is None:
        memory, matched_prompt, match_score = _resolve_memory_entry(
            memories,
            task_prompt,
            match_cutoff=match_cutoff,
        )
        memory_match_method = "local_fuzzy"
        memory_match_raw = None
    else:
        matched_prompt, memory_match_raw = llm_match
        matching_memories = [
            memory for memory in memories if _memory_task_prompt(memory) == matched_prompt
        ]
        if not matching_memories:
            raise ValueError(f"No ready memory found for LLM-selected task: {matched_prompt!r}")
        memory = matching_memories[0]
        match_score = 1.0
        memory_match_method = "llm"
    print(
        "[ReadyStateAgent] memory match: "
        f"requested={task_prompt!r} matched={matched_prompt!r} "
        f"method={memory_match_method} score={match_score:.3f} "
        f"memory_id={memory.get('memory_id', memory.get('id'))!r} "
        f"episode={memory.get('episode_index')!r}"
    )
    frames = _frames_from_memory_entry(memory, base_dir=db_path.parent)
    search_result = _run_window_search_on_frames(
        frames=frames,
        task_prompt=matched_prompt,
        window_size=window_size,
        min_frame_ratio=min_frame_ratio,
        max_iterations=max_iterations,
        llm_config=llm_config,
        client=client,
    )
    selected_index = search_result.selected_index
    judgement = search_result.judgement
    history = search_result.history
    selected_frame = frames[selected_index]
    print(
        "[ReadyStateAgent] selected target_state: "
        f"requested={task_prompt!r} matched={matched_prompt!r} "
        f"ready_frame={selected_frame['frame_index']} local_index={selected_index}"
    )
    target_state = np.asarray(selected_frame["state"], dtype=np.float64).reshape(-1).tolist()
    selected_image_path = selected_frame["image_path"]
    result = {
        "agent": "ReadyStateAgent",
        "retrieval_source": "ready_memory_index",
        "requested_task_prompt": task_prompt,
        "matched_task_prompt": matched_prompt,
        "match_score": match_score,
        "memory_match_method": memory_match_method,
        "memory_match_raw": memory_match_raw,
        "memory_db_path": str(db_path),
        "memory_id": memory.get("memory_id", memory.get("id")),
        "episode_index": memory.get("episode_index"),
        "ready_frame_index": selected_frame["frame_index"],
        "ready_frame_local_index": selected_index,
        "fallback_to_initial_frame": search_result.fallback_to_initial_frame,
        "fallback_reason": search_result.fallback_reason,
        "target_state": target_state,
        "target_qpos": target_state,
        "target_front_image_path": selected_image_path,
        "target_front_image_paths": [selected_image_path],
        "judgement": None
        if judgement is None
        else {
            "B_is_target_state": judgement.b_is_target_state,
            "A_exceeded_ready": judgement.a_exceeded_ready,
            "B_exceeded_ready": judgement.b_exceeded_ready,
            "target_object": judgement.target_object,
            "operation_region": judgement.operation_region,
            "expected_pre_contact_relation": judgement.expected_pre_contact_relation,
            "A_analysis": judgement.a_analysis,
            "B_analysis": judgement.b_analysis,
            "confidence": judgement.confidence,
            "reason": judgement.reason,
            "raw": judgement.raw,
        },
        "search_history": history,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return result


def _resolve_task_prompt(sample: dict[str, Any], tasks_map: dict[int, str], fallback: str | None) -> str:
    if fallback:
        return fallback
    return _extract_prompt(sample, tasks_map)


def retrieve_ready_memory_from_episode(
    *,
    repo_id: str,
    task_prompt: str | None,
    episode_index: int | None,
    window_size: float,
    output_path: Path,
    image_output_dir: Path,
    front_image_key: str,
    min_frame_ratio: float,
    max_iterations: int,
    llm_config: dict | None = None,
) -> dict[str, Any]:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

    dataset_meta = LeRobotDatasetMetadata(repo_id)
    dataset = LeRobotDataset(repo_id)
    tasks_map = {int(k): str(v) for k, v in dataset_meta.tasks.items()}
    episode_bounds = _get_episode_bounds(dataset)

    selected_bound = None
    best_score = -1.0
    for start, end in episode_bounds:
        sample0 = dataset[start]
        sample_episode_index = _scalar_to_int(sample0.get("episode_index"))
        sample_prompt = _resolve_task_prompt(sample0, tasks_map, None)
        if episode_index is not None and sample_episode_index != episode_index:
            continue
        score = 1.0 if not task_prompt else _prompt_similarity(task_prompt, sample_prompt)
        if score > best_score:
            selected_bound = (start, end, sample_episode_index, sample_prompt)
            best_score = score

    if selected_bound is None or (task_prompt and best_score < 0.5):
        raise ValueError(
            f"No episode matched task_prompt={task_prompt!r} episode_index={episode_index!r}"
        )

    start, end, resolved_episode_index, resolved_task_prompt = selected_bound
    print(
        "[ReadyStateAgent] episode match: "
        f"requested={task_prompt!r} matched={resolved_task_prompt!r} "
        f"score={best_score:.3f} episode={resolved_episode_index!r}"
    )
    length = end - start
    if length <= 1:
        raise ValueError(f"Episode {resolved_episode_index} is too short: length={length}")

    min_index = min(length - 1, max(1, int(round(float(min_frame_ratio) * length))))
    window_percent = float(window_size)
    n = _window_percent_to_frame_count(window_percent, length)
    b = max(min_index, int(0.5 * (length - 1)))
    a = max(0, b - n)
    client = None
    history = []
    last_judgement = None
    accepted = False

    for iteration in range(max(1, int(max_iterations))):
        sample_a = dataset[start + a]
        sample_b = dataset[start + b]
        frame_a_path = image_output_dir / f"episode_{resolved_episode_index}_iter_{iteration:02d}_A_{a}.jpg"
        frame_b_path = image_output_dir / f"episode_{resolved_episode_index}_iter_{iteration:02d}_B_{b}.jpg"
        _write_sample_front_image(sample_a, path=frame_a_path, front_image_key=front_image_key)
        _write_sample_front_image(sample_b, path=frame_b_path, front_image_key=front_image_key)

        judgement = judge_ready_pair(
            task_prompt=resolved_task_prompt,
            frame_a_path=frame_a_path,
            frame_b_path=frame_b_path,
            llm_config=llm_config,
            client=client,
        )
        last_judgement = judgement
        action, next_a, next_b, next_n = _advance_window_from_judgement(
            a,
            b,
            n,
            min_index=min_index,
            a_exceeded_ready=judgement.a_exceeded_ready,
            b_exceeded_ready=judgement.b_exceeded_ready,
        )
        history.append(
            {
                "iteration": iteration,
                "A_frame_index": a,
                "B_frame_index": b,
                "window_size_percent": window_percent,
                "window_size_frames": n,
                "B_is_target_state": judgement.b_is_target_state,
                "A_exceeded_ready": judgement.a_exceeded_ready,
                "B_exceeded_ready": judgement.b_exceeded_ready,
                "target_object": judgement.target_object,
                "operation_region": judgement.operation_region,
                "expected_pre_contact_relation": judgement.expected_pre_contact_relation,
                "A_analysis": judgement.a_analysis,
                "B_analysis": judgement.b_analysis,
                "confidence": judgement.confidence,
                "reason": judgement.reason,
                "window_action": action,
                "frame_a_path": str(frame_a_path),
                "frame_b_path": str(frame_b_path),
            }
        )

        if action == "accept_b":
            accepted = True
            break
        a, b, n = next_a, next_b, next_n

    fallback_to_initial_frame = not accepted
    fallback_reason = None
    if fallback_to_initial_frame:
        b = 0
        fallback_reason = "max_iterations_exhausted"
        print(
            "[ReadyStateAgent] fallback to initial frame 0: "
            f"no acceptable pre-contact B after {len(history)} judgements"
        )

    selected_sample = dataset[start + b]
    target_state = _extract_state(selected_sample)
    selected_frame_path = image_output_dir / f"episode_{resolved_episode_index}_selected_B_{b}.jpg"
    _write_sample_front_image(selected_sample, path=selected_frame_path, front_image_key=front_image_key)
    print(
        "[ReadyStateAgent] selected target_state: "
        f"requested={task_prompt!r} matched={resolved_task_prompt!r} "
        f"ready_frame={b} episode={resolved_episode_index!r}"
    )

    result = {
        "agent": "ReadyStateAgent",
        "retrieval_source": "lerobot_episode",
        "requested_task_prompt": task_prompt,
        "matched_task_prompt": resolved_task_prompt,
        "match_score": best_score,
        "task": resolved_task_prompt,
        "episode_index": resolved_episode_index,
        "episode_start_index": start,
        "episode_end_index": end,
        "episode_length": length,
        "ready_frame_index": b,
        "fallback_to_initial_frame": fallback_to_initial_frame,
        "fallback_reason": fallback_reason,
        "target_state": target_state,
        "target_qpos": target_state,
        "ready_front_image_path": str(selected_frame_path),
        "target_front_image_path": str(selected_frame_path),
        "target_front_image_paths": [str(selected_frame_path)],
        "judgement": None
        if last_judgement is None
        else {
            "B_is_target_state": last_judgement.b_is_target_state,
            "A_exceeded_ready": last_judgement.a_exceeded_ready,
            "B_exceeded_ready": last_judgement.b_exceeded_ready,
            "target_object": last_judgement.target_object,
            "operation_region": last_judgement.operation_region,
            "expected_pre_contact_relation": last_judgement.expected_pre_contact_relation,
            "A_analysis": last_judgement.a_analysis,
            "B_analysis": last_judgement.b_analysis,
            "confidence": last_judgement.confidence,
            "reason": last_judgement.reason,
            "raw": last_judgement.raw,
        },
        "search_history": history,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return result


def _load_state_b(path: str | Path | None) -> list[float] | None:
    if path is None:
        return None
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("state", payload.get("target_state", payload.get("qpos")))
    if payload is None:
        return None
    return np.asarray(payload, dtype=np.float64).reshape(-1).tolist()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retrieve one coarse ready memory from visual A/B frame judgements.")
    parser.add_argument("--memory-db", type=Path, default=None, help="Pre-extracted ready memory trajectory index JSON")
    parser.add_argument("--repo_id", type=str, default=None, help="LeRobot dataset repo id for episode search")
    parser.add_argument("--task-prompt", type=str, default=None, help="Target atomic task prompt")
    parser.add_argument("--episode-index", type=int, default=None, help="Optional episode_index to search")
    parser.add_argument(
        "--window-size",
        type=float,
        default=20.0,
        help="Initial A/B window length as a trajectory percentage N; 20 means 20%% of the memory length",
    )
    parser.add_argument("--min-frame-ratio", type=float, default=0.05, help="Do not select frame 0 as ready memory")
    parser.add_argument("--max-iterations", type=int, default=4)
    parser.add_argument("--front-image-key", type=str, default="observation/image")
    parser.add_argument("--output", type=Path, default=Path("logs/lerobot_ready_memory_selected.json"))
    parser.add_argument(
        "--image-output-dir",
        type=Path,
        default=Path("logs/lerobot_ready_memory_images"),
    )

    parser.add_argument("--frame-a", type=Path, default=None, help="Pair-only mode: earlier frame A")
    parser.add_argument("--frame-b", type=Path, default=None, help="Pair-only mode: later frame B")
    parser.add_argument("--state-b-json", type=Path, default=None, help="Pair-only mode: JSON containing B state")

    parser.add_argument("--llm-model-name", type=str, default=None)
    parser.add_argument("--llm-base-url", type=str, default=None)
    parser.add_argument("--llm-api-key", type=str, default=None)
    parser.add_argument("--llm-backend-mode", choices=["auto", "responses", "chat"], default="auto")
    parser.add_argument("--llm-thinking", choices=["auto", "on", "off"], default="auto")
    parser.add_argument("--llm-temperature", type=float, default=None)
    parser.add_argument("--llm-max-tokens", type=int, default=None)
    parser.add_argument("--llm-max-image-side", type=int, default=768)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    llm_config = {
        "model_name": args.llm_model_name,
        "base_url": args.llm_base_url,
        "api_key": args.llm_api_key,
        "backend_mode": args.llm_backend_mode,
        "thinking": args.llm_thinking,
        "temperature": args.llm_temperature,
        "max_tokens": args.llm_max_tokens,
        "max_image_side": args.llm_max_image_side,
    }

    if args.frame_a is not None or args.frame_b is not None:
        if args.frame_a is None or args.frame_b is None:
            raise ValueError("--frame-a and --frame-b must be provided together")
        if not args.task_prompt:
            raise ValueError("--task-prompt is required in pair-only mode")
        judgement = judge_ready_pair(
            task_prompt=args.task_prompt,
            frame_a_path=args.frame_a,
            frame_b_path=args.frame_b,
            llm_config=llm_config,
        )
        result = {
            "B_is_target_state": judgement.b_is_target_state,
            "target_object": judgement.target_object,
            "operation_region": judgement.operation_region,
            "expected_pre_contact_relation": judgement.expected_pre_contact_relation,
            "A_analysis": judgement.a_analysis,
            "B_analysis": judgement.b_analysis,
            "target_state": _load_state_b(args.state_b_json) if judgement.b_is_target_state else None,
            "A_exceeded_ready": judgement.a_exceeded_ready,
            "B_exceeded_ready": judgement.b_exceeded_ready,
            "confidence": judgement.confidence,
            "reason": judgement.reason,
            "raw": judgement.raw,
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if args.memory_db is not None:
        if not args.task_prompt:
            raise ValueError("--task-prompt is required with --memory-db")
        result = retrieve_ready_memory_from_index(
            memory_db_path=args.memory_db,
            task_prompt=args.task_prompt,
            window_size=args.window_size,
            output_path=args.output,
            min_frame_ratio=args.min_frame_ratio,
            max_iterations=args.max_iterations,
            llm_config=llm_config,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        print(f"Saved ready memory to: {args.output}")
        return

    if not args.repo_id:
        raise ValueError("Either --memory-db, --repo_id, or pair-only --frame-a/--frame-b must be provided")

    result = retrieve_ready_memory_from_episode(
        repo_id=args.repo_id,
        task_prompt=args.task_prompt,
        episode_index=args.episode_index,
        window_size=args.window_size,
        output_path=args.output,
        image_output_dir=args.image_output_dir,
        front_image_key=args.front_image_key,
        min_frame_ratio=args.min_frame_ratio,
        max_iterations=args.max_iterations,
        llm_config=llm_config,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Saved ready memory to: {args.output}")


if __name__ == "__main__":
    main()
