import os
import base64
import io
import mimetypes
from pathlib import Path
from openai import OpenAI
import ast
import numpy as np
import json
import re
import textwrap
from difflib import SequenceMatcher
from typing import Any, Callable
from urllib.parse import urlparse

from camera_calibration_enhancement import (
    estimate_obstacles_from_vlm_output,
    format_calibration_for_llm,
    format_spatial_context_for_llm,
)

MIN_TRANSITION_MOTION_STEPS = 100
DEFAULT_TRANSITION_WAIT_STEPS = MIN_TRANSITION_MOTION_STEPS


class NoValidQposCandidateError(ValueError):
    def __init__(
        self,
        message: str,
        *,
        fallback_qpos=None,
        fallback_selection: dict | None = None,
    ):
        super().__init__(message)
        self.fallback_qpos = fallback_qpos
        self.fallback_selection = fallback_selection


_TRANSITION_EXPERT_SELF_ALLOWLIST = {
    "act_id",
    "act_name",
    "act_span",
    "action_indices",
    "base_name",
    "data",
    "dof",
    "dt",
    "freq",
    "get_site_pose",
    "gripper_control",
    "gripper_id",
    "gripper_jnt_adr",
    "ik",
    "interpolate",
    "jnt_adr",
    "jnt_name",
    "jnt_span",
    "knob_site",
    "lever_jntlimit",
    "lever_joint",
    "lever_qposadr",
    "lever_site",
    "lid_force_knob_joint",
    "lid_force_knob_qposadr",
    "lid_jntlimit",
    "lid_joint",
    "lid_lock",
    "lid_qpos_min",
    "lid_qposadr",
    "model",
    "move_to",
    "move_to_rrt",
    "move_to_target_qpos",
    "move_to_target_qpos_rrt",
    "path_follow",
    "period",
    "planner",
    "rotate_gripper",
    "execute_transition_commands",
    "set_gripper",
    "translate_ee",
    "rotate_ee",
    "wait_steps",
    "site_id",
    "site_name",
    "state_indices",
    "task",
}

_DIRECT_EXECUTE_MOTION_CALLS = {
    "interpolate",
    "move_to",
    "move_to_target_qpos",
    "path_follow",
}


def _self_attr_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "self":
        return node.attr
    return None


def _find_unknown_self_attribute_loads(tree: ast.AST) -> list[tuple[str, int]]:
    unknown: list[tuple[str, int]] = []
    for class_node in [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]:
        allowed = set(_TRANSITION_EXPERT_SELF_ALLOWLIST)
        for node in class_node.body:
            if isinstance(node, ast.FunctionDef):
                allowed.add(node.name)
        for node in ast.walk(class_node):
            if isinstance(node, ast.Attribute) and isinstance(node.ctx, (ast.Store, ast.AugStore)):
                attr_name = _self_attr_name(node)
                if attr_name is not None:
                    allowed.add(attr_name)

        seen = set()
        for node in ast.walk(class_node):
            if not isinstance(node, ast.Attribute) or not isinstance(node.ctx, ast.Load):
                continue
            attr_name = _self_attr_name(node)
            if attr_name is None or attr_name in allowed:
                continue
            key = (attr_name, getattr(node, "lineno", 0))
            if key in seen:
                continue
            seen.add(key)
            unknown.append(key)
    return unknown


def _find_direct_execute_motion_calls(tree: ast.AST) -> list[tuple[str, int]]:
    calls: list[tuple[str, int]] = []
    for class_node in [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]:
        for method in [node for node in class_node.body if isinstance(node, ast.FunctionDef) and node.name == "execute"]:
            seen = set()
            for node in ast.walk(method):
                if not isinstance(node, ast.Call):
                    continue
                attr_name = _self_attr_name(node.func)
                if attr_name not in _DIRECT_EXECUTE_MOTION_CALLS:
                    continue
                key = (attr_name, getattr(node, "lineno", 0))
                if key in seen:
                    continue
                seen.add(key)
                calls.append(key)
    return calls


def _sanitize_error_text(error: Exception, max_len: int = 500) -> str:
    text = str(error)
    # Replace massive data URLs in provider error payloads.
    text = re.sub(
        r"data:image/[a-zA-Z0-9.+-]+;base64,[A-Za-z0-9+/=\\n\\r]+",
        "<image-data-url-elided>",
        text,
    )
    if len(text) > max_len:
        return text[:max_len] + "...<truncated>"
    return text

def file_to_data_url(
    path: str,
    *,
    max_image_side: int | None = None,
    image_quality: int = 80,
) -> str:
    """
    Read a local image file and convert it to a base64-encoded data URL:
    data:image/jpeg;base64,...
    """
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
            print("[LLM] PIL not available; sending original image without compression.")

    mime, _ = mimetypes.guess_type(str(p))
    mime = mime or "image/png"  # fallback

    b64 = base64.b64encode(p.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def _resolve_first_existing_path(paths: list | None, *, base_dir: Path) -> Path | None:
    if not isinstance(paths, list):
        return None
    for value in paths:
        if not value:
            continue
        path = Path(str(value))
        if not path.is_absolute():
            path = base_dir / path
        if path.exists():
            return path
    return None


def _load_calibration_prompt_text(path: str | Path = "logs/transition_calibration.json") -> str:
    payload = _load_calibration_payload(path)
    return format_calibration_for_llm(payload) if payload else ""


def _load_calibration_payload(path: str | Path = "logs/transition_calibration.json") -> dict | None:
    calibration_path = Path(path)
    if not calibration_path.exists():
        return None
    try:
        return json.loads(calibration_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[Calibration] Failed to read {calibration_path}: {e}")
        return None


def _rounded_float_vector(values, digits: int = 4) -> list[float] | None:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size != 3 or not np.isfinite(arr).all():
        return None
    return [round(float(value), digits) for value in arr]


def _target_end_effector_payload_from_resolver(
    target_joint_pos: np.ndarray,
    target_ee_position_resolver: Callable[[np.ndarray], Any] | None,
    *,
    site_name: str | None,
) -> dict | None:
    if target_ee_position_resolver is None:
        return None
    try:
        position = target_ee_position_resolver(np.asarray(target_joint_pos, dtype=np.float64).copy())
    except Exception as e:
        print(f"[Calibration] Failed to resolve target end-effector position: {_sanitize_error_text(e)}")
        return None

    position_world = _rounded_float_vector(position)
    if position_world is None:
        print(f"[Calibration] Invalid target end-effector position from resolver: {position!r}")
        return None

    payload = {
        "position_world": position_world,
        "source": "target_qpos_fk",
    }
    if site_name:
        payload["site_name"] = site_name
    return payload


def _write_json(path: str | Path, payload: dict) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def validate_code(code):
    try:
        # 尝试解析代码
        tree = ast.parse(code)
        direct_motion_calls = _find_direct_execute_motion_calls(tree)
        if direct_motion_calls:
            details = ", ".join(
                f"self.{name} (line {line})" for name, line in direct_motion_calls[:5]
            )
            return (
                False,
                "Direct non-RRT motion call in execute: "
                f"{details}. Use execute_transition_commands, move_to_rrt, or move_to_target_qpos_rrt.",
            )

        unknown_self_attrs = _find_unknown_self_attribute_loads(tree)
        if unknown_self_attrs:
            details = ", ".join(
                f"self.{attr} (line {line})" for attr, line in unknown_self_attrs[:5]
            )
            return (
                False,
                "Unknown self attribute in generated code: "
                f"{details}. Use a local variable or an attribute initialized by TransitionExpert.",
            )
        
        return True, "The code syntax is correct and the structure is complete."
        
    except SyntaxError as e:
        return False, f"Syntax error: {str(e)}"
    except IndentationError as e:
        return False, f"Indentation error: {str(e)}"
    except Exception as e:
        return False, f"Code validation failed: {str(e)}"

def read_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"❌ Error reading file:{e}")
        exit(1)
def _extract_json_object(text: str) -> dict:
    text = text.strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # Try fenced JSON block first, e.g. ```json {...}```.
    fenced = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text, re.IGNORECASE)
    if fenced:
        try:
            obj = json.loads(fenced.group(1))
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

    # Robust fallback: scan for the first decodable JSON object and ignore trailing extra text.
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(text[i:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj

    preview = text[:400].replace("\n", "\\n")
    raise ValueError(f"No JSON object found in LLM output. Preview: {preview}")


def _get_response_text(resp) -> str:
    text = getattr(resp, "output_text", None)
    if isinstance(text, str) and text.strip():
        return text

    chunks = []
    outputs = getattr(resp, "output", None) or []
    for out in outputs:
        content_items = getattr(out, "content", None)
        if content_items is None and isinstance(out, dict):
            content_items = out.get("content", [])
        for item in content_items or []:
            if isinstance(item, dict):
                item_type = item.get("type")
                item_text = item.get("text", "")
            else:
                item_type = getattr(item, "type", None)
                item_text = getattr(item, "text", "")

            if item_type in {"output_text", "text"} and isinstance(item_text, str):
                chunks.append(item_text)

    return "\n".join([c for c in chunks if c]).strip()


def _get_chat_completion_text(resp) -> str:
    choices = getattr(resp, "choices", None) or []
    if not choices:
        return ""

    message = getattr(choices[0], "message", None)
    if message is None and isinstance(choices[0], dict):
        message = choices[0].get("message", {})

    content = getattr(message, "content", None)
    if content is None and isinstance(message, dict):
        content = message.get("content", "")

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        chunks = []
        for item in content:
            if isinstance(item, dict):
                item_text = item.get("text", "")
            else:
                item_text = getattr(item, "text", "")
            if isinstance(item_text, str) and item_text:
                chunks.append(item_text)
        return "\n".join(chunks).strip()

    return ""


def _responses_input_to_chat_messages(
    request_input,
    force_string_content: bool = False,
) -> list[dict[str, Any]]:
    chat_messages: list[dict[str, Any]] = []
    for entry in request_input or []:
        role = entry.get("role", "user") if isinstance(entry, dict) else "user"
        content_items = entry.get("content", []) if isinstance(entry, dict) else []
        chat_content = []
        text_chunks = []
        for item in content_items:
            if not isinstance(item, dict):
                continue
            item_type = item.get("type")
            if item_type == "input_text":
                text = item.get("text", "")
                text_chunks.append(text)
                if not force_string_content:
                    chat_content.append({"type": "text", "text": text})
            elif item_type == "input_image":
                if force_string_content:
                    # Strict backends may only accept string content for chat messages.
                    text_chunks.append("[image omitted for text-only backend]")
                else:
                    image_url = item.get("image_url")
                    if image_url:
                        chat_content.append({"type": "image_url", "image_url": {"url": image_url}})
        if force_string_content:
            content = "\n".join([chunk for chunk in text_chunks if chunk]).strip()
            if content:
                chat_messages.append({"role": role, "content": content})
        elif chat_content:
            chat_messages.append({"role": role, "content": chat_content})

    return chat_messages


def _should_fallback_to_chat(error: Exception) -> bool:
    message = str(error).lower()
    fallback_markers = [
        "responses",
        "response_format",
        "unsupported",
        "not support",
        "not implemented",
        "unknown",
        "404",
        "no route",
        "attributeerror",
    ]
    return any(marker in message for marker in fallback_markers)


def _should_retry_chat_with_string_content(error: Exception) -> bool:
    message = str(error).lower()
    markers = [
        "input should be a valid string",
        "string_type",
        "validation errors",
        "messages",
        "content",
    ]
    return any(marker in message for marker in markers)


def _should_disable_chat_response_format(error: Exception) -> bool:
    message = str(error).lower()
    markers = [
        "response_format",
        "json_object",
        "unsupported",
        "not support",
        "unknown field",
    ]
    return any(marker in message for marker in markers)


def _build_chat_generation_kwargs(
    temperature: float | None,
    top_p: float | None,
    max_tokens: int | None,
    timeout: float | None,
    thinking_mode: str | None = None,
) -> dict:
    kwargs = {}
    if temperature is not None:
        kwargs["temperature"] = temperature
    if top_p is not None:
        kwargs["top_p"] = top_p
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    if timeout is not None and timeout > 0:
        kwargs["timeout"] = timeout
    enable_thinking = _thinking_mode_to_bool(thinking_mode)
    if enable_thinking is not None:
        kwargs["extra_body"] = {
            "chat_template_kwargs": {
                "enable_thinking": enable_thinking,
            }
        }
    return kwargs


def _build_responses_generation_kwargs(
    temperature: float | None,
    top_p: float | None,
    max_tokens: int | None,
    timeout: float | None,
    thinking_mode: str | None = None,
) -> dict:
    kwargs = {}
    if temperature is not None:
        kwargs["temperature"] = temperature
    if top_p is not None:
        kwargs["top_p"] = top_p
    if max_tokens is not None:
        # Responses API expects max_output_tokens rather than max_tokens.
        kwargs["max_output_tokens"] = max_tokens
    if timeout is not None and timeout > 0:
        kwargs["timeout"] = timeout
    enable_thinking = _thinking_mode_to_bool(thinking_mode)
    if enable_thinking is not None:
        kwargs["extra_body"] = {
            "chat_template_kwargs": {
                "enable_thinking": enable_thinking,
            }
        }
    return kwargs


def _normalize_backend_mode(backend_mode: str | None) -> str:
    mode = (backend_mode or "auto").strip().lower()
    if mode not in {"auto", "responses", "chat"}:
        raise ValueError(f"Invalid backend mode: {backend_mode}")
    return mode


def _is_local_base_url(base_url: str | None) -> bool:
    if not base_url:
        return False
    try:
        host = (urlparse(base_url).hostname or "").lower()
    except Exception:
        return False
    return host in {"localhost", "127.0.0.1", "0.0.0.0", "::1"}


def _resolve_backend_mode(mode: str, base_url: str | None) -> str:
    if mode != "auto":
        return mode
    if _is_local_base_url(base_url):
        return "chat"
    return "responses"


def _normalize_thinking_mode(thinking_mode: str | None) -> str:
    mode = (thinking_mode or "auto").strip().lower()
    if mode not in {"auto", "on", "off"}:
        raise ValueError(f"Invalid thinking mode: {thinking_mode}")
    return mode


def _thinking_mode_to_bool(thinking_mode: str | None) -> bool | None:
    mode = _normalize_thinking_mode(thinking_mode)
    if mode == "auto":
        return None
    return mode == "on"


def _strip_think_blocks(text: str) -> str:
    if not text:
        return text
    text = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"<thinking>[\s\S]*?</thinking>", "", text, flags=re.IGNORECASE)
    return text.strip()


def _request_json_object(
    client: OpenAI,
    model_name: str,
    request_input,
    stage_name: str,
    max_attempts: int = 3,
    temperature: float | None = None,
    top_p: float | None = None,
    max_tokens: int | None = None,
    timeout: float | None = None,
    backend_mode: str = "auto",
    thinking_mode: str = "auto",
) -> dict:
    mode = _normalize_backend_mode(backend_mode)
    use_chat_mode = mode == "chat"
    force_chat_string_content = False
    use_chat_response_format = True
    last_text = ""
    last_error = None
    for attempt in range(1, max_attempts + 1):
        text = ""
        try:
            if use_chat_mode:
                request_kwargs = _build_chat_generation_kwargs(
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    timeout=timeout,
                    thinking_mode=thinking_mode,
                )
                messages = _responses_input_to_chat_messages(
                    request_input,
                    force_string_content=force_chat_string_content,
                )
                chat_kwargs = {
                    "model": model_name,
                    "messages": messages,
                    **request_kwargs,
                }
                if use_chat_response_format:
                    chat_kwargs["response_format"] = {"type": "json_object"}
                resp = client.chat.completions.create(**chat_kwargs)
                text = _get_chat_completion_text(resp)
            else:
                request_kwargs = _build_responses_generation_kwargs(
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    timeout=timeout,
                    thinking_mode=thinking_mode,
                )
                resp = client.responses.create(model=model_name, input=request_input, **request_kwargs)
                text = _get_response_text(resp)
        except Exception as e:
            if use_chat_mode and (not force_chat_string_content) and _should_retry_chat_with_string_content(e):
                force_chat_string_content = True
                print(f"[{stage_name}] Chat backend expects string content; retrying with text-only messages.")
                continue
            if use_chat_mode and use_chat_response_format and _should_disable_chat_response_format(e):
                use_chat_response_format = False
                print(f"[{stage_name}] Chat backend rejected response_format; retrying without it.")
                continue
            if (not use_chat_mode) and mode == "auto" and _should_fallback_to_chat(e):
                use_chat_mode = True
                print(
                    f"[{stage_name}] Responses API unavailable, "
                    f"fallback to chat.completions: {_sanitize_error_text(e)}"
                )
                continue
            last_error = e
            print(
                f"[{stage_name}] Attempt {attempt}/{max_attempts}: "
                f"request failed: {_sanitize_error_text(e)}"
            )
            continue

        if _normalize_thinking_mode(thinking_mode) == "off":
            text = _strip_think_blocks(text)

        if not text.strip():
            print(f"[{stage_name}] Attempt {attempt}/{max_attempts}: empty model text response, retrying...")
            continue
        try:
            return _extract_json_object(text)
        except Exception as e:
            last_text = text
            last_error = e
            print(f"[{stage_name}] Attempt {attempt}/{max_attempts}: JSON parse failed: {e}")

    preview = last_text[:400].replace("\n", "\\n")
    raise ValueError(
        f"[{stage_name}] failed to produce valid JSON after {max_attempts} attempts. "
        f"Last error: {last_error}. Preview: {preview}"
    )


def _request_text(
    client: OpenAI,
    model_name: str,
    request_input,
    stage_name: str,
    max_attempts: int = 3,
    temperature: float | None = None,
    top_p: float | None = None,
    max_tokens: int | None = None,
    timeout: float | None = None,
    backend_mode: str = "auto",
    thinking_mode: str = "auto",
) -> str:
    mode = _normalize_backend_mode(backend_mode)
    use_chat_mode = mode == "chat"
    force_chat_string_content = False
    for attempt in range(1, max_attempts + 1):
        text = ""
        try:
            if use_chat_mode:
                request_kwargs = _build_chat_generation_kwargs(
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    timeout=timeout,
                    thinking_mode=thinking_mode,
                )
                messages = _responses_input_to_chat_messages(
                    request_input,
                    force_string_content=force_chat_string_content,
                )
                resp = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    **request_kwargs,
                )
                text = _get_chat_completion_text(resp)
            else:
                request_kwargs = _build_responses_generation_kwargs(
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    timeout=timeout,
                    thinking_mode=thinking_mode,
                )
                resp = client.responses.create(model=model_name, input=request_input, **request_kwargs)
                text = _get_response_text(resp)
        except Exception as e:
            if use_chat_mode and (not force_chat_string_content) and _should_retry_chat_with_string_content(e):
                force_chat_string_content = True
                print(f"[{stage_name}] Chat backend expects string content; retrying with text-only messages.")
                continue
            if (not use_chat_mode) and mode == "auto" and _should_fallback_to_chat(e):
                use_chat_mode = True
                print(
                    f"[{stage_name}] Responses API unavailable, "
                    f"fallback to chat.completions: {_sanitize_error_text(e)}"
                )
                continue
            print(
                f"[{stage_name}] Attempt {attempt}/{max_attempts}: "
                f"request failed: {_sanitize_error_text(e)}"
            )
            continue

        if _normalize_thinking_mode(thinking_mode) == "off":
            text = _strip_think_blocks(text)

        if text.strip():
            return text
        print(f"[{stage_name}] Attempt {attempt}/{max_attempts}: empty model text response, retrying...")

    raise ValueError(f"[{stage_name}] failed to produce non-empty text after {max_attempts} attempts")




def _extract_code_from_response(text: str) -> str:
    content = text.strip()
    match = re.search(r"```python(.*?)```", content, re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r"```(.*?)```", content, re.DOTALL)
    if match:
        return match.group(1).strip()
    return content


def _axis_and_sign(axis_value) -> tuple[str, float]:
    axis_text = str(axis_value).strip().lower().replace("_axis", "").replace("-axis", "")
    sign = 1.0
    if axis_text.startswith("+"):
        axis_text = axis_text[1:]
    elif axis_text.startswith("-"):
        sign = -1.0
        axis_text = axis_text[1:]
    if axis_text not in {"x", "y", "z"}:
        raise ValueError(f"Invalid transition command axis: {axis_value!r}")
    return axis_text, sign


def _optional_positive_int(value, *, default: int | None = None, name: str) -> int | None:
    if value is None:
        return default
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _motion_steps_or_none(value) -> int | None:
    steps = _optional_positive_int(value, default=None, name="steps")
    if steps is None:
        return None
    return max(MIN_TRANSITION_MOTION_STEPS, steps)


def _compact_number(value: float) -> int | float:
    value = float(value)
    return int(value) if value.is_integer() else value


def _normalize_transition_command(
    command: dict,
    *,
    enforce_plan_constraints: bool = True,
    allow_schema_defaults: bool = False,
) -> dict | None:
    if not isinstance(command, dict):
        raise ValueError(f"Transition command must be an object, got {type(command).__name__}")
    op = str(command.get("op", command.get("action", ""))).strip().lower()
    op = op.replace("-", "_").replace(" ", "_")

    if op in {"restore", "restore_target", "restore_target_state", "target_restore"}:
        return None

    if op in {"open_gripper", "release_gripper", "free_gripper"}:
        normalized = {"op": "open_gripper"}
        delay = _optional_positive_int(command.get("delay"), default=None, name="delay")
        if delay is not None:
            normalized["delay"] = delay
        return normalized

    if op in {"close_gripper", "grasp"}:
        normalized = {"op": "close_gripper"}
        delay = _optional_positive_int(command.get("delay"), default=None, name="delay")
        if delay is not None:
            normalized["delay"] = delay
        return normalized

    if op in {"set_gripper", "gripper"}:
        if "value" not in command:
            raise ValueError("set_gripper command requires value")
        value = float(command["value"])
        if not np.isfinite(value) or value < 0.0 or value > 255.0:
            raise ValueError(f"Invalid gripper value: {value}")
        normalized = {"op": "set_gripper", "value": _compact_number(value)}
        delay = _optional_positive_int(command.get("delay"), default=None, name="delay")
        if delay is not None:
            normalized["delay"] = delay
        return normalized

    if op in {"translate", "translate_ee", "move", "move_ee"}:
        axis, sign = _axis_and_sign(command.get("axis", command.get("direction", "")))
        if "distance_m" in command:
            distance_m = float(command["distance_m"])
        elif "distance_cm" in command:
            distance_m = float(command["distance_cm"]) / 100.0
        elif "distance" in command:
            distance_m = float(command["distance"])
        else:
            raise ValueError("translate command requires distance_m")
        distance_m *= sign
        if not np.isfinite(distance_m):
            raise ValueError(f"Invalid translate distance_m: {distance_m}")
        if enforce_plan_constraints and abs(distance_m) > 0.25:
            raise ValueError(f"Invalid translate distance_m: {distance_m}")
        normalized = {"op": "translate", "axis": axis, "distance_m": _compact_number(distance_m)}
        steps = _motion_steps_or_none(command.get("steps"))
        if steps is not None:
            normalized["steps"] = steps
        return normalized

    if op in {"rotate", "rotate_ee"}:
        axis, sign = _axis_and_sign(command.get("axis", ""))
        if "angle_deg" in command:
            angle_deg = float(command["angle_deg"])
        elif "angle" in command:
            angle_deg = float(command["angle"])
        else:
            raise ValueError("rotate command requires angle_deg")
        angle_deg *= sign
        if not np.isfinite(angle_deg):
            raise ValueError(f"Invalid rotate angle_deg: {angle_deg}")
        if enforce_plan_constraints and abs(angle_deg) > 180.0:
            raise ValueError(f"Invalid rotate angle_deg: {angle_deg}")
        normalized = {"op": "rotate", "axis": axis, "angle_deg": _compact_number(angle_deg)}
        steps = _motion_steps_or_none(command.get("steps"))
        if steps is not None:
            normalized["steps"] = steps
        return normalized

    if op in {"wait", "hold"}:
        default_steps = DEFAULT_TRANSITION_WAIT_STEPS if allow_schema_defaults else None
        steps = _optional_positive_int(command.get("steps", command.get("delay")), default=default_steps, name="steps")
        if steps is None:
            raise ValueError("wait command requires steps")
        return {"op": "wait", "steps": steps}

    raise ValueError(f"Unsupported transition command op: {op!r}")


def _normalize_transition_commands(
    commands: list,
    *,
    enforce_plan_constraints: bool = True,
    allow_schema_defaults: bool = False,
) -> list[dict]:
    if not isinstance(commands, list):
        raise ValueError("Transition commands must be a list")
    normalized = []
    for command in commands:
        normalized_command = _normalize_transition_command(
            command,
            enforce_plan_constraints=enforce_plan_constraints,
            allow_schema_defaults=allow_schema_defaults,
        )
        if normalized_command is not None:
            normalized.append(normalized_command)
    if not normalized:
        raise ValueError("Transition commands are empty after removing host-handled restore commands")
    return normalized


def _commands_from_plan_steps(
    plan_steps: list[str],
    *,
    enforce_plan_constraints: bool = True,
    allow_schema_defaults: bool = False,
) -> list[dict]:
    commands = []
    for raw_step in plan_steps:
        step = str(raw_step).strip().lower()
        if not step:
            continue
        if any(token in step for token in ("open gripper", "release gripper", "free gripper", "gripper free")):
            commands.append({"op": "open_gripper"})
            continue
        if "close gripper" in step:
            commands.append({"op": "close_gripper"})
            continue

        move_match = re.search(
            r"(?:move|translate).*?(?P<value>[0-9]+(?:\.[0-9]+)?)\s*(?P<unit>cm|m).*?(?P<axis>[+\-]?[xyz])(?:\b|-axis)",
            step,
        )
        if move_match is None:
            move_match = re.search(
                r"(?:move|translate).*?(?P<axis>[+\-][xyz]|\b[xyz]\b)(?:\b|-axis)",
                step,
            )
        if move_match:
            value_text = move_match.groupdict().get("value")
            unit = move_match.groupdict().get("unit") or "m"
            distance_m = 0.05 if value_text is None else float(value_text)
            if unit == "cm":
                distance_m /= 100.0
            axis, sign = _axis_and_sign(move_match.group("axis"))
            commands.append({"op": "translate", "axis": axis, "distance_m": distance_m * sign})
            continue

        rotate_match = re.search(
            r"rotate.*?(?P<value>[+\-]?[0-9]+(?:\.[0-9]+)?)\s*(?:degrees?|deg).*?(?:around|about)\s*(?P<axis>[+\-]?[xyz])(?:\b|-axis)",
            step,
        )
        if rotate_match:
            axis, sign = _axis_and_sign(rotate_match.group("axis"))
            commands.append(
                {
                    "op": "rotate",
                    "axis": axis,
                    "angle_deg": float(rotate_match.group("value")) * sign,
                }
            )
    return _normalize_transition_commands(
        commands,
        enforce_plan_constraints=enforce_plan_constraints,
        allow_schema_defaults=allow_schema_defaults,
    )


def _commands_from_plan_obj(
    plan_obj: dict,
    *,
    enforce_plan_constraints: bool = True,
    allow_schema_defaults: bool = False,
) -> list[dict]:
    commands = plan_obj.get("commands")
    if commands is not None:
        return _normalize_transition_commands(
            commands,
            enforce_plan_constraints=enforce_plan_constraints,
            allow_schema_defaults=allow_schema_defaults,
        )
    plan_steps = plan_obj.get("plan_steps", [])
    if not isinstance(plan_steps, list) or len(plan_steps) == 0:
        raise ValueError("Stage-1 planning output missing non-empty commands")
    return _commands_from_plan_steps(
        plan_steps,
        enforce_plan_constraints=enforce_plan_constraints,
        allow_schema_defaults=allow_schema_defaults,
    )


def _commands_to_execute_body(commands: list[dict], *, enforce_plan_constraints: bool = True) -> str:
    normalized = _normalize_transition_commands(
        commands,
        enforce_plan_constraints=enforce_plan_constraints,
    )
    commands_literal = json.dumps(normalized, ensure_ascii=False)
    return f"self.execute_transition_commands({commands_literal})"


def _build_plan_multimodal_content(
    prompt_text: str,
    front_image_data_url: str,
    side_image_data_url: str,
    target_front_image_data_url: str | None = None,
) -> list[dict]:
    content = [
        {"type": "input_text", "text": prompt_text},
        {"type": "input_image", "image_url": front_image_data_url},
        {"type": "input_image", "image_url": side_image_data_url},
    ]
    if target_front_image_data_url is not None:
        content.append({"type": "input_image", "image_url": target_front_image_data_url})
    return content


def _format_ee_reachability_for_prompt(enabled: bool) -> str:
    return (
        "Planning verification only enforces per-command movement limits: "
        "translate abs(distance_m) <= 0.25m and rotate abs(angle_deg) <= 180. "
        f"Use at least {MIN_TRANSITION_MOTION_STEPS} steps for every translate/rotate command. "
        "It does not check IK reachability."
    )


def _is_known_invalid_plan_verification_issue(issue: dict) -> bool:
    problem = str(issue.get("problem", "")).lower()
    required_fix = str(issue.get("required_fix", "")).lower()
    combined = f"{problem} {required_fix}"

    gripper_semantics_wrong = (
        "final_target_gripper" in combined
        and (
            "0.0 implies closed" in combined
            or "0 implies closed" in combined
            or "1.0 (open)" in combined
            or "1.0 open" in combined
            or "normalized 0/1" in combined
        )
    )
    if gripper_semantics_wrong:
        return True

    restore_target_misread = (
        ("final_target_qpos" in combined or "final_target_gripper" in combined)
        and any(token in combined for token in ("cumulative motion", "last transition command", "final state after"))
    )
    return restore_target_misread


def _is_known_invalid_plan_revision_instruction(instruction: str) -> bool:
    text = str(instruction).lower()
    if "final_target_gripper" in text and ("1.0 (open)" in text or "1.0 open" in text):
        return True
    if ("final_target_qpos" in text or "final_target_gripper" in text) and "cumulative motion" in text:
        return True
    return False


def _is_plan_constraint_verification_issue(issue: dict) -> bool:
    problem = str(issue.get("problem", "")).lower()
    required_fix = str(issue.get("required_fix", "")).lower()
    combined = f"{problem} {required_fix}"

    next_task_semantic_constraint = (
        any(
            token in combined
            for token in (
                "next atomic task",
                "next-task",
                "semantic task action",
                "transition-only",
                "only doing transition",
                "pressing buttons",
                "turning knobs",
                "opening or closing lids",
                "placing objects",
                "inserting or removing",
                "screwing or unscrewing",
            )
        )
        and any(token in combined for token in ("execute", "remove", "do not", "not execute", "instead of"))
    )
    if next_task_semantic_constraint:
        return True

    numeric_or_schema_tokens = (
        "distance_m",
        "0.25",
        "25cm",
        "25 cm",
        "angle_deg",
        "180",
        "axis",
        "x/y/z",
        "steps",
        "delay",
        "positive",
        "allowed command",
        "unsupported",
        "schema",
        "invalid op",
        "command op",
        '"op"',
        "'op'",
        "required field",
        "missing",
        "invalid",
        "not a list",
        "empty",
        "numeric",
        "number",
        "float",
        "type",
        "range",
        "0..255",
        "0-255",
        "255",
    )
    restore_target_type_check = (
        ("final_target_qpos" in combined or "final_target_gripper" in combined)
        and any(token in combined for token in ("missing", "invalid", "list", "numeric", "number", "float", "type"))
    )
    plan_steps_check = "plan_steps" in combined and any(
        token in combined for token in ("match", "order", "missing", "invalid", "not a list", "empty")
    )
    return restore_target_type_check or plan_steps_check or any(token in combined for token in numeric_or_schema_tokens)


def _is_plan_constraint_revision_instruction(instruction: str) -> bool:
    text = str(instruction).lower()
    next_task_semantic_constraint = (
        any(
            token in text
            for token in (
                "next atomic task",
                "next-task",
                "semantic task action",
                "transition-only",
                "pressing buttons",
                "turning knobs",
                "opening or closing lids",
                "placing objects",
                "inserting or removing",
                "screwing or unscrewing",
            )
        )
        and any(token in text for token in ("remove", "do not", "not execute", "only"))
    )
    if next_task_semantic_constraint:
        return True

    numeric_or_schema_tokens = (
        "distance_m",
        "0.25",
        "25cm",
        "25 cm",
        "angle_deg",
        "180",
        "axis",
        "steps",
        "delay",
        "allowed command",
        "unsupported",
        "schema",
        "invalid op",
        "command op",
        '"op"',
        "'op'",
        "required field",
        "missing",
        "invalid",
        "numeric",
        "range",
        "0..255",
        "0-255",
        "255",
    )
    restore_target_type_check = (
        ("final_target_qpos" in text or "final_target_gripper" in text)
        and any(token in text for token in ("missing", "invalid", "list", "numeric", "number", "float", "type"))
    )
    plan_steps_check = "plan_steps" in text and any(
        token in text for token in ("match", "order", "missing", "invalid", "not a list", "empty")
    )
    return restore_target_type_check or plan_steps_check or any(token in text for token in numeric_or_schema_tokens)


def _normalize_plan_verification_result(result: dict, command_count: int) -> dict:
    if not isinstance(result, dict):
        raise ValueError("Plan verifier output must be a JSON object")

    issues = result.get("issues", [])
    if issues is None:
        issues = []
    if not isinstance(issues, list):
        raise ValueError("Plan verifier output field 'issues' must be a list")

    normalized_issues = []
    bad_indices = set()
    ignored_bad_indices = set()
    filtered_known_invalid_issue = False
    for issue in issues:
        if not isinstance(issue, dict):
            issue = {"command_index": None, "problem": str(issue), "required_fix": ""}
        if _is_known_invalid_plan_verification_issue(issue) or not _is_plan_constraint_verification_issue(issue):
            filtered_known_invalid_issue = True
            try:
                ignored_bad_indices.add(int(issue.get("command_index")))
            except Exception:
                pass
            continue
        raw_index = issue.get("command_index")
        command_index = None
        if raw_index is not None:
            try:
                command_index = int(raw_index)
            except Exception:
                command_index = None
            if command_index is not None and 0 <= command_index < command_count:
                bad_indices.add(command_index)
        normalized_issues.append(
            {
                "command_index": command_index,
                "problem": str(issue.get("problem", "")).strip(),
                "required_fix": str(issue.get("required_fix", "")).strip(),
            }
        )

    for raw_index in result.get("bad_command_indices", []) or []:
        try:
            command_index = int(raw_index)
        except Exception:
            continue
        if command_index in ignored_bad_indices:
            continue
        if 0 <= command_index < command_count:
            bad_indices.add(command_index)

    revision_instructions = result.get("revision_instructions", [])
    if revision_instructions is None:
        revision_instructions = []
    if not isinstance(revision_instructions, list):
        revision_instructions = [str(revision_instructions)]
    revision_instructions = [
        str(item).strip()
        for item in revision_instructions
        if (
            str(item).strip()
            and not _is_known_invalid_plan_revision_instruction(str(item))
            and _is_plan_constraint_revision_instruction(str(item))
        )
    ]

    passed = bool(result.get("passed", False))
    if normalized_issues or bad_indices:
        passed = False
    elif filtered_known_invalid_issue:
        passed = True

    return {
        "passed": passed,
        "issues": normalized_issues,
        "bad_command_indices": sorted(bad_indices),
        "revision_instructions": revision_instructions,
    }


def _local_plan_verification_failure(error: Exception) -> dict:
    return {
        "passed": False,
        "issues": [
            {
                "command_index": None,
                "problem": f"Local command validation failed: {error}",
                "required_fix": (
                    "Regenerate a complete plan whose commands follow the allowed schema, "
                    "single translate abs(distance_m) <= 0.25, single rotate abs(angle_deg) <= 180, "
                    f"and translate/rotate steps >= {MIN_TRANSITION_MOTION_STEPS} when steps are present."
                ),
            }
        ],
        "bad_command_indices": [],
        "revision_instructions": ["Regenerate the complete plan with only valid transition commands."],
    }


def _build_plan_verification_prompt(
    *,
    task_prompt: str,
    planning_prompt: str,
    plan_obj: dict,
    commands: list[dict],
) -> str:
    plan_payload = dict(plan_obj)
    plan_payload["commands"] = commands
    return f"""You are a robot transition plan constraint-only checker.

Task prompt:
{task_prompt}

Verifier task:
Check only whether the proposed transition plan satisfies the hard planning-agent constraints and command limits. Do not judge whether an action choice is reasonable, useful, optimal, safe, or task-appropriate.

Original planning prompt:
{planning_prompt}

Verification rules:
1. Check command schema only: commands must be a list of allowed ops with required fields.
2. Allowed command ops are: open_gripper, close_gripper, set_gripper, translate, rotate, wait, restore_target_state.
3. Verify plan_steps match the executable commands in order.
4. Verify translate commands use x/y/z and abs(distance_m) <= 0.3.
5. Verify rotate commands use x/y/z and abs(angle_deg) <= 180.
6. Verify translate/rotate steps, when present, are positive integers >= {MIN_TRANSITION_MOTION_STEPS}; delay and wait steps must be positive integers.
7. Verify set_gripper values, when present, are numeric and in range 0..255.
8. Gripper actuator semantics in this simulator: 0 means fully open, 255 means fully closed. It is not a normalized 0/1 flag. open_gripper maps to value 0; close_gripper maps to value 255.
9. Do not check IK reachability or cumulative workspace reachability.
10. final_target_qpos and final_target_gripper are host restore targets for the next task's retrieved initial state. They are not produced by integrating transition commands and must not be forced to match the last transition command.
11. Do not reject a plan merely because final_target_qpos/final_target_gripper differ from the cumulative transition-command endpoint.
12. The transition-only constraint is hard: reject commands that explicitly execute the next atomic task instead of restoring/preparing for it, including next-task semantic actions such as grasping a next-task object, placing objects, opening or closing lids, pressing buttons, turning knobs, screwing or unscrewing knobs, or inserting/removing plates or tubes.
13. Do not reject a plan because a gripper action seems unsafe, unnecessary, or semantically questionable unless it explicitly violates the transition-only constraint.
14. Do not reject a plan because a movement direction, approach, lift, retract, grasp, or release seems unreasonable unless it explicitly violates the transition-only constraint.
15. If a hard constraint is violated, identify the corresponding zero-based command_index whenever possible.
16. Do not rewrite the plan. Only report whether it passes and what hard constraint the planning agent must fix.

Return strictly one JSON object:
{{
  "passed": true,
  "issues": [
    {{"command_index": 1, "problem": "short reason", "required_fix": "specific fix"}}
  ],
  "bad_command_indices": [1],
  "revision_instructions": ["specific instruction for the planner"]
}}

Plan JSON to verify:
{json.dumps(plan_payload, ensure_ascii=False, indent=2)}"""


def _request_plan_verification(
    *,
    client: OpenAI,
    model_name: str,
    planning_prompt: str,
    task_prompt: str,
    plan_obj: dict,
    commands: list[dict],
    front_image_data_url: str,
    side_image_data_url: str,
    target_front_image_data_url: str | None,
    request_json_object: Callable[..., dict],
    max_attempts: int = 3,
    temperature: float | None = None,
    top_p: float | None = None,
    max_tokens: int | None = None,
    timeout: float | None = None,
    backend_mode: str = "auto",
    thinking_mode: str = "auto",
) -> dict:
    verifier_prompt = _build_plan_verification_prompt(
        task_prompt=task_prompt,
        planning_prompt=planning_prompt,
        plan_obj=plan_obj,
        commands=commands,
    )
    verifier_obj = request_json_object(
        client=client,
        model_name=model_name,
        request_input=[{
            "role": "user",
            "content": _build_plan_multimodal_content(
                verifier_prompt,
                front_image_data_url,
                side_image_data_url,
                target_front_image_data_url,
            ),
        }],
        stage_name="stage-1.5-plan-verifier",
        max_attempts=max_attempts,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        timeout=timeout,
        backend_mode=backend_mode,
        thinking_mode=thinking_mode,
    )
    return _normalize_plan_verification_result(verifier_obj, len(commands))


def _summarize_plan_verification_issues(verification: dict) -> str:
    parts = []
    for issue in verification.get("issues", []):
        parts.append(
            "command_index={idx}; problem={problem}; required_fix={fix}".format(
                idx=issue.get("command_index"),
                problem=issue.get("problem", ""),
                fix=issue.get("required_fix", ""),
            )
        )
    for instruction in verification.get("revision_instructions", []):
        parts.append(f"instruction={instruction}")
    return " | ".join(parts) if parts else "Verifier rejected the plan without detailed issues."


def _format_plan_revision_feedback(plan_obj: dict, commands: list[dict], verification: dict) -> str:
    plan_steps = plan_obj.get("plan_steps", [])
    if not isinstance(plan_steps, list):
        plan_steps = []

    lines = [
        "Previous verifier feedback:",
        "The previous plan failed verification. Regenerate a complete corrected plan, not a patch.",
    ]
    for issue in verification.get("issues", []):
        command_index = issue.get("command_index")
        command_text = None
        step_text = None
        if isinstance(command_index, int) and 0 <= command_index < len(commands):
            command_text = json.dumps(commands[command_index], ensure_ascii=False)
        if isinstance(command_index, int) and 0 <= command_index < len(plan_steps):
            step_text = str(plan_steps[command_index])
        lines.append(
            "- command_index={idx}; command={command}; plan_step={step}; problem={problem}; required_fix={fix}".format(
                idx=command_index,
                command=command_text,
                step=step_text,
                problem=issue.get("problem", ""),
                fix=issue.get("required_fix", ""),
            )
        )

    for instruction in verification.get("revision_instructions", []):
        lines.append(f"- revision_instruction={instruction}")
    return "\n".join(lines)


def _generate_verified_transition_plan(
    *,
    client: OpenAI,
    model_name: str,
    planning_prompt: str,
    task_prompt: str,
    front_image_data_url: str,
    side_image_data_url: str,
    target_front_image_data_url: str | None,
    verifier_enabled: bool = False,
    max_plan_revisions: int = 2,
    request_json_object: Callable[..., dict] = _request_json_object,
    max_attempts: int = 3,
    temperature: float | None = None,
    top_p: float | None = None,
    max_tokens: int | None = None,
    timeout: float | None = None,
    backend_mode: str = "auto",
    thinking_mode: str = "auto",
    write_logs: bool = False,
) -> tuple[dict, list[dict], dict]:
    max_plan_revisions = max(0, int(max_plan_revisions))
    feedback_text = ""
    last_verification = None

    for revision_index in range(max_plan_revisions + 1):
        effective_planning_prompt = planning_prompt
        if feedback_text:
            effective_planning_prompt = f"{planning_prompt}\n\n{feedback_text}"

        print(
            f"🚀 Stage 1: Generating path planning list using {model_name} "
            f"(attempt {revision_index + 1}/{max_plan_revisions + 1})..."
        )
        plan_obj = request_json_object(
            client=client,
            model_name=model_name,
            request_input=[{
                "role": "user",
                "content": _build_plan_multimodal_content(
                    effective_planning_prompt,
                    front_image_data_url,
                    side_image_data_url,
                    target_front_image_data_url,
                ),
            }],
            stage_name="stage-1-planning",
            max_attempts=max_attempts,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            timeout=timeout,
            backend_mode=backend_mode,
            thinking_mode=thinking_mode,
        )

        try:
            transition_commands = _commands_from_plan_obj(
                plan_obj,
                enforce_plan_constraints=verifier_enabled,
                allow_schema_defaults=not verifier_enabled,
            )
            plan_obj["commands"] = transition_commands
        except Exception as e:
            transition_commands = []
            if not verifier_enabled:
                raise ValueError(
                    "Stage-1 planning output could not be compiled into executable transition commands: "
                    f"{e}"
                ) from e
            verification = _local_plan_verification_failure(e)
        else:
            if not verifier_enabled:
                verification = {
                    "passed": True,
                    "issues": [],
                    "bad_command_indices": [],
                    "revision_instructions": [],
                    "verifier_enabled": False,
                }
                if write_logs:
                    _write_json(f"logs/transition_plan_attempt_{revision_index + 1}.json", plan_obj)
                    _write_json(f"logs/transition_plan_verification_{revision_index + 1}.json", verification)
                print("Stage 1.5: plan verifier disabled; accepted executable plan without verifier constraint checks.")
                return plan_obj, transition_commands, verification

            try:
                llm_verification = _request_plan_verification(
                    client=client,
                    model_name=model_name,
                    planning_prompt=planning_prompt,
                    task_prompt=task_prompt,
                    plan_obj=plan_obj,
                    commands=transition_commands,
                    front_image_data_url=front_image_data_url,
                    side_image_data_url=side_image_data_url,
                    target_front_image_data_url=target_front_image_data_url,
                    request_json_object=request_json_object,
                    max_attempts=max_attempts,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    timeout=timeout,
                    backend_mode=backend_mode,
                    thinking_mode=thinking_mode,
                )
                verification = llm_verification
            except Exception as e:
                transition_commands = []
                verification = _local_plan_verification_failure(e)

        last_verification = verification
        if write_logs:
            _write_json(f"logs/transition_plan_attempt_{revision_index + 1}.json", plan_obj)
            _write_json(f"logs/transition_plan_verification_{revision_index + 1}.json", verification)

        if verification.get("passed"):
            print("✅ Stage 1.5: plan verifier passed.")
            return plan_obj, transition_commands, verification

        print(f"❌ Stage 1.5: plan verifier failed: {_summarize_plan_verification_issues(verification)}")
        if revision_index >= max_plan_revisions:
            label = "revision" if max_plan_revisions == 1 else "revisions"
            raise ValueError(
                f"Plan verification failed after {max_plan_revisions} {label}: "
                f"{_summarize_plan_verification_issues(verification)}"
            )
        feedback_text = _format_plan_revision_feedback(plan_obj, transition_commands, verification)

    raise ValueError(
        "Plan verification failed unexpectedly: "
        f"{_summarize_plan_verification_issues(last_verification or {})}"
    )


def _strip_execute_prelude(body_code: str) -> str:
    lines = body_code.splitlines()
    filtered = []
    for line in lines:
        if "self.ik.initial_qpos" in line:
            continue
        if line.strip().startswith("# Initial IK"):
            continue
        filtered.append(line)
    return "\n".join(filtered).strip()


def _strip_execute_final_restore(body_code: str) -> str:
    lines = body_code.splitlines()
    filtered = []
    for line in lines:
        stripped = line.strip()
        # The final target restoration is hard-inserted by host code, not LLM output.
        if "move_to_target_qpos(" in stripped:
            continue
        if stripped.startswith("target_qpos"):
            continue
        if stripped.startswith("target_gripper"):
            continue
        filtered.append(line)

    # Trim trailing blank lines to keep a clean boundary before host-inserted final block.
    while filtered and filtered[-1].strip() == "":
        filtered.pop()
    return "\n".join(filtered)


def _replace_execute_body(
    template_code: str,
    execute_body_code: str,
    final_target_qpos: list[float],
    final_target_gripper: float | None,
    include_final_restore: bool = True,
    final_target_qpos_candidates: list | None = None,
    target_top_k: int = 3,
) -> str:
    lines = template_code.splitlines()
    start = None
    end = len(lines)
    for i, line in enumerate(lines):
        if line.startswith("    def execute("):
            start = i
            break
    if start is None:
        raise ValueError("execute method not found in transition template")

    for i in range(start + 1, len(lines)):
        if lines[i].startswith("    def "):
            end = i
            break

    raw_body = _extract_code_from_response(execute_body_code)
    if raw_body.startswith("def execute"):
        parts = raw_body.splitlines()
        raw_body = "\n".join(parts[1:])
    raw_body = _strip_execute_prelude(raw_body)
    raw_body = _strip_execute_final_restore(raw_body)
    raw_body = textwrap.dedent(raw_body).strip("\n")

    new_method = [
        "    def execute(self):",
        "        # Initial IK, must not be removed",
        "        self.ik.initial_qpos = self.data.qpos[self.jnt_span]",
    ]

    if raw_body:
        new_method.append("")
        for line in raw_body.splitlines():
            if line.strip() == "":
                new_method.append("")
            else:
                new_method.append(f"        {line}")
    else:
        new_method.append("")
        new_method.append("        # No transition step generated by LLM.")

    if include_final_restore:
        new_method.append("")
        new_method.append("        # Restore to target pose (hard-inserted from planning JSON).")
        if final_target_qpos_candidates:
            candidate_literal = json.dumps(final_target_qpos_candidates, ensure_ascii=False)
            new_method.append("        from transition_generation import select_target_qpos_after_transition, validate_qpos_rrt_path")
            new_method.append(f"        target_qpos_candidates = {candidate_literal}")
            new_method.append("        target_selection = select_target_qpos_after_transition(")
            new_method.append("            target_qpos_candidates,")
            new_method.append("            self.data.qpos[self.jnt_span],")
            new_method.append(f"            top_k={max(1, int(target_top_k or 3))},")
            new_method.append("            path_validator=lambda candidate_qpos, *, selected_index: validate_qpos_rrt_path(")
            new_method.append("                self.model,")
            new_method.append("                self.data,")
            new_method.append("                self.jnt_span,")
            new_method.append("                candidate_qpos,")
            new_method.append("            ),")
            new_method.append("        )")
            new_method.append("        target_qpos_full = np.asarray(target_selection[\"selected_qpos\"], dtype=np.float64).reshape(-1)")
            new_method.append("        target_qpos = target_qpos_full[:self.dof].tolist()")
            new_method.append("        target_gripper = float(target_qpos_full[-1]) if target_qpos_full.size > self.dof else None")
        else:
            new_method.append(f"        target_qpos = {list(final_target_qpos)}")
            if final_target_gripper is None:
                new_method.append("        target_gripper = None")
            else:
                new_method.append(f"        target_gripper = {float(final_target_gripper)}")
        new_method.append("        self.move_to_target_qpos_rrt(target_qpos)")
        new_method.append("        self.gripper_control(target_gripper)")
    else:
        new_method.append("")
        new_method.append("        # Final target restore is disabled by no_interpolation/no_retrieval mode.")
        if final_target_gripper is not None:
            new_method.append(f"        target_gripper = {float(final_target_gripper)}")
            new_method.append("        self.gripper_control(target_gripper)")

    updated_lines = lines[:start] + new_method + lines[end:]
    out = "\n".join(updated_lines)
    if template_code.endswith("\n"):
        out += "\n"
    return out


def _resolve_plan_restore_targets(
    *,
    plan_obj: dict,
    target_arm_qpos: list[float],
    target_gripper_state: float | None,
    include_final_restore: bool,
) -> tuple[list[float], float | None]:
    if include_final_restore:
        plan_target_qpos = plan_obj.get("final_target_qpos", target_arm_qpos)
        if not isinstance(plan_target_qpos, list) or len(plan_target_qpos) == 0:
            raise ValueError("Stage-1 planning output missing valid final_target_qpos")
        try:
            plan_target_qpos = [float(x) for x in plan_target_qpos]
        except Exception as e:
            raise ValueError(f"Invalid final_target_qpos in plan output: {e}")
    else:
        plan_target_qpos = [float(x) for x in target_arm_qpos]
        plan_target_gripper = target_gripper_state
        if plan_target_gripper is not None:
            try:
                plan_target_gripper = float(plan_target_gripper)
            except Exception as e:
                raise ValueError(f"Invalid final_target_gripper in host target state: {e}")
        return plan_target_qpos, plan_target_gripper

    plan_target_gripper = plan_obj.get("final_target_gripper", target_gripper_state)
    if plan_target_gripper is not None:
        try:
            plan_target_gripper = float(plan_target_gripper)
        except Exception as e:
            raise ValueError(f"Invalid final_target_gripper in plan output: {e}")

    return plan_target_qpos, plan_target_gripper


def _format_restore_schema_fields(
    *,
    no_retrieval: bool,
    target_arm_qpos: list[float],
    target_gripper_state: float | None,
) -> str:
    if no_retrieval:
        return '"restore": false'
    return f'"final_target_qpos": {target_arm_qpos},\n    "final_target_gripper": {target_gripper_state}'


def _build_transition_planning_prompt(
    *,
    target_reference_text: str,
    target_binding_text: str,
    calibration_prompt_text: str,
    spatial_context_prompt_text: str,
    reachability_prompt_text: str,
    restore_schema_fields: str,
    motion_constraint_rule: str,
) -> str:
    return f'''
You are a robot transition planner.

Task:
Generate a concise path-planning list for transition execution, not code.

Inputs:
front camera image: the current front camera image
side camera image: the current side camera image
target front reference image: {target_reference_text}

Planning objective:
- Safety-first, collision-avoidance.
- Main sequence: obstacle clearance, then movement toward the restored target preparation pose for the next atomic task.
- Restoration target: the target pose/image is the retrieved preparation pose for the next atomic task, i.e. the retrieved starting pose for the next atomic task. It describes where the robot should be restored before the next atomic task begins or resumes, not an instruction to execute that task.
- After clearance, move the EE toward the restored target gripper world position using the calibrated target delta when available.

Planning Rules:
First, analyze the states of the objects, the robotic arm, and the gripper within the scene.
Next, determine whether the gripper must be released only to make the transition safe.
Subsequently, maneuver the End-Effector (EE) away from all obstacles through translational and rotational movements.
Finally, if target delta is available in Tool-derived spatial context, include translate commands that reduce the remaining x/y/z delta toward the restored target gripper position.
Do not stop after clearance unless the target delta is unavailable or unsafe.

Transition-only constraint:
- Do not execute the next atomic task. The plan must only prepare or restore the robot to the next task's starting pose.
- Do not perform semantic task actions for the next prompt, including grasping a next-task object, placing objects, opening or closing lids, pressing buttons, turning knobs, screwing or unscrewing knobs, inserting or removing plates/tubes, or otherwise changing task-object state.
- Gripper changes are allowed only when needed for transition safety or clearance, not to begin a next-task grasp/manipulation.

Image binding for this request:
- The first image is the CURRENT FRONT view.
- The second image is the CURRENT SIDE view.
- {target_binding_text}
- You must jointly reason over the current views and target reference before generating the plan.

Calibrated camera geometry:
{calibration_prompt_text if calibration_prompt_text else "No calibrated camera geometry is available for this transition."}

Tool-derived spatial context:
{spatial_context_prompt_text if spatial_context_prompt_text else "No VLM+calibration obstacle context is available for this transition."}

Planning movement limits:
{reachability_prompt_text}

Return strictly one JSON object with schema:
{{
    "commands": [
        {{"op": "open_gripper", "delay": 100}},
        {{"op": "translate", "axis": "z", "distance_m": 0.08, "steps": {MIN_TRANSITION_MOTION_STEPS}}},
        {{"op": "rotate", "axis": "z", "angle_deg": 30, "steps": {MIN_TRANSITION_MOTION_STEPS}}},
        {{"op": "close_gripper", "delay": 100}}
    ],
    "plan_steps": [
        "step 1: gripper free...",
        "step 2: move...",
        "step 3: move...",
        ...
        "step n: move...",
    ],
    "safety_notes": ["...", "..."],
    {restore_schema_fields}
}}

Rules:
1. commands is required and is the only executable transition representation.
2. Allowed command ops are: open_gripper, close_gripper, set_gripper, translate, rotate, wait, restore_target_state.
3. translate must use axis x/y/z and distance_m in meters; use negative distance_m for negative axis motion.
   The absolute value of distance_m for every single translate command MUST be <= 0.25m.
   If a longer translation is needed, split it into multiple translate commands, each with abs(distance_m) <= 0.25m.
   Example: do not output {{"op": "translate", "axis": "x", "distance_m": -0.5}}; output two translate commands with distance_m -0.25 each.
4. rotate must use axis x/y/z and angle_deg in degrees; use negative angle_deg for negative rotation.
5. Keep every single translate distance absolute value <= 0.25m and every single rotation <= 180 degrees.
6. Translate/rotate commands must use at least {MIN_TRANSITION_MOTION_STEPS} steps. Use more steps for larger movements; never use fewer for speed.
7. {motion_constraint_rule}
8. plan_steps are only human-readable comments matching commands.
9. In no-retrieval mode, restore is false and no final target-state restoration will be executed.
10. Do not output code.
11. Do not output markdown.
'''


def _normalize_prompt(text: str) -> str:
    return " ".join(text.lower().strip().split())


def _find_local_prompt_match(
    task_prompt: str,
    prompt_choices: list[str],
    *,
    cutoff: float = 0.72,
) -> tuple[str, float] | None:
    if not prompt_choices:
        return None

    norm_prompt = _normalize_prompt(task_prompt)
    norm_map = {_normalize_prompt(prompt): prompt for prompt in prompt_choices}
    if norm_prompt in norm_map:
        return norm_map[norm_prompt], 1.0

    best_norm = None
    best_score = 0.0
    for choice_norm in norm_map:
        score = SequenceMatcher(None, norm_prompt, choice_norm).ratio()
        if score > best_score:
            best_norm = choice_norm
            best_score = score

    if best_norm is None or best_score < float(cutoff):
        return None
    return norm_map[best_norm], float(best_score)


def _pick_nearest_index(stacked_qpos: list, current_joint_pos: np.ndarray) -> int:
    if len(stacked_qpos) == 0:
        raise ValueError("No qpos candidates provided")

    cur = np.asarray(current_joint_pos, dtype=np.float64).reshape(-1)
    best_idx = 0
    best_dist = float("inf")
    for i, q in enumerate(stacked_qpos):
        q_arr = np.asarray(q, dtype=np.float64).reshape(-1)
        dim = min(len(cur), len(q_arr))
        if dim == 0:
            continue
        dist = float(np.linalg.norm(cur[:dim] - q_arr[:dim]))
        if dist < best_dist:
            best_dist = dist
            best_idx = i
    return best_idx


def _as_bool_validation(validation) -> tuple[bool, dict | None]:
    if isinstance(validation, dict):
        for key in ("valid", "success", "is_valid"):
            if key in validation:
                return bool(validation[key]), validation
        return bool(validation), validation
    return bool(validation), None


def _qpos_joint_distance(candidate_qpos, current_joint_pos: np.ndarray) -> float:
    cur = np.asarray(current_joint_pos, dtype=np.float64).reshape(-1)
    q_arr = np.asarray(candidate_qpos, dtype=np.float64).reshape(-1)
    dim = min(6, len(cur), len(q_arr))
    if dim <= 0:
        return float("inf")
    if not np.isfinite(cur[:dim]).all() or not np.isfinite(q_arr[:dim]).all():
        return float("inf")
    return float(np.linalg.norm(cur[:dim] - q_arr[:dim]))


def select_target_qpos_candidate(
    stacked_qpos: list,
    current_joint_pos: np.ndarray,
    *,
    top_k: int = 3,
    path_validator: Callable[..., bool | dict] | None = None,
) -> dict:
    if not isinstance(stacked_qpos, list) or len(stacked_qpos) == 0:
        raise ValueError("No qpos candidates provided")

    effective_top_k = max(1, int(top_k or 3))
    ranked_candidates = []
    for i, qpos in enumerate(stacked_qpos):
        ranked_candidates.append(
            {
                "index": i,
                "distance": _qpos_joint_distance(qpos, current_joint_pos),
                "qpos": np.asarray(qpos, dtype=np.float64).reshape(-1),
            }
        )
    ranked_candidates.sort(key=lambda item: (item["distance"], item["index"]))
    top_candidates = ranked_candidates[: min(effective_top_k, len(ranked_candidates))]

    if path_validator is None:
        selected = top_candidates[0]
        return {
            "selected_index": selected["index"],
            "selected_qpos": selected["qpos"],
            "selected_distance": selected["distance"],
            "top_k": effective_top_k,
            "validation": None,
            "top_candidates": [
                {"index": item["index"], "distance": item["distance"]}
                for item in top_candidates
            ],
        }

    validation_records = [
        {"index": item["index"], "distance": item["distance"]}
        for item in top_candidates
    ]
    for candidate in top_candidates:
        try:
            validation = path_validator(
                candidate["qpos"],
                selected_index=candidate["index"],
            )
        except Exception as e:
            validation = {
                "valid": False,
                "reason": "validator_exception",
                "error": str(e),
            }
        is_valid, validation_payload = _as_bool_validation(validation)
        record_index = next(
            i for i, item in enumerate(validation_records)
            if item["index"] == candidate["index"]
        )
        record = {
            "index": candidate["index"],
            "distance": candidate["distance"],
            "valid": is_valid,
        }
        if validation_payload is not None:
            record.update(validation_payload)
        validation_records[record_index] = record
        if is_valid:
            return {
                "selected_index": candidate["index"],
                "selected_qpos": candidate["qpos"],
                "selected_distance": candidate["distance"],
                "top_k": effective_top_k,
                "validation": record,
                "top_candidates": validation_records,
            }

    fallback = top_candidates[0]
    fallback_record = {
        "index": fallback["index"],
        "distance": fallback["distance"],
        "valid": False,
        "reason": "validation_failed_fallback",
    }
    fallback_selection = {
        "selected_index": fallback["index"],
        "selected_qpos": fallback["qpos"],
        "selected_distance": fallback["distance"],
        "top_k": effective_top_k,
        "validation": fallback_record,
        "top_candidates": validation_records,
        "fallback_reason": "top_k_validation_exhausted",
    }
    raise NoValidQposCandidateError(
        "No valid qpos candidate found in Top-K validation. "
        f"top_k={effective_top_k}, validations={validation_records}",
        fallback_qpos=fallback["qpos"],
        fallback_selection=fallback_selection,
    )


def select_target_qpos_after_transition(
    stacked_qpos: list,
    current_joint_pos: np.ndarray,
    *,
    top_k: int = 3,
    path_validator: Callable[..., bool | dict] | None = None,
) -> dict:
    try:
        return select_target_qpos_candidate(
            stacked_qpos,
            current_joint_pos,
            top_k=top_k,
            path_validator=path_validator,
        )
    except NoValidQposCandidateError as e:
        if e.fallback_selection is not None:
            return e.fallback_selection
        if e.fallback_qpos is not None:
            fallback_qpos = np.asarray(e.fallback_qpos, dtype=np.float64).reshape(-1)
            return {
                "selected_index": 0,
                "selected_qpos": fallback_qpos,
                "selected_distance": _qpos_joint_distance(fallback_qpos, current_joint_pos),
                "top_k": max(1, int(top_k or 3)),
                "validation": {
                    "index": 0,
                    "distance": _qpos_joint_distance(fallback_qpos, current_joint_pos),
                    "valid": False,
                    "reason": "validation_failed_fallback",
                },
                "top_candidates": [],
                "fallback_reason": "top_k_validation_exhausted",
            }
        raise


def _is_no_valid_qpos_candidate_error(error: Exception) -> bool:
    return isinstance(error, NoValidQposCandidateError) or "No valid qpos candidate" in str(error)


def _retrieve_target_qpos_with_retry(
    retrieve_once: Callable[[int], np.ndarray],
    *,
    max_transition_regeneration_attempts: int = 1,
) -> np.ndarray:
    max_retries = max(0, int(max_transition_regeneration_attempts))
    total_attempts = max_retries + 1
    last_error = None
    for attempt_index in range(total_attempts):
        try:
            return retrieve_once(attempt_index)
        except Exception as e:
            if not _is_no_valid_qpos_candidate_error(e):
                raise
            last_error = e
            if attempt_index >= max_retries:
                break
            print(
                "[Transition] Top-K qpos validation failed for all candidates; "
                f"regenerating transition action ({attempt_index + 1}/{max_retries}). "
                f"Reason: {e}"
            )
    fallback_qpos = getattr(last_error, "fallback_qpos", None)
    if fallback_qpos is not None:
        print(
            "[Transition] Top-K qpos validation still failed after retry budget; "
            "falling back to nearest qpos candidate."
        )
        return np.asarray(fallback_qpos, dtype=np.float64).reshape(-1)
    raise last_error


def _contact_pairs(data) -> set[tuple[int, int]]:
    pairs = set()
    for i in range(int(getattr(data, "ncon", 0))):
        contact = data.contact[i]
        geom1 = int(contact.geom1)
        geom2 = int(contact.geom2)
        pairs.add(tuple(sorted((geom1, geom2))))
    return pairs


def _warning_counts(data) -> np.ndarray:
    warning = getattr(data, "warning", None)
    number = getattr(warning, "number", None)
    if number is None:
        return np.zeros(0, dtype=np.int64)
    return np.asarray(number)


def validate_qpos_interpolation_path(
    model,
    data,
    jnt_span,
    candidate_qpos,
    *,
    num_steps: int = 100,
    allow_existing_contacts: bool = True,
) -> dict:
    import mujoco

    try:
        jnt_indices = list(jnt_span)
        if len(jnt_indices) == 0:
            return {"valid": False, "reason": "empty_jnt_span"}

        target = np.asarray(candidate_qpos, dtype=np.float64).reshape(-1)[: len(jnt_indices)]
        if target.size != len(jnt_indices) or not np.isfinite(target).all():
            return {"valid": False, "reason": "invalid_target_qpos"}

        sim_data = mujoco.MjData(model)
        sim_data.qpos[:] = data.qpos
        sim_data.qvel[:] = data.qvel
        if getattr(sim_data, "ctrl", None) is not None and getattr(data, "ctrl", None) is not None:
            sim_data.ctrl[:] = data.ctrl
        mujoco.mj_forward(model, sim_data)

        start = np.asarray(sim_data.qpos[jnt_indices], dtype=np.float64).copy()
        baseline_contacts = _contact_pairs(sim_data) if allow_existing_contacts else set()
        steps = max(2, int(num_steps))

        for alpha in np.linspace(0.0, 1.0, steps):
            sim_data.qpos[jnt_indices] = start + alpha * (target - start)
            sim_data.qvel[:] = 0.0
            mujoco.mj_forward(model, sim_data)

            if not np.isfinite(sim_data.qpos).all() or not np.isfinite(sim_data.qvel).all():
                return {"valid": False, "reason": "nonfinite_state"}

            warnings = _warning_counts(sim_data)
            if warnings.size > 0 and np.any(warnings):
                return {
                    "valid": False,
                    "reason": "mujoco_warning",
                    "warnings": warnings.astype(int).tolist(),
                }

            current_contacts = _contact_pairs(sim_data)
            new_contacts = current_contacts - baseline_contacts
            if new_contacts:
                return {
                    "valid": False,
                    "reason": "new_collision",
                    "new_contacts": [list(pair) for pair in sorted(new_contacts)],
                }

        return {"valid": True, "reason": "path_validated", "num_steps": steps}
    except Exception as e:
        return {"valid": False, "reason": "validator_exception", "error": str(e)}


def validate_qpos_rrt_path(
    model,
    data,
    jnt_span,
    candidate_qpos,
    *,
    num_steps_per_segment: int = 100,
) -> dict:
    from non_llm_transition import (
        joint_ranges_from_model,
        plan_joint_path_rrt,
        validate_joint_path_in_mujoco,
    )

    try:
        jnt_indices = list(jnt_span)
        if len(jnt_indices) == 0:
            return {"valid": False, "reason": "empty_jnt_span"}

        start = np.asarray(data.qpos[jnt_indices], dtype=np.float64).reshape(-1)
        target = np.asarray(candidate_qpos, dtype=np.float64).reshape(-1)[: len(jnt_indices)]
        if target.size != len(jnt_indices) or not np.isfinite(target).all():
            return {"valid": False, "reason": "invalid_target_qpos"}

        path_plan = plan_joint_path_rrt(
            start,
            target,
            path_validator=lambda path: validate_joint_path_in_mujoco(
                model,
                data,
                jnt_span,
                path,
                num_steps_per_segment=int(num_steps_per_segment),
            ),
            joint_ranges=joint_ranges_from_model(model, jnt_span),
        )
        valid = path_plan.status != "RRT_FAILED_SKIP_ACTION" and bool(path_plan.waypoints)
        return {
            "valid": valid,
            "reason": "rrt_path_validated" if valid else "rrt_failed",
            "planner_status": path_plan.status,
            "planner_validation": path_plan.validation,
            "waypoint_count": len(path_plan.waypoints),
        }
    except Exception as e:
        return {"valid": False, "reason": "validator_exception", "error": str(e)}


def _build_task_prompt_index(qpos_db: dict | list):
    tasks = qpos_db if isinstance(qpos_db, list) else qpos_db.get("tasks", [])
    if not isinstance(tasks, list):
        raise ValueError("Invalid qpos database format: expected a list")

    by_prompt = {}
    for task in tasks:
        prompt = str(task.get("task", task.get("task_prompt", ""))).strip()
        if not prompt:
            continue
        by_prompt[prompt] = task
    if not by_prompt:
        raise ValueError("No valid task prompts found in qpos database")
    return by_prompt


def _candidate_qpos_from_selection(stacked_qpos: list, selection: dict) -> list[list[float]]:
    candidates = []
    for item in selection.get("top_candidates", []):
        index = int(item["index"])
        qpos = np.asarray(stacked_qpos[index], dtype=np.float64).reshape(-1)
        candidates.append(qpos.tolist())
    return candidates


def _candidate_values_from_selection(stacked_values: list | None, selection: dict) -> list:
    if not isinstance(stacked_values, list):
        return []
    candidates = []
    for item in selection.get("top_candidates", []):
        index = int(item["index"])
        candidates.append(stacked_values[index] if index < len(stacked_values) else None)
    return candidates


def _stacked_front_image_paths(matched: dict) -> list:
    paths = matched.get("initial_front_image_paths")
    if isinstance(paths, list):
        return paths
    entries = matched.get("entries", [])
    if isinstance(entries, list):
        return [entry.get("initial_front_image_path") for entry in entries]
    return []


def _fallback_find_qpos(
    db: dict | list,
    task_prompt: str,
    current_joint_pos: np.ndarray,
    *,
    top_k: int = 3,
    path_validator: Callable[..., bool | dict] | None = None,
    match_cutoff: float = 0.5,
):
    by_prompt = _build_task_prompt_index(db)

    local_match = _find_local_prompt_match(
        task_prompt,
        list(by_prompt.keys()),
        cutoff=match_cutoff,
    )
    if local_match is None:
        raise ValueError(f"No task prompt matched for: {task_prompt}")
    matched_prompt, _ = local_match

    matched = by_prompt[matched_prompt]
    stacked_qpos = matched.get("initial_qpos")
    if not isinstance(stacked_qpos, list) or len(stacked_qpos) == 0:
        # Backward compatibility: old format stored entries list.
        entries = matched.get("entries", [])
        if not entries:
            raise ValueError(f"Matched task has no qpos entries: {matched_prompt}")
        stacked_qpos = [entry.get("initial_qpos") for entry in entries if entry.get("initial_qpos") is not None]
        if not stacked_qpos:
            raise ValueError(f"Matched task entries have no initial_qpos: {matched_prompt}")
        selection = select_target_qpos_candidate(
            stacked_qpos,
            current_joint_pos,
            top_k=top_k,
            path_validator=path_validator,
        )
        selected_index = selection["selected_index"]
        qpos = stacked_qpos[selected_index]
        if qpos is None:
            raise ValueError(f"Matched task entry has no initial_qpos: {matched_prompt}")
        return matched_prompt, qpos, len(stacked_qpos), selected_index, selection

    selection = select_target_qpos_candidate(
        stacked_qpos,
        current_joint_pos,
        top_k=top_k,
        path_validator=path_validator,
    )
    selected_index = selection["selected_index"]
    return matched_prompt, stacked_qpos[selected_index], len(stacked_qpos), selected_index, selection


def retrieve_target_qpos_with_agent(
    client: OpenAI,
    model_name: str,
    task_prompt: str,
    current_joint_pos: np.ndarray,
    *,
    max_attempts: int = 3,
    temperature: float | None = None,
    top_p: float | None = None,
    max_tokens: int | None = None,
    timeout: float | None = None,
    backend_mode: str = "auto",
    thinking_mode: str = "auto",
    top_k: int = 3,
    path_validator: Callable[..., bool | dict] | None = None,
    local_retrieval_first: bool = False,
    local_retrieval_cutoff: float = 0.72,
    return_selection: bool = False,
):
    qpos_db_path = Path("logs/lerobot_initial_qpos.json")
    if not qpos_db_path.exists():
        raise FileNotFoundError(
            f"Qpos database not found at {qpos_db_path}. Please run export_lerobot_initial_qpos.py first."
        )

    with open(qpos_db_path, "r", encoding="utf-8") as f:
        qpos_db = json.load(f)

    by_prompt = _build_task_prompt_index(qpos_db)
    task_prompt_list = list(by_prompt.keys())
    retrieval_source = "llm"

    if local_retrieval_first:
        local_match = _find_local_prompt_match(
            task_prompt,
            task_prompt_list,
            cutoff=local_retrieval_cutoff,
        )
        if local_match is not None:
            try:
                matched_task_prompt, selected_qpos, candidate_count, selected_index, selection = _fallback_find_qpos(
                    qpos_db,
                    task_prompt,
                    current_joint_pos,
                    top_k=top_k,
                    path_validator=path_validator,
                    match_cutoff=local_retrieval_cutoff,
                )
                matched = by_prompt[matched_task_prompt]
                stacked_qpos = matched.get("initial_qpos")
                if not isinstance(stacked_qpos, list) or len(stacked_qpos) == 0:
                    entries = matched.get("entries", [])
                    stacked_qpos = [
                        entry.get("initial_qpos")
                        for entry in entries
                        if entry.get("initial_qpos") is not None
                    ]
                requested_prompt = task_prompt
                retrieval_source = "local_fuzzy"
                print(
                    "[qpos-retrieval] local match: "
                    f"{matched_task_prompt} (score={local_match[1]:.3f})"
                )
                selected_qpos_arr = np.asarray(selected_qpos, dtype=np.float64).reshape(-1)
                if selected_qpos_arr.size == 0:
                    raise ValueError("Retrieved selected_qpos is empty")

                selected_payload = {
                    "requested_task_prompt": requested_prompt,
                    "matched_task_prompt": matched_task_prompt,
                    "retrieval_source": retrieval_source,
                    "selected_index": selected_index,
                    "candidate_count": candidate_count,
                    "top_k": max(1, int(top_k or 3)),
                    "selected_qpos": selected_qpos_arr.tolist(),
                    "selection": {
                        **selection,
                        "selected_qpos": np.asarray(selection["selected_qpos"], dtype=np.float64).reshape(-1).tolist(),
                    },
                    "target_qpos_candidates": _candidate_qpos_from_selection(stacked_qpos, selection),
                    "target_front_image_paths": _candidate_values_from_selection(
                        _stacked_front_image_paths(matched),
                        selection,
                    ),
                }
                out_path = Path("logs/target_qpos_selected.json")
                out_path.parent.mkdir(parents=True, exist_ok=True)
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(selected_payload, f, ensure_ascii=False, indent=2)

                print(f"Saved retrieved target qpos to: {out_path}")
                if return_selection:
                    return selected_qpos_arr, selected_payload
                return selected_qpos_arr
            except NoValidQposCandidateError:
                raise
            except Exception as e:
                print(f"Local qpos retrieval failed, fallback to LLM retrieval: {e}")

    retrieval_model = model_name
    retrieval_prompt = f"""
You are a retrieval agent for robot transition initialization.

Target task prompt to retrieve: {task_prompt}

Candidate task prompt list:
{json.dumps(task_prompt_list, ensure_ascii=False)}

Return strictly one JSON object with this exact schema:
{{
  "requested_task_prompt": "...",
  "matched_task_prompt": "...",
  "selection_reason": "short reason"
}}

Rules:
1. Only choose matched_task_prompt from the provided candidate list.
2. Match by semantic equivalence of the target prompt.
3. Do not output markdown, do not output extra text.
""".strip()

    try:
        retrieval_obj = _request_json_object(
            client=client,
            model_name=retrieval_model,
            request_input=[{"role": "user", "content": [{"type": "input_text", "text": retrieval_prompt}]}],
            stage_name="qpos-retrieval",
            max_attempts=max_attempts,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            timeout=timeout,
            backend_mode=backend_mode,
            thinking_mode=thinking_mode,
        )
        print(retrieval_obj)
        requested_prompt = str(retrieval_obj.get("requested_task_prompt", task_prompt))
        matched_task_prompt = str(retrieval_obj["matched_task_prompt"]).strip()

        if matched_task_prompt not in by_prompt:
            # If LLM returns a near variant, use local fuzzy normalization.
            matched_task_prompt, _, _, _, _ = _fallback_find_qpos(
                qpos_db,
                matched_task_prompt,
                current_joint_pos,
                top_k=top_k,
                path_validator=path_validator,
                match_cutoff=0.5,
            )

        matched = by_prompt[matched_task_prompt]
        stacked_qpos = matched.get("initial_qpos")
        if not isinstance(stacked_qpos, list) or len(stacked_qpos) == 0:
            entries = matched.get("entries", [])
            if not entries:
                raise ValueError(f"Matched task has no qpos entries: {matched_task_prompt}")
            stacked_qpos = [entry.get("initial_qpos") for entry in entries if entry.get("initial_qpos") is not None]
            if not stacked_qpos:
                raise ValueError(f"Matched task entries have no initial_qpos: {matched_task_prompt}")
            selection = select_target_qpos_candidate(
                stacked_qpos,
                current_joint_pos,
                top_k=top_k,
                path_validator=path_validator,
            )
            selected_index = selection["selected_index"]
            selected_qpos = stacked_qpos[selected_index]
            candidate_count = len(stacked_qpos)
        else:
            selection = select_target_qpos_candidate(
                stacked_qpos,
                current_joint_pos,
                top_k=top_k,
                path_validator=path_validator,
            )
            selected_index = selection["selected_index"]
            selected_qpos = stacked_qpos[selected_index]
            candidate_count = len(stacked_qpos)
    except Exception as e:
        print(f"LLM retrieval failed, fallback to local retrieval: {e}")
        requested_prompt = task_prompt
        matched_task_prompt, selected_qpos, candidate_count, selected_index, selection = _fallback_find_qpos(
            qpos_db,
            task_prompt,
            current_joint_pos,
            top_k=top_k,
            path_validator=path_validator,
            match_cutoff=0.5,
        )
        matched = by_prompt[matched_task_prompt]
        stacked_qpos = matched.get("initial_qpos")
        if not isinstance(stacked_qpos, list) or len(stacked_qpos) == 0:
            entries = matched.get("entries", [])
            stacked_qpos = [entry.get("initial_qpos") for entry in entries if entry.get("initial_qpos") is not None]
        retrieval_source = "local_fallback"

    selected_qpos_arr = np.asarray(selected_qpos, dtype=np.float64).reshape(-1)
    if selected_qpos_arr.size == 0:
        raise ValueError("Retrieved selected_qpos is empty")

    selected_payload = {
        "requested_task_prompt": requested_prompt,
        "matched_task_prompt": matched_task_prompt,
        "retrieval_source": retrieval_source,
        "selected_index": selected_index,
        "candidate_count": candidate_count,
        "top_k": max(1, int(top_k or 3)),
        "selected_qpos": selected_qpos_arr.tolist(),
        "selection": {
            **selection,
            "selected_qpos": np.asarray(selection["selected_qpos"], dtype=np.float64).reshape(-1).tolist(),
        },
        "target_qpos_candidates": _candidate_qpos_from_selection(stacked_qpos, selection),
        "target_front_image_paths": _candidate_values_from_selection(
            _stacked_front_image_paths(matched),
            selection,
        ),
    }
    out_path = Path("logs/target_qpos_selected.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(selected_payload, f, ensure_ascii=False, indent=2)

    print(f"Saved retrieved target qpos to: {out_path}")
    if return_selection:
        return selected_qpos_arr, selected_payload
    return selected_qpos_arr

def transition_code_generation(
    task_prompt: str,
    no_planning: bool = False,
    no_interpolation: bool = False,
    no_retrieval: bool = False,
    llm_config: dict | None = None,
    target_top_k: int = 3,
    qpos_path_validator: Callable[..., bool | dict] | None = None,
    max_transition_regeneration_attempts: int = 1,
    target_ee_position_resolver: Callable[[np.ndarray], Any] | None = None,
):
    llm_config = llm_config or {}

    def _pick_config(key: str, env_key: str, default=None):
        value = llm_config.get(key)
        if value is None:
            value = os.environ.get(env_key)
        return default if value is None else value

    def _to_optional_float(value, field: str):
        if value is None or value == "":
            return None
        try:
            return float(value)
        except Exception as e:
            raise ValueError(f"Invalid {field}: {value}") from e

    def _to_optional_int(value, field: str):
        if value is None or value == "":
            return None
        try:
            return int(value)
        except Exception as e:
            raise ValueError(f"Invalid {field}: {value}") from e

    def _to_bool(value, field: str) -> bool:
        if isinstance(value, bool):
            return value
        text = str(value).strip().lower()
        if text in {"1", "true", "yes", "y", "on"}:
            return True
        if text in {"0", "false", "no", "n", "off"}:
            return False
        raise ValueError(f"Invalid {field}: {value}")

    base_url = _pick_config("base_url", "BASE_URL")
    api_key = _pick_config("api_key", "API_KEY")
    model_name = _pick_config("model_name", "MODEL_NAME")
    temperature = _to_optional_float(_pick_config("temperature", "LLM_TEMPERATURE"), "temperature")
    top_p = _to_optional_float(_pick_config("top_p", "LLM_TOP_P"), "top_p")
    max_tokens = _to_optional_int(_pick_config("max_tokens", "LLM_MAX_TOKENS"), "max_tokens")
    max_attempts = _to_optional_int(_pick_config("max_attempts", "LLM_MAX_ATTEMPTS", 3), "max_attempts")
    timeout = _to_optional_float(_pick_config("timeout", "LLM_TIMEOUT"), "timeout")
    backend_mode = _normalize_backend_mode(str(_pick_config("backend_mode", "LLM_BACKEND_MODE", "auto")))
    effective_backend_mode = _resolve_backend_mode(backend_mode, base_url)
    thinking_mode = _normalize_thinking_mode(str(_pick_config("thinking", "LLM_THINKING", "auto")))
    image_max_side = _to_optional_int(_pick_config("image_max_side", "LLM_IMAGE_MAX_SIDE", 768), "image_max_side")
    image_quality = _to_optional_int(_pick_config("image_quality", "LLM_IMAGE_QUALITY", 80), "image_quality")
    verifier_enabled = _to_bool(
        _pick_config("verifier_enabled", "LLM_PLAN_VERIFIER", False),
        "verifier_enabled",
    )
    max_plan_verification_revisions = _to_optional_int(
        _pick_config("max_plan_verification_revisions", "LLM_PLAN_VERIFICATION_REVISIONS", 2),
        "max_plan_verification_revisions",
    )
    if max_plan_verification_revisions is None:
        max_plan_verification_revisions = 2
    if max_plan_verification_revisions < 0:
        raise ValueError("max_plan_verification_revisions must be >= 0")
    local_retrieval_first = _to_bool(
        _pick_config("local_retrieval_first", "LLM_LOCAL_RETRIEVAL_FIRST", False),
        "local_retrieval_first",
    )
    local_retrieval_cutoff = _to_optional_float(
        _pick_config("local_retrieval_cutoff", "LLM_LOCAL_RETRIEVAL_CUTOFF", 0.72),
        "local_retrieval_cutoff",
    )
    ready_memory_enabled = _to_bool(
        _pick_config("ready_memory_enabled", "LLM_READY_MEMORY_ENABLED", False),
        "ready_memory_enabled",
    )
    ready_memory_db_path = _pick_config("ready_memory_db_path", "READY_MEMORY_DB", None)
    ready_memory_repo_id = _pick_config("ready_memory_repo_id", "READY_MEMORY_REPO_ID", None)
    ready_memory_episode_index = _to_optional_int(
        _pick_config("ready_memory_episode_index", "READY_MEMORY_EPISODE_INDEX", None),
        "ready_memory_episode_index",
    )
    ready_memory_window_size = _to_optional_float(
        _pick_config("ready_memory_window_size", "READY_MEMORY_WINDOW_SIZE", 20.0),
        "ready_memory_window_size",
    )
    ready_memory_min_frame_ratio = _to_optional_float(
        _pick_config("ready_memory_min_frame_ratio", "READY_MEMORY_MIN_FRAME_RATIO", 0.05),
        "ready_memory_min_frame_ratio",
    )
    ready_memory_max_iterations = _to_optional_int(
        _pick_config("ready_memory_max_iterations", "READY_MEMORY_MAX_ITERATIONS", 4),
        "ready_memory_max_iterations",
    )
    ready_memory_match_cutoff = _to_optional_float(
        _pick_config("ready_memory_match_cutoff", "READY_MEMORY_MATCH_CUTOFF", 0.5),
        "ready_memory_match_cutoff",
    )
    ready_memory_front_image_key = str(
        _pick_config("ready_memory_front_image_key", "READY_MEMORY_FRONT_IMAGE_KEY", "observation/image")
    )
    ready_memory_output_path = Path(
        str(_pick_config("ready_memory_output_path", "READY_MEMORY_OUTPUT", "logs/target_ready_state_selected.json"))
    )
    ready_memory_image_output_dir = Path(
        str(_pick_config("ready_memory_image_output_dir", "READY_MEMORY_IMAGE_DIR", "logs/ready_memory_images"))
    )
    if ready_memory_window_size is None:
        ready_memory_window_size = 20.0
    if ready_memory_min_frame_ratio is None:
        ready_memory_min_frame_ratio = 0.05
    if ready_memory_max_iterations is None:
        ready_memory_max_iterations = 4
    if ready_memory_match_cutoff is None:
        ready_memory_match_cutoff = 0.5

    if backend_mode == "auto":
        if effective_backend_mode == "chat":
            print("[LLM] auto backend mode: local endpoint detected, using chat.completions")
        else:
            print("[LLM] auto backend mode: remote endpoint detected, using responses API")

    if model_name is None or str(model_name).strip() == "":
        raise ValueError("MODEL_NAME (or --llm-model-name) must be provided for transition generation")
    if max_attempts is None:
        max_attempts = 3

    client_kwargs = {}
    if api_key is not None:
        client_kwargs["api_key"] = api_key
    if base_url is not None:
        client_kwargs["base_url"] = base_url
    if timeout is not None and timeout > 0:
        client_kwargs["timeout"] = timeout
    client = OpenAI(**client_kwargs)

    current_joint_pos_arr = np.asarray(np.load('logs/current_joint.npy'), dtype=np.float64).reshape(-1)
    if no_retrieval:
        target_joint_pos_arr = current_joint_pos_arr.copy()
        target_retrieval_payload = {
            "requested_task_prompt": task_prompt,
            "matched_task_prompt": None,
            "retrieval_source": "disabled",
            "target_qpos_candidates": [],
            "target_front_image_paths": [],
        }
        target_qpos_candidates = []
        target_front_image_paths = []
    elif ready_memory_enabled:
        from ready_memory_retrieval_agent import (
            retrieve_ready_memory_from_episode,
            retrieve_ready_memory_from_index,
        )

        ready_llm_config = {
            "model_name": model_name,
            "base_url": base_url,
            "api_key": api_key,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "max_attempts": max_attempts,
            "timeout": timeout,
            "backend_mode": effective_backend_mode,
            "thinking": thinking_mode,
            "max_image_side": image_max_side,
        }
        if ready_memory_db_path:
            target_retrieval_payload = retrieve_ready_memory_from_index(
                memory_db_path=ready_memory_db_path,
                task_prompt=task_prompt,
                window_size=ready_memory_window_size,
                output_path=ready_memory_output_path,
                min_frame_ratio=ready_memory_min_frame_ratio,
                max_iterations=ready_memory_max_iterations,
                llm_config=ready_llm_config,
                match_cutoff=ready_memory_match_cutoff,
                client=client,
            )
        elif ready_memory_repo_id:
            target_retrieval_payload = retrieve_ready_memory_from_episode(
                repo_id=str(ready_memory_repo_id),
                task_prompt=task_prompt,
                episode_index=ready_memory_episode_index,
                window_size=ready_memory_window_size,
                output_path=ready_memory_output_path,
                image_output_dir=ready_memory_image_output_dir,
                front_image_key=ready_memory_front_image_key,
                min_frame_ratio=ready_memory_min_frame_ratio,
                max_iterations=ready_memory_max_iterations,
                llm_config=ready_llm_config,
            )
        else:
            raise ValueError(
                "Ready memory retrieval is enabled, but neither READY_MEMORY_DB "
                "nor READY_MEMORY_REPO_ID is configured."
            )

        target_state = target_retrieval_payload.get(
            "target_state",
            target_retrieval_payload.get("target_qpos"),
        )
        target_joint_pos_arr = np.asarray(target_state, dtype=np.float64).reshape(-1)
        if target_joint_pos_arr.size == 0 or not np.isfinite(target_joint_pos_arr).all():
            raise ValueError("ReadyStateAgent returned an invalid empty/nonfinite target_state")
        target_retrieval_payload = {
            **target_retrieval_payload,
            "retrieval_source": target_retrieval_payload.get("retrieval_source", "ready_memory"),
            "selected_qpos": target_joint_pos_arr.tolist(),
            "target_qpos_candidates": [target_joint_pos_arr.tolist()],
            "target_front_image_paths": target_retrieval_payload.get("target_front_image_paths")
            or [target_retrieval_payload.get("target_front_image_path")],
        }
        _write_json("logs/target_qpos_selected.json", target_retrieval_payload)
        target_qpos_candidates = target_retrieval_payload.get("target_qpos_candidates", [])
        target_front_image_paths = target_retrieval_payload.get("target_front_image_paths", [])
    else:
        def retrieve_once(_attempt_index: int):
            return retrieve_target_qpos_with_agent(
                client,
                model_name,
                task_prompt,
                current_joint_pos_arr,
                max_attempts=max_attempts,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                timeout=timeout,
                backend_mode=effective_backend_mode,
                thinking_mode=thinking_mode,
                top_k=target_top_k,
                path_validator=None,
                local_retrieval_first=local_retrieval_first,
                local_retrieval_cutoff=local_retrieval_cutoff or 0.72,
                return_selection=True,
            )

        target_joint_pos_arr, target_retrieval_payload = _retrieve_target_qpos_with_retry(
            retrieve_once,
            max_transition_regeneration_attempts=max_transition_regeneration_attempts,
        )
        target_qpos_candidates = target_retrieval_payload.get("target_qpos_candidates", [])
        target_front_image_paths = target_retrieval_payload.get("target_front_image_paths", [])
    target_joint_pos = str(target_joint_pos_arr.tolist())
    target_arm_qpos = target_joint_pos_arr[:6].tolist()
    target_gripper_state = None if no_retrieval else (
        float(target_joint_pos_arr[-1]) if target_joint_pos_arr.size > 6 else None
    )
    current_joint_pos = str(current_joint_pos_arr.tolist())
    template_code = read_file('scripts/autobio_scripts/transition_template.py')
    front_image_data_url = None
    side_image_data_url = None
    target_front_image_data_url = None
    calibration_payload = _load_calibration_payload()
    if calibration_payload is not None and not no_retrieval:
        target_ee_payload = _target_end_effector_payload_from_resolver(
            target_joint_pos_arr,
            target_ee_position_resolver,
            site_name=(calibration_payload.get("end_effector") or {}).get("site_name"),
        )
        if target_ee_payload is not None:
            calibration_payload = dict(calibration_payload)
            calibration_payload["target_end_effector"] = target_ee_payload
            _write_json("logs/transition_calibration.json", calibration_payload)
    calibration_prompt_text = format_calibration_for_llm(calibration_payload) if calibration_payload else ""
    spatial_context_prompt_text = ""
    reachability_prompt_text = _format_ee_reachability_for_prompt(False)
    motion_constraint_rule = (
        "The planning verifier only enforces per-command movement limits: every translate "
        "abs(distance_m) <= 0.25m and every rotate abs(angle_deg) <= 180. "
        f"Every translate/rotate command must use steps >= {MIN_TRANSITION_MOTION_STEPS} when steps are present. "
        "It does not check IK reachability or cumulative workspace reachability."
    )
    if not no_planning:
        front_image_path = _resolve_first_existing_path(
            ["current_front_calibrated.png", "current_view.png"],
            base_dir=Path("logs"),
        ) or Path("logs/current_view.png")
        side_image_path = _resolve_first_existing_path(
            ["current_side_calibrated.png", "current_side_view.png"],
            base_dir=Path("logs"),
        ) or Path("logs/current_side_view.png")
        front_image_data_url = file_to_data_url(
            str(front_image_path),
            max_image_side=image_max_side,
            image_quality=image_quality or 80,
        )
        side_image_data_url = file_to_data_url(
            str(side_image_path),
            max_image_side=image_max_side,
            image_quality=image_quality or 80,
        )
        target_front_image_path = _resolve_first_existing_path(
            target_front_image_paths,
            base_dir=Path("logs"),
        )
        if target_front_image_path is not None:
            target_front_image_data_url = file_to_data_url(
                str(target_front_image_path),
                max_image_side=image_max_side,
                image_quality=image_quality or 80,
            )

        if calibration_payload is not None:
            obstacle_prompt = f"""
You are a visual obstacle perception tool for robot transition planning.

Task prompt:
{task_prompt}

Use the calibrated front and side images to identify physical objects that may obstruct the gripper when moving from the current task state toward the retrieved target preparation pose for the next atomic task.

Return strictly one JSON object:
{{
  "obstacles": [
    {{
      "name": "short object name",
      "front_bbox": [x1, y1, x2, y2],
      "side_bbox": [x1, y1, x2, y2],
      "confidence": 0.0,
      "risk_reason": "why this object may block the gripper"
    }}
  ]
}}

Rules:
1. Use pixel coordinates in the provided images.
2. Include only physical objects that could affect the robot gripper path.
3. Do not include robot links, background, image labels, or text overlays as obstacles.
4. If no obstacle is visible, return {{"obstacles": []}}.
5. Do not output markdown or extra text.
""".strip()
            try:
                obstacle_vlm_output = _request_json_object(
                    client=client,
                    model_name=model_name,
                    request_input=[{
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": obstacle_prompt},
                            {"type": "input_image", "image_url": front_image_data_url},
                            {"type": "input_image", "image_url": side_image_data_url},
                        ],
                    }],
                    stage_name="stage-0-obstacle-perception",
                    max_attempts=max_attempts,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    timeout=timeout,
                    backend_mode=effective_backend_mode,
                    thinking_mode=thinking_mode,
                )
                _write_json("logs/transition_obstacles_vlm.json", obstacle_vlm_output)
                current_gripper_position = (calibration_payload.get("end_effector") or {}).get("position_world")
                target_gripper_position = (calibration_payload.get("target_end_effector") or {}).get("position_world")
                spatial_context = estimate_obstacles_from_vlm_output(
                    obstacle_vlm_output,
                    calibration_payload,
                    current_gripper_position=current_gripper_position,
                    target_gripper_position=target_gripper_position,
                )
                _write_json("logs/transition_spatial_context.json", spatial_context)
                spatial_context_prompt_text = format_spatial_context_for_llm(spatial_context)
            except Exception as e:
                print(f"[Calibration] stage-0 obstacle perception failed: {_sanitize_error_text(e)}")

    if no_retrieval:
        target_reference_text = (
            "unavailable. No retrieved target qpos or target reference image is provided in this no-retrieval ablation."
        )
        target_binding_text = (
            "The TARGET FRONT reference view is unavailable; plan only from the current views and target prompt."
        )
    elif ready_memory_enabled:
        target_reference_text = (
            "the target task's ReadyStateAgent-retrieved READY FRONT reference image, aligned with the retrieved target_state."
        )
        target_binding_text = (
            "The third image, when present, is the TARGET READY FRONT reference view selected by ReadyStateAgent."
        )
    else:
        target_reference_text = (
            "the target task's retrieved initial front-view image, aligned with the retrieved target state candidates."
        )
        target_binding_text = (
            "The third image, when present, is the TARGET INITIAL FRONT reference view."
        )
    restore_schema_fields = _format_restore_schema_fields(
        no_retrieval=no_retrieval,
        target_arm_qpos=target_arm_qpos,
        target_gripper_state=target_gripper_state,
    )

    planning_prompt = _build_transition_planning_prompt(
        target_reference_text=target_reference_text,
        target_binding_text=target_binding_text,
        calibration_prompt_text=calibration_prompt_text,
        spatial_context_prompt_text=spatial_context_prompt_text,
        reachability_prompt_text=reachability_prompt_text,
        restore_schema_fields=restore_schema_fields,
        motion_constraint_rule=motion_constraint_rule,
    )
    
    print("Next task prompt:", task_prompt)
    print("Current joint pos:", current_joint_pos)
    if no_retrieval:
        print("Target joint pos: retrieval disabled (no-retrieval mode)")
    else:
        print("Target joint pos:", target_joint_pos)

    if no_planning:
        print("[Transition] no_planning=True: skipping planning and code generation, only applying target-qpos restore block.")
        code = _replace_execute_body(
            template_code,
            execute_body_code="",
            final_target_qpos=target_arm_qpos,
            final_target_gripper=target_gripper_state,
            include_final_restore=(not no_interpolation and not no_retrieval),
            final_target_qpos_candidates=target_qpos_candidates,
            target_top_k=target_top_k,
        )
        is_valid, validation_msg = validate_code(code)
        if is_valid:
            with open('scripts/autobio_scripts/transition_template.py', 'w', encoding='utf-8') as f:
                f.write(code)
            print("Updated transition_template.py")
            return
        raise ValueError(f"Generated template code is invalid in no_planning mode: {validation_msg}")

    plan_obj, transition_commands, plan_verification = _generate_verified_transition_plan(
        client=client,
        model_name=model_name,
        planning_prompt=planning_prompt,
        task_prompt=task_prompt,
        front_image_data_url=front_image_data_url,
        side_image_data_url=side_image_data_url,
        target_front_image_data_url=target_front_image_data_url,
        verifier_enabled=verifier_enabled,
        max_plan_revisions=max_plan_verification_revisions,
        max_attempts=max_attempts,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        timeout=timeout,
        backend_mode=effective_backend_mode,
        thinking_mode=thinking_mode,
        write_logs=True,
    )
    plan_steps = plan_obj.get("plan_steps", [])
    if plan_steps is not None and not isinstance(plan_steps, list):
        raise ValueError("Stage-1 planning output has invalid plan_steps")

    include_final_restore = not no_interpolation and not no_retrieval
    plan_target_qpos, plan_target_gripper = _resolve_plan_restore_targets(
        plan_obj=plan_obj,
        target_arm_qpos=target_arm_qpos,
        target_gripper_state=target_gripper_state,
        include_final_restore=include_final_restore,
    )

    plan_out_path = Path("logs/transition_plan.json")
    plan_out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(plan_out_path, "w", encoding="utf-8") as f:
        json.dump(plan_obj, f, ensure_ascii=False, indent=2)
    print(f"Saved transition plan to: {plan_out_path}")
    _write_json("logs/transition_plan_verification.json", plan_verification)

    print("🚀 Stage 2: Compiling planner commands into transition primitive calls...")
    execute_body = _commands_to_execute_body(
        transition_commands,
        enforce_plan_constraints=verifier_enabled,
    )
    code = _replace_execute_body(
        template_code,
        execute_body,
        final_target_qpos=plan_target_qpos,
        final_target_gripper=plan_target_gripper,
        include_final_restore=include_final_restore,
        final_target_qpos_candidates=target_qpos_candidates,
        target_top_k=target_top_k,
    )

    is_valid, validation_msg = validate_code(code)
    if is_valid:
        print(f"✅ The generated code is syntactically correct.")
        with open('scripts/autobio_scripts/transition_template.py', 'w', encoding='utf-8') as f:
            f.write(code)
        print("Updated transition_template.py")
    else:
        last_error = validation_msg
        print(f"❌ The generated code contains syntax errors. Error message:")
        print(f"   {validation_msg[:300]}...")
