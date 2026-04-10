import os
import base64
import mimetypes
from pathlib import Path
from openai import OpenAI
import ast
import numpy as np
import json
import re
import textwrap
from difflib import get_close_matches
from typing import Any
from urllib.parse import urlparse


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

def file_to_data_url(path: str) -> str:
    """
    Read a local image file and convert it to a base64-encoded data URL:
    data:image/jpeg;base64,...
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Image not found: {p.resolve()}")

    mime, _ = mimetypes.guess_type(str(p))
    mime = mime or "image/png"  # fallback

    b64 = base64.b64encode(p.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{b64}"

def validate_code(code):
    try:
        # 尝试解析代码
        tree = ast.parse(code)
        
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


def _collect_execute_allowed_apis(template_code: str) -> list[str]:
    api_names = []
    for name in re.findall(r"^\s{4}def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", template_code, flags=re.MULTILINE):
        if name not in {"__init__", "execute"}:
            api_names.append(name)
    return api_names


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
        new_method.append(f"        target_qpos = {list(final_target_qpos)}")
        if final_target_gripper is None:
            new_method.append("        target_gripper = None")
        else:
            new_method.append(f"        target_gripper = {float(final_target_gripper)}")
        new_method.append("        self.move_to_target_qpos(target_qpos)")
        new_method.append("        self.gripper_control(target_gripper)")
    else:
        new_method.append("")
        new_method.append("        # Final move_to_target_qpos is disabled by no_interpolation mode.")
        if final_target_gripper is not None:
            new_method.append(f"        target_gripper = {float(final_target_gripper)}")
            new_method.append("        self.gripper_control(target_gripper)")

    updated_lines = lines[:start] + new_method + lines[end:]
    out = "\n".join(updated_lines)
    if template_code.endswith("\n"):
        out += "\n"
    return out


def _normalize_prompt(text: str) -> str:
    return " ".join(text.lower().strip().split())


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


def _fallback_find_qpos(db: dict | list, task_prompt: str, current_joint_pos: np.ndarray):
    by_prompt = _build_task_prompt_index(db)

    norm_prompt = _normalize_prompt(task_prompt)
    norm_map = {_normalize_prompt(k): k for k in by_prompt}

    if norm_prompt in norm_map:
        matched_prompt = norm_map[norm_prompt]
    else:
        choices = list(norm_map.keys())
        candidates = get_close_matches(norm_prompt, choices, n=1, cutoff=0.5)
        if not candidates:
            raise ValueError(f"No task prompt matched for: {task_prompt}")
        matched_prompt = norm_map[candidates[0]]

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
        selected_index = _pick_nearest_index(stacked_qpos, current_joint_pos)
        qpos = stacked_qpos[selected_index]
        if qpos is None:
            raise ValueError(f"Matched task entry has no initial_qpos: {matched_prompt}")
        return matched_prompt, qpos, len(stacked_qpos), selected_index

    selected_index = _pick_nearest_index(stacked_qpos, current_joint_pos)
    return matched_prompt, stacked_qpos[selected_index], len(stacked_qpos), selected_index


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
            matched_task_prompt, _, _, _ = _fallback_find_qpos(
                qpos_db, matched_task_prompt, current_joint_pos
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
            selected_index = _pick_nearest_index(stacked_qpos, current_joint_pos)
            selected_qpos = stacked_qpos[selected_index]
            candidate_count = len(stacked_qpos)
        else:
            selected_index = _pick_nearest_index(stacked_qpos, current_joint_pos)
            selected_qpos = stacked_qpos[selected_index]
            candidate_count = len(stacked_qpos)
    except Exception as e:
        print(f"LLM retrieval failed, fallback to local retrieval: {e}")
        requested_prompt = task_prompt
        matched_task_prompt, selected_qpos, candidate_count, selected_index = _fallback_find_qpos(
            qpos_db, task_prompt, current_joint_pos
        )

    selected_qpos_arr = np.asarray(selected_qpos, dtype=np.float64).reshape(-1)
    if selected_qpos_arr.size == 0:
        raise ValueError("Retrieved selected_qpos is empty")

    selected_payload = {
        "requested_task_prompt": requested_prompt,
        "matched_task_prompt": matched_task_prompt,
        "selected_index": selected_index,
        "candidate_count": candidate_count,
        "selected_qpos": selected_qpos_arr.tolist(),
    }
    out_path = Path("logs/target_qpos_selected.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(selected_payload, f, ensure_ascii=False, indent=2)

    print(f"Saved retrieved target qpos to: {out_path}")
    return selected_qpos_arr

def transition_code_generation(
    task_prompt: str,
    no_planning: bool = False,
    no_interpolation: bool = False,
    llm_config: dict | None = None,
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
    target_joint_pos_arr = retrieve_target_qpos_with_agent(
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
    )
    target_joint_pos = str(target_joint_pos_arr.tolist())
    target_arm_qpos = target_joint_pos_arr[:6].tolist()
    target_gripper_state = float(target_joint_pos_arr[-1]) if target_joint_pos_arr.size > 6 else None
    current_joint_pos = str(current_joint_pos_arr.tolist())
    template_code = read_file('scripts/autobio_scripts/transition_template.py')
    allowed_apis = _collect_execute_allowed_apis(template_code)
    front_image_data_url = None
    side_image_data_url = None
    if not no_planning:
        front_image_data_url = file_to_data_url('logs/current_view.png')
        side_image_data_url = file_to_data_url('logs/current_side_view.png')

    planning_prompt = f'''
You are a robot transition planner.

Task:
Generate a concise path-planning list for transition execution, not code.

Inputs:
front camera image: front-to-back is x-axis, left-to-right is y-axis, and up-and-down is z-axis.
left camera image: use this as complementary geometric evidence for occlusions, depth relation, and side clearance.

Planning objective:
- Safety-first, collision-avoidance.
- Main sequence: obstacle clearance.

Planning Rules:
First, the current image is observed to analyze the states of the objects, the robotic arm, and the gripper within the scene. 
Next, a determination is made as to whether the gripper requires releasing. (mostly should be freed) 
Subsequently, the End-Effector (EE) is maneuvered away from all obstacles visible from any viewpoint through a combination of translational and rotational movements. 

Image binding for this request:
- The first image is the FRONT view.
- The second image is the LEFT view.
- You must jointly reason over both images before generating the plan.

Return strictly one JSON object with schema:
{{
    "plan_steps": [
        "step 1: gripper free...",
        "step 2: move...",
        "step 3: move...",
        ...
        "step n: move...",
    ],
    "safety_notes": ["...", "..."],
    "final_target_qpos": {target_arm_qpos},
    "final_target_gripper": {target_gripper_state}
}}

Rules:
1. plan_steps must be ordered, actionable, and short.
2. Each step should be a movement instruction for the robot, such as "move EE 10cm along +x", "rotate EE 90 degrees around z-axis", "open gripper", "close gripper", etc.
3. Do not output code.
4. Do not output markdown.
'''

    codegen_prompt_template = '''
You are a professional Python programmer.

Task:
Please generate the corresponding instruction code based on the plan_steps defined in the plan JSON file.

Planning list (must be followed):
{plan_steps_json}

Allowed APIs:
1. get_site_pose(self, data: mujoco.MjData) -> Pose # get the current pose of the specified site, which can be the EE or any obstacle.
2. interpolate(self, start: Pose, end: Pose, num_steps: int) -> list[Pose] # generate a list of linearly interpolated poses between start and end, with num_steps in total.
3. path_follow(self, path: list[Pose]) # follow the given path by controlling the robot in a closed-loop manner. The path is generated by interpolate() and can be updated online by replanning.
4. move_to(self, pose: Pose, num_steps: int = 100) # move the EE to the specified pose by interpolation. This is a simplified wrapper of interpolate() + path_follow() for direct point-to-point movement.
5. gripper_control(self, value: float, delay: int = 300) # control the gripper to the specified value (0~250, where 0 is fully open and 250 is fully closed), and hold for delay steps.
6. rotate_gripper(self, angle, axis, cur_quat) -> target_quat # rotate the gripper by the specified angle (in degrees) around the specified axis (x/y/z) from the current gripper orientation, and return the target gripper quaternion.

Here is some reference code; you can use this style as a guide for your own implementation:
```python
# Gripper control example: (0~250) 0:open, 250:fully close
# free the gripper
self.gripper_control(0)

# move to a safe place. make gripper away from objects.
# X-axis translation example, from current EE pose
cur_pose = self.get_site_pose(self.data)
end_pose = Pose(pos=cur_pose.pos + (0.1, 0.0, 0.0), quat=cur_pose.quat)
path = self.interpolate(cur_pose, end_pose, 100)
self.path_follow(path)

# Y-axis translation example, from current EE pose
cur_pose = self.get_site_pose(self.data)
end_pose = Pose(pos=cur_pose.pos + (0.0, 0.1, 0.0), quat=cur_pose.quat)
path = self.interpolate(cur_pose, end_pose, 100)
self.path_follow(path)

# Z-axis translation example, from current EE pose
cur_pose = self.get_site_pose(self.data)
end_pose = Pose(pos=cur_pose.pos + (0.0, 0.0, 0.1), quat=cur_pose.quat)
path = self.interpolate(cur_pose, end_pose, 100)
self.path_follow(path)
```

Output rules:
1. Do not output class/function definitions.
2. Do not include markdown fences.
3. Do not include this line (it will be inserted automatically): self.ik.initial_qpos = self.data.qpos[self.jnt_span]
4. Do not generate final target restoration lines (`target_qpos`, `target_gripper`, `move_to_target_qpos`). They are inserted by host code.
'''
    
    print("Next task prompt:", task_prompt)
    print("Current joint pos:", current_joint_pos)
    print("Target joint pos:", target_joint_pos)

    if no_planning:
        print("[Transition] no_planning=True: skipping planning and code generation, only applying target-qpos restore block.")
        code = _replace_execute_body(
            template_code,
            execute_body_code="",
            final_target_qpos=target_arm_qpos,
            final_target_gripper=target_gripper_state,
            include_final_restore=(not no_interpolation),
        )
        is_valid, validation_msg = validate_code(code)
        if is_valid:
            with open('scripts/autobio_scripts/transition_template.py', 'w', encoding='utf-8') as f:
                f.write(code)
            print("Updated transition_template.py")
            return
        raise ValueError(f"Generated template code is invalid in no_planning mode: {validation_msg}")

    print(f"🚀 Stage 1: Generating path planning list using {model_name}...")
    plan_obj = _request_json_object(
        client=client,
        model_name=model_name,
        request_input=[{
            "role": "user",
            "content": [
                {"type": "input_text", "text": planning_prompt},
                {"type": "input_image", "image_url": front_image_data_url},
                {"type": "input_image", "image_url": side_image_data_url},
            ],
        }],
        stage_name="stage-1-planning",
        max_attempts=max_attempts,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        timeout=timeout,
        backend_mode=effective_backend_mode,
        thinking_mode=thinking_mode,
    )
    plan_steps = plan_obj.get("plan_steps", [])
    if not isinstance(plan_steps, list) or len(plan_steps) == 0:
        raise ValueError("Stage-1 planning output missing non-empty plan_steps")

    plan_target_qpos = plan_obj.get("final_target_qpos", target_arm_qpos)
    if not isinstance(plan_target_qpos, list) or len(plan_target_qpos) == 0:
        raise ValueError("Stage-1 planning output missing valid final_target_qpos")
    try:
        plan_target_qpos = [float(x) for x in plan_target_qpos]
    except Exception as e:
        raise ValueError(f"Invalid final_target_qpos in plan output: {e}")

    plan_target_gripper = plan_obj.get("final_target_gripper", target_gripper_state)
    if plan_target_gripper is not None:
        try:
            plan_target_gripper = float(plan_target_gripper)
        except Exception as e:
            raise ValueError(f"Invalid final_target_gripper in plan output: {e}")

    plan_out_path = Path("logs/transition_plan.json")
    plan_out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(plan_out_path, "w", encoding="utf-8") as f:
        json.dump(plan_obj, f, ensure_ascii=False, indent=2)
    print(f"Saved transition plan to: {plan_out_path}")

    print(f"🚀 Stage 2: Using {model_name} to generate execute instruction code based on the plan list...")
    codegen_prompt = codegen_prompt_template.format(
        plan_steps_json=json.dumps(plan_steps, ensure_ascii=False, indent=2),
        allowed_apis_json=json.dumps(allowed_apis, ensure_ascii=False),
    )
    codegen_text = _request_text(
        client=client,
        model_name=model_name,
        request_input=[
            {
                "role": "user",
                "content": [{"type": "input_text", "text": codegen_prompt}],
            }
        ],
        stage_name="stage-2-codegen",
        max_attempts=max_attempts,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        timeout=timeout,
        backend_mode=effective_backend_mode,
        thinking_mode=thinking_mode,
    )

    execute_body = _extract_code_from_response(codegen_text)
    code = _replace_execute_body(
        template_code,
        execute_body,
        final_target_qpos=plan_target_qpos,
        final_target_gripper=plan_target_gripper,
        include_final_restore=(not no_interpolation),
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
