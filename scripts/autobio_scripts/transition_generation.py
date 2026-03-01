import os
import base64
import mimetypes
from pathlib import Path
from openai import OpenAI
import zstandard as zstd
import ast
import numpy as np
import io
import json

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
        
        # 基本检查：确保有必要的结构（如函数或类定义）
        has_function = any(isinstance(node, ast.FunctionDef) for node in ast.walk(tree))
        has_class = any(isinstance(node, ast.ClassDef) for node in ast.walk(tree))
        
        if not (has_function or has_class):
            return False, "The code should contain at least one function or class definition."
        
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
        print(f"❌ 读取文件出错: {e}")
        exit(1)

def find_target_state(task_file_path):
    with open(task_file_path, "rb") as f:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(f) as reader:
            buffer = io.BytesIO(reader.read())
            data = np.load(buffer)
    return data[0,1:40][11:17]


def find_target_path(target_prompt):
    log_dir = Path("logs")
    if not log_dir.exists():
        return None

    for info_path in log_dir.rglob("info.json"):
        try:
            with open(info_path, "r", encoding="utf-8") as f:
                info = json.load(f)
        except Exception:
            continue
        
        if info['task']['prefix'] == target_prompt:
            return str(info_path.parent / "states.npy.zst")

    return None

def transition_code_generation(task_prompt: str):
    base_url = os.environ.get('BASE_URL', "https://api.linkapi.org/v1")
    api_key = os.environ.get('API_KEY')
    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
    )


    target_joint_path = find_target_path(task_prompt)
    target_joint_pos = str(find_target_state(target_joint_path))
    current_joint_pos = str(np.load('logs/current_joint.npy').tolist())
    template_code = read_file('scripts/autobio_scripts/transition_template.py')
    image_data_url = file_to_data_url('logs/current_view.png')

    prompt = f'''
You are a professional Python programmer. Please generate complete, executable Python code: a transition task expert who can perform actions from the current robotic arm state to the initial state of the target task.
Logic: 
Input: 
1. target joint position vector: {target_joint_pos}
2. current joint position vector: {current_joint_pos}
3. target task prompt: {task_prompt}
4. current front camara image: The front-to-back direction is the x-axis, the left-to-right direction is the y-axis, and the up-and-down direction is the z-axis.
5. output code template: \n```python{template_code}```\n

Analyze all these informations, strictly follow the given code template, generate code so that it can execute: current state -> given task initial state safely.
Note: 
1. Fully mimic the coding style of the template.
2. The transition execution needs safety concern (not collision).
'''
    print("Next task prompt:", task_prompt)
    print("Current joint pos:", current_joint_pos)
    print("Target joint pos:", target_joint_pos)
    print("🚀 开始使用gpt-5.2生成代码...")
    resp = client.responses.create(
        model="gpt-5.2",
        input=[{
            "role": "user",
            "content": [
                {"type": "input_text", "text": prompt},
                {"type": "input_image", "image_url": image_data_url},
            ],
        }],
    )

    content = resp.output_text
    import re
    match = re.search(r"```python(.*?)```", content, re.DOTALL)
    if match:
        code = match.group(1).strip()
    else:
        match = re.search(r"```(.*?)```", content, re.DOTALL)
        if match:
            code = match.group(1).strip()
        else:
            code = content.replace("```python", "").replace("```", "").strip()

    is_valid, validation_msg = validate_code(code)
    if is_valid:
        print(f"✅ 生成的代码语法正确！")
        with open('scripts/autobio_scripts/transition.py', 'w', encoding='utf-8') as f:
            f.write(code)
    else:
        last_error = validation_msg
        print(f"❌ 生成的代码有语法错误，错误信息：")
        print(f"   {validation_msg[:200]}...")  # 只显示前200个字符的错误信息