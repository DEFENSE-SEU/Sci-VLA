import os
import re
import sys

def main():
    # 获取当前工作目录（假设在项目根目录运行）
    workspace_root = os.getcwd()
    
    # 定义日志文件路径和搜索目录
    log_file_path = os.path.join(workspace_root, 'logs', 'disconnected_task.txt')
    search_dir = os.path.join(workspace_root, 'scripts', 'autobio_scripts')
    
    # 检查日志文件是否存在
    if not os.path.exists(log_file_path):
        print(f"Error: Log file not found at {log_file_path}")
        return

    # 读取日志文件内容
    try:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
    except Exception as e:
        print(f"Error reading log file: {e}")
        return

    # 使用正则表达式提取关键字
    # 匹配 "but_" 和 "_fail" 之间的内容
    match = re.search(r'but_(.*?)_fail', content)
    
    if not match:
        print(f"No pattern 'but_..._fail' found in {log_file_path}")
        print(f"Content was: {content}")
        return
    
    keyword = match.group(1)
    # replace underscores with hyphens
    # keyword = keyword.replace('_', ' ')
    print(f"Extracted keyword: {keyword}")
    
    if not keyword:
        print("Extracted keyword is empty.")
        return

    # 检查搜索目录是否存在
    if not os.path.exists(search_dir):
        print(f"Error: Search directory not found at {search_dir}")
        return

    # 搜索匹配的文件
    found_files = []
    for filename in os.listdir(search_dir):
        if filename.endswith(".py"):
            file_path = os.path.join(search_dir, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    file_content = f.read()
                    if keyword in file_content and 'transition' not in filename.lower() and 'task' not in filename.lower():
                        found_files.append(file_path)
            except Exception as e:
                print(f"Warning: Could not read {filename}: {e}")
    
    # 输出结果
    if found_files:
        print(f"Found {len(found_files)} matching script(s):")
        for file_path in found_files:
            print(file_path)

    if len(found_files) == 1:
        # save the found file path to a txt file
        output_path = os.path.join(workspace_root, 'logs', 'found_failed_task_script.txt')
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(found_files[0])
            print(f"Saved the found script path to {output_path}")
        except Exception as e:
            print(f"Error writing to {output_path}: {e}")
    else:
        print(f"No python scripts found containing '{keyword}' in {search_dir}")

if __name__ == "__main__":
    main()
