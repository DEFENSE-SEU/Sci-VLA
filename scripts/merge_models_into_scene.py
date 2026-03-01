#!/usr/bin/env python3
"""
把 model/ 目录下的 XML 资产以 <include file="..."/> 的方式加入到指定的 scene.xml 中。

用法示例（在仓库根目录执行或指定绝对路径）:
  python scripts/merge_models_into_scene.py \
    --model-dir model \
    --scene universal_robots_ur5e/scene.xml \
    --dry-run

会先生成备份（scene.xml.bak.YYYYmmddHHMMSS）除非禁用 --no-backup。
"""
import argparse
import os
import sys
import datetime
import xml.etree.ElementTree as ET


def find_xml_files(model_dir, include_patterns=None, exclude_patterns=None):
    xml_files = []
    for root, dirs, files in os.walk(model_dir):
        for f in files:
            if not f.lower().endswith('.xml'):
                continue
            path = os.path.join(root, f)
            rel = os.path.relpath(path, model_dir)
            # 排序/过滤策略：默认包含所有 xml，可按需要调整
            allowed = True
            if include_patterns:
                allowed = any(p in rel for p in include_patterns)
            if exclude_patterns and any(p in rel for p in exclude_patterns):
                allowed = False
            if allowed:
                xml_files.append(path)
    xml_files.sort()
    return xml_files


def backup_file(path):
    ts = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    backup_path = f"{path}.bak.{ts}"
    with open(path, 'rb') as r, open(backup_path, 'wb') as w:
        w.write(r.read())
    return backup_path


def insert_includes(scene_path, xml_paths, dry_run=True, no_backup=False):
    # 解析 scene.xml
    tree = ET.parse(scene_path)
    root = tree.getroot()

    scene_dir = os.path.dirname(os.path.abspath(scene_path))

    # 生成 <include/> 元素列表（使用相对路径，转为正斜杠）
    include_elements = []
    for p in xml_paths:
        rel = os.path.relpath(os.path.abspath(p), scene_dir)
        rel = rel.replace('\\', '/')
        el = ET.Element('include')
        el.set('file', rel)
        include_elements.append(el)

    if dry_run:
        print('DRY RUN: 将要向', scene_path, '插入以下 include:')
        for el in include_elements:
            print('  <include file="%s"/>' % el.get('file'))
        return True

    # 备份
    if not no_backup:
        backup = backup_file(scene_path)
        print('备份已创建：', backup)

    # 找到 worldbody 或 asset 的插入点，优先在 worldbody 之前插入
    insert_index = None
    for idx, child in enumerate(list(root)):
        tag = child.tag.lower()
        if tag == 'worldbody':
            insert_index = idx
            break

    # 如果找到了索引，我们需要重建 root 的子元素顺序以插入
    children = list(root)
    # 移除所有子节点，再按顺序插回，插入 include
    for c in children:
        root.remove(c)

    new_children = []
    for i, c in enumerate(children):
        if insert_index is not None and i == insert_index:
            new_children.extend(include_elements)
        new_children.append(c)

    # 如果没有 worldbody，则追加到 root 末尾
    if insert_index is None:
        new_children.extend(include_elements)

    for c in new_children:
        root.append(c)

    # 写回文件（保持默认 ElementTree 输出）
    tree.write(scene_path, encoding='utf-8', xml_declaration=True)
    print('已将 %d 个 include 插入到 %s' % (len(include_elements), scene_path))
    return True


def main():
    p = argparse.ArgumentParser(description='将 model 下 XML 资产插入到 scene.xml 中')
    p.add_argument('--model-dir', default='model', help='模型资产根目录（相对或绝对）')
    p.add_argument('--scene', default='universal_robots_ur5e/scene.xml', help='目标 scene.xml 路径')
    p.add_argument('--dry-run', action='store_true', help='只打印将要插入的 include，不修改文件')
    p.add_argument('--no-backup', action='store_true', help='不创建备份（默认创建备份）')
    p.add_argument('--include-pattern', action='append', help='仅包含路径中含有该子串的 xml（可多次）')
    p.add_argument('--exclude-pattern', action='append', help='排除路径中含有该子串的 xml（可多次）')
    args = p.parse_args()

    model_dir = args.model_dir
    scene = args.scene

    if not os.path.isdir(model_dir):
        print('错误：model-dir 不存在：', model_dir)
        sys.exit(2)
    if not os.path.isfile(scene):
        print('错误：scene 文件不存在：', scene)
        sys.exit(2)

    xmls = find_xml_files(model_dir, args.include_pattern, args.exclude_pattern)
    if not xmls:
        print('未找到任何 XML 文件于', model_dir)
        sys.exit(0)

    # 过滤掉目标 scene 本身（如果 scene 位于 model tree 下）
    xmls = [x for x in xmls if os.path.abspath(x) != os.path.abspath(scene)]

    insert_includes(scene, xmls, dry_run=args.dry_run, no_backup=args.no_backup)


if __name__ == '__main__':
    main()
