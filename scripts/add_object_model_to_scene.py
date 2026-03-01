#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/add_object_model_to_scene.py

把一个物体 XML 模型注册到 scene.xml 的 <asset> 中，并在 <worldbody> 中实例化：
  <asset>
    <model name="X" file="../object/X.xml" content_type="text/xml"/>
  </asset>

  <worldbody>
    <body name="X" pos="..." quat="...">
      <attach model="X" body="X" prefix="/"/>
    </body>
  </worldbody>

项目约定（已按你的要求写死默认）：
- 根 body 的命名规则：永远等于文件名/等于 model name
  => attach 的 body 默认等于 model name（不再用 "body"）

特性：
- 自动备份 scene.xml -> scene.xml.bak.YYYYmmddHHMMSS（可 --no-backup 关闭）
- 自动创建 <asset>（若不存在），并尽量放在 <worldbody> 前
- <worldbody> 不存在则报错
- 去重冲突：默认遇到同名会自动改名；可用 --replace 覆盖（删除旧的同名条目再插入）
- “合适位置”启发式：优先挂到 worldbody 下常见容器 body（objects/workspace/table 等），否则挂到 <worldbody> 根
"""

import argparse
import datetime
import os
import sys
import xml.etree.ElementTree as ET


def backup_file(path: str) -> str:
    ts = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    backup_path = f"{path}.bak.{ts}"
    with open(path, "rb") as r, open(backup_path, "wb") as w:
        w.write(r.read())
    return backup_path


def indent(elem: ET.Element, level: int = 0) -> None:
    """Pretty indentation (in-place) for ElementTree."""
    i = "\n" + level * "  "
    if len(elem):
        if not (elem.text and elem.text.strip()):
            elem.text = i + "  "
        for child in elem:
            indent(child, level + 1)
        if not (elem.tail and elem.tail.strip()):
            elem.tail = i
    else:
        if not (elem.tail and elem.tail.strip()):
            elem.tail = i


def ensure_asset_before_worldbody(root: ET.Element) -> ET.Element:
    """
    Ensure <asset> exists. If creating it, place it before <worldbody> if possible.
    """
    asset = root.find("asset")
    if asset is not None:
        return asset

    worldbody = root.find("worldbody")
    asset = ET.Element("asset")

    # Insert <asset> before <worldbody> if worldbody exists; otherwise append.
    if worldbody is None:
        root.append(asset)
        return asset

    children = list(root)
    root.clear()
    inserted = False
    for c in children:
        if (not inserted) and c is worldbody:
            root.append(asset)
            inserted = True
        root.append(c)
    if not inserted:
        root.append(asset)
    return asset


def require_worldbody(root: ET.Element) -> ET.Element:
    wb = root.find("worldbody")
    if wb is None:
        raise ValueError("scene 文件中未找到 <worldbody>，无法实例化物体。")
    return wb


def default_name_from_object_path(p: str) -> str:
    """
    你们项目规则：name = 文件名去后缀。
    例：
      pipette_tip.gen.xml -> pipette_tip.gen
      CoaterMSC150T.xml   -> CoaterMSC150T
    """
    base = os.path.basename(p)
    name, _ = os.path.splitext(base)
    return (name.strip().replace(" ", "_")) or "object"


def existing_model_names(asset: ET.Element) -> set[str]:
    names = set()
    for m in asset.findall("model"):
        n = m.get("name")
        if n:
            names.add(n)
    return names


def existing_body_names(parent: ET.Element) -> set[str]:
    names = set()
    for b in parent.findall("./body"):
        n = b.get("name")
        if n:
            names.add(n)
    return names


def choose_parent_body(worldbody: ET.Element) -> ET.Element:
    """
    启发式：尽量把新物体挂到更“合适”的容器 body 下。
    优先匹配常见命名：objects/workspace/table 等；找不到就挂到 <worldbody> 根。
    """
    candidates = [
        "objects", "object", "props", "prop",
        "workspace", "workbench",
        "table", "desk",
        "scene", "lab",
        "world",
    ]

    # 第一层
    for b in worldbody.findall("./body"):
        name = (b.get("name") or "").strip().lower()
        if name in candidates:
            return b

    # 更深层（谨慎一点，但能适配一些工程结构）
    for b in worldbody.findall(".//body"):
        name = (b.get("name") or "").strip().lower()
        if name in candidates:
            return b

    return worldbody


def unique_name(desired: str, used: set[str]) -> str:
    if desired not in used:
        return desired
    k = 2
    while f"{desired}_{k}" in used:
        k += 1
    return f"{desired}_{k}"


def add_or_replace_model_asset(asset: ET.Element, model_name: str, model_file_rel: str, replace: bool) -> None:
    if replace:
        for m in list(asset.findall("model")):
            if m.get("name") == model_name:
                asset.remove(m)

    el = ET.Element("model")
    el.set("name", model_name)
    el.set("file", model_file_rel)
    el.set("content_type", "text/xml")
    asset.append(el)


def add_or_replace_body_attach(parent: ET.Element,
                               body_name: str,
                               model_name: str,
                               pos: str,
                               quat: str,
                               prefix: str,
                               replace: bool) -> None:
    if replace:
        for b in list(parent.findall("./body")):
            if b.get("name") == body_name:
                parent.remove(b)

    body = ET.Element("body")
    body.set("name", body_name)
    body.set("pos", pos)
    body.set("quat", quat)

    attach = ET.Element("attach")
    attach.set("model", model_name)
    # 你们规则：根 body 名永远等于 model name
    attach.set("body", model_name)
    attach.set("prefix", prefix)

    body.append(attach)
    parent.append(body)


def main() -> int:
    ap = argparse.ArgumentParser(description="向场景 XML 中注册并 attach 一个物体 XML 模型。")
    ap.add_argument("--object", required=True, help="物体 XML 文件路径（相对或绝对）")
    ap.add_argument("--scene", required=True, help="场景 XML 文件路径（将被原地修改）")
    ap.add_argument("--name", default=None, help="模型/实例名称（默认取 object 文件名去后缀）")
    ap.add_argument("--pos", default="0 0 0", help='body 的 pos，默认 "0 0 0"')
    ap.add_argument("--quat", default="1 0 0 0", help='body 的 quat，默认 "1 0 0 0" (w x y z)')
    ap.add_argument("--prefix", default="/", help='attach 的 prefix=，默认 "/"')
    ap.add_argument("--replace", action="store_true", help="若同名已存在，则覆盖（删除旧的再插入）")
    ap.add_argument("--no-backup", action="store_true", help="不备份原 scene.xml（默认会备份）")
    ap.add_argument("--dry-run", action="store_true", help="只打印将要修改的内容，不写回文件")
    args = ap.parse_args()

    obj_path = args.object
    scene_path = args.scene

    if not os.path.isfile(obj_path):
        print(f"错误：object 文件不存在：{obj_path}")
        return 2
    if not os.path.isfile(scene_path):
        print(f"错误：scene 文件不存在：{scene_path}")
        return 2

    tree = ET.parse(scene_path)
    root = tree.getroot()

    asset = ensure_asset_before_worldbody(root)
    worldbody = require_worldbody(root)

    scene_dir = os.path.dirname(os.path.abspath(scene_path))
    obj_abs = os.path.abspath(obj_path)
    obj_rel = os.path.relpath(obj_abs, scene_dir).replace("\\", "/")

    desired_name = (args.name or default_name_from_object_path(obj_path)).strip()
    if not desired_name:
        desired_name = "object"

    parent_for_body = choose_parent_body(worldbody)

    # 冲突处理
    model_name = desired_name
    body_name = desired_name
    if not args.replace:
        model_name = unique_name(model_name, existing_model_names(asset))
        body_name = unique_name(body_name, existing_body_names(parent_for_body))

    # 输出将要修改的内容
    print("将要进行的修改：")
    print(f"- scene:  {scene_path}")
    print(f"- object: {obj_path}")
    print(f"- <asset> 添加: <model name=\"{model_name}\" file=\"{obj_rel}\" content_type=\"text/xml\"/>")
    if parent_for_body is worldbody:
        print("- <worldbody> 根下插入：")
    else:
        print(f"- <worldbody> 子节点 body(name=\"{parent_for_body.get('name','')}\") 下插入：")
    print(f"  <body name=\"{body_name}\" pos=\"{args.pos}\" quat=\"{args.quat}\">")
    print(f"    <attach model=\"{model_name}\" body=\"{model_name}\" prefix=\"{args.prefix}\"/>")
    print("  </body>")

    if args.dry_run:
        print("DRY RUN：未写回文件。")
        return 0

    if not args.no_backup:
        bk = backup_file(scene_path)
        print(f"已创建备份：{bk}")

    # 写入
    add_or_replace_model_asset(asset, model_name, obj_rel, replace=args.replace)
    add_or_replace_body_attach(
        parent_for_body,
        body_name=body_name,
        model_name=model_name,
        pos=args.pos,
        quat=args.quat,
        prefix=args.prefix,
        replace=args.replace,
    )

    indent(root, 0)
    tree.write(scene_path, encoding="utf-8", xml_declaration=True)
    print("完成：已写回 scene 文件。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
