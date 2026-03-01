import os
import argparse
import shutil
from datetime import datetime

XML_TEMPLATE = """<mujoco model="{name}">
  <compiler angle="radian" meshdir="../../assets"/>

  <option gravity="0 0 -9.81"/>

  <asset>
    <mesh
      name="{mesh_name}"
      file="{asset_name}/base.obj"
      scale="0.001 0.001 0.001"/>
  </asset>

  <worldbody>
    <body name="{body_name}">

      <geom
        name="{visual_name}"
        type="mesh"
        mesh="{mesh_name}"
        contype="0"
        conaffinity="0"/>

      <geom
        name="{collision_name}"
        type="mesh"
        mesh="{mesh_name}"
        rgba="0.8 0.8 0.8 1"
        contype="1"
        conaffinity="1"
        friction="1.0 0.3 0.001"/>

    </body>
  </worldbody>
</mujoco>
"""


def generate_xml(model_dir: str):
    model_dir = os.path.normpath(model_dir)

    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    obj_path = os.path.join(model_dir, "base.obj")
    if not os.path.isfile(obj_path):
        raise FileNotFoundError(f"base.obj not found in {model_dir}")

    asset_name = os.path.basename(model_dir)
    project_root = os.path.abspath(os.path.join(model_dir, "..", ".."))

    object_dir = os.path.join(project_root, "model", "object")
    os.makedirs(object_dir, exist_ok=True)

    xml_path = os.path.join(object_dir, f"{asset_name}.xml")

    # 备份已有 XML
    if os.path.exists(xml_path):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = xml_path + f".bak_{timestamp}"
        shutil.copy2(xml_path, backup_path)
        print(f"[INFO] Existing XML backed up to: {backup_path}")

    xml_content = XML_TEMPLATE.format(
        name=asset_name,
        asset_name=asset_name,
        mesh_name=f"{asset_name.lower()}_mesh",
        body_name=asset_name.lower(),
        visual_name=f"{asset_name.lower()}_visual",
        collision_name=f"{asset_name.lower()}_collision",
    )

    with open(xml_path, "w", encoding="utf-8") as f:
        f.write(xml_content)

    print(f"[SUCCESS] XML generated: {xml_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate MuJoCo object XML from asset directory (expects base.obj)"
    )
    parser.add_argument(
        "model_dir",
        type=str,
        help="Path to asset directory, e.g. autolab_sim/assets/HotPlateHPR6"
    )

    args = parser.parse_args()
    generate_xml(args.model_dir)
