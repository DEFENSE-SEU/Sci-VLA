import mujoco
import mujoco.viewer
import argparse
mujoco.mj_loadPluginLibrary('./libmjlab.so.3.3.0')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load and visualize MuJoCo XML model")
    parser.add_argument("--xml", required=True, type=str, help="Path to the XML model file")
    args = parser.parse_args()
    
    model = mujoco.MjModel.from_xml_path(args.xml)
    data = mujoco.MjData(model)

    # 启动交互式查看器
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            # 步进仿真
            mujoco.mj_step(model, data)
            # 同步查看器
            viewer.sync()