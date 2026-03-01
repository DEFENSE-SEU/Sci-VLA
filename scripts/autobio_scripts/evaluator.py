from collections import deque
from contextlib import contextmanager
from typing import Callable, TypeAlias

import numpy as np
import mujoco
import pathlib #
import imageio #
import time #
from datetime import datetime #
from task import Task
from serialize import STATE_SPEC

Policy: TypeAlias = Callable[[dict], np.ndarray]

def make_thermal_mixer_extra(task: Task):
    from copy import deepcopy
    from instrument import Thermal_mixer_eppendorf_c
    thermal_mixers: list[Thermal_mixer_eppendorf_c] = task.manager.systems_by_type.get(Thermal_mixer_eppendorf_c, [])
    if len(thermal_mixers) == 0:
        return

    temp_model = deepcopy(task.model)

    context = [(mixer.ui_state, None, *mixer.ui_state.make_canvas(), mixer.display) for mixer in thermal_mixers]
    def update_texture(ui, last_ui, fig, ax, target, render_context):
        if last_ui == ui:
            return ui
        ui.draw(ax)
        img = ui.render_canvas(fig)
        temp_model.tex(target).data[...] = img
        mujoco.mjr_uploadTexture(temp_model, render_context, target)
        return deepcopy(ui)
    
    def update(_, render_context):
        for i in range(len(context)):
            ui, last_ui, fig, ax, target = context[i]
            last_ui = update_texture(ui, last_ui, fig, ax, target, render_context)
            context[i] = (ui, last_ui, fig, ax, target)
    
    def finish():
        import matplotlib.pyplot as plt
        for i in range(len(context)):
            ui, last_ui, fig, ax, target = context[i]
            plt.close(fig)

    return update, finish

def make_liquid_extra(task: Task):
    from skimage.measure import EllipseModel
    from liquid import ContainerSystem, Container
    container_systems: list[ContainerSystem] = task.manager.systems_by_type.get(ContainerSystem, [])
    if len(container_systems) == 0:
        return

    def update(scene: mujoco.MjvScene, _):
        for container_system in container_systems:
            container = container_system.container
            if container.liquid is not None:
                add_liquid_surface(scene, container)

    def add_liquid_surface(scene: mujoco.MjvScene, container: Container):
        meshplane = container.liquid.meshplane
        surface_distance = container.liquid.surface.distance
        position = container.position
        rotation_matrix = container.rotation_matrix

        def compose(local_pos, local_mat, global_pos=None, global_mat=None):
            if global_mat is not None:
                local_mat = global_mat @ local_mat
                local_pos = global_mat @ local_pos
            if global_pos is not None:
                local_pos = global_pos + local_pos
            return local_pos, local_mat

        liquid_mesh = meshplane.calculate_mesh(surface_distance)
        surface = liquid_mesh.vertices[liquid_mesh.boundary]
        local_frame = container.liquid.surface.frame
        planar_surface = surface @ local_frame 
        em = EllipseModel()
        em.estimate(planar_surface[:, :2])
        xc, yc, a, b, theta = em.params
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        circle_pos = np.array([xc, yc, surface_distance])
        circle_mat = np.array([
            [cos_theta * a, -sin_theta * b, 0.0],
            [sin_theta * a,  cos_theta * b, 0.0],
            [0.0, 0.0, 1.0],
        ])  # Transform unit circle to ellipse
        circle_pos, circle_mat = compose(circle_pos, circle_mat, None, local_frame)
        circle_pos, circle_mat = compose(circle_pos, circle_mat, position, rotation_matrix)
        circle_size = np.array([1.0, 1e-4, 0.0])
        mujoco.mjv_initGeom(
            scene.geoms[scene.ngeom],
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=circle_size,
            pos=circle_pos,
            mat=circle_mat.ravel(),
            rgba=(0, 0, 1, 1),
        )
        scene.ngeom += 1
    
    def finish():
        pass

    return update, finish

@contextmanager
def set_history(model, data, qpos):
    state_size = mujoco.mj_stateSize(model, STATE_SPEC)
    old_states = np.zeros(state_size)
    mujoco.mj_getState(model, data, old_states, STATE_SPEC)
    data.qpos[:] = qpos
    mujoco.mj_kinematics(model, data)
    mujoco.mj_camlight(model, data)
    try:
        yield
    finally:
        mujoco.mj_setState(model, data, old_states, STATE_SPEC)
        mujoco.mj_kinematics(model, data)
        mujoco.mj_camlight(model, data)

class Evaluator:
    def __init__(
        self, task: Task,
        *,
        image_height: int = 224,
        image_width: int = 224,
        image_history: int = 0,
        video_out_path: str = "./videos",  # 新增视频输出路径参数
    ):
        self.task = task
        self.model = task.model
        self.data = task.data
        self.renderer = mujoco.Renderer(self.model, image_height, image_width)
        self.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
        self.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = False
        self.renderer._scene_option.sitegroup[:] = False
        self.image_history = image_history
        self.history_states = deque(maxlen=image_history)

        self.video_out_path = pathlib.Path(video_out_path)
        self.video_out_path.mkdir(parents=True, exist_ok=True)
        self.replay_images = []  # 存储用于生成视频的图像

    def make_render_extra(self, scene: Task):
        extras = []
        for extra in [
            make_thermal_mixer_extra,
            # make_liquid_extra,
        ]:
            extra_func = extra(scene)
            if extra_func is not None:
                extras.append(extra_func)
        def render_extra(scene: mujoco.MjvScene, render_context: mujoco.MjrContext):
            for update_func, _ in extras:
                update_func(scene, render_context)
        def render_finish():
            for _, finish_func in extras:
                finish_func()
        return render_extra, render_finish
    
    def get_image(self, camera_key):
        if camera_key not in self.cameras:
            return None
        camera = self.cameras[camera_key]
        mujoco.mjv_updateCamera(self.model, self.data, camera, self.renderer._scene)
        image = self.renderer.render()
        return image
    
    def get_images(self):
        self.renderer.update_scene(self.data)
        if self.render_extra is not None:
            self.render_extra(self.renderer.scene, self.renderer._mjr_context)
        return {
            "observation/image": self.get_image("image"),
            "observation/wrist_image": self.get_image("wrist_image"),
            "observation/wrist_image_2": self.get_image("wrist_image_2"),
        }

    def get_observation(self):
        obs = {
            "observation/state": self.data.qpos[self.task_info['state_indices']],
            **self.get_images(),
            "prompt": self.task_info['prefix'],
        }

        main_image = self.get_image("image")
        if main_image is not None:
            # 转换为uint8格式，与之前代码保持一致
            self.replay_images.append(main_image.astype(np.uint8))

        if self.image_history > 0:
            for i in range(self.image_history):
                if i < len(self.history_states):
                    with set_history(self.model, self.data, self.history_states[i]):
                        history_images = self.get_images()
                else:
                    history_images = {
                        "observation/image": None,
                        "observation/wrist_image": None,
                        "observation/wrist_image_2": None,
                    }
                
                j = i - self.image_history
                obs.update({
                    f"observation/{j}/image": history_images["observation/image"],
                    f"observation/{j}/wrist_image": history_images["observation/wrist_image"],
                    f"observation/{j}/wrist_image_2": history_images["observation/wrist_image_2"],
                })
        return obs

    def reset(self):
        self.task_info = self.task.task_info
        mujoco.mj_forward(self.model, self.data)
        self.render_extra, self.render_finish = self.make_render_extra(self.task)
        self.cameras = {}
        for camera_key, camera_name in self.task_info["camera_mapping"].items():
            camera_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
            assert camera_id >= 0, f"Camera {camera_name} not found in model"
            camera = mujoco.MjvCamera()
            camera.fixedcamid = camera_id
            camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
            self.cameras[camera_key] = camera
        self.history_states.clear()
        self.history_states.append(self.data.qpos)

        self.replay_images = []

    def save_video(self, success: bool, filename_override: str | None = None): #
        """保存评估视频"""
        if not self.replay_images:
            return
            
        # 使用时间戳生成唯一文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # 精确到毫秒
        
        if filename_override:
            filename = f"{filename_override}_{timestamp}.mp4"
        else:
            task_name = self.task_info['prefix'].replace(" ", "_").replace("/", "_")
            suffix = "success" if success else "failure"
            filename = f"rollout_{task_name}_{timestamp}_{suffix}.mp4"
        
        # 保存视频
        video_path = self.video_out_path / filename
        try:
            imageio.mimwrite(
                video_path,
                [np.asarray(img) for img in self.replay_images],
                # fps=5,  # 与之前代码相同的帧率
            )
            print(f"Video saved: {video_path}")
        except Exception as e:
            print(f"Failed to save video: {e}") #

    def evaluate(self, policy: Policy, time_limit: float | None = None, prompts: list[str] | None = None):
        if time_limit is None:
            time_limit = self.task.time_limit

        self.reset()

        def step():
            """Step the task, return True if simulation is healthy."""
            try:
                self.task.step_and_log({})
                if self.data.warning.number.any():
                    # Simulation diverge, etc.
                    return False
                return True
            except mujoco.FatalError as e:
                print(f"MuJoCo simulation error: {e}")
                return False
            except Exception as e:
                # Other parts of simulation
                print(f"Unexpected error: {e}")
                return False
            except:
                import traceback
                traceback.print_exc()
                return False

        def run_prompt(prompt: str | None, current_time_limit: float):
            start_time = self.data.time
            
            if self.task.early_stop:
                # check_mode = prompt if prompt else ''
                # shall_continue = lambda: not self.task.check() and (self.data.time - start_time) < current_time_limit
                shall_continue = lambda: (self.data.time - start_time) < current_time_limit
            else:
                shall_continue = lambda: (self.data.time - start_time) < current_time_limit

            while shall_continue():
                observation = self.get_observation()
                actions = policy(observation)
                assert actions.ndim == 2 and actions.shape[1] == len(self.task_info['action_indices']), breakpoint()
                # if abs(actions[-1]-actions[0]) < 1e-2:
                #     return True
                for action in actions:
                    self.data.ctrl[self.task_info['action_indices']] = action
                    self.history_states.append(self.data.qpos)
                    for _ in range(10):
                        healthy = step()
                        if not healthy:
                            return False
            return True

            
        
        
        # 如果没有提供 prompts，则执行默认逻辑（单任务）
        if prompts is None:
            healthy = run_prompt(None, time_limit)
            task_success = self.task.check() if healthy else False
            self.task.finish()
            self.render_finish()
            self.save_video(task_success)
            if not healthy:
                return False
            return task_success
        
        
        if len(prompts)>1:
            all_success = True
            
            # 多 prompt 逻辑
            for i in range(len(prompts)):
                prompt = prompts[i]
                print(f"Executing prompt: {prompt}")
                self.task_info['prefix'] = prompt
                # task_nums = len(prompts)
                healthy = run_prompt(prompt, time_limit)
                
                if not healthy:
                    all_success = False
                    failed_prompt = prompt
                    break

                current_view = self.replay_images[-1]
                imageio.imwrite('logs/current_view.png', current_view)
                current_joint_pos = self.data.qpos[range(self.model.joint('/ur:shoulder_pan').qposadr.item(), self.model.joint('/ur:shoulder_pan').qposadr.item() + 6)]
                np.save('logs/current_joint.npy', current_joint_pos)

                if prompt!=prompts[-1]:
                    from transition_generation import transition_code_generation
                    transition_code_generation(prompts[i+1])
                    from transition import TransitionExpert
                    expert = TransitionExpert(self.model, self.data, self.task)
                    expert.execute(prompts[i+1])
            
            self.task.finish()
            self.render_finish()

            combined_prompts = ",".join([p.replace(" ", "_").replace("/", "_") for p in prompts])
            # if all_success:
            #     self.save_video(True, filename_override=f"{combined_prompts}_success")
            # else:
            #     safe_prompt = failed_prompt.replace(" ", "_").replace("/", "_")
            #     self.save_video(False, filename_override=f"{combined_prompts},but_{safe_prompt}_fail")

            self.save_video(True, filename_override=f"{combined_prompts[:35]}")

            return all_success
                    
        else:
            task_success = True
            prompt = prompts[0]
            print(f"Executing prompt: {prompt}")
            self.task_info['prefix'] = prompt
            healthy = run_prompt(prompt, time_limit)
            self.task.finish()
            self.render_finish()
            if not healthy:
                task_success = False
            prompt = prompt.replace(" ", "_").replace("/", "_")
            self.save_video(task_success, filename_override=f"{prompt}")
            return task_success

        
