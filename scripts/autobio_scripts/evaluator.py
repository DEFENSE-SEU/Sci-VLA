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
        video_out_path: str = "./videos",
        video_fps: int = 20,
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
        self.video_fps = max(1, int(video_fps))
        self.replay_images = []
        self.replay_times = []

    def _capture_replay_frame(self):
        if "image" not in self.cameras:
            return

        frame = self.get_image("image")
        if frame is None:
            return
        self.replay_images.append(frame.astype(np.uint8))
        self.replay_times.append(float(self.data.time))

    def _resample_replay_frames(self):
        if not self.replay_images:
            return [], np.array([], dtype=np.int64)

        if len(self.replay_images) == 1 or len(self.replay_times) != len(self.replay_images):
            return list(self.replay_images), np.arange(len(self.replay_images), dtype=np.int64)

        times = np.asarray(self.replay_times, dtype=np.float64)
        start_t = float(times[0])
        end_t = float(times[-1])
        if end_t <= start_t:
            return [self.replay_images[0]], np.array([0], dtype=np.int64)

        dt = 1.0 / self.video_fps
        target_times = np.arange(start_t, end_t + 1e-9, dt, dtype=np.float64)

        right_idx = np.searchsorted(times, target_times, side="left")
        right_idx = np.clip(right_idx, 0, len(times) - 1)
        left_idx = np.maximum(right_idx - 1, 0)

        choose_right = np.abs(times[right_idx] - target_times) <= np.abs(times[left_idx] - target_times)
        sampled_indices = np.where(choose_right, right_idx, left_idx)
        sampled_indices = np.unique(sampled_indices)

        sampled_frames = [self.replay_images[i] for i in sampled_indices]
        return sampled_frames, sampled_indices

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

    def get_image_by_camera_name(self, camera_name: str):
        camera_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        if camera_id < 0:
            return None
        camera = mujoco.MjvCamera()
        camera.fixedcamid = camera_id
        camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
        mujoco.mjv_updateCamera(self.model, self.data, camera, self.renderer._scene)
        return self.renderer.render()

    def get_transition_views(self):
        # Prefer canonical viewpoints if they exist in the model.
        front_view = self.get_image_by_camera_name("table_cam_front")
        side_view = self.get_image_by_camera_name("table_cam_left")

        # Fallback: use the task primary image camera when canonical front does not exist.
        if front_view is None:
            front_view = self.get_image("image")

        # Fallback: if no explicit side camera exists, keep transition pipeline usable.
        if side_view is None:
            side_view = front_view

        return front_view, side_view
    
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
        images = self.get_images()
        obs = {
            "observation/state": self.data.qpos[self.task_info['state_indices']],
            **images,
            "prompt": self.task_info['prefix'],
        }

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
        self.replay_times = []
        self._capture_replay_frame()

    def save_video(self, success: bool, filename_override: str | None = None, action_count: int | None = None): #
        if not self.replay_images:
            return

        raw_frame_count = len(self.replay_images)
        mean_frame_diff = 0.0
        low_motion_ratio = 0.0
        effective_capture_fps = 0.0

        if raw_frame_count > 1:
            diffs = []
            for i in range(1, raw_frame_count):
                prev = self.replay_images[i - 1].astype(np.int16)
                curr = self.replay_images[i].astype(np.int16)
                diffs.append(float(np.mean(np.abs(curr - prev))))
            mean_frame_diff = float(np.mean(diffs))
            low_motion_ratio = float(np.mean(np.asarray(diffs) < 0.5))

            if len(self.replay_times) == raw_frame_count:
                sim_span = self.replay_times[-1] - self.replay_times[0]
                if sim_span > 0:
                    effective_capture_fps = (raw_frame_count - 1) / sim_span

        sampled_frames, sampled_indices = self._resample_replay_frames()
        output_frame_count = len(sampled_frames)

        print(
            f"[VideoDiag] raw_frames={raw_frame_count} | output_frames={output_frame_count} | "
            f"mean_abs_diff={mean_frame_diff:.4f} | "
            f"low_motion_ratio={low_motion_ratio:.2%} | "
            f"effective_capture_fps={effective_capture_fps:.3f}"
        )
        if action_count is not None:
            print(
                f"[ActionDiag] actions={action_count}"
            )
            
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
                [np.asarray(img) for img in sampled_frames],
                fps=self.video_fps,
            )
            print(f"Video saved: {video_path}")
        except Exception as e:
            print(f"Failed to save video: {e}") #

    def evaluate(
        self,
        policy: Policy,
        time_limit: float | None = None,
        prompts: list[str] | None = None,
        use_transition_generation: bool = True,
    ):
        if time_limit is None:
            time_limit = self.task.time_limit

        self.reset()
        episode_start_wall = time.perf_counter()
        transition_infer_total = 0.0
        transition_total = 0.0
        executed_action_count = 0
        settle_duration = 2.0

        def print_timing_summary():
            episode_total = time.perf_counter() - episode_start_wall
            transition_ratio = (transition_total / episode_total * 100.0) if episode_total > 0 else 0.0
            print(
                f"[Timing] transition planning total: {transition_infer_total:.3f}s | "
                f"transition total: {transition_total:.3f}s | "
                f"episode total: {episode_total:.3f}s | "
                f"transition ratio: {transition_ratio:.2f}%"
            )

        def step():
            """Step the task, return True if simulation is healthy."""
            try:
                self.task.step_and_log({})
                if self.data.warning.number.any():
                    # Simulation diverge, etc.
                    return False
                self._capture_replay_frame()
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

        # Let the scene settle for 2 seconds of simulation time before the first action.
        settle_steps = max(1, int(round(settle_duration / self.model.opt.timestep)))
        for _ in range(settle_steps):
            healthy = step()
            if not healthy:
                self.task.finish()
                self.render_finish()
                self.save_video(False, action_count=executed_action_count)
                print_timing_summary()
                return False
        # self._capture_replay_frame()

        def run_prompt(prompt: str | None, current_time_limit: float):
            nonlocal executed_action_count
            start_time = self.data.time

            while (self.data.time - start_time) < current_time_limit:
                observation = self.get_observation()
                actions = policy(observation)
                assert actions.ndim == 2 and actions.shape[1] == len(self.task_info['action_indices']), breakpoint()
                # if abs(actions[-1]-actions[0]) < 1e-2:
                #     return True
                for action in actions:
                    self.data.ctrl[self.task_info['action_indices']] = action
                    executed_action_count += 1
                    self.history_states.append(self.data.qpos)
                    for _ in range(10):
                        healthy = step()
                        if not healthy:
                            return False
            return True

            
        
        if prompts is None:
            healthy = run_prompt(None, time_limit)
            task_success = self.task.check() if healthy else False
            self.task.finish()
            self.render_finish()
            self.save_video(task_success, action_count=executed_action_count)
            print_timing_summary()
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

                current_view, current_side_view = self.get_transition_views()
                if current_view is not None:
                    imageio.imwrite('logs/current_view.png', current_view)
                if current_side_view is not None:
                    imageio.imwrite('logs/current_side_view.png', current_side_view)
                current_joint_pos = self.data.qpos[range(self.model.joint('/ur:shoulder_pan').qposadr.item(), self.model.joint('/ur:shoulder_pan').qposadr.item() + 6)]
                np.save('logs/current_joint.npy', current_joint_pos)

                if prompt!=prompts[-1] and use_transition_generation:
                    transition_start_wall = time.perf_counter()
                    from transition_generation import transition_code_generation
                    transition_infer_start_wall = time.perf_counter()
                    transition_code_generation(prompts[i+1])
                    transition_infer_elapsed = time.perf_counter() - transition_infer_start_wall
                    transition_infer_total += transition_infer_elapsed
                    from transition_template import TransitionExpert
                    expert = TransitionExpert(self.model, self.data, self.task)
                    original_step_and_log = self.task.step_and_log

                    def step_and_log_with_capture(info: dict):
                        original_step_and_log(info)
                        self._capture_replay_frame()

                    self.task.step_and_log = step_and_log_with_capture
                    try:
                        self._capture_replay_frame()
                        expert.execute()
                        self._capture_replay_frame()
                    finally:
                        self.task.step_and_log = original_step_and_log
                    transition_elapsed = time.perf_counter() - transition_start_wall
                    transition_total += transition_elapsed
                    print(
                        f"[Timing] transition to next prompt took {transition_elapsed:.3f}s "
                        f"(inference {transition_infer_elapsed:.3f}s)"
                    )
                elif prompt != prompts[-1]:
                    print("[Transition] Skipped transition generation (disabled by flag).")
            
            self.task.finish()
            self.render_finish()

            combined_prompts = ",".join([p.replace(" ", "_").replace("/", "_") for p in prompts])

            self.save_video(True, filename_override=f"{combined_prompts[:35]}", action_count=executed_action_count)
            print_timing_summary()

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
            self.save_video(task_success, filename_override=f"{prompt}", action_count=executed_action_count)
            print_timing_summary()
            return task_success

        
