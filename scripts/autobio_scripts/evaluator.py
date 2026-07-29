from collections import deque
from contextlib import contextmanager
import importlib
import inspect
import os
import sys
from typing import Any, Callable, TypeAlias

import numpy as np
import mujoco
import non_llm_transition
import pathlib #
import imageio #
import time #
from datetime import datetime #
from task import Task
from serialize import STATE_SPEC
from problem_validation_demo import (
    ProblemValidationDemoConfig,
    PromptRunController,
    execute_problem_validation_sequence,
    sample_problem_validation_state,
)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

Policy: TypeAlias = Callable[[dict], np.ndarray]


class _RequiredVideoWriteError(RuntimeError):
    pass


class _CompletionModelPredicate:
    """Online wrapper around the trained intervention-switch completion model."""

    def __init__(self, config: dict[str, Any]):
        checkpoint_path = pathlib.Path(config["checkpoint_path"]).expanduser().resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Completion-model checkpoint not found: {checkpoint_path}")

        switch_dir = pathlib.Path(__file__).resolve().parents[1] / "intervention_switch"
        if str(switch_dir) not in sys.path:
            sys.path.insert(0, str(switch_dir))
        try:
            import torch
            from PIL import Image
            from transformers import AutoProcessor
            from model import load_checkpoint
        except ImportError as exc:
            raise ImportError(
                "Completion-model inference requires torch, transformers, and Pillow."
            ) from exc

        requested_device = str(config.get("device", "auto"))
        self._torch = torch
        self._image_type = Image
        self.device = torch.device(
            "cuda" if requested_device == "auto" and torch.cuda.is_available() else
            ("cpu" if requested_device == "auto" else requested_device)
        )
        self.model, metadata = load_checkpoint(checkpoint_path, device=self.device)
        self.processor = AutoProcessor.from_pretrained(metadata["model_name"])
        self.threshold = float(
            metadata["threshold"] if config.get("threshold") is None else config["threshold"]
        )
        self.consecutive_positive = max(1, int(config.get("consecutive_positive", 3)))
        self.check_interval = float(config.get("check_interval", 0.2))
        self._initial_frame = None
        self._initial_feature = None
        self._text_feature = None
        self._prompt = None
        self._positive_count = 0
        self._last_check_time: float | None = None
        self.last_probability: float | None = None
        print(
            "[CompletionModel] "
            f"checkpoint={checkpoint_path} device={self.device} threshold={self.threshold:.3f} "
            f"consecutive_positive={self.consecutive_positive} check_interval={self.check_interval:.3f}s"
        )

    def start_prompt(self, prompt: str, initial_frame: np.ndarray) -> None:
        if initial_frame is None:
            raise RuntimeError("Completion model needs the primary RGB observation at prompt start.")
        self._prompt = str(prompt)
        self._initial_frame = np.asarray(initial_frame, dtype=np.uint8).copy()
        initial_inputs = self.processor.image_processor(
            images=[self._image_type.fromarray(self._initial_frame)], return_tensors="pt"
        )
        text_inputs = self.processor.tokenizer(
            [self._prompt], padding=True, truncation=True, return_tensors="pt"
        )
        with self._torch.inference_mode():
            self._initial_feature = self.model.encode_image(
                initial_inputs["pixel_values"].to(self.device)
            )
            self._text_feature = self.model.encode_text(
                text_inputs["input_ids"].to(self.device),
                text_inputs["attention_mask"].to(self.device),
            )
        self._positive_count = 0
        self._last_check_time = None
        self.last_probability = None

    def check(self, prompt: str, current_frame: np.ndarray, current_time: float) -> bool:
        if current_frame is None:
            raise RuntimeError("Completion model needs the primary RGB observation for inference.")
        if self._initial_frame is None or self._prompt != str(prompt):
            self.start_prompt(str(prompt), current_frame)
        if (
            self._last_check_time is not None
            and current_time - self._last_check_time < self.check_interval
        ):
            return False

        image_inputs = self.processor.image_processor(
            images=[self._image_type.fromarray(np.asarray(current_frame, dtype=np.uint8))],
            return_tensors="pt",
        )
        with self._torch.inference_mode():
            current_feature = self.model.encode_image(image_inputs["pixel_values"].to(self.device))
            logits = self.model.classify_features(
                current_feature, self._initial_feature, self._text_feature
            )
            self.last_probability = float(self._torch.sigmoid(logits).item())
        self._last_check_time = float(current_time)
        self._positive_count = self._positive_count + 1 if self.last_probability >= self.threshold else 0
        return self._positive_count >= self.consecutive_positive


def _geom_name(model, geom_id: int) -> str:
    geom_id = int(geom_id)
    geom_names = getattr(model, "geom_names", None)
    if isinstance(geom_names, dict):
        return str(geom_names.get(geom_id, ""))
    try:
        return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
    except Exception:
        try:
            return str(model.geom(geom_id).name)
        except Exception:
            return ""


def _is_robot_geom_name(name: str) -> bool:
    normalized = str(name)
    return (
        normalized.startswith("/ur:")
        or normalized.startswith("1/ur:")
        or normalized.startswith("2/ur:")
        or "/ur:" in normalized
        or normalized.startswith("ur5")
    )


def count_robot_object_collision_contacts(model, data) -> int:
    """Count unique robot-object contact pairs in the current MuJoCo contact buffer."""
    robot_object_pairs: set[tuple[int, int]] = set()
    for i in range(int(getattr(data, "ncon", 0))):
        contact = data.contact[i]
        geom1 = int(contact.geom1)
        geom2 = int(contact.geom2)
        geom1_is_robot = _is_robot_geom_name(_geom_name(model, geom1))
        geom2_is_robot = _is_robot_geom_name(_geom_name(model, geom2))
        if geom1_is_robot == geom2_is_robot:
            continue
        robot_object_pairs.add(tuple(sorted((geom1, geom2))))
    return len(robot_object_pairs)


def capture_prompt_initial_state(data, state_indices, action_indices) -> dict:
    return {
        "qpos": np.asarray(data.qpos[list(state_indices)], dtype=np.float64).copy(),
        "ctrl": np.asarray(data.ctrl[list(action_indices)], dtype=np.float64).copy(),
    }


def restore_robot_state_direct(
    *,
    task,
    data,
    state_indices,
    action_indices,
    target_state,
    num_steps: int = 250,
) -> int:
    state_indices = list(state_indices)
    action_indices = list(action_indices)
    if len(action_indices) == 0:
        return 0

    target = np.asarray(target_state, dtype=np.float64).reshape(-1)
    if target.size != len(action_indices) or not np.isfinite(target).all():
        raise ValueError(
            f"Restore target must contain {len(action_indices)} finite values, got {target.size}"
        )

    current = np.asarray(data.qpos[state_indices], dtype=np.float64).reshape(-1)
    if current.size != target.size:
        current = np.asarray(data.ctrl[action_indices], dtype=np.float64).reshape(-1)
    if current.size != target.size or not np.isfinite(current).all():
        raise ValueError(
            f"Current restore state must contain {target.size} finite values, got {current.size}"
        )

    steps = max(1, int(num_steps))
    for alpha in np.linspace(1.0 / steps, 1.0, steps):
        data.ctrl[action_indices] = current + alpha * (target - current)
        task.step_and_log({})
    return steps


def restore_robot_state_rrt(
    *,
    model,
    task,
    data,
    state_indices,
    action_indices,
    target_state,
    rng: np.random.Generator,
    steps_per_segment: int = 250,
    validation_steps_per_segment: int = 100,
    gripper_settle_steps: int = 50,
) -> dict:
    """Restore six arm joints with RRT, then restore the gripper separately."""
    state_indices = list(state_indices)
    action_indices = list(action_indices)
    target = np.asarray(target_state, dtype=np.float64).reshape(-1)
    if len(state_indices) != 7 or len(action_indices) != 7:
        raise ValueError(
            "Problem-validation RRT restore requires six arm joints and one gripper "
            f"(state={len(state_indices)}, action={len(action_indices)})"
        )
    if target.size != 7 or not np.isfinite(target).all():
        raise ValueError(
            f"Problem-validation RRT target must contain 7 finite values, got {target.size}"
        )

    arm_state_indices = state_indices[:-1]
    arm_action_indices = action_indices[:-1]
    current_arm = np.asarray(data.qpos[arm_state_indices], dtype=np.float64).reshape(-1)
    target_arm = target[:-1]
    if current_arm.size != 6 or not np.isfinite(current_arm).all():
        raise ValueError(
            "Problem-validation RRT start must contain 6 finite arm-joint values, "
            f"got {current_arm.size}"
        )

    def validate_path(waypoints):
        return non_llm_transition.validate_joint_path_in_mujoco(
            model,
            data,
            arm_state_indices,
            waypoints,
            num_steps_per_segment=validation_steps_per_segment,
        )

    path_plan = non_llm_transition.plan_joint_path_rrt(
        current_arm,
        target_arm,
        path_validator=validate_path,
        joint_ranges=non_llm_transition.joint_ranges_from_model(model, arm_state_indices),
        rng=rng,
    )
    rrt_failed = path_plan.status == "RRT_FAILED_SKIP_ACTION" or not path_plan.waypoints
    if rrt_failed:
        print("[ProblemValidationDemo] RRT_FAILED_SKIP_ACTION")

    if rrt_failed:
        executed_arm_steps = 0
    else:
        executed_arm_steps = non_llm_transition.execute_interpolated_joint_path(
            task=task,
            data=data,
            act_span=arm_action_indices,
            waypoints=path_plan.waypoints,
            steps_per_segment=steps_per_segment,
        )

    executed_gripper_steps = max(0, int(gripper_settle_steps))
    data.ctrl[action_indices[-1]] = float(target[-1])
    for _ in range(executed_gripper_steps):
        task.step_and_log({})

    return {
        "planner_status": path_plan.status,
        "planner_validation": path_plan.validation,
        "waypoint_count": len(path_plan.waypoints),
        "executed_arm_steps": executed_arm_steps,
        "executed_gripper_steps": executed_gripper_steps,
        "fallback": rrt_failed,
    }


def restore_prompt_initial_state_direct(
    *,
    task,
    data,
    state_indices,
    action_indices,
    initial_state: dict,
    num_steps: int = 250,
) -> int:
    state_indices = list(state_indices)
    action_indices = list(action_indices)
    if len(action_indices) == 0:
        return 0

    target = np.asarray(initial_state["qpos"], dtype=np.float64).reshape(-1)
    if target.size != len(action_indices):
        target = np.asarray(initial_state["ctrl"], dtype=np.float64).reshape(-1)
    if target.size != len(action_indices):
        raise ValueError(
            f"Initial restore target dim {target.size} does not match action dim {len(action_indices)}"
        )

    return restore_robot_state_direct(
        task=task,
        data=data,
        state_indices=state_indices,
        action_indices=action_indices,
        target_state=target,
        num_steps=num_steps,
    )


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
        render_video: bool = True,
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

        self.render_video = bool(render_video)
        self.video_out_path = pathlib.Path(video_out_path)
        if self.render_video:
            self.video_out_path.mkdir(parents=True, exist_ok=True)
        self.video_fps = max(1, int(video_fps))
        self.replay_images = []
        self.replay_left_images = []
        self.replay_times = []
        self._next_replay_capture_time = 0.0

    def _update_render_scene(self):
        self.renderer.update_scene(self.data)
        if self.render_extra is not None:
            self.render_extra(self.renderer.scene, self.renderer._mjr_context)

    def _capture_replay_frame(self, force: bool = False):
        if not self.render_video:
            return

        current_time = float(self.data.time)
        capture_interval = 1.0 / self.video_fps
        next_capture_time = getattr(self, "_next_replay_capture_time", current_time)
        if not force and current_time + 1e-9 < next_capture_time:
            return

        if "image" not in self.cameras:
            return

        self._update_render_scene()
        front_frame = self.get_image("image")
        if front_frame is None:
            return

        left_frame = self.get_image_by_camera_name("table_cam_left")

        self.replay_images.append(front_frame.astype(np.uint8))
        if left_frame is not None:
            self.replay_left_images.append(left_frame.astype(np.uint8))
        self.replay_times.append(current_time)
        self._next_replay_capture_time = current_time + capture_interval

    def _resample_replay_frames(self, replay_images):
        if not replay_images:
            return [], np.array([], dtype=np.int64)

        if len(replay_images) == 1 or len(self.replay_times) != len(replay_images):
            return list(replay_images), np.arange(len(replay_images), dtype=np.int64)

        times = np.asarray(self.replay_times, dtype=np.float64)
        start_t = float(times[0])
        end_t = float(times[-1])
        if end_t <= start_t:
            return [replay_images[0]], np.array([0], dtype=np.int64)

        dt = 1.0 / self.video_fps
        target_times = np.arange(start_t, end_t + 1e-9, dt, dtype=np.float64)

        right_idx = np.searchsorted(times, target_times, side="left")
        right_idx = np.clip(right_idx, 0, len(times) - 1)
        left_idx = np.maximum(right_idx - 1, 0)

        choose_right = np.abs(times[right_idx] - target_times) <= np.abs(times[left_idx] - target_times)
        sampled_indices = np.where(choose_right, right_idx, left_idx)
        sampled_indices = np.unique(sampled_indices)

        sampled_frames = [replay_images[i] for i in sampled_indices]
        return sampled_frames, sampled_indices

    def _build_video_filename(self, success: bool, filename_override: str | None, camera_suffix: str = ""):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        suffix_part = f"_{camera_suffix}" if camera_suffix else ""

        if filename_override:
            return f"{filename_override}{suffix_part}_{timestamp}.mp4"

        task_name = self.task_info['prefix'].replace(" ", "_").replace("/", "_")
        status = "success" if success else "failure"
        return f"rollout_{task_name}_{timestamp}_{status}{suffix_part}.mp4"

    def _write_video(self, frames, success: bool, filename_override: str | None, camera_suffix: str = ""):
        if not frames:
            return None

        filename = self._build_video_filename(success, filename_override, camera_suffix)
        video_path = self.video_out_path / filename
        try:
            imageio.mimwrite(
                video_path,
                [np.asarray(img) for img in frames],
                fps=self.video_fps,
            )
            print(f"Video saved ({camera_suffix or 'front'}): {video_path}")
            return video_path
        except Exception as e:
            print(f"Failed to save video ({camera_suffix or 'front'}): {e}")
            return None

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
        self._update_render_scene()
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
        self.replay_left_images = []
        self.replay_times = []
        self._next_replay_capture_time = float(self.data.time)
        if self.render_video:
            self._capture_replay_frame(force=True)

    def save_video(
        self,
        success: bool,
        filename_override: str | None = None,
        action_count: int | None = None,
        strict: bool = False,
    ): #
        if not self.render_video:
            if action_count is not None:
                print(f"[ActionDiag] actions={action_count}")
            print("[VideoDiag] Replay video rendering disabled; skipping video save.")
            if strict:
                raise _RequiredVideoWriteError(
                    "Problem-validation demo required video write failed for front: "
                    "replay video rendering is disabled"
                )
            return

        def left_camera_exists():
            try:
                return (
                    mujoco.mj_name2id(
                        self.model,
                        mujoco.mjtObj.mjOBJ_CAMERA,
                        "table_cam_left",
                    )
                    >= 0
                )
            except Exception:
                return False

        if not self.replay_images:
            if strict:
                failed_required_cameras = ["front"]
                if self.replay_left_images:
                    sampled_left_frames, _ = self._resample_replay_frames(
                        self.replay_left_images
                    )
                    left_path = self._write_video(
                        sampled_left_frames,
                        success,
                        filename_override,
                        camera_suffix="left",
                    )
                    if left_path is None:
                        failed_required_cameras.append("left")
                elif left_camera_exists():
                    self._write_video(
                        [],
                        success,
                        filename_override,
                        camera_suffix="left",
                    )
                    failed_required_cameras.append("left")
                raise _RequiredVideoWriteError(
                    "Problem-validation demo required video write failed for "
                    + ", ".join(failed_required_cameras)
                    + ": no front replay frames were captured"
                )
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

        sampled_front_frames, sampled_indices = self._resample_replay_frames(self.replay_images)
        output_frame_count = len(sampled_front_frames)

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

        failed_required_cameras = []
        front_path = self._write_video(sampled_front_frames, success, filename_override)
        if strict and front_path is None:
            failed_required_cameras.append("front")

        if len(self.replay_left_images) == raw_frame_count:
            sampled_left_frames = [self.replay_left_images[i] for i in sampled_indices]
            left_path = self._write_video(
                sampled_left_frames,
                success,
                filename_override,
                camera_suffix="left",
            )
            if strict and left_path is None:
                failed_required_cameras.append("left")
        elif len(self.replay_left_images) > 0:
            sampled_left_frames, _ = self._resample_replay_frames(self.replay_left_images)
            left_path = self._write_video(
                sampled_left_frames,
                success,
                filename_override,
                camera_suffix="left",
            )
            if strict and left_path is None:
                failed_required_cameras.append("left")
        else:
            left_camera_present = strict and left_camera_exists()
            if left_camera_present:
                left_path = self._write_video(
                    [],
                    success,
                    filename_override,
                    camera_suffix="left",
                )
                if left_path is None:
                    failed_required_cameras.append("left")
            else:
                print("[VideoDiag] table_cam_left not found; skipping left camera video.")

        if failed_required_cameras:
            raise _RequiredVideoWriteError(
                "Problem-validation demo required video write failed for "
                + ", ".join(failed_required_cameras)
            )

    def evaluate(
        self,
        policy: Policy,
        time_limit: float | None = None,
        prompts: list[str] | None = None,
        use_transition_generation: bool = True,
        transition_mode: str = "auto",
        no_planning: bool = False,
        no_interpolation: bool = False,
        no_retrieval: bool = False,
        control_fps: float = 50.0,
        llm_config: dict | None = None,
        use_task_judge: bool = False,
        max_prompt_retries: int = 1,
        judge_confidence_threshold: float = 0.6,
        judge_on_error: str = "fail",
        intervention_mode: str = "non_timeout",
        completion_model_config: dict[str, Any] | None = None,
        transition_seed: int | None = None,
        problem_validation_demo_config: ProblemValidationDemoConfig | None = None,
    ):
        if time_limit is None:
            time_limit = self.task.time_limit

        if problem_validation_demo_config is not None:
            config = problem_validation_demo_config
            if prompts is None or tuple(prompts) != config.prompts:
                raise ValueError(
                    "Problem-validation demo prompts must exactly match the fixed config: "
                    f"expected {config.prompts!r}, got {prompts!r}"
                )

        self.reset()
        completion_predicate = None
        if completion_model_config is not None:
            cached_config = getattr(self, "_completion_model_config", None)
            if cached_config != completion_model_config:
                self._completion_model_predicate = _CompletionModelPredicate(completion_model_config)
                self._completion_model_config = dict(completion_model_config)
            completion_predicate = self._completion_model_predicate
        if control_fps <= 0:
            raise ValueError(f"control_fps must be positive, got {control_fps}")
        action_repeat_steps = max(1, int(round(1.0 / control_fps / self.model.opt.timestep)))
        effective_control_fps = 1.0 / (action_repeat_steps * self.model.opt.timestep)
        print(
            f"[ControlDiag] requested_control_fps={control_fps:.3f} | "
            f"action_repeat_steps={action_repeat_steps} | "
            f"effective_control_fps={effective_control_fps:.3f} | "
            f"timestep={self.model.opt.timestep:.6f}"
        )
        episode_start_wall = time.perf_counter()
        transition_infer_total = 0.0
        transition_total = 0.0
        transition_count = 0
        transition_collision_counts: dict[int, int] = {}
        executed_action_count = 0
        atomic_task_results: list[dict] = []
        settle_duration = 0.0
        effective_transition_mode = (transition_mode or "auto").strip().lower()
        if effective_transition_mode == "auto":
            effective_transition_mode = "llm" if use_transition_generation else "none"
        if effective_transition_mode not in {
            "none",
            "llm",
            "retrieval_interp",
            "retrieval_collision_planner",
            "random_future_task_pose_collision_planner",
            "random_future_task_pose_rrt",
            "random_dataset_task_pose_rrt",
            "ready_memory_initial_pose_rrt",
        }:
            raise ValueError(f"Unsupported transition_mode: {transition_mode}")
        effective_intervention_mode = (intervention_mode or "non_timeout").strip().lower().replace("-", "_")
        if effective_intervention_mode not in {"non_timeout", "timeout"}:
            raise ValueError(f"Unsupported intervention_mode: {intervention_mode}")
        transition_rng = np.random.default_rng(transition_seed)
        def build_timing_stats():
            episode_total = time.perf_counter() - episode_start_wall
            transition_ratio = (transition_total / episode_total * 100.0) if episode_total > 0 else 0.0
            planning_avg_per_transition = (transition_infer_total / transition_count) if transition_count > 0 else 0.0
            transition_avg_per_transition = (transition_total / transition_count) if transition_count > 0 else 0.0
            return {
                "transition_planning_total": float(transition_infer_total),
                "transition_total": float(transition_total),
                "transition_count": int(transition_count),
                "transition_planning_avg_per_transition": float(planning_avg_per_transition),
                "transition_avg_per_transition": float(transition_avg_per_transition),
                "episode_total": float(episode_total),
                "transition_ratio": float(transition_ratio),
                "atomic_task_results": list(atomic_task_results),
                "transition_collision_counts": {
                    str(index): int(count)
                    for index, count in sorted(transition_collision_counts.items())
                },
            }

        def print_timing_summary():
            timing = build_timing_stats()
            print(
                f"[Timing] transition planning total: {timing['transition_planning_total']:.3f}s | "
                f"transition total: {timing['transition_total']:.3f}s | "
                f"transition count: {timing['transition_count']} | "
                f"planning avg/transition: {timing['transition_planning_avg_per_transition']:.3f}s | "
                f"transition avg/transition: {timing['transition_avg_per_transition']:.3f}s | "
                f"episode total: {timing['episode_total']:.3f}s | "
                f"transition ratio: {timing['transition_ratio']:.2f}%"
            )

        def finalize_problem_validation_demo(success: bool):
            cleanup_errors = []
            video_error = None
            try:
                self.task.finish()
            except BaseException as exc:
                cleanup_errors.append(exc)
            try:
                self.render_finish()
            except BaseException as exc:
                cleanup_errors.append(exc)
            try:
                self.save_video(
                    success,
                    filename_override=config.video_filename_prefix,
                    action_count=executed_action_count,
                    strict=True,
                )
            except BaseException as exc:
                cleanup_errors.append(exc)
                if isinstance(exc, _RequiredVideoWriteError):
                    video_error = exc
            return video_error or (cleanup_errors[0] if cleanup_errors else None)

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

        def check_prompt_success(prompt: str | None):
            check_func = getattr(self.task, "check", None)
            if not callable(check_func):
                return None
            try:
                signature = inspect.signature(check_func)
            except (TypeError, ValueError):
                return bool(check_func()) if prompt is None else None

            if "prompt" in signature.parameters:
                return bool(check_func(prompt=prompt))
            if prompt is None:
                return bool(check_func())
            return None

        def check_prompt_completion(prompt: str | None):
            """Return the signal that controls VLA intervention, not task scoring."""
            if completion_predicate is None:
                return check_prompt_success(prompt)
            completion_prompt = str(prompt or self.task_info.get("prefix", "")).strip()
            if not completion_prompt:
                raise ValueError("Completion-model inference requires a non-empty task prompt.")
            current_frame = self.get_images()["observation/image"]
            complete = completion_predicate.check(
                completion_prompt, current_frame, current_time=float(self.data.time)
            )
            if complete:
                print(
                    f"[CompletionModel] prompt={completion_prompt!r} "
                    f"probability={completion_predicate.last_probability:.3f} complete=True"
                )
            return complete

        def record_prompt_start(prompt: str | None):
            record_func = getattr(self.task, "record_atomic_start", None)
            if callable(record_func):
                record_func(prompt)
            if completion_predicate is not None:
                completion_prompt = str(prompt or self.task_info.get("prefix", "")).strip()
                if not completion_prompt:
                    raise ValueError("Completion-model inference requires a non-empty task prompt.")
                initial_frame = self.get_images()["observation/image"]
                completion_predicate.start_prompt(completion_prompt, initial_frame)

        # Let the scene settle for 2 seconds of simulation time before the first action.
        settle_steps = max(1, int(round(settle_duration / self.model.opt.timestep)))
        for _ in range(settle_steps):
            healthy = step()
            if not healthy:
                if problem_validation_demo_config is not None:
                    cleanup_error = finalize_problem_validation_demo(False)
                    if cleanup_error is not None:
                        print(f"[ProblemValidationDemo] cleanup_error={cleanup_error}")
                        if isinstance(cleanup_error, _RequiredVideoWriteError):
                            raise cleanup_error
                else:
                    self.task.finish()
                    self.render_finish()
                    self.save_video(False, action_count=executed_action_count)
                print_timing_summary()
                return False, build_timing_stats()
        # self._capture_replay_frame()

        def run_prompt(
            prompt: str | None,
            current_time_limit: float,
            post_success_seconds: float = 0.0,
            force_non_timeout: bool = False,
        ):
            nonlocal executed_action_count
            start_time = self.data.time
            controller = PromptRunController(
                start_time=start_time,
                time_limit=current_time_limit,
                post_success_seconds=post_success_seconds,
            )
            use_non_timeout_controller = (
                # The completion model is an intervention switch: stop as soon as it
                # predicts completion, but retain current_time_limit as a hard timeout
                # when it never produces a positive prediction.
                force_non_timeout
                or completion_predicate is not None
                or effective_intervention_mode == "non_timeout"
            )

            def prompt_should_continue():
                if not use_non_timeout_controller:
                    return (self.data.time - start_time) < current_time_limit
                return controller.should_continue(self.data.time)

            while prompt_should_continue():
                observation = self.get_observation()
                actions = policy(observation)
                assert actions.ndim == 2 and actions.shape[1] == len(self.task_info['action_indices']), breakpoint()
                # if abs(actions[-1]-actions[0]) < 1e-2:
                #     return True
                for action in actions:
                    self.data.ctrl[self.task_info['action_indices']] = action
                    executed_action_count += 1
                    self.history_states.append(self.data.qpos)
                    for _ in range(action_repeat_steps):
                        healthy = step()
                        if not healthy:
                            return False, False
                    prompt_completion = check_prompt_completion(prompt)
                    previous_success_time = controller.success_time
                    tail_complete = controller.observe(
                        self.data.time,
                        success=prompt_completion,
                    )
                    if previous_success_time is None and controller.success_time is not None:
                        if post_success_seconds > 0.0:
                            print(
                                "[ProblemValidationDemo] "
                                f"first_success_time={controller.success_time:.6f}"
                            )
                    if tail_complete:
                        return True, check_prompt_success(prompt)
            return True, check_prompt_success(prompt)

        if problem_validation_demo_config is not None:
            demo_success = False
            sequence_result = None
            pending_error = None
            pending_traceback = None

            def run_demo_prompt(prompt: str, post_success_seconds: float):
                prompt_index = len(atomic_task_results)
                atomic_result = {
                    "prompt_index": int(prompt_index),
                    "prompt": prompt,
                    "success": False,
                    "attempt_index": 0,
                }
                atomic_task_results.append(atomic_result)
                print(f"[ProblemValidationDemo] prompt_index={prompt_index} prompt={prompt!r}")
                self.task_info["prefix"] = prompt
                healthy = False
                try:
                    record_prompt_start(prompt)
                    healthy, local_success = run_prompt(
                        prompt,
                        time_limit,
                        post_success_seconds=post_success_seconds,
                        force_non_timeout=True,
                    )
                    atomic_result["success"] = bool(local_success)
                    return bool(healthy), bool(local_success)
                finally:
                    print(
                        f"[ProblemValidationDemo] prompt_index={prompt_index} "
                        f"attempt_index=0 healthy={bool(healthy)} "
                        f"success={atomic_result['success']} prompt={prompt!r}"
                    )

            def restore_demo_state(sampled_state, restore_steps_per_segment: int):
                nonlocal transition_total, transition_count
                print(
                    "[ProblemValidationDemo] "
                    f"task={sampled_state.task!r} "
                    f"dataset_root={sampled_state.dataset_root} "
                    f"episode={sampled_state.episode_index} "
                    f"length={sampled_state.episode_length} "
                    f"prefix_size={sampled_state.prefix_frame_count} "
                    f"frame={sampled_state.frame_index} "
                    f"ratio={sampled_state.frame_ratio:.6f}"
                )
                print(
                    "[ProblemValidationDemo] state="
                    + np.array2string(np.asarray(sampled_state.state), precision=8)
                )
                transition_start_wall = time.perf_counter()
                transition_collision_count = 0
                original_step_and_log = self.task.step_and_log

                def step_and_log_with_capture(info: dict):
                    nonlocal transition_collision_count
                    original_step_and_log(info)
                    if self.data.warning.number.any():
                        warning_number = np.asarray(self.data.warning.number).copy()
                        raise RuntimeError(
                            "Problem-validation RRT restore stopped after a MuJoCo warning: "
                            f"warning.number={warning_number.tolist()}"
                        )
                    transition_collision_count += count_robot_object_collision_contacts(
                        self.model,
                        self.data,
                    )
                    self._capture_replay_frame()

                self.task.step_and_log = step_and_log_with_capture
                try:
                    restore_result = restore_robot_state_rrt(
                        model=self.model,
                        task=self.task,
                        data=self.data,
                        state_indices=self.task_info["state_indices"],
                        action_indices=self.task_info["action_indices"],
                        target_state=sampled_state.state,
                        rng=transition_rng,
                        steps_per_segment=restore_steps_per_segment,
                    )
                finally:
                    self.task.step_and_log = original_step_and_log

                transition_elapsed = time.perf_counter() - transition_start_wall
                transition_total += transition_elapsed
                transition_count += 1
                transition_collision_counts[1] = int(transition_collision_count)
                print(
                    "[ProblemValidationDemo] "
                    f"planner_status={restore_result['planner_status']} "
                    f"planner_validation={restore_result['planner_validation']} "
                    f"waypoints={restore_result['waypoint_count']} "
                    f"arm_steps={restore_result['executed_arm_steps']} "
                    f"gripper_steps={restore_result['executed_gripper_steps']} "
                    f"fallback={restore_result['fallback']} "
                    f"robot_object_contacts={transition_collision_count}"
                )

            try:
                sequence_result = execute_problem_validation_sequence(
                    config=config,
                    rng=transition_rng,
                    run_prompt=run_demo_prompt,
                    sample_state=sample_problem_validation_state,
                    restore_state=restore_demo_state,
                )
                demo_success = bool(sequence_result.success)
            except BaseException as exc:
                pending_error = exc
                pending_traceback = exc.__traceback__
            finally:
                cleanup_error = finalize_problem_validation_demo(demo_success)
                print_timing_summary()
                if pending_error is None and cleanup_error is not None:
                    pending_error = cleanup_error
                    pending_traceback = cleanup_error.__traceback__
                elif pending_error is not None and cleanup_error is not None:
                    print(f"[ProblemValidationDemo] cleanup_error={cleanup_error}")

            if pending_error is not None:
                raise pending_error.with_traceback(pending_traceback)
            return demo_success, build_timing_stats()

        if prompts is None:
            record_prompt_start(None)
            healthy, prompt_success = run_prompt(None, time_limit)
            task_success = bool(prompt_success) if prompt_success is not None else (self.task.check() if healthy else False)
            atomic_task_results.append(
                {
                    "prompt_index": 0,
                    "prompt": None,
                    "success": bool(task_success),
                    "attempt_index": 0,
                }
            )
            self.task.finish()
            self.render_finish()
            self.save_video(task_success, action_count=executed_action_count)
            print_timing_summary()
            if not healthy:
                return False, build_timing_stats()
            return task_success, build_timing_stats()
        
        
        if len(prompts)>1:
            all_success = True
            max_prompt_retries = max(0, int(max_prompt_retries))

            def save_transition_state(prompt_index: int, attempt_index: int):
                current_view, current_side_view = self.get_transition_views()
                logs_dir = pathlib.Path("logs")
                logs_dir.mkdir(parents=True, exist_ok=True)
                judgement_dir = logs_dir / "task_judgement_images"
                judgement_dir.mkdir(parents=True, exist_ok=True)

                front_path = judgement_dir / f"prompt_{prompt_index:02d}_attempt_{attempt_index:02d}_front.png"
                side_path = judgement_dir / f"prompt_{prompt_index:02d}_attempt_{attempt_index:02d}_side.png"

                if current_view is not None:
                    imageio.imwrite('logs/current_view.png', current_view)
                    imageio.imwrite(front_path, current_view)
                if current_side_view is not None:
                    imageio.imwrite('logs/current_side_view.png', current_side_view)
                    imageio.imwrite(side_path, current_side_view)
                current_joint_pos = self.data.qpos[range(self.model.joint('/ur:shoulder_pan').qposadr.item(), self.model.joint('/ur:shoulder_pan').qposadr.item() + 6)]
                np.save('logs/current_joint.npy', current_joint_pos)
                try:
                    from camera_calibration_enhancement import (
                        build_transition_calibration_payload,
                        write_calibration_assets,
                    )

                    calibration_payload = build_transition_calibration_payload(
                        model=self.model,
                        data=self.data,
                        front_image=current_view,
                        side_image=current_side_view,
                    )
                    calibration_assets = write_calibration_assets(
                        calibration_payload,
                        {"front": current_view, "side": current_side_view},
                        output_dir=logs_dir,
                    )
                    annotated_paths = calibration_assets.get("annotated_image_paths", {})
                    if annotated_paths.get("front"):
                        front_path = pathlib.Path(annotated_paths["front"])
                    if annotated_paths.get("side"):
                        side_path = pathlib.Path(annotated_paths["side"])
                except Exception as e:
                    print(f"[Calibration] Failed to build transition calibration assets: {e}")
                return front_path, side_path

            def execute_transition_to(target_prompt: str, timing_label: str, transition_index: int | None = None):
                nonlocal transition_infer_total, transition_total, transition_count
                if effective_transition_mode == "none":
                    print(f"[Transition] Skipped transition to {timing_label} (transition_mode=none).")
                    return

                transition_start_wall = time.perf_counter()
                transition_infer_start_wall = time.perf_counter()
                transition_collision_count = 0

                def record_transition_collision_sample():
                    nonlocal transition_collision_count
                    transition_collision_count += count_robot_object_collision_contacts(self.model, self.data)

                ur_joint_start = self.model.joint('/ur:shoulder_pan').qposadr.item()
                ur_jnt_span = range(ur_joint_start, ur_joint_start + 6)

                if effective_transition_mode == "llm":
                    from transition_generation import transition_code_generation, validate_qpos_interpolation_path

                    def qpos_path_validator(candidate_qpos, *, selected_index: int):
                        return validate_qpos_interpolation_path(
                            self.model,
                            self.data,
                            ur_jnt_span,
                            candidate_qpos,
                        )

                    def target_ee_position_resolver(candidate_qpos):
                        candidate = np.asarray(candidate_qpos, dtype=np.float64).reshape(-1)
                        if candidate.size < len(ur_jnt_span):
                            raise ValueError(
                                f"Target qpos has {candidate.size} values, expected at least {len(ur_jnt_span)}"
                            )
                        site_id = self.model.site('/ur:2f85:pinch').id
                        original_qpos = np.asarray(self.data.qpos, dtype=np.float64).copy()
                        try:
                            self.data.qpos[ur_jnt_span] = candidate[: len(ur_jnt_span)]
                            mujoco.mj_forward(self.model, self.data)
                            return np.asarray(self.data.site_xpos[site_id], dtype=np.float64).copy()
                        finally:
                            self.data.qpos[:] = original_qpos
                            mujoco.mj_forward(self.model, self.data)

                    transition_code_generation(
                        target_prompt,
                        no_planning=no_planning,
                        no_interpolation=no_interpolation,
                        no_retrieval=no_retrieval,
                        llm_config=llm_config,
                        target_top_k=3,
                        qpos_path_validator=qpos_path_validator,
                        target_ee_position_resolver=target_ee_position_resolver,
                    )
                    transition_infer_elapsed = time.perf_counter() - transition_infer_start_wall
                    transition_infer_total += transition_infer_elapsed

                    import transition_template

                    importlib.invalidate_caches()
                    transition_template = importlib.reload(transition_template)
                    TransitionExpert = transition_template.TransitionExpert
                    expert = TransitionExpert(self.model, self.data, self.task)
                    original_step_and_log = self.task.step_and_log

                    def step_and_log_with_capture(info: dict):
                        original_step_and_log(info)
                        record_transition_collision_sample()
                        self._capture_replay_frame()

                    self.task.step_and_log = step_and_log_with_capture
                    try:
                        self._capture_replay_frame()
                        expert.execute()
                        self._capture_replay_frame()
                    finally:
                        self.task.step_and_log = original_step_and_log
                else:
                    from non_llm_transition import execute_non_llm_transition

                    original_step_and_log = self.task.step_and_log

                    def step_and_log_with_capture(info: dict):
                        original_step_and_log(info)
                        record_transition_collision_sample()
                        self._capture_replay_frame()

                    self.task.step_and_log = step_and_log_with_capture
                    try:
                        self._capture_replay_frame()
                        non_llm_result = execute_non_llm_transition(
                            model=self.model,
                            data=self.data,
                            task=self.task,
                            target_prompt=target_prompt,
                            mode=effective_transition_mode,
                            target_top_k=3,
                            rng=transition_rng,
                            ready_memory_db_path=pathlib.Path(
                                (llm_config or {}).get("ready_memory_db_path")
                                or "logs/ready_memory_index.json"
                            ),
                        )
                        self._capture_replay_frame()
                    finally:
                        self.task.step_and_log = original_step_and_log
                    transition_infer_elapsed = float(
                        non_llm_result.get(
                            "planning_elapsed",
                            time.perf_counter() - transition_infer_start_wall,
                        )
                    )
                    transition_infer_total += transition_infer_elapsed
                transition_elapsed = time.perf_counter() - transition_start_wall
                transition_total += transition_elapsed
                transition_count += 1
                if transition_index is not None:
                    transition_collision_counts[int(transition_index)] = (
                        transition_collision_counts.get(int(transition_index), 0)
                        + int(transition_collision_count)
                    )
                    print(
                        f"[TransitionCollision] transition_{int(transition_index)} "
                        f"robot_object_contacts={int(transition_collision_count)}"
                    )
                print(
                    f"[Timing] transition to {timing_label} took {transition_elapsed:.3f}s "
                    f"(inference {transition_infer_elapsed:.3f}s)"
                )

            def execute_retry_restore_to_initial(initial_state: dict, timing_label: str):
                nonlocal transition_infer_total, transition_total, transition_count
                transition_start_wall = time.perf_counter()
                original_step_and_log = self.task.step_and_log

                def step_and_log_with_capture(info: dict):
                    original_step_and_log(info)
                    self._capture_replay_frame()

                self.task.step_and_log = step_and_log_with_capture
                try:
                    self._capture_replay_frame()
                    restored_steps = restore_prompt_initial_state_direct(
                        task=self.task,
                        data=self.data,
                        state_indices=self.task_info["state_indices"],
                        action_indices=self.task_info["action_indices"],
                        initial_state=initial_state,
                    )
                    self._capture_replay_frame()
                finally:
                    self.task.step_and_log = original_step_and_log

                transition_elapsed = time.perf_counter() - transition_start_wall
                transition_total += transition_elapsed
                transition_count += 1
                print(
                    f"[Timing] retry restore to {timing_label} took {transition_elapsed:.3f}s "
                    f"(inference 0.000s, steps {restored_steps})"
                )
            
            # 多 prompt 逻辑
            prompt_index = 0
            attempt_counts = [0 for _ in prompts]
            while prompt_index < len(prompts):
                prompt = prompts[prompt_index]
                attempt_index = attempt_counts[prompt_index]
                print(f"Executing prompt: {prompt} (attempt {attempt_index + 1})")
                self.task_info['prefix'] = prompt
                record_prompt_start(prompt)
                prompt_initial_state = capture_prompt_initial_state(
                    self.data,
                    self.task_info["state_indices"],
                    self.task_info["action_indices"],
                )
                # task_nums = len(prompts)
                healthy, local_prompt_success = run_prompt(prompt, time_limit)
                atomic_task_results.append(
                    {
                        "prompt_index": int(prompt_index),
                        "prompt": prompt,
                        "success": bool(local_prompt_success),
                        "attempt_index": int(attempt_index),
                    }
                )
                
                if not healthy:
                    all_success = False
                    failed_prompt = prompt
                    break

                prompt_failed_but_continuing = False
                if local_prompt_success is False:
                    all_success = False
                    failed_prompt = prompt
                    if effective_intervention_mode == "timeout":
                        prompt_failed_but_continuing = True
                        print(
                            f"[TaskCheck] prompt_index={prompt_index} attempt={attempt_index} "
                            "timed out without satisfying the local success predicate; "
                            "marking episode failed but continuing to subsequent prompts."
                        )
                    else:
                        print(
                            f"[TaskCheck] prompt_index={prompt_index} attempt={attempt_index} "
                            "timed out without satisfying the local success predicate; "
                            "skipping transition and failing episode."
                        )
                        break

                front_path, side_path = save_transition_state(prompt_index, attempt_index)

                if local_prompt_success is True:
                    action = "advance"
                    target_prompt = prompts[prompt_index + 1] if prompt_index + 1 < len(prompts) else None
                    print(
                        f"[TaskCheck] prompt_index={prompt_index} attempt={attempt_index} "
                        "local_success=True action=advance"
                    )
                elif prompt_failed_but_continuing:
                    action = "advance"
                    target_prompt = prompts[prompt_index + 1] if prompt_index + 1 < len(prompts) else None
                elif use_task_judge:
                    from transition_judgement import (
                        append_judgement_log,
                        decide_prompt_action,
                        judge_task_success,
                        resolve_judge_error_success,
                        select_transition_target_prompt,
                    )
                    judgement = None
                    error_text = None
                    try:
                        judgement = judge_task_success(
                            prompt=prompt,
                            front_image_path=front_path,
                            side_image_path=side_path,
                            llm_config=llm_config,
                            threshold=judge_confidence_threshold,
                        )
                        prompt_success = bool(judgement["prompt_success"])
                    except Exception as e:
                        error_text = str(e)
                        prompt_success = resolve_judge_error_success(judge_on_error)
                        judgement = {
                            "success": prompt_success,
                            "confidence": 1.0 if prompt_success else 0.0,
                            "reason": f"judge error handled with judge_on_error={judge_on_error}",
                            "failure_mode": "judge_error",
                            "prompt_success": prompt_success,
                            "raw": None,
                        }
                        print(f"[TaskJudge] Judge failed for prompt {prompt_index}: {error_text}")

                    action = decide_prompt_action(
                        prompt_success=prompt_success,
                        attempt_index=attempt_index,
                        max_prompt_retries=max_prompt_retries,
                        is_final_prompt=prompt_index == len(prompts) - 1,
                    )
                    append_judgement_log(
                        {
                            "task_name": getattr(self.task, "name", None),
                            "prompt_index": prompt_index,
                            "prompt": prompt,
                            "attempt_index": attempt_index,
                            "success": judgement["success"],
                            "prompt_success": judgement["prompt_success"],
                            "confidence": judgement["confidence"],
                            "reason": judgement["reason"],
                            "failure_mode": judgement["failure_mode"],
                            "front_image_path": str(front_path),
                            "side_image_path": str(side_path),
                            "action": action,
                            "raw_judge_result": judgement.get("raw"),
                            "error_text": error_text,
                        }
                    )
                    print(
                        f"[TaskJudge] prompt_index={prompt_index} attempt={attempt_index} "
                        f"success={judgement['prompt_success']} confidence={judgement['confidence']:.3f} "
                        f"action={action}"
                    )
                    target_prompt = select_transition_target_prompt(prompts, prompt_index, action)
                else:
                    action = "advance"
                    target_prompt = prompts[prompt_index + 1] if prompt_index + 1 < len(prompts) else None

                if action == "fail_episode":
                    all_success = False
                    failed_prompt = prompt
                    break

                if target_prompt is not None:
                    timing_label = "current prompt retry" if action == "retry" else "next prompt"
                    if use_task_judge and action == "retry":
                        execute_retry_restore_to_initial(prompt_initial_state, timing_label)
                    else:
                        print(
                            f"[Transition] prompt_index={prompt_index} "
                            f"completed_prompt={prompt!r} target_prompt={target_prompt!r}"
                        )
                        execute_transition_to(target_prompt, timing_label, transition_index=prompt_index + 1)

                if action == "retry":
                    attempt_counts[prompt_index] += 1
                    continue

                prompt_index += 1
            
            self.task.finish()
            self.render_finish()

            combined_prompts = ",".join([p.replace(" ", "_").replace("/", "_") for p in prompts])

            self.save_video(all_success, filename_override=f"{combined_prompts[:35]}", action_count=executed_action_count)
            print_timing_summary()

            return all_success, build_timing_stats()
                    
        else:
            task_success = True
            prompt = prompts[0]
            print(f"Executing prompt: {prompt}")
            self.task_info['prefix'] = prompt
            record_prompt_start(prompt)
            healthy, local_prompt_success = run_prompt(prompt, time_limit)
            atomic_task_results.append(
                {
                    "prompt_index": 0,
                    "prompt": prompt,
                    "success": bool(local_prompt_success),
                    "attempt_index": 0,
                }
            )
            self.task.finish()
            self.render_finish()
            if not healthy:
                task_success = False
            elif local_prompt_success is not None:
                task_success = bool(local_prompt_success)
            prompt = prompt.replace(" ", "_").replace("/", "_")
            self.save_video(task_success, filename_override=f"{prompt}", action_count=executed_action_count)
            print_timing_summary()
            return task_success, build_timing_stats()

        
