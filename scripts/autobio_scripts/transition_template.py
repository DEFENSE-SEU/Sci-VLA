'''
This is a example of transition expert between two tasks.
The transition action is to move the end-effector to the lever position of the thermal cycler
'''

import numpy as np
import mujoco
from task import Task
from kinematics import IK, Pose, slerp, FK
from topp import Topp
from scipy.spatial.transform import Rotation as R
import zstandard as zstd
import io

class TransitionExpert:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, task):
        self.task = task
        self.model = model
        self.data = data
        self.jnt_name = f'/ur:shoulder_pan'
        self.act_name = f'/ur:shoulder_pan'
        self.site_name = f'/ur:2f85:pinch'
        self.base_name = f'/ur:base'
        self.jnt_adr = model.joint(self.jnt_name).qposadr.item()
        self.act_id = model.actuator(self.act_name).id
        self.site_id = model.site(self.site_name).id
        self.gripper_id = model.actuator(f'/ur:2f85:fingers_actuator').id
        self.gripper_jnt_adr = model.joint(f'/ur:2f85:right_driver_joint').qposadr.item()
        self.dof = 6
        self.jnt_span = range(self.jnt_adr, self.jnt_adr + self.dof)
        self.act_span = range(self.act_id, self.act_id + self.dof)
        self.state_indices = list(self.jnt_span) + [self.gripper_jnt_adr]
        self.action_indices = list(self.act_span) + [self.gripper_id]
        self.ik = IK(self.dof, self.model, data, self.base_name, self.site_name)
        self.dt = self.model.opt.timestep

        self.freq = 20
        self.period = int(round(1.0 / self.dt / self.freq))
        self.planner = Topp(
            dof=self.dof,
            qc_vel=1.5,
            qc_acc=1.0,
            ik=self.ik.solve
        )

    def get_site_pose(self, data: mujoco.MjData) -> Pose:
        mat = data.site_xmat[self.site_id]
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, mat)
        return Pose(data.site_xpos[self.site_id], quat)

    def interpolate(self, start: Pose, end: Pose, num_steps: int) -> list[Pose]:
        path = []
        for i in range(num_steps + 1):
            t = i / num_steps
            pos = (1 - t) * start.pos + t * end.pos
            quat = slerp(start.quat, end.quat, t)
            path.append(Pose(pos, quat))
        return path

    def path_follow(self, path: list[Pose]):
        trajectory = self.planner.jnt_traj(path)
        run_time = trajectory.duration + 0.2
        num_steps = int(run_time / self.dt)
        for step in range(num_steps):
            if step % self.period == 0:
                t = step * self.dt
                ctrl = self.planner.query(trajectory, t)
                self.data.ctrl[self.act_span] = ctrl
            self.task.step_and_log({})
    
    def move_to(self, pose: Pose, num_steps: int = 100):
        cur_pos = self.get_site_pose(self.data)
        path = self.interpolate(cur_pos, pose, num_steps)
        self.path_follow(path)

    def move_to_rrt(self, pose: Pose, num_steps: int = 100):
        self.ik.initial_qpos = np.asarray(self.data.qpos[self.jnt_span], dtype=np.float64).copy()
        target_qpos = self.ik.solve(pose.pos, pose.quat)
        self.move_to_target_qpos_rrt(target_qpos, num_steps=num_steps)

    def gripper_control(self, value: float, delay: int = 300):
        self.data.ctrl[self.gripper_id] = value
        for _ in range(delay):
            self.task.step_and_log({})

    def rotate_gripper(self, angle, axis, cur_quat):
        rotation_angle = angle
        rotation_axis = axis

        rotate_90 = R.from_euler(rotation_axis, rotation_angle, degrees=True)
        cur_quat = np.asarray(cur_quat, dtype=np.float64).reshape(4)
        cur_quat_xyzw = np.array([cur_quat[1], cur_quat[2], cur_quat[3], cur_quat[0]], dtype=np.float64)
        target_quat_xyzw = (rotate_90 * R.from_quat(cur_quat_xyzw)).as_quat()
        target_quat = np.array(
            [target_quat_xyzw[3], target_quat_xyzw[0], target_quat_xyzw[1], target_quat_xyzw[2]],
            dtype=np.float64,
        )

        return target_quat

    def set_gripper(self, value: float, delay: int = 300):
        self.gripper_control(float(value), delay=delay)

    def translate_ee(self, axis: str, distance_m: float, steps: int = 100):
        axis_to_delta = {
            "x": np.array([1.0, 0.0, 0.0], dtype=np.float64),
            "y": np.array([0.0, 1.0, 0.0], dtype=np.float64),
            "z": np.array([0.0, 0.0, 1.0], dtype=np.float64),
        }
        axis_key = str(axis).lower()
        if axis_key not in axis_to_delta:
            raise ValueError(f"Invalid translate axis: {axis}")
        cur_pose = self.get_site_pose(self.data)
        end_pose = Pose(
            pos=cur_pose.pos + axis_to_delta[axis_key] * float(distance_m),
            quat=cur_pose.quat,
        )
        self.move_to_rrt(end_pose, num_steps=int(steps))

    def rotate_ee(self, axis: str, angle_deg: float, steps: int = 100):
        axis_key = str(axis).lower()
        if axis_key not in {"x", "y", "z"}:
            raise ValueError(f"Invalid rotate axis: {axis}")
        cur_pose = self.get_site_pose(self.data)
        target_quat = self.rotate_gripper(float(angle_deg), axis_key, cur_pose.quat)
        end_pose = Pose(pos=cur_pose.pos, quat=target_quat)
        self.move_to_rrt(end_pose, num_steps=int(steps))

    def wait_steps(self, steps: int):
        for _ in range(int(steps)):
            self.task.step_and_log({})

    def execute_transition_commands(self, commands: list[dict]):
        for command in commands:
            op = command.get("op")
            if op == "open_gripper":
                self.set_gripper(0.0, delay=int(command.get("delay", 300)))
            elif op == "close_gripper":
                self.set_gripper(255.0, delay=int(command.get("delay", 300)))
            elif op == "set_gripper":
                self.set_gripper(command["value"], delay=int(command.get("delay", 300)))
            elif op == "translate":
                self.translate_ee(
                    command["axis"],
                    command["distance_m"],
                    steps=int(command.get("steps", 100)),
                )
            elif op == "rotate":
                self.rotate_ee(
                    command["axis"],
                    command["angle_deg"],
                    steps=int(command.get("steps", 100)),
                )
            elif op == "wait":
                self.wait_steps(int(command["steps"]))
            elif op == "restore_target_state":
                continue
            else:
                raise ValueError(f"Unsupported transition command op: {op}")

    def move_to_target_qpos(self, q_target, num_steps=1000):
        q_curr = self.data.qpos[self.jnt_span]
        traj = np.linspace(q_curr, q_target, num_steps)

        for step in range(1, num_steps):
            self.data.ctrl[self.act_span] = traj[step]
            self.task.step_and_log({})

    def move_to_target_qpos_rrt(
        self,
        q_target,
        num_steps=1000,
        validation_steps_per_segment: int = 100,
    ):
        from non_llm_transition import (
            execute_interpolated_joint_path,
            joint_ranges_from_model,
            plan_joint_path_rrt,
            validate_joint_path_in_mujoco,
        )

        q_curr = np.asarray(self.data.qpos[self.jnt_span], dtype=np.float64).copy()
        target_qpos = np.asarray(q_target, dtype=np.float64).reshape(-1)[: self.dof]
        path_plan = plan_joint_path_rrt(
            q_curr,
            target_qpos,
            path_validator=lambda path: validate_joint_path_in_mujoco(
                self.model,
                self.data,
                self.jnt_span,
                path,
                num_steps_per_segment=int(validation_steps_per_segment),
            ),
            joint_ranges=joint_ranges_from_model(self.model, self.jnt_span),
        )
        if path_plan.status == "RRT_FAILED_SKIP_ACTION" or not path_plan.waypoints:
            print("[Transition] RRT FAILED; skipping transition action.")
            return 0
        execute_interpolated_joint_path(
            task=self.task,
            data=self.data,
            act_span=self.act_span,
            waypoints=path_plan.waypoints,
            steps_per_segment=int(num_steps),
        )


    # The real action needs to be replaced
    def execute(self):
        # Initial IK, must not be removed
        self.ik.initial_qpos = self.data.qpos[self.jnt_span]

        self.execute_transition_commands([{"op": "open_gripper", "delay": 100}, {"op": "translate", "axis": "z", "distance_m": 0.15, "steps": 120}, {"op": "rotate", "axis": "z", "angle_deg": 20, "steps": 120}, {"op": "translate", "axis": "x", "distance_m": -0.25, "steps": 160}, {"op": "translate", "axis": "y", "distance_m": 0.1, "steps": 120}, {"op": "translate", "axis": "z", "distance_m": 0.25, "steps": 180}, {"op": "translate", "axis": "x", "distance_m": -0.25, "steps": 160}, {"op": "translate", "axis": "y", "distance_m": 0.05, "steps": 100}, {"op": "translate", "axis": "z", "distance_m": 0.25, "steps": 180}, {"op": "translate", "axis": "x", "distance_m": -0.25, "steps": 160}, {"op": "translate", "axis": "y", "distance_m": 0.0063, "steps": 100}, {"op": "translate", "axis": "z", "distance_m": 0.1412, "steps": 120}])

        # Restore to target pose (hard-inserted from planning JSON).
        from transition_generation import select_target_qpos_after_transition, validate_qpos_rrt_path
        target_qpos_candidates = [[-6.283143043518066, -1.3121533393859863, 0.18979614973068237, 0.1861255019903183, -0.6473639607429504, -3.868467092514038, 0.0025952330324798822]]
        target_selection = select_target_qpos_after_transition(
            target_qpos_candidates,
            self.data.qpos[self.jnt_span],
            top_k=3,
            path_validator=lambda candidate_qpos, *, selected_index: validate_qpos_rrt_path(
                self.model,
                self.data,
                self.jnt_span,
                candidate_qpos,
            ),
        )
        target_qpos_full = np.asarray(target_selection["selected_qpos"], dtype=np.float64).reshape(-1)
        target_qpos = target_qpos_full[:self.dof].tolist()
        target_gripper = float(target_qpos_full[-1]) if target_qpos_full.size > self.dof else None
        self.move_to_target_qpos_rrt(target_qpos)
        self.gripper_control(target_gripper)
