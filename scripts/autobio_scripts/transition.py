'''
This is a example of transition expert between two tasks.
The transition action is to move the end-effector to the initial pose/joint state of the centrifuge tube pick-and-place task
'''

import numpy as np
import mujoco
from task import Task
from kinematics import IK, Pose, slerp, FK
from topp import Topp
from scipy.spatial.transform import Rotation as R
import zstandard as zstd
import io


'''
Transition expert class in order to perform transition action between tasks.
Arm: UR5e
Instrument: centrifuge5910 (environment includes rack with balance centrifuge tube)
Task 1: (current) unknown
Task 2: pick the balance centrifuge tube from rack and place it into the centrifuge5910
Transition: move EE to a safe lifted configuration, then go to Task 2 initial joint state.
'''
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

    def gripper_control(self, value: float, delay: int = 300):
        self.data.ctrl[self.gripper_id] = value
        for _ in range(delay):
            self.task.step_and_log({})

    def rotate_gripper(self, angle, axis, cur_quat):
        rotation_angle = angle  # 度
        rotation_axis = axis  # 绕垂直轴旋转
        # 创建旋转四元数
        rotate_90 = R.from_euler(rotation_axis, rotation_angle, degrees=True)
        target_quat = (rotate_90 * R.from_quat(cur_quat)).as_quat()

        return target_quat

    def move_to_target_qpos(self, q_target, num_steps=1000):
        q_curr = self.data.qpos[self.jnt_span]
        traj = np.linspace(q_curr, q_target, num_steps)

        for step in range(1, num_steps):
            self.data.ctrl[self.act_span] = traj[step]
            self.task.step_and_log({})

    def find_inital_state(self, subtask_file_path):
        with open(subtask_file_path, "rb") as f:
            dctx = zstd.ZstdDecompressor()
            with dctx.stream_reader(f) as reader:
                buffer = io.BytesIO(reader.read())
                data = np.load(buffer)
        return data[10, 1:40][11:17]

    def _clip_to_joint_limits(self, q):
        q = np.asarray(q, dtype=float).copy()
        for i in range(self.dof):
            jnt_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f'/ur:{["shoulder_pan","shoulder_lift","elbow","wrist_1","wrist_2","wrist_3"][i]}')
            if jnt_id >= 0:
                lo, hi = self.model.jnt_range[jnt_id]
                # Some models use 0,0 for unlimited; only clip if range looks valid
                if not (np.isclose(lo, 0.0) and np.isclose(hi, 0.0)):
                    q[i] = float(np.clip(q[i], lo, hi))
        return q.tolist()

    # The real action needs to be replaced
    def execute(self, target_task_prompt):
        # Initial IK, must not be removed
        self.ik.initial_qpos = self.data.qpos[self.jnt_span]

        # Safety-first transition:
        # 1) Open gripper to avoid accidental grasp/collision with rack/centrifuge lid.
        # 2) Lift EE in Z to clear table-top objects (rack/centrifuge).
        # 3) Slight retreat in X (front/back) and Y (left/right) to reduce proximity risk.
        # 4) Move to the target task initial joint position smoothly.

        # Gripper control example: (0~250) 0:open, 250:fully close
        self.gripper_control(0)

        # Z-axis translation: lift up to clear obstacles
        cur_pose = self.get_site_pose(self.data)
        end_pose = Pose(pos=cur_pose.pos + (0.0, 0.0, 0.12), quat=cur_pose.quat)
        path = self.interpolate(cur_pose, end_pose, 120)
        self.path_follow(path)

        # X-axis translation: retreat slightly (safer before large joint motion)
        cur_pose = self.get_site_pose(self.data)
        end_pose = Pose(pos=cur_pose.pos + (0.08, 0.0, 0.0), quat=cur_pose.quat)
        path = self.interpolate(cur_pose, end_pose, 100)
        self.path_follow(path)

        # Y-axis translation: shift slightly away from nearby objects
        cur_pose = self.get_site_pose(self.data)
        end_pose = Pose(pos=cur_pose.pos + (0.0, -0.06, 0.0), quat=cur_pose.quat)
        path = self.interpolate(cur_pose, end_pose, 100)
        self.path_follow(path)

        # Second move to the target joint position. You need to find the target joint position from the target task initial state.
        # Provided target joint position vector:
        # [ 0.00488135 -1.4635216   1.48107634 -1.56855584 -1.44370356 -1.54162118]
        target_qpos = [0.00488135, -1.4635216, 1.48107634, -1.56855584, -1.44370356, -1.54162118]
        target_qpos = self._clip_to_joint_limits(target_qpos)

        # Use a slightly longer trajectory for safety around the centrifuge/rack area
        self.move_to_target_qpos(target_qpos, num_steps=1400)