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

        self.lid_lock = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_EQUALITY, '/thermal_cycler_biorad_c1000:lid-lock')
        self.lid_joint = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, '/thermal_cycler_biorad_c1000:lid')
        self.lid_qpos_min = model.jnt_range[self.lid_joint, 0].item()
        self.lid_qposadr = model.jnt_qposadr[self.lid_joint].item()
        self.lid_force_knob_joint = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, '/thermal_cycler_biorad_c1000:lid-force-knob')
        self.lid_force_knob_qposadr = model.jnt_qposadr[self.lid_force_knob_joint].item()
        self.lid_jntlimit = model.jnt_range[self.lid_joint]
        self.lever_joint = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, '/thermal_cycler_biorad_c1000:lid-lever')
        self.lever_qposadr = model.jnt_qposadr[self.lever_joint].item()
        self.lever_jntlimit = model.jnt_range[self.lever_joint]
        self.lever_site = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, '/thermal_cycler_biorad_c1000:lid-lever')
        self.knob_site = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, '/thermal_cycler_biorad_c1000:lid-force-knob')

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
        rotation_angle = angle
        rotation_axis = axis

        rotate_90 = R.from_euler(rotation_axis, rotation_angle, degrees=True)
        target_quat = (rotate_90 * R.from_quat(cur_quat)).as_quat()

        return target_quat

    def move_to_target_qpos(self, q_target, num_steps=1000):
        q_curr = self.data.qpos[self.jnt_span]
        traj = np.linspace(q_curr, q_target, num_steps)

        for step in range(1, num_steps):
            self.data.ctrl[self.act_span] = traj[step]
            self.task.step_and_log({})


    # The real action needs to be replaced
    def execute(self):
        # Initial IK, must not be removed
        self.ik.initial_qpos = self.data.qpos[self.jnt_span]

        # step 1: open gripper
        self.gripper_control(0)

        # step 2: move EE 15cm along +z
        cur_pose = self.get_site_pose(self.data)
        end_pose = Pose(pos=cur_pose.pos + (0.0, 0.0, 0.15), quat=cur_pose.quat)
        path = self.interpolate(cur_pose, end_pose, 100)
        self.path_follow(path)

        # step 3: move EE 20cm along +x
        cur_pose = self.get_site_pose(self.data)
        end_pose = Pose(pos=cur_pose.pos + (0.2, 0.0, 0.0), quat=cur_pose.quat)
        path = self.interpolate(cur_pose, end_pose, 100)
        self.path_follow(path)

        # step 4: move EE 20cm along +y
        cur_pose = self.get_site_pose(self.data)
        end_pose = Pose(pos=cur_pose.pos + (0.0, 0.2, 0.0), quat=cur_pose.quat)
        path = self.interpolate(cur_pose, end_pose, 100)
        self.path_follow(path)

        # Restore to target pose (hard-inserted from planning JSON).
        target_qpos = [-1.1447508335113525, -1.696751594543457, 1.5192636251449585, -1.6308640241622925, -1.3747559785842896, -1.718564748764038]
        target_gripper = 0.0
        self.move_to_target_qpos(target_qpos)
        self.gripper_control(target_gripper)