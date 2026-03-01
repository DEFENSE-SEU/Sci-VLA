import numpy as np
import mujoco
mujoco.mj_loadPluginLibrary('./libmjlab.so.3.3.0')

from kinematics import IK, Pose, slerp, FK
from topp import Topp
from task import Task, Expert, Manager, SCENE_ROOT

def set_gravcomp(body: mujoco.MjsBody):
    body.gravcomp = 1
    for child in body.bodies:
        set_gravcomp(child)


class UR5eArm:
    # 6-DOF arm
    def __init__(self, model: mujoco.MjModel, prefix: str):
        self.model = model
        self.prefix = prefix
        self.jnt_name = f'{prefix}shoulder_pan'
        self.act_name = f'{prefix}shoulder_pan'
        self.site_name = f'{prefix}2f85:pinch'
        self.base_name = f'{prefix}base'
        self.jnt_adr = model.joint(self.jnt_name).qposadr.item()
        self.act_id = model.actuator(self.act_name).id
        self.site_id = model.site(self.site_name).id
        self.gripper_id = model.actuator(f'{prefix}2f85:fingers_actuator').id
        self.gripper_jnt_adr = model.joint(f'{prefix}2f85:right_driver_joint').qposadr.item()
        self.dof = 6
        self.jnt_span = range(self.jnt_adr, self.jnt_adr + self.dof)
        self.act_span = range(self.act_id, self.act_id + self.dof)
        self.state_indices = list(self.jnt_span) + [self.gripper_jnt_adr]
        self.action_indices = list(self.act_span) + [self.gripper_id]
        self.ik: IK = None

    def register_ik(self, data: mujoco.MjData):
        self.ik = IK(self.dof, self.model, data, self.base_name, self.site_name)
    
    def get_site_pose(self, data: mujoco.MjData) -> Pose:
        mat = data.site_xmat[self.site_id]
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, mat)
        return Pose(data.site_xpos[self.site_id], quat)

    def qpos_perturb(self):
        lows = (-1.2, -0.2, -0.1, -0.5, -0.2, -0.2)
        highs = (0.0, 0.0, 0.1, 0.2, 0.2,  0.2)
        perturbation = np.random.uniform(lows, highs)
        return perturbation


class Cube:
    def __init__(self, model: mujoco.MjModel, name: str):
        self.model = model
        self.name = name
        self.site_name = f'/{name}:top'
        self.body_name = name
        self.jnt_name = f'{name}_joint'
        self.site_id = model.site(self.site_name).id
        self.body_id = model.body(self.body_name).id
        self.jnt_adr = model.joint(self.jnt_name).qposadr.item()
        
    def get_pose(self, data: mujoco.MjData) -> Pose:
        pos = data.xpos[self.body_id]
        quat = data.xquat[self.body_id]
        return Pose(pos, quat)

    def get_top_site_pose(self, data: mujoco.MjData) -> Pose:
        pos = data.site_xpos[self.site_id]
        mat = data.site_xmat[self.site_id]
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, mat)
        return Pose(pos, quat)
        
    def get_grasp_pose(self, data: mujoco.MjData, height_offset=0.0) -> Pose:
        site_pose = self.get_top_site_pose(data)
        
        # Relative rotation: 180 degrees around Y axis to point gripper down
        rel_quat = np.zeros(4)
        mujoco.mju_axisAngle2Quat(rel_quat, np.array([0.0, 1.0, 0.0]), np.pi)
        
        res_quat = np.zeros(4)
        mujoco.mju_mulQuat(res_quat, site_pose.quat, rel_quat)
        
        pos = site_pose.pos.copy()
        pos[2] += height_offset
        
        return Pose(pos, res_quat)


class CubeStack(Task):
    default_scene = SCENE_ROOT / "mani_cube_stack.xml"
    default_task = "stack_cube"

    early_stop = True

    @classmethod
    def prepare(cls, spec: mujoco.MjSpec) -> mujoco.MjSpec:
        body = spec.body('/ur:world')
        set_gravcomp(body)
        return spec

    def __init__(self, spec: mujoco.MjSpec):
        manager = Manager.from_spec(spec, [])
        super().__init__(manager)
        self.arm = UR5eArm(self.model, '/ur:')
        self.red_cube = Cube(self.model, 'cube_red')
        self.blue_cube = Cube(self.model, 'cube_blue')
        self.yellow_cube = Cube(self.model, 'cube_yellow_large')
        self.plate = Cube(self.model, 'plate_white')

    def reset(self, seed: int | None = None):
        super().reset(seed=seed)
        self.manager.reset(keyframe=0)

        # Randomize the arm joint position
        perturbation = self.arm.qpos_perturb()
        self.data.qpos[self.arm.jnt_span] += perturbation
        self.data.ctrl[self.arm.act_span] += perturbation
        
        # Randomize cube positions
        # Red cube
        self.data.qpos[self.red_cube.jnt_adr] = 0.0
        self.data.qpos[self.red_cube.jnt_adr+1] = -0.2
        self.data.qpos[self.red_cube.jnt_adr+2] = 0.844
        self.data.qpos[self.red_cube.jnt_adr:self.red_cube.jnt_adr+2] += np.random.uniform(-0.05, 0.05, 2)
        
        # Blue cube
        self.data.qpos[self.blue_cube.jnt_adr] = 0.0
        self.data.qpos[self.blue_cube.jnt_adr+1] = 0.2
        self.data.qpos[self.blue_cube.jnt_adr+2] = 0.844
        self.data.qpos[self.blue_cube.jnt_adr:self.blue_cube.jnt_adr+2] += np.random.uniform(-0.05, 0.05, 2)

        # Yellow cube
        self.data.qpos[self.yellow_cube.jnt_adr] = -0.3
        self.data.qpos[self.yellow_cube.jnt_adr+1] = 0.0
        self.data.qpos[self.yellow_cube.jnt_adr+2] = 0.864
        self.data.qpos[self.yellow_cube.jnt_adr:self.yellow_cube.jnt_adr+2] += np.random.uniform(-0.05, 0.05, 2)

        # Plate
        self.data.qpos[self.plate.jnt_adr] = 0.3
        self.data.qpos[self.plate.jnt_adr+1] = 0.0
        self.data.qpos[self.plate.jnt_adr+2] = 0.834
        self.data.qpos[self.plate.jnt_adr:self.plate.jnt_adr+2] += np.random.uniform(-0.05, 0.05, 2)

        mujoco.mj_kinematics(self.model, self.data)

        self.time_limit = 20.0
        prefix = 'pick the red cube and place it on the blue cube'

        self.task_info = {
            'prefix': prefix,
            'state_indices': self.arm.state_indices,
            'action_indices': self.arm.action_indices,
            'camera_mapping': {
                'image': 'table_cam_left',
                'wrist_image': '/ur:wrist_cam'
            },
            'seed': seed,
        }

        return self.task_info

    def check(self):
        red_pos = self.red_cube.get_pose(self.data).pos
        blue_pos = self.blue_cube.get_pose(self.data).pos
        
        dist_xy = np.linalg.norm(red_pos[:2] - blue_pos[:2])
        dist_z = red_pos[2] - blue_pos[2]
        
        # Red cube (0.04 height) on blue cube (0.04 height).
        # Centers should be 0.04 apart in Z.
        if dist_xy < 0.02 and 0.035 < dist_z < 0.045:
            return True
        return False


class CubeStackExpert(CubeStack, Expert):
    def __init__(self, spec: mujoco.MjSpec, freq: int = 20):
        super().__init__(spec)
        self.freq = freq
        self.period = int(round(1.0 / self.dt / self.freq))
        self.arm.register_ik(self.data)
        self.planner = Topp(
            dof=self.arm.dof,
            qc_vel=1.5,
            qc_acc=1.0,
            ik=self.arm.ik.solve
        )

    def interpolate(self, start: Pose, end: Pose, num_steps: int) -> list[Pose]:
        path = []
        for i in range(num_steps + 1):
            t = i / num_steps
            pos = (1 - t) * start.pos + t * end.pos
            quat = slerp(start.quat, end.quat, t)
            path.append(Pose(pos, quat))
        return path

    def interpolate2(self, start: Pose, end: Pose, num_steps: int, height: float = None) -> list[Pose]:
        path = []
        p1 = start.pos
        p2 = end.pos
        horizon_vec = np.array([p2[0]-p1[0], p2[1]-p1[1], 0.0])
        horizon_dis = np.linalg.norm(horizon_vec)
        origin = p1.copy()
        origin[2] = 0.0
        basis1 = horizon_vec / horizon_dis
        basis2 = np.array([0.0, 0.0, 1.0])
        p1_ = np.array([0.0, p1[2]])
        p2_ = np.array([horizon_dis, p2[2]])
        if height is None:
            height = horizon_dis / 4.0
        p3_ = (p1_ + p2_) / 2.0
        p3_[1] += height
        x = np.array([p1_[0], p3_[0], p2_[0]])
        y = np.array([p1_[1], p3_[1], p2_[1]])
        coef = np.polyfit(x, y, 2)
        x_eval = np.linspace(p1_[0], p2_[0], num_steps + 1)
        y_eval = np.polyval(coef, x_eval)
        for i in range(num_steps + 1):
            t = i / num_steps
            quat = slerp(start.quat, end.quat, t)
            pos = x_eval[i] * basis1 + y_eval[i] * basis2 + origin
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
                self.data.ctrl[self.arm.act_span] = ctrl
            self.step_and_log({})
    
    def move_to(self, pose: Pose, num_steps: int = 2):
        cur_pos = self.arm.get_site_pose(self.data)
        path = self.interpolate(cur_pos, pose, num_steps)
        self.path_follow(path)

    def gripper_control(self, value: float, delay: int = 300):
        self.data.ctrl[self.arm.gripper_id] = value
        for _ in range(delay):
            self.step_and_log({})

    def execute(self):
        self.arm.ik.initial_qpos = self.data.qpos[self.arm.jnt_span]
        
        # 1. Move to above red cube
        red_top = self.red_cube.get_grasp_pose(self.data, height_offset=0.1)
        startpose = self.arm.get_site_pose(self.data)
        path = self.interpolate2(startpose, red_top, num_steps=10)
        self.path_follow(path)
        
        # 2. Move down to grasp
        red_grasp = self.red_cube.get_grasp_pose(self.data, height_offset=0.0)
        self.move_to(red_grasp, num_steps=5)
        
        # 3. Close gripper
        self.gripper_control(240)
        
        # 4. Move up
        self.move_to(red_top, num_steps=5)
        
        # 5. Move to above blue cube
        blue_top = self.blue_cube.get_grasp_pose(self.data, height_offset=0.15)
        path = self.interpolate2(red_top, blue_top, num_steps=10)
        self.path_follow(path)
        
        # 6. Move down to place
        # Place red cube on blue cube.
        # Red cube height 0.04, Blue cube height 0.04.
        # Target gripper Z = Blue_Top_Z + Red_Height = Blue_Top_Z + 0.04.
        # blue_cube.get_grasp_pose(0) is Blue_Top_Z.
        place_pose = self.blue_cube.get_grasp_pose(self.data, height_offset=0.04)
        self.move_to(place_pose, num_steps=5)
        
        # 7. Open gripper
        self.gripper_control(0)
        
        # 8. Move up
        self.move_to(blue_top, num_steps=5)

        self.finish()

CubeStack.Expert = CubeStackExpert

if __name__ == "__main__":
    from tqdm import trange
    spec = CubeStack.load()
    expert = CubeStack.Expert(spec)
    for i in trange(2):
        expert.reset(i)
        expert.set_serializer()
        expert.execute()
