import numpy as np
import mujoco
mujoco.mj_loadPluginLibrary('./libmjlab.so.3.3.0')
from kinematics import IK, Pose, slerp, FK
from topp import Topp
from task import Task, Expert, Manager, SCENE_ROOT
from instrument import ThermalCyclerBioradC1000
import time

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
        '''
        lows = (- 1.618, - 1.618, - 1.618, - 1.618, - 1.618, - 1.618)
        highs = ( 1.618,  1.618,  1.618,  1.618,  1.618,   1.618)
        '''
        perturbation = np.random.uniform(lows, highs)
        return perturbation
    
    def set_to_knob_position(self, data: mujoco.MjData):
        """设置机械臂到旋钮上方的预设位置"""
        # 这是一个接近旋钮上方的关节位置，你可能需要根据实际场景调整
        #knob_qpos = np.array([-0.5, -1.0, 1.5, -1.8, -1.57, 0.0])
        #knob_qpos = np.array([-1.8, -1.5, 1.5, -1.8, -1.57, 0.0])
        #knob_qpos = np.array([-1.8, -1.55, 1.6, -1.5, -1.57, 1.0])
        '''
        lows = (-1.8, -1.55, 1.6, -1.5, -1.57, -1.0)
        highs = (-1.8, -1.55, 1.6, -1.5, -1.57, 1.0)
        knob_qpos = np.random.uniform(lows, highs)
        '''
        knob_qpos = np.array([-0.5, -1.0, 1.5, -1.8, -1.57, 0.0])
        data.qpos[self.jnt_span] = knob_qpos
        data.ctrl[self.act_span] = knob_qpos


class ThermalCycler(ThermalCyclerBioradC1000):
    def _reset(self, data):
        super()._reset(data)
        self.fk_lever = FK(2, self.model, data, f'{self.local_prefix}body', f'{self.local_prefix}lid-lever')

    def fk(self, qpos: np.ndarray) -> Pose:
        return self.fk_lever.forward(qpos)
    
    def qpos_interpolate(self, qpos_list: list[np.ndarray], num_steps: list[int]) -> list[np.ndarray]:
        assert len(qpos_list) == len(num_steps) + 1
        interpolated_qpos = []
        for i in range(len(num_steps)):
            start_qpos = qpos_list[i]
            end_qpos = qpos_list[i + 1]
            step_qpos = (end_qpos - start_qpos) / num_steps[i]
            for step in range(num_steps[i]):
                interpolated_qpos.append(start_qpos + step * step_qpos)
        interpolated_qpos.append(qpos_list[-1])
        return interpolated_qpos

    def lever_path(self, data: mujoco.MjData, mode: str = '1/close') -> list[Pose]:
        # site path for open/close the lid
        cur_qpos = data.qpos[[self.lid_qposadr, self.lever_qposadr]]
        match mode:
            case '1/close':
                qpos1 = np.array([self.lid_jntlimit[0], cur_qpos[1]])
                qpos2 = np.array([self.lid_jntlimit[0], self.lever_jntlimit[0] / 4 * 3])
                qpos_list = [cur_qpos, qpos1, qpos2]
                num_steps = [15, 10]
                qpos_list = self.qpos_interpolate(qpos_list, num_steps)
                path = [self.get_eefpose_lever(self.fk(qpos), '1/grip') for qpos in qpos_list]
                path.append(self.get_eefpose_lever(self.fk(qpos2), '1/detach'))
            case '2/close':
                qpos1 = np.array([cur_qpos[0], self.lever_jntlimit[0]])
                qpos_list = [cur_qpos, qpos1]
                num_steps = [10]
                qpos_list = self.qpos_interpolate(qpos_list, num_steps)
                path = [self.get_eefpose_lever(self.fk(qpos), '2/grip') for qpos in qpos_list]
            case '1/open':
                qpos1 = np.array([self.lid_jntlimit[1], cur_qpos[1]])
                qpos_list = [cur_qpos, qpos1]
                num_steps = [15]
                qpos_list = self.qpos_interpolate(qpos_list, num_steps)
                path = [self.get_eefpose_lever(self.fk(qpos), '1/grip') for qpos in qpos_list]

            case '2/open':
                qpos1 = np.array([cur_qpos[0], self.lever_jntlimit[1]])
                qpos_list = [cur_qpos, qpos1]
                num_steps = [10]
                qpos_list = self.qpos_interpolate(qpos_list, num_steps)
                path = [self.get_eefpose_lever(self.fk(qpos), '2/grip') for qpos in qpos_list]
            case _:
                raise ValueError(f"Unknown lever path mode: {mode}")
        return path
    
    def knob_path(self, data: mujoco.MjData, mode: str = 'tighten') -> list[Pose]:
        site_pos = data.site_xpos[self.knob_site]
        site_mat = data.site_xmat[self.knob_site]
        site_quat = np.zeros(4)
        mujoco.mju_mat2Quat(site_quat, site_mat)
        rel_quat = np.zeros(4)
        match mode:
            case 'tighten':
                mujoco.mju_axisAngle2Quat(rel_quat, np.array([0.0, 0.0, 1.0]), -np.pi / 2)
            case 'loosen':
                mujoco.mju_axisAngle2Quat(rel_quat, np.array([0.0, 0.0, 1.0]), np.pi / 2)
            case _:
                raise ValueError(f"Unknown knob path mode: {mode}")
        res_quat = np.zeros(4)
        mujoco.mju_mulQuat(res_quat, site_quat, rel_quat)
        siteposes = [Pose(site_pos, site_quat), Pose(site_pos, res_quat)]
        return [self.get_eefpose_knob(sitepose, mode='grip') for sitepose in siteposes]

    def get_eefpose_lever(self, sitepose: Pose, mode: str = '1/detach') -> Pose:
        rel_quat = np.zeros(4)
        rel_quat[0] = 1.0
        match mode:
            case '1/detach':
                mujoco.mju_axisAngle2Quat(rel_quat, np.array([1.0, 0.0, 0.0]), np.pi / 2)
                rel_pos = np.array([0.0, 0.04, 0.0])
            case '1/grip':
                mujoco.mju_axisAngle2Quat(rel_quat, np.array([1.0, 0.0, 0.0]), np.pi / 2)
                rel_pos = np.zeros(3)
            case '2/detach':
                rel_pos = np.array([0.0, 0.0, -0.04])
            case '2/grip':
                rel_pos = np.array([0.0, 0.0, 0.0])
            case _:
                raise ValueError(f"Unknown approach mode: {mode}")
        res_pos, res_quat = np.zeros(3), np.zeros(4)
        mujoco.mju_mulPose(
            res_pos, res_quat,
            sitepose.pos, sitepose.quat,
            rel_pos, rel_quat
        )
        return Pose(res_pos, res_quat)
    
    def get_eefpose_knob(self, sitepose: Pose, mode: str = 'detach') -> Pose:
        site_pos, site_quat = sitepose.pos, sitepose.quat
        if mode == 'detach':
            pos_bias = np.array([0.0, 0.0, 0.06])
        elif mode == 'grip':
            pos_bias = np.array([0.0, 0.0, 0.038])
        rel_quat = np.zeros(4)
        mujoco.mju_axisAngle2Quat(rel_quat, np.array([0.0, 1.0, 0.0]), np.pi)
        res_pos, res_quat = np.zeros(3), np.zeros(4)
        mujoco.mju_mulPose(
            res_pos, res_quat,
            site_pos, site_quat,
            pos_bias, rel_quat
        )
        return Pose(res_pos, res_quat)

    def get_eef_pose(self, data: mujoco.MjData, loc: str, mode: str = 'detach', random: bool = False) -> Pose:
        match loc:
            case 'knob':
                site_pos = data.site_xpos[self.knob_site]
                site_mat = data.site_xmat[self.knob_site]
                quat = np.zeros(4)
                mujoco.mju_mat2Quat(quat, site_mat)
                return self.get_eefpose_knob(Pose(site_pos, quat), mode)
            case 'lever':
                site_pos = data.site_xpos[self.lever_site]
                site_mat = data.site_xmat[self.lever_site]
                quat = np.zeros(4)
                mujoco.mju_mat2Quat(quat, site_mat)
                return self.get_eefpose_lever(Pose(site_pos, quat), mode)
            case _:
                raise ValueError(f"Unknown location: {loc}")


class ThermalCyclerManipulate(Task):
    default_scene = SCENE_ROOT / "mani_thermal_cycler.xml"
    default_task = "thermal_cycler_close_to_open_transition"
    # default_task = "thermal_cycler_close"
    # default_task = "thermal_cycler_open"

    early_stop = True

    @classmethod
    def prepare(cls, spec: mujoco.MjSpec) -> mujoco.MjSpec:
        body = spec.body('/ur:world')
        set_gravcomp(body)
        return spec

    def __init__(self, spec: mujoco.MjSpec):
        self.instrument = ThermalCycler('/thermal_cycler_biorad_c1000:')
        manager = Manager.from_spec(spec, [self.instrument])
        super().__init__(manager)
        self.arm = UR5eArm(self.model, '/ur:')
        self.time_limit = 31.5

    def reset(self, seed: int | None = None):
        super().reset(seed=seed)
        self.manager.reset(keyframe=0)

        # Randomize the arm joint position
        perturbation = self.arm.qpos_perturb()
        self.data.qpos[self.arm.jnt_span] += perturbation
        self.data.ctrl[self.arm.act_span] += perturbation
        mujoco.mj_kinematics(self.model, self.data)

        # Task-specific setup
        match self.task:
            case "thermal_cycler_close":
                prefix = 'close the lid of the thermal cycler'
                #self.time_limit = 31.5
                self.time_limit = 40
            case "thermal_cycler_open":
                prefix = 'open the lid of the thermal cycler'
                self.data.qpos[self.instrument.lid_qposadr] = self.instrument.lid_jntlimit[0]
                self.data.qpos[self.instrument.lever_qposadr] = self.instrument.lever_jntlimit[0]
                mujoco.mj_kinematics(self.model, self.data)
                self.time_limit = 22.5
            case "thermal_cycler_close_to_open_transition":
                # prefix = 'transition to lever position after tightening knob'
                prefix = 'The preliminary step before opening the thermal cycler lid after it has been closed'
                self.data.qpos[self.instrument.lid_qposadr] = self.instrument.lid_jntlimit[0]
                self.data.qpos[self.instrument.lever_qposadr] = self.instrument.lever_jntlimit[0]
                mujoco.mj_kinematics(self.model, self.data)
                self.time_limit = 15.0
                self.arm.set_to_knob_position(self.data)
                mujoco.mj_kinematics(self.model, self.data)
            case _:
                raise ValueError(f"Unknown task: {self.task}")

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

    def check(self, mode: str = ''):
        cur_qpos = self.data.qpos[[self.instrument.lid_qposadr, self.instrument.lever_qposadr]]
        #match self.task:
        match mode:
            case 'thermal_cycler_close':
                if abs(cur_qpos[0] + 1.9) < 0.01 and abs(cur_qpos[1] + 0.94) < 0.01:
                    return True
            case 'thermal_cycler_open':
                if abs(cur_qpos[0]) < 0.003 and abs(cur_qpos[1]) < 0.003:
                    return True
            case 'thermal_cycler_close_to_open_transition':
                # For transition task, we check if the arm has reached the lever position
                '''
                lever_pose = self.instrument.get_eef_pose(self.data, loc='lever', mode='1/detach')
                current_pose = self.arm.get_site_pose(self.data)
                distance = np.linalg.norm(current_pose.pos - lever_pose.pos)
                if distance < 0.04:  # 2cm threshold
                    return True
                '''
                # 检查是否成功到达杠杆位置并夹住
                # 1. 检查机械臂是否接近杠杆的夹取位置
                
                current_pose = self.arm.get_site_pose(self.data)
                target_pose = self.instrument.get_eef_pose(self.data, loc='lever', mode='2/grip')
                # 位置误差检查（单位：米）
                pos_error = np.linalg.norm(current_pose.pos - target_pose.pos)
                vertical_distance = abs(current_pose.pos[2] - target_pose.pos[2])
                # 姿态误差检查（使用四元数点积）
                quat_error = 1 - abs(np.dot(current_pose.quat, target_pose.quat))
                # 2. 检查夹爪是否闭合（夹住了杠杆）
                gripper_closed = self.data.ctrl[self.arm.gripper_id] > 200  # 夹爪控制值大于200表示夹紧
                # 3. 检查杠杆位置是否在预期范围内（杠杆可能被轻微移动）
                lever_in_position = abs(cur_qpos[1] - self.instrument.lever_jntlimit[0]) < 0.1
                # 如果所有条件都满足，则认为任务成功
                '''print(f"current_pose: {current_pose}")
                print(f"target_pose: {target_pose}")
                print(f"pos_error: {pos_error}")
                print(f"quat_error: {quat_error}")
                print(f"lever_in_position: {lever_in_position}")'''
                if (gripper_closed and pos_error < 0.01 and  # 位置误差小于2cm
                    quat_error < 0.01 and  # 姿态误差较小
                    lever_in_position and vertical_distance <0.01):   # 杠杆在预期位置
                    return True  
                '''if (pos_error < 0.02 and  # 位置误差小于2cm
                    quat_error < 0.05 and  # 姿态误差较小
                    lever_in_position ):   # 杠杆在预期位置
                    return True'''  
        return False


class ThermalCyclerManipulateExpert(ThermalCyclerManipulate, Expert):
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
        if self.task == 'thermal_cycler_close':
            # approach and grip the lever
            endpose = self.instrument.get_eef_pose(self.data, loc='lever', mode='1/detach')
            startpose = self.arm.get_site_pose(self.data)
            path = self.interpolate2(startpose, endpose, num_steps=10)
            self.path_follow(path)
            target_pose = self.instrument.get_eef_pose(self.data, loc='lever', mode='1/grip')
            self.move_to(target_pose, num_steps=2)
            self.gripper_control(240)

            # close the lid
            path = self.instrument.lever_path(self.data, mode='1/close')
            self.path_follow(path[:-1])
            self.gripper_control(0)
            self.move_to(path[-1], num_steps=5)

            # screw the knob
            self.move_to(self.instrument.get_eef_pose(self.data, loc='knob', mode='detach'), 5)
            self.move_to(self.instrument.get_eef_pose(self.data, loc='knob', mode='grip'), 3)
            self.gripper_control(240)
            path = self.instrument.knob_path(self.data, mode='tighten')
            self.move_to(path[1], 5)
            self.gripper_control(100)
            self.move_to(self.instrument.get_eef_pose(self.data, loc='knob', mode='detach'), 5)
        elif self.task == 'thermal_cycler_open':
            self.move_to(self.instrument.get_eef_pose(self.data, loc='lever', mode='2/detach'))
            self.gripper_control(170)
            self.move_to(self.instrument.get_eef_pose(self.data, loc='lever', mode='2/grip'))
            self.gripper_control(240)
            path = self.instrument.lever_path(self.data, mode='2/open')
            self.path_follow(path)
            self.gripper_control(170)
            self.move_to(self.instrument.get_eef_pose(self.data, loc='lever', mode='2/detach'))
            self.move_to(self.instrument.get_eef_pose(self.data, loc='lever', mode='1/detach'))
            self.move_to(self.instrument.get_eef_pose(self.data, loc='lever', mode='1/grip'))
            self.gripper_control(240)
            path = self.instrument.lever_path(self.data, mode='1/open')
            self.path_follow(path)
            self.gripper_control(170)
            self.move_to(self.instrument.get_eef_pose(self.data, loc='lever', mode='1/detach'))
        elif self.task == 'thermal_cycler_close_to_open_transition':
            # Transition task: move from knob to lever position
            # Start from current position (assumed to be near knob after tightening)
            '''
            lever_detach_pose = self.instrument.get_eef_pose(self.data, loc='lever', mode='1/detach')
            self.move_to(lever_detach_pose, num_steps=10)
            
            # Optionally move to grip position for readiness
            lever_grip_pose = self.instrument.get_eef_pose(self.data, loc='lever', mode='1/grip')
            self.move_to(lever_grip_pose, num_steps=3)
            '''

            '''
            self.move_to(self.instrument.get_eef_pose(self.data, loc='knob', mode='detach'), 5)
            self.move_to(self.instrument.get_eef_pose(self.data, loc='lever', mode='2/detach'))
            '''
            # 计算接近旋钮的路径
            # endpose = self.instrument.get_eef_pose(self.data, loc='knob', mode='detach')
            # startpose = self.arm.get_site_pose(self.data)
            # path = self.interpolate2(startpose, endpose, num_steps=10)
            # self.path_follow(path)  # 沿着路径移动
            # self.gripper_control(240)
            
            # self.move_to(self.instrument.get_eef_pose(self.data, loc='knob', mode='detach'), 5)
            # self.move_to(self.instrument.get_eef_pose(self.data, loc='knob', mode='grip'), 3)
            # self.gripper_control(240)
            
            #旋转
            # path = self.instrument.knob_path(self.data, mode='tighten')
            # self.move_to(path[1], 5)
            # self.gripper_control(100)
            # path = self.instrument.knob_path(self.data, mode='tighten')
            # self.move_to(path[1], 5)
            self.gripper_control(100)
            self.move_to(self.instrument.get_eef_pose(self.data, loc='knob', mode='detach'), 5)
            path = self.instrument.knob_path(self.data, mode='tighten')
            self.move_to(path[1], 5)
            # 计算接近杠杆的路径
            endpose = self.instrument.get_eef_pose(self.data, loc='lever', mode='2/detach')
            startpose = self.arm.get_site_pose(self.data)
            path = self.interpolate2(startpose, endpose, num_steps=10)
            self.path_follow(path)  # 沿着路径移动
            #夹住移动
            # self.move_to(self.instrument.get_eef_pose(self.data, loc='lever', mode='2/detach'))
            self.gripper_control(170)
            self.move_to(self.instrument.get_eef_pose(self.data, loc='lever', mode='2/grip'))
            # self.gripper_control(240)

        self.finish()

ThermalCyclerManipulate.Expert = ThermalCyclerManipulateExpert

if __name__ == "__main__":
    from tqdm import trange
    spec = ThermalCyclerManipulate.load()
    expert = ThermalCyclerManipulate.Expert(spec)
    for i in trange(100):
        expert.reset(i)
        expert.set_serializer()
        expert.execute()
        time.sleep(1)
