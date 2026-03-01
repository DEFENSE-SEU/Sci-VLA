import numpy as np
import mujoco
mujoco.mj_loadPluginLibrary('./libmjlab.so.3.3.0')

from kinematics import IK, Pose, slerp, FK
from topp import Topp
from task import Task, Expert, Manager, SCENE_ROOT
from instrument import Centrifuge_tiangen_tgear_mini
from scipy.spatial.transform import Rotation as R

def set_gravcomp(body: mujoco.MjsBody):
    body.gravcomp = 1
    for child in body.bodies:
        set_gravcomp(child)

def compose_quaternions(*qs):
    result = np.array([1.0, 0.0, 0.0, 0.0])
    for q in reversed(qs):
        mujoco.mju_mulQuat(result, result, q)
    return result

class GridSlot:

    def __init__(self, model: mujoco.MjModel, prefix: str):
        self.model = model
        self.prefix = prefix
        self.grids = dict()
        for i in range(self.model.nsite):
            site_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_SITE, i)
            if not site_name.startswith(prefix):
                continue
            parts = site_name.split('-')
            slot_type = parts[1]
            if slot_type not in self.grids:
                info = dict()
                info['origin'] = i
                user = self.model.site_user[i]
                info['rows'] = user[1]
                info['row_gap'] = user[2]
                info['cols'] = user[3]
                info['col_gap'] = user[4]
                info['height'] = user[5]
                self.grids[slot_type] = info
            else:
                print(f"Duplicate grid slot type '{slot_type}' found in '{site_name}'")
        if len(self.grids) == 0:
            raise ValueError(f"No grid slots found with prefix '{prefix}'")

    def get_position(self, data: mujoco.MjData, row: int, col: int, slot_type: str='default', hei: int=0) -> np.ndarray:
        grid = self.grids[slot_type]
        origin = data.site_xpos[grid['origin']]
        frame = data.site_xmat[grid['origin']].reshape(3, 3)
        row_direction = frame[:, 0]
        col_direction = frame[:, 1]
        bias = row * grid['row_gap'] * row_direction + col * grid['col_gap'] * col_direction + np.array([0.0, 0.0, grid['height'] * hei])
        return origin + bias
    
    def random_row_col(self, slot_type: str = 'default') -> tuple:
        grid = self.grids[slot_type]
        row = np.random.randint(grid['rows'])
        col = np.random.randint(grid['cols'])
        return row, col

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
        lows = (-2, 0.0, -0.8, -0.025, 0.0, -0.1)
        highs = (-1.8, 0.15, -0.7, 0.025, 0.3,  0.1)
        perturbation = np.random.uniform(lows, highs)
        return perturbation

class CentrifugeTube:
    def __init__(self, model: mujoco.MjModel, cap_prefix: str, body_prefix: str, prefix: str = 'centrifuge_1-5ml_screw'):
        self.model = model
        self.prefix = prefix
        self.cap_id = model.body(f'{cap_prefix}centrifuge_1-5ml_screw_cap').id
        self.body_id = model.body(f'{body_prefix}centrifuge_1-5ml_screw_body').id
        self.joint_id = model.joint('centrifuge_1-5ml_screw_joint').qposadr.item()
        self.pos_span = range(self.joint_id, self.joint_id + 3)
        self.quat_span = range(self.joint_id + 3, self.joint_id + 7)
    
    def get_pose(self, data: mujoco.MjData) -> Pose:
        pos = data.qpos[self.pos_span]
        quat = data.qpos[self.quat_span]
        return Pose(pos, quat)

    def set_pose(self, data: mujoco.MjData, pose: Pose):
        data.qpos[self.pos_span] = pose.pos
        data.qpos[self.quat_span] = pose.quat

    def get_cap_pose(self, data: mujoco.MjData) -> Pose:
        return Pose(data.xpos[self.cap_id], data.xquat[self.cap_id])
    
    def get_body_pose(self, data: mujoco.MjData) -> Pose:
        return Pose(data.xpos[self.body_id], data.xquat[self.body_id])
    
    def get_begin_effector_pose(self, data: mujoco.MjData, random: bool=False) -> Pose:
        cap_pos = self.get_cap_pose(data).pos
        body_quat = self.get_body_pose(data).quat
        bias = np.array([0.0, 0.0, -0.005])
        pos = cap_pos + bias
        quat = np.zeros(4)
        quat_rel = np.zeros(4)
        axis = np.array([0.0, 1.0, 0.0])
        mujoco.mju_axisAngle2Quat(quat_rel, axis, np.pi / 2)
        mujoco.mju_mulQuat(quat, quat_rel, body_quat)
        trans_quat = compose_quaternions(
            np.array([0.7071, 0.0, 0.0, 0.7071]),
            np.array([0.7071, 0.0, 0.7071, 0.0]),
            np.array([0.7933533, 0.0, 0.0, -0.6087614]),
            quat
        )
        return Pose(pos, trans_quat)

    def get_end_effector_pose(self, data: mujoco.MjData, random: bool=False) -> Pose:
        cap_pos = self.get_cap_pose(data).pos
        body_quat = self.get_body_pose(data).quat
        bias = np.array([0.0, 0.0, -0.001])
        pos = cap_pos + bias
        quat = np.zeros(4)
        quat_rel = np.zeros(4)
        axis = np.array([0.0, 1.0, 0.0])
        mujoco.mju_axisAngle2Quat(quat_rel, axis, np.pi / 2)
        mujoco.mju_mulQuat(quat, quat_rel, body_quat)
        trans_quat = compose_quaternions(
            np.array([0.7071, 0.0, 0.0, 0.7071]),
            np.array([0.7071, 0.0, 0.7071, 0.0]),
            quat
        )
        return Pose(pos, trans_quat)

class Centrifuge_mini(Centrifuge_tiangen_tgear_mini):

    def _reset(self, data):
        super()._reset(data)
        self.fk_lever = FK(1, self.model, data, f'{self.local_prefix}body', f'{self.local_prefix}lid')

    def fk(self, qpos: np.ndarray) -> Pose:
        return self.fk_lever.forward(qpos)
    
    def qpos_interpolate(self, qpos_list: list[np.ndarray], num_steps: list[int]) -> list[np.ndarray]:
        interpolated_qpos = []
        for i in range(len(num_steps)):
            start_qpos = qpos_list[i]
            end_qpos = qpos_list[i + 1]
            step_qpos = (end_qpos - start_qpos) / num_steps[i]
            for step in range(num_steps[i]):
                interpolated_qpos.append(start_qpos + step * step_qpos)
        interpolated_qpos.append(qpos_list[-1])
        return interpolated_qpos

    def lever_path(self, data: mujoco.MjData, mode: str='1/close') -> list[Pose]:
        # site path for open/close the lid
        cur_qpos = data.qpos[self.lid_qposadr]
        cur_qpos_arr = np.asarray(cur_qpos).reshape(1)
        match mode:
            case '1/close':
                qpos1 = np.array([self.lid_jntlimit[1]]) 
                qpos_list = [cur_qpos_arr, qpos1]
                num_steps = [15]
                qpos_list = self.qpos_interpolate(qpos_list, num_steps)
                path = [self.get_eefpose_lever(self.fk(qpos), 'grip') for qpos in qpos_list]
                path.append(self.get_eefpose_lever(self.fk(qpos1), '2/detach'))
            case 'open':  # 新增打开模式
                # 打开的目标位置（关节角度）
                open_qpos = np.array([self.lid_jntlimit[0]])  # 最小限制，完全打开
                
                # 生成从当前位置到打开位置的插值
                qpos_list = [cur_qpos_arr, open_qpos]
                num_steps = [20]  # 可以调整步数控制速度
                
                qpos_interp = self.qpos_interpolate(qpos_list, num_steps)
                
                # 生成路径点
                path = []
                quarter_index = len(qpos_interp) // 4  
                for i, qpos in enumerate(qpos_interp[:quarter_index]):
                    pose_mode = 'grip'
                    pose = self.get_eefpose_lever(self.fk(qpos), pose_mode)
                    path.append(pose)
            case _:
                raise ValueError(f"Unknown lever path mode: {mode}")
        return path

    def get_eefpose_lever(self, sitepose: Pose, mode: str='1/detach') -> Pose:
        rel_quat = np.array([0.0 ,-0.7071 ,0.7071 ,0.0])
        match mode:
            case '1/detach':
                rel_pos = np.array([0.0, 0.008, 0.017])
            case '2/detach':
                rel_pos = np.array([0.0, 0.03, 0.03])
            case 'grip':
                rel_pos = np.array([0.0, 0.0, 0.03])
            case 'lock':
                rel_pos = np.array([0.0, 0.2, -0.017])
            case 'open':
                #rel_quat = np.array([0.7071, 0.7071, 0.0, 0.0])
                euler_angles = [90, 90, 0]  # X, Y, Z 旋转角度（度）
                rotation = R.from_euler('xyz', euler_angles, degrees=True)
                rel_quat = np.roll(rotation.as_quat(), 1)
                rel_pos = np.array([0.0, 0.045, -0.05])
            case _:
                print("failure", mode)
                raise ValueError(f"Unknown approach mode: {mode}")
        res_pos, res_quat = np.zeros(3), np.zeros(4)
        mujoco.mju_mulPose(
            res_pos, res_quat,
            sitepose.pos, sitepose.quat,
            rel_pos, rel_quat
        )
        return Pose(res_pos, res_quat)
    def get_hole_pose(self, data: mujoco.MjData, hole_id: int = 0) -> Pose:
        # 获取转子（rotor）的位姿
        rotor_body_id = data.body("/centrifuge_tiangen_tgear_mini:rotor").id
        
        # 获取转子在世界坐标系中的位置和四元数
        rotor_pos = data.xpos[rotor_body_id].copy()
        rotor_quat = data.xquat[rotor_body_id].copy()
        
        # 从XML中可以看到孔在转子局部坐标系中的位置：
        # <geom type="cylinder" name="test_hole" size="0.005 0.01" pos="0 0 0.01" ... />
        # 孔在转子局部坐标系中的位置是 (0, 0, 0.01)
        hole_local_pos = np.array([0.03, 0.01, -0.0])+np.array([-0.04, 0.01, 0.04])
        
        # 计算转子的旋转矩阵
        rotor_mat = np.zeros(9)
        mujoco.mju_quat2Mat(rotor_mat, rotor_quat)
        rotor_mat = rotor_mat.reshape(3, 3)
        
        # 将局部坐标转换到世界坐标
        hole_world_pos = rotor_pos + np.dot(rotor_mat, hole_local_pos)
        
        # 孔的朝向与转子相同
        hole_quat = rotor_quat.copy()
        
        return Pose(rotor_pos+hole_local_pos, hole_quat)
    
    def get_hole_pose1(self, data: mujoco.MjData, hole_id: int = 0) -> Pose:
        # 获取转子body
        rotor_body_id = data.body("/centrifuge_tiangen_tgear_mini:rotor").id
        rotor_pos = data.xpos[rotor_body_id].copy()
        rotor_quat = data.xquat[rotor_body_id].copy()
        
        # 从模型中获取孔的geom信息
        geom_id = data.geom("/centrifuge_tiangen_tgear_mini:test_hole").id
        
        # 获取孔在转子局部坐标系中的位置（从geom获取）
        geom_local_pos = self.model.geom_pos[geom_id].copy()
        geom_local_quat = self.model.geom_quat[geom_id].copy()
        
        # 将局部坐标转换到世界坐标
        rotor_mat = np.zeros(9)
        mujoco.mju_quat2Mat(rotor_mat, rotor_quat)
        rotor_mat = rotor_mat.reshape(3, 3)
        
        # 计算世界坐标系下的位置
        hole_world_pos = rotor_pos + np.dot(rotor_mat, geom_local_pos)
        
        # 计算世界坐标系下的旋转（组合转子的旋转和孔的局部旋转）
        hole_world_quat = np.zeros(4)
        mujoco.mju_mulQuat(hole_world_quat, rotor_quat, geom_local_quat)
        
        return Pose(hole_world_pos, hole_world_quat)

    def get_eef_pose(self, data: mujoco.MjData, loc: str, mode: str='1/detach', random: bool=False) -> Pose:
        match loc:
            case 'lid':
                site_pos = data.site_xpos[self.lid_site]
                site_mat = data.site_xmat[self.lid_site]
                quat = np.zeros(4)
                mujoco.mju_mat2Quat(quat, site_mat)
                return self.get_eefpose_lever(Pose(site_pos, quat), mode)
            case _:
                raise ValueError(f"Unknown location: {loc}")


class CentrifugeMiniManipulate(Task):
    default_scene = SCENE_ROOT / "centrifuge_mini_tasks.xml"
    default_task = "place_centrifuge_tube_into_centrifuge_mini"

    time_limit = 30.0
    early_stop = True

    @classmethod
    def prepare(cls, spec: mujoco.MjSpec) -> mujoco.MjSpec:
        body = spec.body('/ur:world')
        set_gravcomp(body)
        return spec

    def __init__(self, spec: mujoco.MjSpec):
        self.instrument = Centrifuge_mini('/centrifuge_tiangen_tgear_mini:')
        manager = Manager.from_spec(spec, [self.instrument])
        super().__init__(manager)
        self.rack = GridSlot(self.model, 'rack/')
        self.object = CentrifugeTube(self.model, 'tubecap', 'tubebody')
        self.arm = UR5eArm(self.model, '/ur:')
        self.hole_pose=np.array([0.0, 0.05, 0.824])

    def reset(self, seed: int | None = None):
        super().reset(seed=seed)
        self.manager.reset(keyframe=0)

        # Randomize the arm joint position
        perturbation = self.arm.qpos_perturb()
        self.data.qpos[self.arm.jnt_span] += perturbation
        self.data.ctrl[self.arm.act_span] += perturbation
        
        match self.task:
            case "close_centrifuge_mini_lid":
                hole_pose = self.instrument.get_hole_pose1(self.data, hole_id=0)
                tube_pos=hole_pose.pos+ np.array([0.0, -0.01, 0.01])
                #tube_pos=np.array([0.031, 0.084, 0.925])
                '''0.0 0.05 0.824
                0 0 0.02340602
                0.00000054 0.00025897 0.08725972
                0.03 0.01 0.0'''
                tube_quat=hole_pose.quat
                #hole_pose.quat=np.array([0.7933533, 0, -0.6087614, 0])

                self.object.set_pose(self.data, Pose(tube_pos, tube_quat))
                #self.object.set_pose(self.data,hole_pose)
                self.tube_pos = tube_pos 
                self.tube_quat = tube_quat
                mujoco.mj_kinematics(self.model, self.data)
                prefix="close the lid of the centrifuge mini"
            case "open_centrifuge_mini_lid":
                close_qpos = self.instrument.lid_jntlimit[1]
                self.data.qpos[self.instrument.lid_qposadr] = close_qpos

                row = np.random.randint(5)
                col = np.random.randint(12)
                tube_pos = self.rack.get_position(self.data, row, col, slot_type='0')
                self.data.qpos[self.object.pos_span] = tube_pos
                mujoco.mj_kinematics(self.model, self.data)
                prefix="open the lid of the centrifuge mini"
            case "place_centrifuge_tube_into_centrifuge_mini":
                self.hole_pose = self.instrument.get_hole_pose1(self.data, hole_id=0)
                
                row = np.random.randint(5)
                col = np.random.randint(12)
                tube_pos = self.rack.get_position(self.data, row, col, slot_type='0')
                self.data.qpos[self.object.pos_span] = tube_pos
                mujoco.mj_kinematics(self.model, self.data)

                prefix="take centrifuge tube out from the centrifuge mini"

        self.task_info = {
            'prefix': '',
            'state_indices': self.arm.state_indices,
            'action_indices': self.arm.action_indices,
            'camera_mapping': {
                'image': 'table_cam_front',
                'wrist_image': '/ur:wrist_cam'
            },
            'seed': seed,
        }

        return self.task_info

    def check(self):
        # Placeholder for task completion logic
        return True


class CentrifugeMiniManipulateExpert(CentrifugeMiniManipulate, Expert):
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

    def move_to(self, pose: Pose, num_steps: int=2):
        cur_pos = self.arm.get_site_pose(self.data)
        path = self.interpolate(cur_pos, pose, num_steps)
        self.path_follow(path)

    def gripper_control(self, value: float):
        self.data.ctrl[self.arm.gripper_id] = value
        for _ in range(50):
            self.step_and_log({})

    def execute(self):
        self.arm.ik.initial_qpos = self.data.qpos[self.arm.jnt_span]
        if self.task == 'close_centrifuge_mini_lid':
            # end_pose = self.instrument.get_eef_pose(self.data, loc='lid', mode='2/detach')
            target_pose = self.instrument.get_eef_pose(self.data, loc='lid', mode='grip')
            # self.move_to(end_pose, num_steps=100)
            self.move_to(target_pose, num_steps=100)

            path = self.instrument.lever_path(self.data, mode='1/close')
            self.path_follow(path[:-1])
            self.move_to(path[-1], num_steps=100)
        elif self.task == 'open_centrifuge_mini_lid':
            end_pose = self.instrument.get_eef_pose(self.data, loc='lid', mode='open')
            target_pose = self.instrument.get_eef_pose(self.data, loc='lid', mode='grip')
            pre_pose = Pose(
                pos=end_pose.pos + np.array([0.1, 0, 0]),
                quat=end_pose.quat  # 保持相同的姿态
            )
            self.move_to(pre_pose, num_steps=100)
            self.move_to(end_pose, num_steps=100)
            self.gripper_control(255)
            path = self.instrument.lever_path(self.data, mode='open')
            self.path_follow(path[:2])
            self.gripper_control(0)
        elif self.task == 'place_centrifuge_tube_into_centrifuge_mini':
            eef_pose = self.object.get_end_effector_pose(self.data)
            eef_pre_pose = Pose(pos=eef_pose.pos + (0.0, 0.0, 0.05), quat=eef_pose.quat)
            cur_pose = self.arm.get_site_pose(self.data)
            path = self.interpolate(cur_pose, eef_pre_pose, 100)
            end_pose=Pose(pos=eef_pose.pos + (0.0, 0.0, 0.01), quat=eef_pose.quat)
            path_ = self.interpolate(eef_pre_pose, end_pose, 100)
            path.extend(path_[1:])
            self.path_follow(path)
            self.gripper_control(240)
            
            cur_pose = self.arm.get_site_pose(self.data)
            lift_pose = Pose(pos=cur_pose.pos + np.array([0.0, 0.0, 0.1]), quat=cur_pose.quat)
            path_lift = self.interpolate(cur_pose, lift_pose, 100)
            self.path_follow(path_lift)
            target_pos = self.hole_pose
            cur_pose = self.arm.get_site_pose(self.data)
            pre_target_pose = Pose(pos=target_pos.pos + np.array([0.0, -0.01, 0.1]), quat=cur_pose.quat)
            path_to_target = self.interpolate(lift_pose, pre_target_pose, 100)
            self.path_follow(path_to_target)
            cur_pose = self.arm.get_site_pose(self.data)
            end_pose = Pose(pos=target_pos.pos + np.array([0.0, -0.01, 0.07]), quat=cur_pose.quat)
            path_to_target = self.interpolate(pre_target_pose, end_pose, 100)
            self.path_follow(path_to_target)
            self.gripper_control(0)

            cur_pose = self.arm.get_site_pose(self.data)
            lift_pose = Pose(pos=cur_pose.pos + np.array([0.0, 0.0, 0.1]), quat=cur_pose.quat)
            path_lift = self.interpolate(cur_pose, lift_pose, 100)
            self.path_follow(path_lift)

        self.finish()

CentrifugeMiniManipulate.Expert = CentrifugeMiniManipulateExpert

if __name__ == "__main__":
    from tqdm import trange
    tasks = [
        "close_centrifuge_mini_lid",
    "open_centrifuge_mini_lid",
    "place_centrifuge_tube_into_centrifuge_mini"
    ]
    spec = CentrifugeMiniManipulate.load()
    expert = CentrifugeMiniManipulate.Expert(spec)
    for task in tasks:
        expert.task = task
        print("processing task: ",task)
        for i in trange(100):
            expert.reset(i)
            expert.set_serializer()
            expert.execute()