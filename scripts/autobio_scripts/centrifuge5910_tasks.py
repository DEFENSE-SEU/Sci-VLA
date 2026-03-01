import numpy as np
import mujoco
mujoco.mj_loadPluginLibrary('./libmjlab.so.3.3.0')

from kinematics import IK, Pose, slerp, mul_pose, neg_pose, FK
from topp import Topp
from task import Task, Expert, Manager, SCENE_ROOT
from instrument import Centrifuge_Eppendorf_5910
from scipy.spatial.transform import Rotation as R
import random

def set_gravcomp(body: mujoco.MjsBody):
    body.gravcomp = 1
    for child in body.bodies:
        set_gravcomp(child)

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

    def get_tube_pose(self, data: mujoco.MjData) -> Pose:
        sitepose = self.get_site_pose(data)
        rel_pos = np.array([0.0, 0.0, 0.01])
        rel_quat = np.zeros(4)
        mujoco.mju_axisAngle2Quat(rel_quat, [0.0, 1.0, 0.0], np.pi)
        res_pos, res_quat = np.zeros(3), np.zeros(4)
        mujoco.mju_mulPose(
            res_pos, res_quat,
            sitepose.pos, sitepose.quat,
            rel_pos, rel_quat
        )
        return Pose(res_pos, res_quat)

    def qpos_perturb(self):
        lows = (-0.05, 0.0, -0.15, -0.025, 0.0, -0.1)
        highs = (0.05, 0.15, -0.05, 0.025, 0.3, 0.1)
        perturbation = np.random.uniform(lows, highs)
        return perturbation


class Centrifuge_5910(Centrifuge_Eppendorf_5910):
    def __init__(self, prefix: str):
        super().__init__(prefix)
        # 不进行任何检测，假设配置固定
    def get_adapter_pose(self, data: mujoco.MjData, adapter_name: str) -> Pose:
        """获取适配器的位姿"""
        adapter_body = self.model.body(adapter_name)
        if adapter_body is None:
            return Pose(np.zeros(3), np.array([1., 0., 0., 0.]))
        
        # 获取适配器的全局位姿
        adapter_xpos = data.xpos[adapter_body.id]
        adapter_xquat = data.xquat[adapter_body.id]
        
        return Pose(adapter_xpos, adapter_xquat)

    def get_slot_pose(self, data: mujoco.MjData, slot_id: int) -> Pose:
        """
        获取槽位位姿 - 根据实际适配器布局
        """
        adapter_index = 0
        # 总共只有4个适配器，每个适配器只有1个中心孔
        if slot_id >= 1:
            adapter_index = 1
        
        # 确定使用哪个适配器
        
        # 适配器名称
        adapter_name = f'{self.local_prefix}adapter-7x50-{adapter_index}'
        # 获取适配器位姿
        adapter_pose = self.get_adapter_pose(data, adapter_name)
        
        # 适配器上孔的位置（根据XML：pos="0 0 0.01"）
        # 在适配器局部坐标系中，孔在中心位置
        hole_local_pos = np.array([0.0, 0.0, 0.01])
        
        # 计算适配器的旋转矩阵
        adapter_mat = np.zeros(9)
        mujoco.mju_quat2Mat(adapter_mat, adapter_pose.quat)
        adapter_mat = adapter_mat.reshape(3, 3)
        
        # 将局部坐标转换到世界坐标
        hole_world_pos = adapter_pose.pos + np.dot(adapter_mat, hole_local_pos)
        
        return Pose(hole_world_pos, adapter_pose.quat)
    
    def get_tube_pose(self, data: mujoco.MjData, slot_id: int, mode: str = "distal") -> Pose:
        """获取离心管位姿"""
        slot_pose = self.get_slot_pose(data, slot_id)
        
        if mode == "distal":
            rel_pos = np.array([0.0, 0.0, 0.01])
        elif mode == "proximal":
            rel_pos = np.array([0.0, 0.0, -0.02])
        else:
            rel_pos = np.array([0.0, 0.0, 0.0])
        
        return Pose(slot_pose.pos + rel_pos, slot_pose.quat)
    
    def rotor_perturb(self):
        """转子扰动"""
        return np.random.uniform(-0.1, 0.1)

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
        cur_qpos = data.qpos[self.lid_qposadr] 
        cur_qpos_arr = np.asarray(cur_qpos).reshape(1)  

        match mode: 
            case '1/close':  
                qpos1 = np.array([self.lid_jntlimit[1] - 0.2]) 
                qpos_list = [cur_qpos_arr, qpos1]  
                num_steps = [15]  
                qpos_list = self.qpos_interpolate(qpos_list, num_steps)  
                path = [self.get_eefpose_lever(self.fk(qpos), 'grip') for qpos in qpos_list]  
                path.append(self.get_eefpose_lever(self.fk(qpos1), '2/detach')) 
            case 'open/full':
                # 完整打开路径：从当前状态完全打开
                qpos_open = np.array([self.lid_jntlimit[0]])  # 几乎完全打开
                qpos_list = [cur_qpos_arr, qpos_open]  
                num_steps = [20]  # 更多步数使动作更平滑
                qpos_list = self.qpos_interpolate(qpos_list, num_steps)  
                path = [self.get_eefpose_lever(self.fk(qpos), 'grip') for qpos in qpos_list]  
                # 打开后移动到准备松开位置
                path.append(self.get_eefpose_lever(self.fk(qpos_open), '1/detach'))
            case _:  
                raise ValueError(f"Unknown lever path mode: {mode}") 
        return path  

    def get_eefpose_lever(self, sitepose: Pose, mode: str='1/detach') -> Pose:
        rel_quat = np.array([1.,0.,0.,0.])
        match mode: 
            case '1/detach': rel_pos = np.array([0., 0., -0.01])  
            case '2/detach': rel_pos = np.array([0.0, 0.0, -0.15]) 
            case 'grip': rel_pos = np.array([0., 0.0, 0.03])  
            case 'lock_pre': rel_pos = np.array([0.0, -0.15, 0.1])  
            case 'lock': rel_pos = np.array([0.02, 0.08, 0.1])  
            case _:
                print("failure", mode) 
                raise ValueError(f"Unknown approach mode: {mode}") 

        res_pos, res_quat = np.zeros(3), np.zeros(4) 
        mujoco.mju_mulPose(res_pos, res_quat, sitepose.pos, sitepose.quat, rel_pos, rel_quat) 
        return Pose(res_pos, res_quat)  

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


class CentrifugeTube:
    def __init__(self, model: mujoco.MjModel, cap_prefix: str, body_prefix: str):
        self.model = model
        cap = model.body(f'{cap_prefix}centrifuge_50ml_screw_cap')
        body = model.body(f'{body_prefix}centrifuge_50ml_screw_body')
        self.cap_id = cap.id
        self.body_id = body.id
        root_id = body.weldid.item()
        root = model.body(root_id)
        self.jnt_adr = model.joint(root.jntadr.item()).qposadr.item()
        self.pos_span = range(self.jnt_adr, self.jnt_adr + 3)
        self.quat_span = range(self.jnt_adr + 3, self.jnt_adr + 7)
        
    def get_pose(self, cap_prefix: str, data: mujoco.MjData) -> Pose:
        pos = data.qpos[self.pos_span]
        quat = data.qpos[self.quat_span]
        return Pose(pos, quat)

    '''def get_tub2_pose(self, cap_prefix: str, data: mujoco.MjData) -> Pose:
        pos = data.qpos[self.pos_span2]
        quat = data.qpos[self.quat_span2]
        return Pose(pos, quat)'''

    def set_pose(self, data: mujoco.MjData, pose: Pose):
        data.qpos[self.pos_span] = pose.pos
        data.qpos[self.quat_span] = pose.quat

    '''def set_tub2_pose(self, data: mujoco.MjData, pose: Pose):
        data.qpos[self.pos_span2] = pose.pos
        data.qpos[self.quat_span2] = pose.quat'''
        

    def set_cap_pose(self, data: mujoco.MjData, pose: Pose):
        """显式设置盖子位置"""
        # 需要找到盖子的独立关节地址（如果存在）
        cap_jnt_adr = self.model.joint("centrifuge_50ml_screw_cap").qposadr.item()
        data.qpos[cap_jnt_adr:cap_jnt_adr+3] = pose.pos
        data.qpos[cap_jnt_adr+3:cap_jnt_adr+7] = pose.quat

    def set_cap2_pose(self, data: mujoco.MjData, pose: Pose):
        """显式设置盖子位置"""
        # 需要找到盖子的独立关节地址（如果存在）
        cap_jnt_adr = self.model.joint("centrifuge_50ml_screw_cap3").qposadr.item()
        data.qpos[cap_jnt_adr:cap_jnt_adr+3] = pose.pos
        data.qpos[cap_jnt_adr+3:cap_jnt_adr+7] = pose.quat

    def get_cap_pose(self, data: mujoco.MjData) -> Pose:
        return Pose(data.xpos[self.cap_id], data.xquat[self.cap_id])
    
    def get_body_pose(self, data: mujoco.MjData) -> Pose:
        return Pose(data.xpos[self.body_id], data.xquat[self.body_id])
    
    def get_eef_pose(self, data: mujoco.MjData) -> Pose:
        cap_pose = self.get_cap_pose(data)
        body_pose = self.get_body_pose(data)
        rel_pos = np.array([0.01, 0.0, 0.01])
        rel_quat = np.zeros(4)
        mujoco.mju_axisAngle2Quat(rel_quat, [0.0, 1.0, 0.0], np.pi + np.pi / 6)
        res_pos, res_quat = np.zeros(3), np.zeros(4)
        mujoco.mju_mulPose(
            res_pos, res_quat,
            cap_pose.pos, body_pose.quat,
            rel_pos, rel_quat
        )
        return Pose(res_pos, res_quat)
    def get_end_effector_pose(self, data: mujoco.MjData, random: bool=False) -> Pose:
        cap_pos = self.get_cap_pose(data).pos
        body_quat = self.get_body_pose(data).quat
        bias = np.array([0.0, 0.0, -0.005])
        pos = cap_pos + bias
        rel_quat = np.zeros(4)
        rel_quat1 = np.zeros(4)
        quat1 = np.zeros(4)
        mujoco.mju_axisAngle2Quat(rel_quat, np.array([1.0, 0.0, 0.0]), np.pi)    
        mujoco.mju_axisAngle2Quat(rel_quat1, np.array([0.0, 0.0, 1.0]), np.pi)   
        mujoco.mju_mulQuat(quat1, body_quat, rel_quat) 
        mujoco.mju_mulQuat(quat1, quat1, rel_quat1)
        return Pose(pos, quat1)

class Centrifuge5910Manipulate(Task):
    default_scene = SCENE_ROOT / "centrifuge_5910_tasks.xml"
    # default_task = "open_centrifuge5910_lid"
    # default_task = "place_experimental_tube_into_centrifuge5910"
    # default_task = "place_balance_tube_into_centrifuge5910"
    # default_task = "close_centrifuge5910_lid"
    # default_task = "press_centrifuge5910_button"
    # default_task = "take_experimental_tube_from_centrifuge5910"
    default_task = "take_balance_tube_from_centrifuge5910"

    time_limit = 30.0
    early_stop = True

    @classmethod
    def prepare(cls, spec: mujoco.MjSpec) -> mujoco.MjSpec:
        body = spec.body('/ur:world')
        set_gravcomp(body)
        return spec

    def __init__(self, spec: mujoco.MjSpec):
        self.instrument = Centrifuge_5910('/centrifuge_eppendorf_5910:')
        manager = Manager.from_spec(spec, [self.instrument])
        super().__init__(manager)
        self.arm = UR5eArm(self.model, '/ur:')
        self.rack1 = GridSlot(self.model, 'rack/')
        self.tube = CentrifugeTube(self.model, "1/", "1/")
        self.tube2 = CentrifugeTube(self.model, "2/", "2/")
        self.tube_end_pos = np.zeros(3)
        self.tube2_end_pos = np.zeros(3)
        self.tube_start_pos = np.zeros(3)
        self.tube2_start_pos = np.zeros(3)

    def reset(self, seed: int | None = None):
        super().reset(seed=seed)
        self.manager.reset(keyframe=0)
        # 随机选择一个槽位（0-13）
        match self.task:
            case 'centrifuge5910_long_task_1':
                end_row=1
                end_col=4
                self.tube_start_pos = self.rack1.get_position(self.data, end_row, end_col, '50ml')
                self.data.qpos[self.tube.pos_span] = self.tube_start_pos
                end_row=0
                end_col=2
                self.tube2_start_pos = self.rack1.get_position(self.data, end_row, end_col, '50ml')
                self.data.qpos[self.tube2.pos_span] = self.tube2_start_pos

                self.data.eq_active[self.instrument.lid_lock] = 0
                lid_qpos = self.instrument.lid_jntlimit[1] - 0.03  # 接近完全关闭的位置
                self.data.qpos[self.instrument.lid_qposadr] = lid_qpos

                self.data.qpos[self.arm.jnt_span] += self.arm.qpos_perturb()
                self.data.ctrl[self.arm.act_span] += self.arm.qpos_perturb()
                # 确保物理正确
                mujoco.mj_forward(self.model, self.data)
                prefix = ''

            case 'centrifuge5910_long_task_2':
                slot_id=1
                tube_pose = self.instrument.get_tube_pose(self.data, slot_id, 'distal')
                self.tube.set_pose(self.data, tube_pose)
                # 将配平离心管放在离心机槽位中
                slot_id=0
                tube_pose2 = self.instrument.get_tube_pose(self.data, slot_id, 'distal')
                self.tube2.set_pose(self.data, tube_pose2)
                end_row=1
                end_col=4
                self.tube_end_pos = self.rack1.get_position(self.data, end_row, end_col, '50ml')

                self.data.eq_active[self.instrument.lid_lock] = 0
                lid_qpos = self.instrument.lid_jntlimit[1] - 0.03  # 接近完全关闭的位置
                self.data.qpos[self.instrument.lid_qposadr] = lid_qpos

                self.data.qpos[self.arm.jnt_span] += self.arm.qpos_perturb()
                self.data.ctrl[self.arm.act_span] += self.arm.qpos_perturb()
                # 确保物理正确
                mujoco.mj_forward(self.model, self.data)
                prefix = ''
            case 'open_centrifuge5910_lid':
                r = random.randint(1,100)
                if r%2==0:
                    end_row=1
                    end_col=4
                    self.tube_start_pos = self.rack1.get_position(self.data, end_row, end_col, '50ml')
                    self.data.qpos[self.tube.pos_span] = self.tube_start_pos
                    end_row=0
                    end_col=2
                    self.tube2_start_pos = self.rack1.get_position(self.data, end_row, end_col, '50ml')
                    self.data.qpos[self.tube2.pos_span] = self.tube2_start_pos
                else:
                    slot_id=1
                    tube_pose = self.instrument.get_tube_pose(self.data, slot_id, 'distal')
                    self.tube.set_pose(self.data, tube_pose)
                    # 将配平离心管放在离心机槽位中
                    slot_id=0
                    tube_pose2 = self.instrument.get_tube_pose(self.data, slot_id, 'distal')
                    self.tube2.set_pose(self.data, tube_pose2)
                    end_row=1
                    end_col=4
                    self.tube_end_pos = self.rack1.get_position(self.data, end_row, end_col, '50ml')

                self.data.eq_active[self.instrument.lid_lock] = 0
                lid_qpos = self.instrument.lid_jntlimit[1] - 0.03  # 接近完全关闭的位置
                self.data.qpos[self.instrument.lid_qposadr] = lid_qpos

                self.data.qpos[self.arm.jnt_span] += self.arm.qpos_perturb()
                self.data.ctrl[self.arm.act_span] += self.arm.qpos_perturb()
                # 确保物理正确
                mujoco.mj_forward(self.model, self.data)
                prefix='open the lid of the centrifuge5910'
            case 'close_centrifuge5910_lid':
                r = random.randint(1,100)
                if r%2==0:
                    slot_id=1
                    tube_pose = self.instrument.get_tube_pose(self.data, slot_id, 'distal')
                    self.tube.set_pose(self.data, tube_pose)
                    # 将配平离心管放在离心机槽位中
                    slot_id=0
                    tube_pose2 = self.instrument.get_tube_pose(self.data, slot_id, 'distal')
                    self.tube2.set_pose(self.data, tube_pose2)
                    end_row=1
                    end_col=4
                    self.tube_end_pos = self.rack1.get_position(self.data, end_row, end_col, '50ml')
                else:
                    end_row=1
                    end_col=4
                    self.tube_start_pos = self.rack1.get_position(self.data, end_row, end_col, '50ml')
                    self.data.qpos[self.tube.pos_span] = self.tube_start_pos
                    end_row=0
                    end_col=2
                    self.tube2_start_pos = self.rack1.get_position(self.data, end_row, end_col, '50ml')
                    self.data.qpos[self.tube2.pos_span] = self.tube2_start_pos

                self.data.qpos[self.arm.jnt_span] += self.arm.qpos_perturb()
                self.data.ctrl[self.arm.act_span] += self.arm.qpos_perturb()
                # 确保物理正确
                mujoco.mj_forward(self.model, self.data)
                prefix='close the lid of the centrifuge5910'
            case 'take_experimental_tube_from_centrifuge5910':
                # 将离心管放在离心机槽位中
                slot_id=1
                tube_pose = self.instrument.get_tube_pose(self.data, slot_id, 'distal')
                self.tube.set_pose(self.data, tube_pose)
                # 将配平离心管放在离心机槽位中
                slot_id=0
                tube_pose2 = self.instrument.get_tube_pose(self.data, slot_id, 'distal')
                self.tube2.set_pose(self.data, tube_pose2)
                end_row=1
                end_col=4
                self.tube_end_pos = self.rack1.get_position(self.data, end_row, end_col, '50ml')

                
                self.data.qpos[self.arm.jnt_span] += self.arm.qpos_perturb()
                self.data.ctrl[self.arm.act_span] += self.arm.qpos_perturb()
                # 确保物理正确
                mujoco.mj_forward(self.model, self.data)
                # 设置目标放置位置（在桌子上）
                self.target_place_pos = np.array([
                    -0.2 + np.random.uniform(-0.01, 0.01),  # X: 0.2-0.4
                    0.3 + np.random.uniform(-0.01, 0.01),  # Y: -0.1-0.1
                    0.854  # Z: 桌子高度
                ])
                prefix='pick the experimental centrifuge tube from the centrifuge5910 and place it on the rack'
            case 'take_balance_tube_from_centrifuge5910':
                # 将离心管放在离心机槽位中
                slot_id=1
                tube_pose = self.instrument.get_tube_pose(self.data, slot_id, 'distal')
                self.tube2.set_pose(self.data, tube_pose)
                end_row=1
                end_col=4
                self.tube_start_pos = self.rack1.get_position(self.data, end_row, end_col, '50ml')
                self.data.qpos[self.tube.pos_span] = self.tube_start_pos
                end_row=0
                end_col=2
                self.tube_end_pos = self.rack1.get_position(self.data, end_row, end_col, '50ml')

                self.data.qpos[self.arm.jnt_span] += self.arm.qpos_perturb()
                self.data.ctrl[self.arm.act_span] += self.arm.qpos_perturb()
                # 确保物理正确
                mujoco.mj_forward(self.model, self.data)
                # 设置目标放置位置（在桌子上）
                self.target_place_pos = np.array([
                    -0.2 + np.random.uniform(-0.01, 0.01),  # X: 0.2-0.4
                    0.3 + np.random.uniform(-0.01, 0.01),  # Y: -0.1-0.1
                    0.854  # Z: 桌子高度
                ])
                prefix='pick the balance centrifuge tube from the centrifuge5910 and place it on the rack'
            case 'place_experimental_tube_into_centrifuge5910':
                end_row=1
                end_col=4
                self.tube_start_pos = self.rack1.get_position(self.data, end_row, end_col, '50ml')
                #self.tube.set_pose(self.data, self.tube_start_pos)
                self.data.qpos[self.tube.pos_span] = self.tube_start_pos
                end_row=0
                end_col=2
                self.tube2_start_pos = self.rack1.get_position(self.data, end_row, end_col, '50ml')
                #self.tube2.set_pose(self.data, self.tube2_start_pos)
                self.data.qpos[self.tube2.pos_span] = self.tube2_start_pos
                slot_id=0
                self.tube_end_pos  = self.instrument.get_tube_pose(self.data, slot_id, 'distal')
                
                self.data.qpos[self.arm.jnt_span] += self.arm.qpos_perturb()
                self.data.ctrl[self.arm.act_span] += self.arm.qpos_perturb()
                # 确保物理正确
                mujoco.mj_forward(self.model, self.data)
                prefix='pick the experimental centrifuge tube from rack and place it into the centrifuge5910'
            case 'place_balance_tube_into_centrifuge5910':
                slot_id=0
                tube_pose = self.instrument.get_tube_pose(self.data, slot_id, 'distal')
                self.tube.set_pose(self.data, tube_pose)
                end_row=0
                end_col=2
                self.tube2_start_pos = self.rack1.get_position(self.data, end_row, end_col, '50ml')
                self.data.qpos[self.tube2.pos_span] = self.tube2_start_pos
                slot_id=1
                self.tube2_end_pos  = self.instrument.get_tube_pose(self.data, slot_id, 'distal')

                self.data.qpos[self.arm.jnt_span] += self.arm.qpos_perturb()
                self.data.ctrl[self.arm.act_span] += self.arm.qpos_perturb()
                # 确保物理正确
                mujoco.mj_forward(self.model, self.data)
                prefix='pick the balance centrifuge tube from rack and place it into the centrifuge5910'
            case 'press_centrifuge5910_button':
                # 将离心管放在离心机槽位中
                slot_id=1
                tube_pose = self.instrument.get_tube_pose(self.data, slot_id, 'distal')
                self.tube.set_pose(self.data, tube_pose)
                # 将配平离心管放在离心机槽位中
                slot_id=0
                tube_pose2 = self.instrument.get_tube_pose(self.data, slot_id, 'distal')
                self.tube2.set_pose(self.data, tube_pose2)
                end_row=0
                end_col=2
                self.tube_end_pos = self.rack1.get_position(self.data, end_row, end_col, '50ml')
                # 设置盖子为关闭状态
                # 设置盖子关节到关闭位置
                lid_qpos = self.instrument.lid_jntlimit[1] - 0.005  # 接近完全关闭的位置
                self.data.qpos[self.instrument.lid_qposadr] = lid_qpos
                # 激活盖子锁定
                self.data.eq_active[self.instrument.lid_lock] = 1
                # 随机扰动机械臂位置
                self.data.qpos[self.arm.jnt_span] += self.arm.qpos_perturb()
                self.data.ctrl[self.arm.act_span] += self.arm.qpos_perturb()
                prefix='press the screen button of the centrifuge5910'
            case _:
                raise ValueError(f"Unknown task: {self.task}")
        self.task_info = {
            'prefix': prefix,
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
        match self.task:
        #match mode:
            case 'close_centrifuge_5910_lid':
                gripper_pose = self.arm.get_site_pose(self.data)
                lid_qpos = self.data.qpos[self.instrument.lid_qposadr]
                lid_joint_limit = self.instrument.lid_jntlimit
                final_pose = self.instrument.get_eef_pose(self.data, loc='lid', mode='lock_pre')
                distance_to_final = np.linalg.norm(gripper_pose.pos[:2] - final_pose.pos[:2])
                if lid_qpos >= lid_joint_limit[1] - 0.05 and self.data.eq_active[self.instrument.lid_lock] == 1 and distance_to_final < 0.08:
                    return False
            case 'take_experimental_tube_from_centrifuge':
                gripper_pose = self.arm.get_site_pose(self.data)
                gripper_value = self.data.ctrl[self.arm.gripper_id] if hasattr(self.arm, 'gripper_id') else 0.0
                target_place_pos = np.array([-0.2,0.3,0.824])
                body_id = self.model.body("1/centrifuge_50ml_screw_cap").id
                pos = self.data.xpos[body_id]
                quat = self.data.xquat[body_id]
                tube_pose = Pose(pos, quat)
                distance_to_target = np.linalg.norm(tube_pose.pos[:2] - target_place_pos[:2])
                distance_to_tube = np.linalg.norm(gripper_pose.pos[:2] - tube_pose.pos[:2])
                if gripper_value < 50.0 and distance_to_tube>0.08 and distance_to_target<0.15:
                    return True
        return False


class Centrifuge5910ManipulateExpert(Centrifuge5910Manipulate, Expert):
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

    def move_to(self, pose: Pose, num_steps: int=100):
        cur_pos = self.arm.get_site_pose(self.data)
        path = self.interpolate(cur_pos, pose, num_steps)
        self.path_follow(path)

    def gripper_control(self, value: float):
        self.data.ctrl[self.arm.gripper_id] = value
        for _ in range(300):
            self.step_and_log({})

    def execute(self):
        self.arm.ik.initial_qpos = self.data.qpos[self.arm.jnt_span]
        match self.task:
            case 'open_centrifuge5910_lid':
                cur_pose = self.arm.get_site_pose(self.data)
                end_pose = Pose(pos=cur_pose.pos + (0.0, 0.0, 0.1), quat=cur_pose.quat)
                path = self.interpolate(cur_pose, end_pose, 20)
                self.path_follow(path)
                cur_pose = self.arm.get_site_pose(self.data)
                rotation_angle = -30  # 度
                rotation_axis = 'y'  # 绕垂直轴旋转
                # 创建旋转四元数
                rotate_90 = R.from_euler(rotation_axis, rotation_angle, degrees=True)
                target_quat = (rotate_90 * R.from_quat(cur_pose.quat)).as_quat()
                end_pose = Pose(pos=np.array([0.2, 0.1, 1.35]), quat=cur_pose.quat)
                path = self.interpolate(cur_pose, end_pose, 20)
                self.path_follow(path)

                target_pose = self.instrument.get_eef_pose(self.data, loc='lid', mode='1/detach')
                self.move_to(target_pose, num_steps=12)
                
                target_pose = self.instrument.get_eef_pose(self.data, loc='lid', mode='grip')
                self.move_to(target_pose, num_steps=12)
                self.gripper_control(240)

                path = self.instrument.lever_path(self.data, mode='open/full')
                self.path_follow(path[:-1])
                self.gripper_control(0)
            case 'close_centrifuge5910_lid':
                target_pose = self.instrument.get_eef_pose(self.data, loc='lid', mode='1/detach')
                self.move_to(target_pose, num_steps=12)
                self.gripper_control(0)

                target_pose = self.instrument.get_eef_pose(self.data, loc='lid', mode='grip')
                self.move_to(target_pose, num_steps=2)  
                self.gripper_control(240)  

                path = self.instrument.lever_path(self.data, mode='1/close')
                self.path_follow(path[:-1])  
                self.gripper_control(0)  
                self.move_to(path[-1], num_steps=15)  

                lock_quat = np.array([0.0, 0.7071, 0.7071, 0.0])  
                lock_pose_pre = self.instrument.get_eef_pose(self.data, loc='lid', mode='lock_pre')
                lock_pose_pre.quat = lock_quat  
                self.gripper_control(255)  
                self.move_to(lock_pose_pre, num_steps=5)  

                lock_pose = self.instrument.get_eef_pose(self.data, loc='lid', mode='lock')
                lock_pose.quat = lock_quat  
                self.move_to(lock_pose, num_steps=5)  
                self.data.eq_active[self.instrument.lid_lock] = 1  
                for _ in range(100):
                    self.step_and_log({})
                final_pose = self.instrument.get_eef_pose(self.data, loc='lid', mode='lock_pre')
                final_pose.quat = lock_quat
                self.move_to(final_pose, num_steps=5)
            case 'take_experimental_tube_from_centrifuge5910':
                slot_id=0
                eef_pose = self.tube.get_end_effector_pose(self.data)
                pre_end_pos = self.tube_end_pos.copy()
                tube_pose = self.instrument.get_tube_pose(self.data, slot_id, 'distal')
                # 步骤1：打开夹爪
                current_pose = self.arm.get_site_pose(self.data)
                self.gripper_control(0)  # UR5e的夹爪控制值范围可能不同，需要调整
                lift_pose = Pose(
                    pos=tube_pose.pos + np.array([0, 0.2, 0.2]),
                    quat=eef_pose.quat
                )
                #quat=current_pose.quat  # 保持相同的姿态
                current_pose = self.arm.get_site_pose(self.data)
                self.move_to(lift_pose, 14)
                # 步骤2：移动到预抓取位置（离心管上方)
                eef_pre_pose = Pose(
                    pos=tube_pose.pos + np.array([0.0, 0.0, 0.15]),  # 上方6cm（与 Pickup 任务相同）
                    quat=eef_pose.quat
                )
                #end_quta=current_pose.quat
                cur_pose = self.arm.get_site_pose(self.data)
                # 移动到预抓取位置（模仿 Pickup 任务的两段路径）
                path_to_pre = self.interpolate(cur_pose, eef_pre_pose, 25)
                self.path_follow(path_to_pre)
                current_pose = self.arm.get_site_pose(self.data)
                target_pose = Pose(
                    pos=tube_pose.pos + np.array([0.0, 0.0, 0.13]),
                    quat=current_pose.quat
                )
                # 步骤3：移动到抓取位置
                path_to_grip = self.interpolate(eef_pre_pose, target_pose, 5)
                self.path_follow(path_to_grip)
                # 步骤4：闭合夹爪
                self.gripper_control(350)
                
                # 步骤5：垂直向上提起离心管
                cur_pose = self.arm.get_site_pose(self.data)
                lift_pose = Pose(
                    pos=tube_pose.pos + np.array([0.0, 0.0, 0.2]),  # 向上15cm
                    quat=current_pose.quat
                )
                path_lift = self.interpolate(cur_pose, lift_pose, 15)
                self.path_follow(path_lift)

                cur_pose = self.arm.get_site_pose(self.data)
                lift_pose = Pose(
                    pos=cur_pose.pos + np.array([0.0, 0.3, 0.1]),  # 向右20cm
                    quat=cur_pose.quat
                )
                path_lift = self.interpolate(cur_pose, lift_pose, 15)
                self.path_follow(path_lift)
                cur_pose = self.arm.get_site_pose(self.data)
                pre_pose = Pose(pos=pre_end_pos + (0.0, 0.0, 0.2), quat=cur_pose.quat)
                path = self.interpolate(cur_pose, pre_pose, 20)
                self.path_follow(path)
                cur_pose = self.arm.get_site_pose(self.data)
                pre_end_pos = Pose(pos=pre_end_pos + (0.0, 0.0, 0.12), quat=cur_pose.quat)
                path = self.interpolate(cur_pose, pre_end_pos, 20)
                self.path_follow(path)
                self.gripper_control(0)
                cur_pose = self.arm.get_site_pose(self.data)
                end_pose = Pose(pos=cur_pose.pos + (0.0, 0.0, 0.1), quat=cur_pose.quat)
                path = self.interpolate(cur_pose, end_pose, 20)
                self.path_follow(path)
                # 等待一会儿
                for _ in range(100):
                    self.step_and_log({})
            case 'take_balance_tube_from_centrifuge5910':
                slot_id=1
                eef_pose = self.tube.get_end_effector_pose(self.data)
                pre_end_pos = self.tube_end_pos.copy()
                tube_pose = self.instrument.get_tube_pose(self.data, slot_id, 'distal')
                # 步骤1：打开夹爪
                current_pose = self.arm.get_site_pose(self.data)
                self.gripper_control(0)  # UR5e的夹爪控制值范围可能不同，需要调整
                lift_pose = Pose(
                    pos=tube_pose.pos + np.array([0, 0.2, 0.2]),
                    quat=eef_pose.quat
                )
                #quat=current_pose.quat  # 保持相同的姿态
                current_pose = self.arm.get_site_pose(self.data)
                self.move_to(lift_pose, 14)
                # 步骤2：移动到预抓取位置（离心管上方)
                eef_pre_pose = Pose(
                    pos=tube_pose.pos + np.array([0.0, 0.0, 0.15]),  # 上方6cm（与 Pickup 任务相同）
                    quat=eef_pose.quat
                )
                #end_quta=current_pose.quat
                cur_pose = self.arm.get_site_pose(self.data)
                # 移动到预抓取位置（模仿 Pickup 任务的两段路径）
                path_to_pre = self.interpolate(cur_pose, eef_pre_pose, 25)
                self.path_follow(path_to_pre)
                current_pose = self.arm.get_site_pose(self.data)
                target_pose = Pose(
                    pos=tube_pose.pos + np.array([0.0, 0.0, 0.13]),
                    quat=current_pose.quat
                )
                # 步骤3：移动到抓取位置
                path_to_grip = self.interpolate(eef_pre_pose, target_pose, 5)
                self.path_follow(path_to_grip)
                # 步骤4：闭合夹爪
                self.gripper_control(350)
                
                # 步骤5：垂直向上提起离心管
                cur_pose = self.arm.get_site_pose(self.data)
                lift_pose = Pose(
                    pos=tube_pose.pos + np.array([0.0, 0.0, 0.17]),  # 向上15cm
                    quat=current_pose.quat
                )
                path_lift = self.interpolate(cur_pose, lift_pose, 15)
                self.path_follow(path_lift)

                cur_pose = self.arm.get_site_pose(self.data)
                lift_pose = Pose(
                    pos=cur_pose.pos + np.array([0.0, 0.2, 0.1]),  # 向右20cm
                    quat=cur_pose.quat
                )
                path_lift = self.interpolate(cur_pose, lift_pose, 15)
                self.path_follow(path_lift)

                cur_pose = self.arm.get_site_pose(self.data)
                lift_pose = Pose(
                    pos=cur_pose.pos + np.array([0.0, 0.3, 0]),  # 向右20cm
                    quat=cur_pose.quat
                )
                path_lift = self.interpolate(cur_pose, lift_pose, 15)
                self.path_follow(path_lift)
                cur_pose = self.arm.get_site_pose(self.data)
                pre_pose = Pose(pos=pre_end_pos + (0.0, 0.0, 0.2), quat=cur_pose.quat)
                path = self.interpolate(cur_pose, pre_pose, 20)
                self.path_follow(path)
                cur_pose = self.arm.get_site_pose(self.data)
                pre_end_pos = Pose(pos=pre_end_pos + (0.0, 0.0, 0.12), quat=cur_pose.quat)
                path = self.interpolate(cur_pose, pre_end_pos, 20)
                self.path_follow(path)
                self.gripper_control(0)
                cur_pose = self.arm.get_site_pose(self.data)
                end_pose = Pose(pos=cur_pose.pos + (0.0, 0.0, 0.1), quat=cur_pose.quat)
                path = self.interpolate(cur_pose, end_pose, 20)
                self.path_follow(path)
                # 等待一会儿
                for _ in range(100):
                    self.step_and_log({})
            case 'place_experimental_tube_into_centrifuge5910':
                eef_pose = self.tube.get_end_effector_pose(self.data)
                lift_pose = Pose(
                    pos=self.tube_start_pos + np.array([0, 0, 0.2]),
                    quat=eef_pose.quat
                )
                self.move_to(lift_pose, 14)
                lift_pose = Pose(
                    pos=self.tube_start_pos + np.array([0, 0, 0.12]),
                    quat=eef_pose.quat
                )
                self.move_to(lift_pose, 14)
                self.gripper_control(250)
                cur_pose = self.arm.get_site_pose(self.data)
                end_pose = Pose(pos=cur_pose.pos + (0.0, 0.0, 0.1), quat=cur_pose.quat)
                path = self.interpolate(cur_pose, end_pose, 20)
                self.path_follow(path)
                pre_end_pos = self.tube_end_pos
                cur_pose = self.arm.get_site_pose(self.data)
                end_pose = Pose(pos=pre_end_pos.pos + (0.0, 0.4, 0.3), quat=cur_pose.quat)
                path = self.interpolate(cur_pose, end_pose, 20)
                self.path_follow(path)
                cur_pose = self.arm.get_site_pose(self.data)
                end_pose = Pose(pos=pre_end_pos.pos + (0.0, 0, 0.2), quat=cur_pose.quat)
                path = self.interpolate(cur_pose, end_pose, 20)
                self.path_follow(path)
                cur_pose = self.arm.get_site_pose(self.data)
                end_pose = Pose(pos=pre_end_pos.pos + (0.0, 0, 0.13), quat=cur_pose.quat)
                path = self.interpolate(cur_pose, end_pose, 20)
                self.path_follow(path)
                self.gripper_control(0)
                cur_pose = self.arm.get_site_pose(self.data)
                end_pose = Pose(pos=cur_pose.pos + (0.0, 0.0, 0.1), quat=cur_pose.quat)
                path = self.interpolate(cur_pose, end_pose, 20)
                self.path_follow(path)
            case 'place_balance_tube_into_centrifuge5910':
                eef_pose = self.tube.get_end_effector_pose(self.data)
                lift_pose = Pose(
                    pos=self.tube2_start_pos + np.array([0, 0, 0.2]),
                    quat=eef_pose.quat
                )
                self.move_to(lift_pose, 14)
                lift_pose = Pose(
                    pos=self.tube2_start_pos + np.array([0, 0, 0.12]),
                    quat=eef_pose.quat
                )
                self.move_to(lift_pose, 14)
                self.gripper_control(250)
                cur_pose = self.arm.get_site_pose(self.data)
                end_pose = Pose(pos=cur_pose.pos + (0.0, 0.0, 0.1), quat=cur_pose.quat)
                path = self.interpolate(cur_pose, end_pose, 20)
                self.path_follow(path)
                pre_end_pos = self.tube2_end_pos
                cur_pose = self.arm.get_site_pose(self.data)
                end_pose = Pose(pos=pre_end_pos.pos + (0.0, 0.4, 0.3), quat=cur_pose.quat)
                path = self.interpolate(cur_pose, end_pose, 20)
                self.path_follow(path)
                cur_pose = self.arm.get_site_pose(self.data)
                end_pose = Pose(pos=pre_end_pos.pos + (0.0, 0, 0.17), quat=cur_pose.quat)
                path = self.interpolate(cur_pose, end_pose, 20)
                self.path_follow(path)
                cur_pose = self.arm.get_site_pose(self.data)
                end_pose = Pose(pos=pre_end_pos.pos + (0.0, 0, 0.13), quat=cur_pose.quat)
                path = self.interpolate(cur_pose, end_pose, 20)
                self.path_follow(path)
                self.gripper_control(0)
                cur_pose = self.arm.get_site_pose(self.data)
                end_pose = Pose(pos=cur_pose.pos + (0.0, 0.0, 0.06), quat=cur_pose.quat)
                path = self.interpolate(cur_pose, end_pose, 20)
                self.path_follow(path)
            case 'press_centrifuge5910_button':
                self.gripper_control(250)  
                cur_pose = self.arm.get_site_pose(self.data)
                end_pose = Pose(pos=cur_pose.pos + (0.0, 0.0, 0.1), quat=cur_pose.quat)
                path = self.interpolate(cur_pose, end_pose, 20)
                self.path_follow(path)
                cur_pose = self.arm.get_site_pose(self.data)
                rotation_angle = 30  # 度
                rotation_axis = 'y'  # 绕垂直轴旋转
                # 创建旋转四元数
                rotate_90 = R.from_euler(rotation_axis, rotation_angle, degrees=True)
                #target_quat = (rotate_90 * R.from_quat(cur_pose.quat)).as_quat()
                end_pose = Pose(pos=np.array([0.45, 0.05, 1.22]), quat=cur_pose.quat)
                path = self.interpolate(cur_pose, end_pose, 20)
                self.path_follow(path)
                cur_pose = self.arm.get_site_pose(self.data)
                end_pose = Pose(pos=np.array([0.35, -0.05, 1]), quat=cur_pose.quat)
                path = self.interpolate(cur_pose, end_pose, 20)
                self.path_follow(path)
        self.finish()

Centrifuge5910Manipulate.Expert = Centrifuge5910ManipulateExpert

if __name__ == "__main__":
    from tqdm import trange
    tasks = ["open_centrifuge5910_lid",
    "place_experimental_tube_into_centrifuge5910",
    "place_balance_tube_into_centrifuge5910",
    "close_centrifuge5910_lid",
    "press_centrifuge5910_button",
    "take_experimental_tube_from_centrifuge5910",
    "take_balance_tube_from_centrifuge5910"]

    spec = Centrifuge5910Manipulate.load()
    expert = Centrifuge5910Manipulate.Expert(spec)
    for task in tasks:
        expert.task = task
        print("processing task: ",task)
        for i in trange(100):
            expert.reset(i)
            expert.set_serializer()
            expert.execute()