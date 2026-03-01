import mujoco
mujoco.mj_loadPluginLibrary('./libmjlab.so.3.3.0')
import numpy as np
from kinematics import IK, Pose, slerp
from topp import Topp
from task import Task, Expert, Manager, SCENE_ROOT

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

    def set_pose(self, data: mujoco.MjData, pose: Pose):
        data.qpos[self.pos_span] = pose.pos
        data.qpos[self.quat_span] = pose.quat

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

class UR5eArm:
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
        lows = (-0.01, 0.0, -0.1, -0.02, 0.0, -0.1)
        highs = (0.01, 0.1, 0.0, 0.02, 0.1,  0.1)
        perturbation = np.random.uniform(lows, highs)
        return perturbation
    

class CleanTable(Task):
    default_scene = SCENE_ROOT / "clean_table.xml"
    default_task = "place_centrifugeTube_into_basket"
    
    time_limit = 20.0
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
        self.rack1 = GridSlot(self.model, 'slot1/')
        self.tube = CentrifugeTube(self.model, "centrifuge/", "centrifuge/")

    def reset(self, seed: int | None = None):
        super().reset(seed=seed)
        self.manager.reset(keyframe=0)

        pcrPlate_body = self.model.body(f'96/pcr_plate_96well')
        pcrPlate_body_id = pcrPlate_body.id
        pcrPlate_jnt_adr = self.model.joint(f'pcr_plate_free').qposadr.item()
        lows = (-0.1, -0.1, 0)
        highs = (0.1, 0.1,  0)
        perturbation = np.random.uniform(lows, highs)
        self.pcrPlate_pos=np.array([0.3,0, 0.834])+perturbation
        self.data.qpos[pcrPlate_jnt_adr:pcrPlate_jnt_adr+3] = self.pcrPlate_pos

        start_row = np.random.randint(2)
        start_col = np.random.randint(5)
        self.tube_start_pos = self.rack1.get_position(self.data, start_row, start_col, '50ml')
        self.data.qpos[self.tube.pos_span] = self.tube_start_pos

        body_id = self.model.body("tipbox/tip_box").id
        tipBox_joint = self.model.joint("tipbox_joint")
        tipBox_qpos_adr = tipBox_joint.qposadr.item()
        lows = (-0.1, -0.1, 0)
        highs = (0.1, 0.1,  0)
        perturbation = np.random.uniform(lows, highs)
        self.data.qpos[tipBox_qpos_adr:tipBox_qpos_adr+3] = np.array([0, -0.3, 0.844])+perturbation

        # Randomize the arm joint position
        perturbation = self.arm.qpos_perturb()
        self.data.qpos[self.arm.jnt_span] += perturbation
        self.data.ctrl[self.arm.act_span] += perturbation
        mujoco.mj_kinematics(self.model, self.data)

        # Task-specific setup
        match self.task:
            case "place_tipBox_into_basket":
                prefix = 'pick the pipette tip box and drop into the basket'
            case "place_centrifugeTube_into_basket":
                prefix = 'pick the centrifuge tube and drop into the basket'
            case "place_pcrPlate_into_basket":
                prefix = 'pick the pcr plate and drop into the basket'
            case _:
                raise ValueError(f"Unknown task: {self.task}")
        self.time_limit = 30

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
    
    def check(self, mode: str = ''):
        return  True


class CleanTableExpert(CleanTable, Expert):
    """专家策略：从侧面抓取架子上的移液器"""
    
    def __init__(self, spec: mujoco.MjSpec, freq: int = 20):
        super().__init__(spec)
        self.freq = freq
        self.period = int(round(1.0 / self.dt / self.freq))
        
        # 注册逆运动学
        self.arm.register_ik(self.data)
        
        # 初始化轨迹规划器
        self.planner = Topp(
            dof=self.arm.dof,
            qc_vel=2.0,
            qc_acc=1.5,
            ik=self.arm.ik.solve
        )
    
    def interpolate(self, start: Pose, end: Pose, num_steps: int) -> list[Pose]:
        """插值生成路径"""
        path = []
        for i in range(num_steps + 1):
            t = i / num_steps
            pos = (1 - t) * start.pos + t * end.pos
            quat = slerp(start.quat, end.quat, t)
            path.append(Pose(pos, quat))
        return path
    
    def path_follow(self, path: list[Pose]):
        """跟随路径"""
        trajectory = self.planner.jnt_traj(path)
        run_time = trajectory.duration + 0.2
        num_steps = int(run_time / self.dt)
        
        for step in range(num_steps):
            if step % self.period == 0:
                t = step * self.dt
                ctrl = self.planner.query(trajectory, t)
                self.data.ctrl[self.arm.act_span] = ctrl
            self.step_and_log({})
    
    def move_to(self, pose: Pose, num_steps: int = 100):
        """移动到指定位姿"""
        cur_pose = self.arm.get_site_pose(self.data)
        path = self.interpolate(cur_pose, pose, num_steps)
        self.path_follow(path)
    
    def gripper_control(self, value: float):
        """控制夹爪"""
        if hasattr(self.arm, 'gripper_id'):
            self.data.ctrl[self.arm.gripper_id] = value
            for _ in range(50):  # 减少等待时间
                self.step_and_log({})
    
    def get_plate_grip_pose_by_site(self, data: mujoco.MjData, site_name: str = 'grip_right') -> Pose:
        """
        通过site获取96孔板的夹取位姿
        """
        site_id = self.model.site(f"grip_{site_name}").id
        
        # 获取site位姿
        site_pos = data.site_xpos[site_id]
        site_mat = data.site_xmat[site_id]
        
        # 转换为四元数
        site_quat = np.zeros(4)
        mujoco.mju_mat2Quat(site_quat, site_mat)
        
        # 计算夹爪相对于site的位姿
        # 夹爪应该在site位置，方向垂直于夹取面
        rel_quat = np.zeros(4)
        if 'right' in site_name:
            # 右侧夹取，夹爪朝向-X方向（夹取面法线）
            mujoco.mju_axisAngle2Quat(rel_quat, np.array([0.0, 1.0, 0.0]), np.pi)
        elif 'left' in site_name:
            # 左侧夹取，夹爪朝向+X方向
            mujoco.mju_axisAngle2Quat(rel_quat, np.array([0.0, 1.0, 0.0]), 0)
        elif 'front' in site_name:
            # 前侧夹取，夹爪朝向-Y方向
            mujoco.mju_axisAngle2Quat(rel_quat, np.array([1.0, 0.0, 0.0]), np.pi)
        elif 'back' in site_name:
            # 后侧夹取，夹爪朝向+Y方向
            mujoco.mju_axisAngle2Quat(rel_quat, np.array([1.0, 0.0, 0.0]), 0)
        
        # 应用相对变换
        grip_pos = site_pos
        grip_quat = np.zeros(4)
        mujoco.mju_mulQuat(grip_quat, site_quat, rel_quat)
        
        return Pose(grip_pos, grip_quat)

    def execute(self):
        self.arm.ik.initial_qpos = self.data.qpos[self.arm.jnt_span]
        if self.task == 'place_tipBox_into_basket':
            body_id = self.model.body("tipbox/tip_box").id
            plate_pos = self.data.xpos[body_id]
            current_pose = self.arm.get_site_pose(self.data)
            lift_pose = Pose(
                pos=plate_pos+ np.array([0.0, 0, 0.06]),  
                quat=current_pose.quat  # 保持相同的姿态
            )
            self.move_to(lift_pose)
            current_pose = self.arm.get_site_pose(self.data)
            lift_pose = Pose(
                pos=plate_pos+ np.array([0.0, 0, 0.0]),  
                quat=current_pose.quat  # 保持相同的姿态
            )
            self.move_to(lift_pose)
            self.gripper_control(250.0)

            current_pose = self.arm.get_site_pose(self.data)
            lift_pose = Pose(
                pos=current_pose.pos + np.array([0.0, 0.0, 0.1]),  
                quat=current_pose.quat  # 保持相同的姿态
            )
            self.move_to(lift_pose)

            current_pose = self.arm.get_site_pose(self.data)
            lift_pose = Pose(
                pos=current_pose.pos + np.array([0.3, 0, 0.2]),  
                quat=current_pose.quat  # 保持相同的姿态
            )
            self.move_to(lift_pose)

            body_id = self.model.body("basket").id
            plate_pos = self.data.xpos[body_id]
            
            current_pose = self.arm.get_site_pose(self.data)
            lift_pose = Pose(
                pos=plate_pos + np.array([0.0, -0.1, 0.3]),  
                quat=current_pose.quat  # 保持相同的姿态
            )
            self.move_to(lift_pose)
            current_pose = self.arm.get_site_pose(self.data)
            lift_pose = Pose(
                pos=plate_pos + np.array([0.0, -0.1, 0.1]),  
                quat=current_pose.quat  # 保持相同的姿态
            )
            self.move_to(lift_pose)
            self.gripper_control(0.0)
            current_pose = self.arm.get_site_pose(self.data)
            lift_pose = Pose(
                pos=current_pose.pos + np.array([0.0, 0.0, 0.3]),  
                quat=current_pose.quat  # 保持相同的姿态
            )
            self.move_to(lift_pose)
        elif self.task == 'place_centrifugeTube_into_basket':
            current_pose = self.arm.get_site_pose(self.data)
            lift_pose = Pose(
                pos=self.tube_start_pos+ np.array([0.0, 0.0, 0.16]),  
                quat=current_pose.quat  # 保持相同的姿态
            )
            self.move_to(lift_pose)

            current_pose = self.arm.get_site_pose(self.data)
            lift_pose = Pose(
                pos=self.tube_start_pos+ np.array([0.0, 0.0, 0.11]),  
                quat=current_pose.quat  # 保持相同的姿态
            )
            self.move_to(lift_pose)
            self.gripper_control(250.0)

            current_pose = self.arm.get_site_pose(self.data)
            lift_pose = Pose(
                pos=current_pose.pos + np.array([0.0, 0.0, 0.1]),  
                quat=current_pose.quat  # 保持相同的姿态
            )
            self.move_to(lift_pose)

            body_id = self.model.body("basket").id
            plate_pos = self.data.xpos[body_id]
            
            current_pose = self.arm.get_site_pose(self.data)
            lift_pose = Pose(
                pos=plate_pos + np.array([0.0, -0.4, 0.4]),  
                quat=current_pose.quat  # 保持相同的姿态
            )
            self.move_to(lift_pose)
            current_pose = self.arm.get_site_pose(self.data)
            lift_pose = Pose(
                pos=plate_pos + np.array([0.0, -0.1, 0.4]),  
                quat=current_pose.quat  # 保持相同的姿态
            )
            self.move_to(lift_pose)
            current_pose = self.arm.get_site_pose(self.data)
            lift_pose = Pose(
                pos=plate_pos + np.array([0.0, -0.1, 0.1]),  
                quat=current_pose.quat  # 保持相同的姿态
            )
            self.move_to(lift_pose)
            self.gripper_control(0.0)
            current_pose = self.arm.get_site_pose(self.data)
            lift_pose = Pose(
                pos=current_pose.pos + np.array([0.0, 0.0, 0.3]),  
                quat=current_pose.quat  # 保持相同的姿态
            )
            self.move_to(lift_pose)
        elif self.task == 'place_pcrPlate_into_basket':
            self.gripper_control(0)
            body_id = self.model.body("pcr_plate_96well").id
            plate_pos = self.data.xpos[body_id]
            cur_pose = self.arm.get_site_pose(self.data)
            grip_pose = self.get_plate_grip_pose_by_site(self.data, 'right')
            terminal_pose = Pose(grip_pose.pos+(0.0, -0.06, 0.1), grip_pose.quat)
            path = self.interpolate(cur_pose, terminal_pose, 6)

            self.path_follow(path)
            cur_pose = self.arm.get_site_pose(self.data)
            terminal_pose = Pose(grip_pose.pos+(0.0, -0.06, 0), grip_pose.quat)
            path = self.interpolate(cur_pose, terminal_pose, 6)
            self.path_follow(path)
            self.gripper_control(150.0)

            current_pose = self.arm.get_site_pose(self.data)
            lift_pose = Pose(
                pos=current_pose.pos + np.array([0.0, 0.0, 0.1]),  
                quat=current_pose.quat  # 保持相同的姿态
            )
            self.move_to(lift_pose)

            current_pose = self.arm.get_site_pose(self.data)
            lift_pose = Pose(
                pos=current_pose.pos + np.array([0.0, 0.0, 0.1]),  
                quat=current_pose.quat  # 保持相同的姿态
            )
            self.move_to(lift_pose)

            current_pose = self.arm.get_site_pose(self.data)
            lift_pose = Pose(
                pos=current_pose.pos + np.array([0.0, 0.0, 0.1]),  
                quat=current_pose.quat  # 保持相同的姿态
            )
            self.move_to(lift_pose)

            current_pose = self.arm.get_site_pose(self.data)
            lift_pose = Pose(
                pos=current_pose.pos + np.array([0.0, 0.0, 0.1]),  
                quat=current_pose.quat  # 保持相同的姿态
            )
            self.move_to(lift_pose)

            body_id = self.model.body("basket").id
            plate_pos = self.data.xpos[body_id]
            
            current_pose = self.arm.get_site_pose(self.data)
            lift_pose = Pose(
                pos=plate_pos + np.array([0.0, -0.1, 0.4]),  
                quat=current_pose.quat  # 保持相同的姿态
            )
            self.move_to(lift_pose)
            current_pose = self.arm.get_site_pose(self.data)
            lift_pose = Pose(
                pos=plate_pos + np.array([0.0, -0.1, 0.1]),  
                quat=current_pose.quat  # 保持相同的姿态
            )
            self.move_to(lift_pose)
            self.gripper_control(0.0)
            current_pose = self.arm.get_site_pose(self.data)
            lift_pose = Pose(
                pos=current_pose.pos + np.array([0.0, 0.0, 0.3]),  
                quat=current_pose.quat  # 保持相同的姿态
            )
            self.move_to(lift_pose)
        
        self.finish()


# 注册专家类
CleanTable.Expert = CleanTableExpert


if __name__ == "__main__":
    """测试新任务"""
    from tqdm import trange
    tasks = [
        "place_tipBox_into_basket",
        "place_centrifugeTube_into_basket",
        "place_pcrPlate_into_basket",
        ]
    try:
        spec = CleanTable.load()
        expert = CleanTable.Expert(spec)
        for task in tasks:
            expert.task = task
            print("processing task: ",task)
            for i in trange(100):
                expert.reset(i)
                expert.set_serializer()
                expert.execute()
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
