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
    

class PickPlacePipette(Task):
    default_scene = SCENE_ROOT / "mani_pipette_stand1.xml"
    default_task = "pick_pipette_from_stand"
    
    time_limit = 20.0
    early_stop = True
    
    @classmethod
    def prepare(cls, spec: mujoco.MjSpec) -> mujoco.MjSpec:
        """准备场景，将移液器放置在架子上"""
        # 启用重力补偿
        ur_body = spec.body("1/ur5e:")
        if ur_body is not None:
            set_gravcomp(ur_body)
        return spec
    
    def __init__(self, spec: mujoco.MjSpec):
        """初始化任务"""
        # 创建管理器
        manager = Manager.from_spec(spec, [])
        super().__init__(manager)
        self.slot_positions = [
            np.array([0, 0.05, 0.15]),    # 假设在横梁上的位置
            np.array([0, 0, 0.15]),       # 中心位置
            np.array([0, -0.05, 0.15])    # 另一侧位置
        ]
        # 初始化机械臂
        self.arm = UR5eArm(self.model, '1/ur:')
        self.stand_base_pos = np.array([0.1, -0.2, 0.854])  # 架子基座位置
        self.stand_offset = np.array([0.02, 0.0, 0.24])      # 架子上的偏移
        self.pipette_site_offset = np.array([0.0, 0.0, 0.15])  # 站点偏移
        
        # 计算理论位置
        self.expected_pipette_pos = self.stand_base_pos + self.stand_offset + self.pipette_site_offset
        # 设置初始机械臂姿态（从场景中读取）
        # 机械臂在桌子右边 (0.5, 0.0, 0.824)
        self.model.key_qpos[0, self.arm.jnt_span] = [
            -1.5, -1.0, 2.0, 2.0, 0.0, -1.5  # 默认安全姿态
        ]
        self.model.key_ctrl[0, self.arm.act_span] = [
            -1.5, -1.0, 2.0, 2.0, 0.0,-1.5
        ]
        # 设置夹爪初始为打开状态
        if hasattr(self.arm, 'gripper_id'):
            self.model.key_ctrl[0, self.arm.gripper_id] = 1.0
        
        self.slot_positions = [
            np.array([0.0, 0.05, 0.15]),    # 右侧插槽
            np.array([0.0, 0.0, 0.15]),     # 中心插槽
            np.array([0.0, -0.05, 0.15])    # 左侧插槽
        ]
        self.pipette_on_stand_pos = None
        # 定义移液器方向（四元数）
        # 水平放置，长轴平行于X轴
        self.pipette_orientation = np.array([1.0, 0.0, 0.0, 0.0])
        # 桌子中心位置（目标位置）
        self.table_center = np.array([0.0, 0.0, 0.15])
        
    def get_pipette_pose(self, data: mujoco.MjData) -> Pose:
        try:
            # 如果body id未知，尝试通过名称查找
            body_id = self.model.body("tl/pipette").id
            pos = data.xpos[body_id]
            quat = data.xquat[body_id]
            return Pose(pos, quat)
        except Exception as e:
            # 如果移液器已经附着到手上，则使用手上的站点
            try:
                return self.arm.get_site_pose(data)
            except:
                print("get position error")
                # 如果都失败，使用估计位置
                return Pose(self.pipette_on_stand_pos if self.pipette_on_stand_pos is not None else np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]))
    
    def is_pipette_grasped(self, data: mujoco.MjData) -> bool:
        """检查移液器是否被夹爪抓住"""
        if hasattr(self.arm, 'gripper_id'):
            gripper_value = data.ctrl[self.arm.gripper_id]
            # 夹爪关闭阈值
            return gripper_value > 150.0
        return False
    
    def compute_side_grasp_pose(self, pipette_pose: Pose) -> Pose:
        """计算从侧面抓取移液器的位姿"""
        # 侧面抓取：夹爪从移液器的侧面接近
        # 移液器长轴方向为X轴，从侧面抓取意味着夹爪从Y方向接近
        
        # 相对位姿：夹爪应该在移液器侧面
        # 根据你的场景，机械臂从桌子右边来，架子在左边
        # 所以应该从架子前面（Y方向）的侧面抓取
        rel_pos = np.array([0.05, 0.0, 0.08])  # 侧面5cm，高度在移液器中部
        rel_quat = np.array([0.707, 0.0, 0.0, 0.707])  # 夹爪朝向移液器
        
        res_pos, res_quat = np.zeros(3), np.zeros(4)
        mujoco.mju_mulPose(
            res_pos, res_quat,
            pipette_pose.pos, pipette_pose.quat,
            rel_pos, rel_quat
        )
        return Pose(res_pos, res_quat)
    
    def compute_front_approach_pose(self, pipette_pose: Pose, distance: float = 0.15) -> Pose:
        """计算从架子前方接近的位姿"""
        # 从架子前方（Y方向）接近
        #approach_pos = pipette_pose.pos + np.array([0.0, distance, 0.0])
        approach_pos = pipette_pose.pos + np.array([distance, 0.0, 0.0])
        # 朝向架子
        approach_quat = np.array([0.707, 0.0, 0.707, 0.0])
        return Pose(approach_pos, approach_quat)

    def compute_side_approach_pose(self, pipette_pose: Pose, distance: float = 0.15) -> Pose:
        """计算从架子前方接近的位姿"""
        # 从架子前方（Y方向）接近
        approach_pos = pipette_pose.pos + np.array([0.0, distance, 0.0])
        #approach_pos = pipette_pose.pos + np.array([distance, 0.0, 0.0])
        # 朝向架子
        approach_quat = np.array([0.707, 0.0, 0.707, 0.0])
        return Pose(approach_pos, approach_quat)
    
    def compute_table_center_pose(self, height_offset: float = 0.1) -> Pose:
        """计算桌子中心的位姿"""
        center_pos = self.table_center + np.array([0.0, 0.0, height_offset])
        center_quat = np.array([0.707, 0.0, 0.707, 0.0])  # 标准朝向
        return Pose(center_pos, center_quat)
    
    def reset(self, seed: int | None = None):
        """重置环境"""
        super().reset(seed=seed)
        # 重置管理器
        self.manager.reset(keyframe=0)
        # 重置任务状态
        self.pipette_grasped = False
        self.reached_table_center = False
        
        prefix = ''
        
        if self.task == "pick_pipette_from_stand":
            perturbation = self.arm.qpos_perturb()
            self.data.qpos[self.arm.jnt_span] += perturbation
            self.data.ctrl[self.arm.act_span] += perturbation
            prefix = 'pick pipette from stand'
            
            # Set pipette on stand
            self.pipette_on_stand_pos = self.stand_base_pos + self.stand_offset
            pipette_jnt_adr = self.model.joint("pipette_joint").qposadr.item()
            self.data.qpos[pipette_jnt_adr:pipette_jnt_adr+3] = self.pipette_on_stand_pos
            pipette_quat_adr = pipette_jnt_adr + 3
            target_quat = np.array([1.0, 0.0, 0.0, 0.0])  # 绕Z轴旋转90度
            self.data.qpos[pipette_quat_adr:pipette_quat_adr+4] = target_quat
            
            # Open gripper
            if hasattr(self.arm, 'gripper_id'):
                self.data.ctrl[self.arm.gripper_id] = 1.0
                
        elif self.task == "place_pipette_on_stand":
            perturbation = self.arm.qpos_perturb()
            self.data.qpos[self.arm.jnt_span] += perturbation
            self.data.ctrl[self.arm.act_span] += perturbation
            prefix = 'place pipette on stand'
            
            # Set gripper closed
            self.data.qpos[self.arm.gripper_jnt_adr] = 0.0
            self.data.ctrl[self.arm.gripper_id] = 300.0
            
            mujoco.mj_forward(self.model, self.data)
            for _ in range(80):
                mujoco.mj_step(self.model, self.data)
            self.data.time = 0.0
            mujoco.mj_forward(self.model, self.data)
            
            # Set pipette in hand
            pipette_jnt_adr = self.model.joint("pipette_joint").qposadr.item()
            gripper_pose = self.arm.get_site_pose(self.data)
            self.data.qpos[pipette_jnt_adr:pipette_jnt_adr+3] = gripper_pose.pos + np.array([-0.05, 0.0, 0.02])
            pipette_quat_adr = pipette_jnt_adr + 3
            target_quat = np.array([1.0, 0.0, 0.0, 0.0])
            self.data.qpos[pipette_quat_adr:pipette_quat_adr+4] = target_quat
            
            self.pipette_grasped = True

        mujoco.mj_forward(self.model, self.data)
        
        # 任务信息
        self.task_info = {
            'prefix': prefix,
            'state_indices': self.arm.state_indices,
            'action_indices': self.arm.action_indices,
            'camera_mapping': {
                'image': 'table_cam_front',
                'wrist_image': '1/ur:wrist_cam'
            },
            'seed': seed,
        }
        
        return self.task_info
    
    def check(self, mode: str = ''):
        #match self.task:
        match mode:
            case 'pick_pipette_from_stand':
                pipette_pose = self.get_pipette_pose(self.data)
                gripper_value = self.data.ctrl[self.arm.gripper_id] if hasattr(self.arm, 'gripper_id') else 0.0
                gripper_pose = self.arm.get_site_pose(self.data)
                stand_offset = np.array([0.08, 0.0, 0.1])
                stand_pos=self.stand_base_pos + stand_offset
                distance_to_stand = np.linalg.norm(gripper_pose.pos[:2] - stand_pos[:2])
                distance_to_pipette = np.linalg.norm(gripper_pose.pos - pipette_pose.pos)

                stand_target_pos = self.stand_base_pos + np.array([0.04, 0.0, 0.09])
                x_distance = abs(gripper_pose.pos[0] - stand_target_pos[0])
                y_distance = abs(gripper_pose.pos[1] - stand_target_pos[1])
                
                if gripper_value>150.0 and distance_to_stand >0.174 and distance_to_pipette<0.13 and x_distance>0.056 and y_distance >0.149:
                    return True
            case 'place_pipette_on_stand':
                pipette_pose = self.get_pipette_pose(self.data)
                stand_target_pos = self.stand_base_pos + np.array([0.04, 0.0, 0.09])
                horizontal_distance = np.linalg.norm(pipette_pose.pos[:2] - stand_target_pos[:2])
                vertical_distance = abs(pipette_pose.pos[2] - stand_target_pos[2])
                gripper_pose = self.arm.get_site_pose(self.data)
                distance_to_pipette = np.linalg.norm(gripper_pose.pos - pipette_pose.pos)
                
                if horizontal_distance < 0.02 and vertical_distance < 0.02 and distance_to_pipette > 0.2:
                    return True
        return  False


class PickPlacePipetteExpert(PickPlacePipette, Expert):
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
    
    def move_to(self, pose: Pose, num_steps: int = 5):
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
    
    def execute(self):
        self.arm.ik.initial_qpos = self.data.qpos[self.arm.jnt_span]
        if self.task == 'pick_pipette_from_stand':
            self.gripper_control(0.030)
            # 获取当前移液器位姿
            pipette_pose = self.get_pipette_pose(self.data)
            stand_front_pose = self.compute_front_approach_pose(pipette_pose, -0.1)
            #stand_front_pose = self.compute_front_approach_pose(pipette_pose, 0.0)
            stand_pose = Pose(
                pos=stand_front_pose.pos + np.array([0.0, 0.0, 0.09]),
                quat=np.array([0.707, 0.0, 0.707, 0.0])  # 保持相同的姿态
            )
            self.move_to(stand_pose, 10)
            self.gripper_control(0.0)
            current_pose = self.arm.get_site_pose(self.data)
            lift_pose = Pose(
                pos=current_pose.pos + np.array([0.11, 0.0, 0.0]),
                quat=current_pose.quat  # 保持相同的姿态
            )
            self.move_to(lift_pose, 8)
            self.gripper_control(260.0)  # 关闭夹爪
            self.pipette_grasped = True
            # 短暂等待确保抓取稳定
            for _ in range(50):
                self.step_and_log({})
            current_pose = self.arm.get_site_pose(self.data)

            lift_pose = Pose(
                pos=current_pose.pos + np.array([0.0, 0.0, 0.2]),  
                quat=current_pose.quat  # 保持相同的姿态
            )
            self.move_to(lift_pose, 8)
        elif self.task == 'place_pipette_on_stand':
            current_pose = self.arm.get_site_pose(self.data)
            lift_pose = Pose(
                pos=current_pose.pos + np.array([-0.2, 0.0, 0.0]),  
                quat=current_pose.quat  # 保持相同的姿态
            )
            self.move_to(lift_pose, 12)
            stand_target_pos = self.stand_base_pos + np.array([0.0, 0.0, 0.3])
            # 定义夹爪姿态（从架子前方接近）
            stand_quat = np.array([0.707, 0.0, 0.707, 0.0])  # 朝向架子
            
            stand_pose = Pose(stand_target_pos, stand_quat)
            self.move_to(stand_pose, 15)
            current_pose = self.arm.get_site_pose(self.data)
            lift_pose = Pose(
                pos=current_pose.pos + np.array([0.08, 0.0, 0.0]),  
                quat=current_pose.quat  # 保持相同的姿态
            )
            self.move_to(lift_pose, 12)
            current_pose = self.arm.get_site_pose(self.data)
            lift_pose = Pose(
                pos=current_pose.pos + np.array([0.0, 0.0, -0.09]),  
                quat=current_pose.quat  # 保持相同的姿态
            )
            self.move_to(lift_pose, 12)
            self.gripper_control(0.0)
        
        self.finish()


# 注册专家类
PickPlacePipette.Expert = PickPlacePipetteExpert


if __name__ == "__main__":
    """测试新任务"""
    from tqdm import trange
    
    try:
        spec = PickPlacePipette.load()
        expert = PickPlacePipette.Expert(spec)
        for i in trange(1):
            expert.reset(i)
            expert.set_serializer()
            expert.execute()
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
