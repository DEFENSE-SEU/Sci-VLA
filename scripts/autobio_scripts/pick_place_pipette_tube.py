import mujoco
mujoco.mj_loadPluginLibrary('./libmjlab.so.3.3.0')
import numpy as np
from kinematics import IK, Pose, slerp
from topp import Topp
from task import Task, Expert, Manager, SCENE_ROOT
import time 

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
    

class PickPlacePipetteAndTube(Task):
    default_scene = SCENE_ROOT / "mani_pipette_tube_rack.xml"
    default_task = "pick_pipette_to_stand"
    
    time_limit = 30.0
    early_stop = True
    
    @classmethod
    def prepare(cls, spec: mujoco.MjSpec) -> mujoco.MjSpec:
        """准备场景"""
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
        
        # 初始化机械臂
        self.arm = UR5eArm(self.model, '1/ur:')
        
        # 位置定义
        self.stand_base_pos = np.array([-0.2, -0.2, 0.854])  # 移液器架子基座位置
        self.tube_rack_base_pos = np.array([0.2, -0.2, 0.854]) # 离心管架子基座位置
        
        # 移液器在架子上的目标位置 (相对于架子基座)
        self.stand_offset = np.array([0.02, 0.0, 0.24])      
        self.pipette_target_on_stand = self.stand_base_pos + self.stand_offset
        
        # 离心管在架子上的目标位置 (相对于架子基座)
        # 假设放在架子中心的一个孔位
        self.tube_target_on_rack = self.tube_rack_base_pos + np.array([0.0, 0.0, 0.1]) 

        # 初始机械臂姿态
        self.model.key_qpos[0, self.arm.jnt_span] = [
            -1.5, -1.0, 2.0, 2.0, 0.0, -1.5
        ]
        self.model.key_ctrl[0, self.arm.act_span] = [
            -1.5, -1.0, 2.0, 2.0, 0.0,-1.5
        ]
        # 设置夹爪初始为打开状态
        if hasattr(self.arm, 'gripper_id'):
            self.model.key_ctrl[0, self.arm.gripper_id] = 1.0
            
    def get_object_pose(self, data: mujoco.MjData, obj_name: str) -> Pose:
        try:
            body_id = self.model.body(obj_name).id
            pos = data.xpos[body_id]
            quat = data.xquat[body_id]
            return Pose(pos, quat)
        except Exception as e:
            return Pose(np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]))
    
    def reset(self, seed: int | None = None):
        """重置环境"""
        super().reset(seed=seed)
        self.manager.reset(keyframe=0)
        
        prefix = ''
        
        # 随机化机械臂初始位置
        perturbation = self.arm.qpos_perturb()
        self.data.qpos[self.arm.jnt_span] += perturbation
        self.data.ctrl[self.arm.act_span] += perturbation
        
        # 打开夹爪
        if hasattr(self.arm, 'gripper_id'):
            self.data.ctrl[self.arm.gripper_id] = 1.0

        if self.task == "pick_pipette_to_stand":
            prefix = 'pick pipette to stand'
            # 将移液器放在桌面上
            pipette_jnt_adr = self.model.joint("pipette_joint").qposadr.item()
            pos = np.array([-0.1, 0.1, 0.9])
            self.data.qpos[pipette_jnt_adr:pipette_jnt_adr+3] = pos
            # 随机角度 (水平放置)
            pipette_quat_adr = pipette_jnt_adr + 3
            # 躺平 (绕Y轴旋转90度)
            self.data.qpos[pipette_quat_adr:pipette_quat_adr+4] = np.array([0.707, 0.0, 0.707, 0.0])
            # # 绕X轴旋转90度
            # self.data.qpos[pipette_quat_adr:pipette_quat_adr+4] = np.array([0.707, 0.707, 0.0, 0.0])
            
            # 离心管放在一边 (也要躺平)
            tube_jnt_adr = self.model.joint("tube_joint").qposadr.item()
            self.data.qpos[tube_jnt_adr:tube_jnt_adr+3] = np.array([0.3, 0.3, 0.85])
            tube_quat_adr = tube_jnt_adr + 3
            self.data.qpos[tube_quat_adr:tube_quat_adr+4] = np.array([0.707, 0.0, 0.707, 0.0])

        elif self.task == "pick_tube_to_rack":
            prefix = 'pick tube to rack'
            # 将离心管放在桌面上
            tube_jnt_adr = self.model.joint("tube_joint").qposadr.item()
            pos = np.array([0.1, 0.1, 0.85])
            self.data.qpos[tube_jnt_adr:tube_jnt_adr+3] = pos
            # 随机角度 (躺平)
            tube_quat_adr = tube_jnt_adr + 3
            # 绕Y轴旋转90度让它躺下
            self.data.qpos[tube_quat_adr:tube_quat_adr+4] = np.array([0.707, 0.0, 0.707, 0.0]) 
            
            # 移液器放在一边 (也要躺平)
            pipette_jnt_adr = self.model.joint("pipette_joint").qposadr.item()
            self.data.qpos[pipette_jnt_adr:pipette_jnt_adr+3] = np.array([-0.3, 0.3, 0.85])
            pipette_quat_adr = pipette_jnt_adr + 3
            self.data.qpos[pipette_quat_adr:pipette_quat_adr+4] = np.array([0.707, 0.0, 0.707, 0.0])

        mujoco.mj_forward(self.model, self.data)
        
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
        match mode:
            case 'pick_pipette_to_stand':
                # 检查移液器是否在架子上
                pipette_pose = self.get_object_pose(self.data, "tl/pipette")
                dist = np.linalg.norm(pipette_pose.pos - self.pipette_target_on_stand)
                # 允许一定的误差
                if dist < 0.1: 
                    return True
            case 'pick_tube_to_rack':
                # 检查离心管是否在架子上
                tube_pose = self.get_object_pose(self.data, "tube/centrifuge_15ml")
                # 检查高度和水平位置
                dist_xy = np.linalg.norm(tube_pose.pos[:2] - self.tube_target_on_rack[:2])
                dist_z = abs(tube_pose.pos[2] - self.tube_target_on_rack[2])
                
                if dist_xy < 0.1 and dist_z < 0.1:
                    return True
        return False


class PickPlacePipetteAndTubeExpert(PickPlacePipetteAndTube, Expert):
    
    def __init__(self, spec: mujoco.MjSpec, freq: int = 20):
        super().__init__(spec)
        self.freq = freq
        self.period = int(round(1.0 / self.dt / self.freq))
        self.arm.register_ik(self.data)
        self.planner = Topp(
            dof=self.arm.dof,
            qc_vel=2.0,
            qc_acc=1.5,
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
    
    def move_to(self, pose: Pose, num_steps: int = 5):
        cur_pose = self.arm.get_site_pose(self.data)
        path = self.interpolate(cur_pose, pose, num_steps)
        self.path_follow(path)
    
    def gripper_control(self, value: float):
        if hasattr(self.arm, 'gripper_id'):
            self.data.ctrl[self.arm.gripper_id] = value
            for _ in range(50):
                self.step_and_log({})
    
    def execute(self):
        self.arm.ik.initial_qpos = self.data.qpos[self.arm.jnt_span]
        
        if self.task == 'pick_pipette_to_stand':
            # 任务开始时执行一些随机小动作等待场景稳定
            for _ in range(600):
                self.step_and_log({})

            # 1. 移动到当前移液器位置上方
            pipette_pose = self.get_object_pose(self.data, "tl/pipette")
            # 抓取姿态：从上方抓取
            grasp_quat = np.array([0.0, 1.0, 0.0, 0.0]) # 垂直向下
            
            hover_pose = Pose(pipette_pose.pos + np.array([0, 0, 0.2]), grasp_quat)
            self.move_to(hover_pose, 50)
            
            # 2. 下降抓取
            # 偏移量调整，确保抓取到移液器手柄
            grasp_pose = Pose(pipette_pose.pos + np.array([0.08, 0.02, 0.0]), grasp_quat)
            self.move_to(grasp_pose, 20)
            self.gripper_control(255.0) # 闭合
            
            # 3. 提起
            lift_pose = Pose(grasp_pose.pos + np.array([0, 0, 0.25]), grasp_quat)
            self.move_to(lift_pose, 30)

            # 4. 移动到架子位置
            # 目标位置
            target_pos = self.pipette_target_on_stand
            # 放置姿态：绕Y轴旋转90度，使夹爪水平，移液器垂直
            place_quat = np.array([0.707, 0.0, 0.707, 0.0])

            # 策略：使用更多中间点，逐步接近目标
            # 中间点1：向Y方向移动，靠近架子Y坐标，保持抓取姿态和当前高度
            curr_pose = self.arm.get_site_pose(self.data)
            mid1_pos = np.array([curr_pose.pos[0], target_pos[1], curr_pose.pos[2]])
            mid1_pose = Pose(mid1_pos, grasp_quat)
            self.move_to(mid1_pose, 40)

            # 中间点2：向X方向移动，靠近架子X坐标，保持抓取姿态
            mid2_pos = np.array([target_pos[0] - 0.08, target_pos[1], curr_pose.pos[2]])
            mid2_pose = Pose(mid2_pos, grasp_quat)
            self.move_to(mid2_pose, 40)

            # 中间点3：调整姿态准备放置，稍微后退
            approach_pos = target_pos + np.array([-0.08, 0.0, 0.0])
            approach_pose = Pose(approach_pos, place_quat)
            self.move_to(approach_pose, 40)

            # 5. 放置
            place_pose = Pose(target_pos, place_quat)
            self.move_to(place_pose, 30)
            
            # 6. 释放
            self.gripper_control(0.0)
            
            # 7. 撤退
            retreat_pose = Pose(target_pos + np.array([-0.15, 0, 0]), place_quat)
            self.move_to(retreat_pose, 20)

        elif self.task == 'pick_tube_to_rack':
            # 1. 移动到离心管上方
            tube_pose = self.get_object_pose(self.data, "tube/centrifuge_15ml")
            grasp_quat = np.array([0.0, 1.0, 0.0, 0.0]) # 垂直向下
            
            hover_pose = Pose(tube_pose.pos + np.array([0, 0, 0.2]), grasp_quat)
            self.move_to(hover_pose, 50)
            
            # 2. 下降抓取
            grasp_pose = Pose(tube_pose.pos + np.array([0, 0, 0.0]), grasp_quat)
            self.move_to(grasp_pose, 20)
            self.gripper_control(255.0)
            
            # 3. 提起
            lift_pose = Pose(tube_pose.pos + np.array([0, 0, 0.3]), grasp_quat)
            self.move_to(lift_pose, 20)
            
            # 4. 移动到架子上方
            rack_hover_pose = Pose(self.tube_target_on_rack + np.array([0, 0, 0.2]), grasp_quat)
            self.move_to(rack_hover_pose, 50)
            
            # 5. 放置
            place_pose = Pose(self.tube_target_on_rack, grasp_quat)
            self.move_to(place_pose, 20)
            
            # 6. 释放
            self.gripper_control(0.0)
            
            # 7. 撤退
            retreat_pose = Pose(self.tube_target_on_rack + np.array([0, 0, 0.2]), grasp_quat)
            self.move_to(retreat_pose, 20)
        
        self.finish()

# 注册专家类
PickPlacePipetteAndTube.Expert = PickPlacePipetteAndTubeExpert

if __name__ == "__main__":
    from tqdm import trange
    try:
        spec = PickPlacePipetteAndTube.load()
        expert = PickPlacePipetteAndTube.Expert(spec)
        for i in trange(1):
            expert.reset(i)
            expert.set_serializer()
            expert.execute()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
