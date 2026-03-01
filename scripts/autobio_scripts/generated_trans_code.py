#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
过渡任务：从pick_pipette_from_stand到place_pipette_on_stand
生成过渡数据代码脚本
"""

import mujoco
# 加载插件库
mujoco.mj_loadPluginLibrary('./libmjlab.so.3.3.0')

import numpy as np
from kinematics import IK, Pose, slerp
from topp import Topp
from task import Task, Expert, Manager, SCENE_ROOT
import random
from tqdm import trange

def set_gravcomp(body: mujoco.MjsBody):
    """设置重力补偿"""
    body.gravcomp = 1
    for child in body.bodies:
        set_gravcomp(child)

class UR5eArm:
    """UR5e机械臂控制类"""
    
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
        """注册逆运动学求解器"""
        self.ik = IK(self.dof, self.model, data, self.base_name, self.site_name)
    
    def get_site_pose(self, data: mujoco.MjData) -> Pose:
        """获取末端执行器位姿"""
        mat = data.site_xmat[self.site_id]
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, mat)
        return Pose(data.site_xpos[self.site_id], quat)

    def qpos_perturb(self):
        """生成关节位置扰动"""
        lows = (-0.01, 0.0, -0.1, -0.02, 0.0, -0.1)
        highs = (0.01, 0.1, 0.0, 0.02, 0.1,  0.1)
        perturbation = np.random.uniform(lows, highs)
        return perturbation

class TransitionPipetteTask(Task):
    """过渡任务：从pick_pipette_from_stand到place_pipette_on_stand"""
    
    default_scene = SCENE_ROOT / "mani_pipette_stand1.xml"
    default_task = "transition"
    
    time_limit = 25.0
    early_stop = True
    
    @classmethod
    def prepare(cls, spec: mujoco.MjSpec) -> mujoco.MjSpec:
        """准备场景，启用重力补偿"""
        ur_body = spec.body("1/ur5e:")
        if ur_body is not None:
            set_gravcomp(ur_body)
        return spec
    
    def __init__(self, spec: mujoco.MjSpec):
        """初始化过渡任务"""
        manager = Manager.from_spec(spec, [])
        super().__init__(manager)
        
        # 初始化机械臂
        self.arm = UR5eArm(self.model, '1/ur:')
        
        # 架子位置参数
        self.stand_base_pos = np.array([0.1, -0.2, 0.854])
        self.stand_offset = np.array([0.02, 0.0, 0.24])
        self.pipette_site_offset = np.array([0.0, 0.0, 0.15])
        
        # 桌子中心位置
        self.table_center = np.array([0.0, 0.0, 0.15])
        
        # 设置初始机械臂姿态
        self.model.key_qpos[0, self.arm.jnt_span] = [
            -1.5, -1.0, 2.0, 2.0, 0.0, -1.5
        ]
        self.model.key_ctrl[0, self.arm.act_span] = [
            -1.5, -1.0, 2.0, 2.0, 0.0, -1.5
        ]
        
        # 任务状态标志
        self.pipette_grasped = False
        self.reached_table_center = False
        
    def get_pipette_pose(self, data: mujoco.MjData) -> Pose:
        """获取移液器位姿"""
        try:
            body_id = self.model.body("tl/pipette").id
            pos = data.xpos[body_id]
            quat = data.xquat[body_id]
            return Pose(pos, quat)
        except Exception as e:
            try:
                return self.arm.get_site_pose(data)
            except:
                return Pose(np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]))
    
    def is_pipette_grasped(self, data: mujoco.MjData) -> bool:
        """检查移液器是否被抓住"""
        if hasattr(self.arm, 'gripper_id'):
            gripper_value = data.ctrl[self.arm.gripper_id]
            return gripper_value > 150.0
        return False
    
    def compute_front_approach_pose(self, pipette_pose: Pose, distance: float = 0.15) -> Pose:
        """计算从前方接近的位姿"""
        approach_pos = pipette_pose.pos + np.array([distance, 0.0, 0.0])
        approach_quat = np.array([0.707, 0.0, 0.707, 0.0])
        return Pose(approach_pos, approach_quat)
    
    def compute_table_center_pose(self, height_offset: float = 0.1) -> Pose:
        """计算桌子中心的位姿"""
        center_pos = self.table_center + np.array([0.0, 0.0, height_offset])
        center_quat = np.array([0.707, 0.0, 0.707, 0.0])
        return Pose(center_pos, center_quat)
    
    def compute_intermediate_pose(self, start_pose: Pose, end_pose: Pose, ratio: float = 0.5) -> Pose:
        """计算中间位姿"""
        pos = start_pose.pos + ratio * (end_pose.pos - start_pose.pos)
        quat = slerp(start_pose.quat, end_pose.quat, ratio)
        return Pose(pos, quat)
    
    def reset(self, seed: int | None = None):
        """重置过渡任务环境"""
        super().reset(seed=seed)
        self.manager.reset(keyframe=0)
        
        # 设置任务前缀
        prefix = 'Transition action from pick_pipette_from_stand to place_pipette_on_stand'
        
        # 添加随机扰动
        perturbation = self.arm.qpos_perturb()
        self.data.qpos[self.arm.jnt_span] += perturbation
        self.data.ctrl[self.arm.act_span] += perturbation
        
        # 设置移液器在机械臂手中（模拟前置任务完成状态）
        self.pipette_grasped = True
        
        # 设置夹爪为闭合状态（抓住移液器）
        if hasattr(self.arm, 'gripper_id'):
            self.data.ctrl[self.arm.gripper_id] = 250.0
            self.data.qpos[self.arm.gripper_jnt_adr] = 0.0
        
        # 设置移液器位置跟随机械臂末端
        mujoco.mj_forward(self.model, self.data)
        for _ in range(80):
            mujoco.mj_step(self.model, self.data)
        self.data.time = 0.0
        mujoco.mj_forward(self.model, self.data)
        
        # 将移液器放置在机械臂末端附近
        pipette_jnt_adr = self.model.joint("pipette_joint").qposadr.item()
        gripper_pose = self.arm.get_site_pose(self.data)
        self.data.qpos[pipette_jnt_adr:pipette_jnt_adr+3] = gripper_pose.pos + np.array([-0.05, 0.0, 0.02])
        pipette_quat_adr = pipette_jnt_adr + 3
        target_quat = np.array([1.0, 0.0, 0.0, 0.0])
        self.data.qpos[pipette_quat_adr:pipette_quat_adr+4] = target_quat
        
        # 设置机械臂在提起位置（模拟前置任务结束状态）
        lift_position = np.array([0.3, -0.1, 0.3])  # 提起位置
        self.data.qpos[self.arm.jnt_adr:self.arm.jnt_adr+3] = lift_position
        
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
        """检查任务完成条件"""
        match mode:
            case 'transition':
                # 检查移液器是否放置在架子上
                pipette_pose = self.get_pipette_pose(self.data)
                stand_target_pos = self.stand_base_pos + np.array([0.04, 0.0, 0.09])
                
                # 计算水平距离和垂直距离
                horizontal_distance = np.linalg.norm(pipette_pose.pos[:2] - stand_target_pos[:2])
                vertical_distance = abs(pipette_pose.pos[2] - stand_target_pos[2])
                
                # 检查夹爪是否打开
                gripper_value = self.data.ctrl[self.arm.gripper_id] if hasattr(self.arm, 'gripper_id') else 0.0
                gripper_open = gripper_value < 50.0
                
                # 检查机械臂是否离开
                gripper_pose = self.arm.get_site_pose(self.data)
                distance_to_pipette = np.linalg.norm(gripper_pose.pos - pipette_pose.pos)
                
                # 完成条件：移液器在架子上，夹爪打开，机械臂离开
                if (horizontal_distance < 0.02 and 
                    vertical_distance < 0.02 and 
                    gripper_open and 
                    distance_to_pipette > 0.15):
                    return True
        return False

class TransitionPipetteExpert(TransitionPipetteTask, Expert):
    """过渡任务专家策略"""
    
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
            for _ in range(30):
                self.step_and_log({})
    
    def execute(self):
        """执行过渡任务策略"""
        self.arm.ik.initial_qpos = self.data.qpos[self.arm.jnt_span]
        
        # 步骤1：从提起位置移动到桌子中心上方
        print("步骤1: 移动到桌子中心上方")
        current_pose = self.arm.get_site_pose(self.data)
        table_center_pose = self.compute_table_center_pose(height_offset=0.25)
        self.move_to(table_center_pose, 10)
        
        # 步骤2：下降到桌子中心
        print("步骤2: 下降到桌子中心")
        table_pose = self.compute_table_center_pose(height_offset=0.1)
        self.move_to(table_pose, 8)
        
        # 步骤3：短暂停留
        print("步骤3: 短暂停留")
        for _ in range(50):
            self.step_and_log({})
        
        # 步骤4：移动到架子前方
        print("步骤4: 移动到架子前方")
        pipette_pose = self.get_pipette_pose(self.data)
        stand_front_pose = self.compute_front_approach_pose(pipette_pose, distance=0.2)
        self.move_to(stand_front_pose, 12)
        
        # 步骤5：精确接近架子
        print("步骤5: 精确接近架子")
        stand_target_pos = self.stand_base_pos + np.array([0.0, 0.0, 0.3])
        stand_quat = np.array([0.707, 0.0, 0.707, 0.0])
        stand_pose = Pose(stand_target_pos, stand_quat)
        self.move_to(stand_pose, 10)
        
        # 步骤6：微调位置准备放置
        print("步骤6: 微调位置准备放置")
        current_pose = self.arm.get_site_pose(self.data)
        fine_tune_pose = Pose(
            pos=current_pose.pos + np.array([0.08, 0.0, 0.0]),
            quat=current_pose.quat
        )
        self.move_to(fine_tune_pose, 8)
        
        # 步骤7：下降到放置位置
        print("步骤7: 下降到放置位置")
        current_pose = self.arm.get_site_pose(self.data)
        place_pose = Pose(
            pos=current_pose.pos + np.array([0.0, 0.0, -0.09]),
            quat=current_pose.quat
        )
        self.move_to(place_pose, 8)
        
        # 步骤8：打开夹爪放置移液器
        print("步骤8: 打开夹爪放置移液器")
        self.gripper_control(0.0)
        
        # 步骤9：抬起机械臂
        print("步骤9: 抬起机械臂")
        current_pose = self.arm.get_site_pose(self.data)
        lift_pose = Pose(
            pos=current_pose.pos + np.array([-0.15, 0.0, 0.1]),
            quat=current_pose.quat
        )
        self.move_to(lift_pose, 8)
        
        # 步骤10：移动到安全位置
        print("步骤10: 移动到安全位置")
        safe_pose = Pose(
            pos=np.array([0.3, 0.0, 0.3]),
            quat=np.array([0.707, 0.0, 0.707, 0.0])
        )
        self.move_to(safe_pose, 10)
        
        # 检查任务是否完成
        if self.check('transition'):
            print("过渡任务完成！")
        else:
            print("过渡任务未完成条件")
        
        self.finish()

# 注册专家类
TransitionPipetteTask.Expert = TransitionPipetteExpert

def generate_transition_data(num_episodes: int = 100):
    """生成过渡数据"""
    print(f"开始生成 {num_episodes} 个过渡任务数据...")
    
    try:
        # 加载场景
        spec = TransitionPipetteTask.load()
        
        # 创建专家实例
        expert = TransitionPipetteTask.Expert(spec)
        
        # 生成多个episode的数据
        for episode in trange(num_episodes):
            try:
                # 重置环境
                expert.reset(seed=episode)
                
                # 设置序列化器
                expert.set_serializer()
                
                # 执行专家策略
                expert.execute()
                
                print(f"Episode {episode + 1}/{num_episodes} 完成")
                
            except Exception as e:
                print(f"Episode {episode + 1} 出错: {e}")
                continue
        
        print("所有过渡数据生成完成！")
        
    except Exception as e:
        print(f"生成数据时出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    """主函数：生成过渡数据"""
    print("=" * 60)
    print("过渡任务数据生成脚本")
    print("任务: transition")
    print("描述: Transition action from pick_pipette_from_stand to place_pipette_on_stand")
    print("=" * 60)
    
    # 生成100次过渡数据
    generate_transition_data(100)