import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class Simulation:
    def __init__(self, scene, robot, history_data, task_info):
        """
        初始化动画模拟器
        :param scene: Scene 对象 (包含绘图元素和障碍物列表)
        :param robot: Robot 对象 (用于 FK 计算)
        :param history_data: 字典，包含 keys: ['q', 'ee', 'phase', 'wrist_mode']
        :param task_info: 字典，包含 keys: ['p_pick', 'p_place'] (用于确定物体位置)
        """
        self.scene = scene
        self.robot = robot

        # 解包历史数据
        self.history_q = history_data['q']
        self.history_ee = history_data['ee']
        self.phase_tracker = history_data['phase']
        self.wrist_modes = history_data['wrist_mode']

        # 任务关键点 (用于绘制静态物体)
        self.p_pick = task_info['p_pick']
        self.p_place = task_info['p_place']

        # 初始化场景中的动态元素
        # 返回: arm_line, path_line, gripper_line, ball_joint_dot, obj_dot
        self.lines = self.scene.init_dynamic_elements()

        # 轨迹线数据缓存
        self.path_x, self.path_y, self.path_z = [], [], []

    def _is_colliding(self, joint_positions):
        """
        简单的可视化碰撞检测 (仅检测肘部和腕部是否在障碍物内)
        用于在动画中将机械臂变红作为警告
        """

        # 定义简单的点碰撞检测内部函数
        def check_point(pt):
            x, y, z = pt
            if z < 0.0: return True  # 地面
            for box in self.scene.global_obstacles:
                cx, cy, cz, w, d, h = box
                if (cx - w / 2 <= x <= cx + w / 2) and \
                        (cy - d / 2 <= y <= cy + d / 2) and \
                        (cz - h / 2 <= z <= cz + h / 2):
                    return True
            return False

        # 检测肘部(p2)和腕部(p3_wrist)
        p2, p3_wrist = joint_positions[2], joint_positions[3]
        if check_point(p2) or check_point(p3_wrist):
            return True
        return False

    def update(self, frame_idx):
        """
        动画帧更新函数
        """
        # [修复] 如果是动画的第一帧，清空之前的轨迹数据
        # 防止动画循环时出现从终点连回起点的直线
        if frame_idx == 0:
            self.path_x, self.path_y, self.path_z = [], [], []

        # 1. 获取当前帧数据
        q_now = self.history_q[frame_idx]
        ee_pos = self.history_ee[frame_idx]
        phase = self.phase_tracker[frame_idx]  # 1=Pick, 2=Transfer, 3=Place
        w_mode = self.wrist_modes[frame_idx]  # 腕部模式

        # 2. 计算机械臂形态 (FK)
        # fk return: [p0, p1, p2, p3_wrist, p3_tool]
        fk_points = self.robot.forward_kinematics(q_now)
        p0, p1, p2, p3_wrist, p3_tool = fk_points

        # 3. 碰撞检测与变色
        is_crash = self._is_colliding(fk_points)
        color_arm = 'red' if is_crash else '#333333'

        # 4. 更新机械臂连杆 (Arm Link)
        self.scene.arm_line.set_data([p[0] for p in fk_points[:4]], [p[1] for p in fk_points[:4]])
        self.scene.arm_line.set_3d_properties([p[2] for p in fk_points[:4]])
        self.scene.arm_line.set_color(color_arm)

        # 5. 更新末端轨迹 (Path Trace)
        self.path_x.append(ee_pos[0])
        self.path_y.append(ee_pos[1])
        self.path_z.append(ee_pos[2])
        self.scene.path_line.set_data(self.path_x, self.path_y)
        self.scene.path_line.set_3d_properties(self.path_z)

        # 6. 计算并更新夹爪 (Gripper) 形态
        t1, t2, t3 = q_now
        # 根据 wrist_mode 计算总俯仰角
        # 如果 w_mode=1 (保持水平), 则 t4 = -(t2+t3)
        total_pitch = t2 + t3 + (- (t2 + t3) * w_mode)

        # 夹爪中心点 (比腕关节再往外延伸一点)
        g_center = p3_wrist + np.array([
            np.cos(total_pitch) * np.cos(t1),
            np.cos(total_pitch) * np.sin(t1),
            np.sin(total_pitch)
        ]) * self.robot.gripper_len

        # 计算垂直于手臂方向的向量，用于画夹爪的张开宽度
        perp_vec = np.array([-np.sin(t1), np.cos(t1), 0])

        # 夹爪开合状态：Phase 2 (搬运中) 闭合，其他张开
        current_width = 0.07 if phase == 2 else 0.12

        # 计算夹爪U型的4个点
        u_pts = np.array([
            g_center + perp_vec * current_width / 2,  # 指尖左
            p3_wrist + perp_vec * current_width / 2,  # 根部左
            p3_wrist - perp_vec * current_width / 2,  # 根部右
            g_center - perp_vec * current_width / 2  # 指尖右
        ])

        self.scene.gripper_line.set_data(u_pts[:, 0], u_pts[:, 1])
        self.scene.gripper_line.set_3d_properties(u_pts[:, 2])

        # 7. 更新关节球 (Visual Joint)
        self.scene.ball_joint_dot.set_data([p3_wrist[0]], [p3_wrist[1]])
        self.scene.ball_joint_dot.set_3d_properties([p3_wrist[2]])

        # 8. 更新物体位置 (Object)
        if phase == 1:
            # 抓取前，物体在取货点
            obj_pos = self.p_pick
        elif phase == 2:
            # 搬运中，物体跟随夹爪中心
            obj_pos = g_center
        else:
            # 放下后，物体在放货点
            obj_pos = self.p_place

        self.scene.obj_dot.set_data([obj_pos[0]], [obj_pos[1]])
        self.scene.obj_dot.set_3d_properties([obj_pos[2]])

        return self.scene.arm_line, self.scene.path_line, self.scene.gripper_line, self.scene.obj_dot

    def run(self, interval=20, save_path=None):
        """
        开始播放动画
        :param interval: 帧间隔 (ms)
        :param save_path: 如果不为None，则保存为mp4/gif (需要安装ffmpeg或imagemagick)
        """
        print(f"Starting Animation ({len(self.history_q)} frames)...")

        # 为了流畅，每隔 5 帧渲染一次 (skip frames)
        skip = 5
        frames = range(0, len(self.history_q), skip)

        ani = FuncAnimation(
            self.scene.fig,
            self.update,
            frames=frames,
            interval=interval,
            blit=False
        )

        if save_path:
            print(f"Saving animation to {save_path}...")
            ani.save(save_path, writer='ffmpeg', fps=30)
        else:
            plt.show()