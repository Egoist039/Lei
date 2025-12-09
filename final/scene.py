import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


class Scene:
    def __init__(self, x_lim=(-0.8, 0.8), y_lim=(-0.8, 0.8), z_lim=(0, 1.2)):
        """
        初始化 3D 场景
        """
        self.fig = plt.figure(figsize=(14, 10))
        self.ax = self.fig.add_subplot(111, projection='3d')

        # 设置坐标轴范围
        self.ax.set_xlim(x_lim)
        self.ax.set_ylim(y_lim)
        self.ax.set_zlim(z_lim)

        # 设置初始视角 (Elev: 仰角, Azim: 方位角)
        self.ax.view_init(elev=30, azim=45)

        self.shelves = []
        self.global_obstacles = []  # 汇总所有障碍物供规划器使用

        # 动态绘图对象 (Line2D)
        self.arm_line = None
        self.path_line = None
        self.gripper_line = None
        self.ball_joint_dot = None
        self.obj_dot = None

    def add_shelf(self, shelf):
        """
        向场景添加货架，并自动提取其障碍物信息
        """
        self.shelves.append(shelf)
        # 将货架拆分出的障碍物加入全局障碍物列表
        self.global_obstacles.extend(shelf.get_obstacle_list())

    def add_obstacle(self, box):
        """
        添加单独的障碍物 [cx, cy, cz, dx, dy, dz]
        """
        self.global_obstacles.append(box)
        # 顺便画出来
        self._plot_box(box, color='orange', label='Extra Obstacle')

    def _plot_box(self, box, color='gray', label=None):
        """
        绘制简单的立方体障碍物 (线框模式)
        """
        cx, cy, cz, w, d, h = box
        xx = [cx - w / 2, cx + w / 2]
        yy = [cy - d / 2, cy + d / 2]
        zz = [cz - h / 2, cz + h / 2]

        # 画上下底面边框
        self.ax.plot([xx[0], xx[1], xx[1], xx[0], xx[0]],
                     [yy[0], yy[0], yy[1], yy[1], yy[0]],
                     [zz[1]] * 5, color=color, lw=1.5, label=label)  # 顶面

        # 画四条棱
        self.ax.plot([xx[0], xx[0]], [yy[0], yy[0]], zz, color=color, lw=1.5)
        self.ax.plot([xx[1], xx[1]], [yy[0], yy[0]], zz, color=color, lw=1.5)
        self.ax.plot([xx[1], xx[1]], [yy[1], yy[1]], zz, color=color, lw=1.5)
        self.ax.plot([xx[0], xx[0]], [yy[1], yy[1]], zz, color=color, lw=1.5)

    def draw_trajectory(self, trajectory, color='green', linestyle='--', label='Planned Path'):
        """
        绘制静态的规划路径
        :param trajectory: 形状为 (N, 3) 的 numpy 数组
        """
        if trajectory is None or len(trajectory) == 0:
            return

        # 提取坐标
        xs = trajectory[:, 0]
        ys = trajectory[:, 1]
        zs = trajectory[:, 2]

        # 绘制虚线
        self.ax.plot(xs, ys, zs, color=color, linestyle=linestyle, lw=1, alpha=0.6, label=label)

        # 更新图例以包含新加的线
        handles, labels = self.ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        self.ax.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=10)

    def draw_static_elements(self):
        """
        绘制静态场景：货架、地板、图例
        """
        # 1. 绘制所有货架
        for i, shelf in enumerate(self.shelves):
            shelf.plot(self.ax)

        # 2. 绘制工作空间底板 (示意)
        xx, yy = np.meshgrid(np.linspace(-0.8, 0.8, 2), np.linspace(-0.8, 0.8, 2))
        self.ax.plot_surface(xx, yy, np.zeros_like(xx), alpha=0.1, color='gray')

        # 3. 设置标签
        self.ax.set_xlabel("X (m)")
        self.ax.set_ylabel("Y (m)")
        self.ax.set_zlabel("Z (m)")
        self.ax.set_title("Robot Arm Pick & Place Simulation", fontsize=14)

    def init_dynamic_elements(self):
        """
        初始化动态元素（机械臂、轨迹、物体），返回对象用于动画更新
        """
        # 机械臂连杆 (黑色粗线)
        self.arm_line, = self.ax.plot([], [], [], 'o-', color='#333333', lw=4, ms=6, label='Robot Link')

        # 实际末端轨迹 (蓝色细线)
        self.path_line, = self.ax.plot([], [], [], '-', color='#1E90FF', lw=1.5, alpha=0.8, label='EE Path')

        # 夹爪 (黑色U型线)
        self.gripper_line, = self.ax.plot([], [], [], 'k-', lw=2.5, label='Gripper')

        # 腕关节球 (灰色点)
        self.ball_joint_dot, = self.ax.plot([], [], [], 'o', color='gray', ms=5)

        # 被抓物体 (红色大点)
        self.obj_dot, = self.ax.plot([], [], [], 'o', color='#FF4500', ms=10, label='Object')

        # 生成图例 (Legend)
        # 注意：这里我们手动过滤重复标签，避免图例太乱
        handles, labels = self.ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        self.ax.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=10, frameon=True)

        return self.arm_line, self.path_line, self.gripper_line, self.ball_joint_dot, self.obj_dot

    def show(self):
        plt.show()