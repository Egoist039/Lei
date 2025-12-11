import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# [Reference]
# Visualization and Animation framework
# Source: Previous Group Assignment (Modified for 3-DOF Arm project)
class Scene:
    def __init__(self, xlim=(-0.8, 0.8), ylim=(-0.8, 0.8), zlim=(0, 1.2)):
        self.fig = plt.figure(figsize=(14, 10))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)
        self.ax.set_zlim(zlim)
        self.ax.view_init(elev=30, azim=45)
        self.shelves = []
        self.obs = []
        self.ln_arm = None
        self.ln_path = None
        self.ln_grip = None
        self.ln_joint = None
        self.ln_obj = None

    def add_shelf(self, s):
        self.shelves.append(s)
        self.obs.extend(s.get_obs())

    def draw_path(self, traj, col='green'):
        if traj is None or len(traj) == 0: return
        xs, ys, zs = traj[:, 0], traj[:, 1], traj[:, 2]
        self.ax.plot(xs, ys, zs, color=col, linestyle='--', lw=1, alpha=0.6, label='Planned Path')
        self.ax.legend(loc='upper right')

    def draw_static(self):
        for s in self.shelves: s.plot(self.ax)
        xx, yy = np.meshgrid(np.linspace(-0.8, 0.8, 2), np.linspace(-0.8, 0.8, 2))
        self.ax.plot_surface(xx, yy, np.zeros_like(xx), alpha=0.1, color='gray')
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_title("Robot Simulation")

    def init_dyn(self):
        self.ln_arm, = self.ax.plot([], [], [], 'o-', color='#333', lw=4, ms=6, label='Link')
        self.ln_path, = self.ax.plot([], [], [], '-', color='#1E90FF', lw=1.5, alpha=0.8, label='Trail')
        self.ln_grip, = self.ax.plot([], [], [], 'k-', lw=2.5, label='Gripper')
        self.ln_joint, = self.ax.plot([], [], [], 'o', color='gray', ms=5)
        self.ln_obj, = self.ax.plot([], [], [], 'o', color='#FF4500', ms=10, label='Object')
        self.ax.legend(loc='upper right', frameon=True)
        return self.ln_arm, self.ln_path, self.ln_grip, self.ln_joint, self.ln_obj

    def show(self):
        plt.show()