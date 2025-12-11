import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class Sim:
    def __init__(self, scene, robot, hist, task):
        self.scene = scene
        self.robot = robot
        self.q_hist = hist['q']
        self.ee_hist = hist['ee']
        self.phase = hist['phase']
        self.modes = hist['mode']
        self.p_pick = task['pick']
        self.p_place = task['place']

        self.lines = self.scene.init_dyn()
        self.tr_x, self.tr_y, self.tr_z = [], [], []

    def check(self, j_pos):
        # 碰撞变色
        def hit(pt):
            x, y, z = pt
            if z < 0: return True
            for b in self.scene.obs:
                cx, cy, cz, w, d, h = b
                if (cx - w / 2 <= x <= cx + w / 2) and (cy - d / 2 <= y <= cy + d / 2) and (
                        cz - h / 2 <= z <= cz + h / 2):
                    return True
            return False

        if hit(j_pos[2]) or hit(j_pos[3]): return True
        return False

    def update(self, i):
        if i == 0: self.tr_x, self.tr_y, self.tr_z = [], [], []

        q = self.q_hist[i]
        ee = self.ee_hist[i]
        ph = self.phase[i]
        md = self.modes[i]

        fk = self.robot.fk(q)
        p0, p1, p2, p_w, p_t = fk

        is_hit = self.check(fk)
        c_arm = 'red' if is_hit else '#333333'

        self.scene.ln_arm.set_data([p[0] for p in fk[:4]], [p[1] for p in fk[:4]])
        self.scene.ln_arm.set_3d_properties([p[2] for p in fk[:4]])
        self.scene.ln_arm.set_color(c_arm)

        self.tr_x.append(ee[0]);
        self.tr_y.append(ee[1]);
        self.tr_z.append(ee[2])
        self.scene.ln_path.set_data(self.tr_x, self.tr_y)
        self.scene.ln_path.set_3d_properties(self.tr_z)

        # Gripper
        t1, t2, t3 = q
        pitch = t2 + t3 + (-(t2 + t3) * md)

        c_g = p_w + np.array(
            [np.cos(pitch) * np.cos(t1), np.cos(pitch) * np.sin(t1), np.sin(pitch)]) * self.robot.grip_len
        vec = np.array([-np.sin(t1), np.cos(t1), 0])
        w = 0.07 if ph == 2 else 0.12  # width

        pts = np.array([
            c_g + vec * w / 2, p_w + vec * w / 2,
            p_w - vec * w / 2, c_g - vec * w / 2
        ])
        self.scene.ln_grip.set_data(pts[:, 0], pts[:, 1])
        self.scene.ln_grip.set_3d_properties(pts[:, 2])

        self.scene.ln_joint.set_data([p_w[0]], [p_w[1]])
        self.scene.ln_joint.set_3d_properties([p_w[2]])

        # Object
        if ph == 1:
            pos = self.p_pick
        elif ph == 2:
            pos = c_g
        else:
            pos = self.p_place

        self.scene.ln_obj.set_data([pos[0]], [pos[1]])
        self.scene.ln_obj.set_3d_properties([pos[2]])

    def run(self, interval=20):
        print(f"Start Anim: {len(self.q_hist)} frames")
        frames = range(0, len(self.q_hist), 5)
        ani = FuncAnimation(self.scene.fig, self.update, frames=frames, interval=interval, blit=False)
        plt.show()