import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


class Shelf:
    def __init__(self, sid, pos, size=(0.3, 0.3), n=3, z0=0.15, space=0.25):
        self.id = sid
        self.cx, self.cy = pos
        self.w, self.d = size
        self.n_layers = n
        self.z0 = z0
        self.space = space
        self.leg_sz = 0.03
        self.plate_h = 0.02
        self.h_list = [z0 + i * space for i in range(n)]
        self.top = self.h_list[-1]

    def get_target(self, idx, obj_sz=0.04):
        i = max(0, min(idx - 1, self.n_layers - 1))
        z = self.h_list[i] + 0.01 + obj_sz / 2 + 0.005
        return np.array([self.cx, self.cy, z])

    def get_approach(self, pos):
        vec = np.array([self.cx, self.cy, 0.0])
        dist = np.linalg.norm(vec)

        if dist < 1e-6:
            dir_in = np.zeros(3)
        else:
            dir_in = -(vec / dist)

        p_act = pos + dir_in * 0.14
        p_ent = p_act + dir_in * 0.12
        return p_act, p_ent

    def get_obs(self):
        obs = []
        # Plates
        for h in self.h_list:
            z = h - self.plate_h / 2
            obs.append([self.cx, self.cy, z, self.w, self.d, self.plate_h])
        # Legs
        hw = self.w / 2 - self.leg_sz / 2
        hd = self.d / 2 - self.leg_sz / 2
        corners = [
            (self.cx - hw, self.cy - hd), (self.cx + hw, self.cy - hd),
            (self.cx + hw, self.cy + hd), (self.cx - hw, self.cy + hd)
        ]
        z_leg = self.top / 2
        for (lx, ly) in corners:
            obs.append([lx, ly, z_leg, self.leg_sz, self.leg_sz, self.top])
        return obs

    def plot(self, ax):
        hw, hd = self.w / 2, self.d / 2
        for h in self.h_list:
            xmin, xmax = self.cx - hw, self.cx + hw
            ymin, ymax = self.cy - hd, self.cy + hd

            xx = [xmin, xmax, xmax, xmin, xmin]
            yy = [ymin, ymin, ymax, ymax, ymin]
            zz = [h] * 5
            ax.plot(xx, yy, zz, color='#5C3317', lw=1.5)

            verts = [list(zip(xx[:-1], yy[:-1], zz[:-1]))]
            ax.add_collection3d(Poly3DCollection(verts, alpha=0.3, facecolors='#8B4513'))

        corners = [
            (self.cx - hw, self.cy - hd), (self.cx + hw, self.cy - hd),
            (self.cx + hw, self.cy + hd), (self.cx - hw, self.cy + hd)
        ]
        for (lx, ly) in corners:
            ax.plot([lx, lx], [ly, ly], [0, self.top], color='black', lw=2)

        ax.text(self.cx, self.cy, self.top + 0.1, f"Shelf {self.id}",
                color='black', fontsize=10, ha='center', weight='bold')