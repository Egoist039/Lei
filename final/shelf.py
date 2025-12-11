import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


class Shelf:
    def __init__(self, shelf_id, center_pos, size=(0.3, 0.3), num_layers=3, base_z=0.15, layer_spacing=0.25):
        self.id = shelf_id
        self.cx, self.cy = center_pos
        self.width, self.depth = size
        self.num_layers = num_layers
        self.base_z = base_z
        self.layer_spacing = layer_spacing
        self.leg_size = 0.03
        self.plate_thickness = 0.02
        self.layer_heights = [self.base_z + i * self.layer_spacing for i in range(num_layers)]
        self.top_z = self.layer_heights[-1]

    def get_target_pos(self, layer_idx, obj_size=0.04):
        idx = max(0, min(layer_idx - 1, self.num_layers - 1))
        z = self.layer_heights[idx] + 0.01 + obj_size / 2 + 0.005
        return np.array([self.cx, self.cy, z])

    def calculate_approach_points(self, target_pos_xyz):

        vec_to_shelf = np.array([self.cx, self.cy, 0.0])
        dist = np.linalg.norm(vec_to_shelf)

        if dist < 1e-6:
            dir_inwards = np.array([0, 0, 0])
        else:
            dir_outwards = vec_to_shelf / dist
            dir_inwards = -dir_outwards  # 指向机械臂基座的方向（即从货架向外）

        # p_act: 目标点向外退 14cm
        p_act = target_pos_xyz + dir_inwards * 0.14
        # p_ent: 再向外退 12cm
        p_ent = p_act + dir_inwards * 0.12

        return p_act, p_ent

    def get_obstacle_list(self):
        obstacles = []
        # 层板障碍物
        for h in self.layer_heights:
            z_center = h - self.plate_thickness / 2
            obstacles.append([self.cx, self.cy, z_center, self.width, self.depth, self.plate_thickness])
        # 支柱障碍物
        hw = self.width / 2 - self.leg_size / 2
        hd = self.depth / 2 - self.leg_size / 2
        corners = [
            (self.cx - hw, self.cy - hd), (self.cx + hw, self.cy - hd),
            (self.cx + hw, self.cy + hd), (self.cx - hw, self.cy + hd)
        ]
        total_height = self.top_z
        z_leg_center = total_height / 2
        for (lx, ly) in corners:
            obstacles.append([lx, ly, z_leg_center, self.leg_size, self.leg_size, total_height])
        return obstacles

    def plot(self, ax):

        hw, hd = self.width / 2, self.depth / 2

        for h in self.layer_heights:
            # 定义四个顶点
            x_min, x_max = self.cx - hw, self.cx + hw
            y_min, y_max = self.cy - hd, self.cy + hd
            z_val = h

            xx = [x_min, x_max, x_max, x_min, x_min]
            yy = [y_min, y_min, y_max, y_max, y_min]
            zz = [z_val] * 5

            ax.plot(xx, yy, zz, color='#5C3317', lw=1.5)

            verts = [list(zip(xx[:-1], yy[:-1], zz[:-1]))]
            # alpha=0.3 半透明，方便看穿
            poly = Poly3DCollection(verts, alpha=0.3, facecolors='#8B4513')
            ax.add_collection3d(poly)

        # 2. 绘制四根支柱
        corners = [
            (self.cx - hw, self.cy - hd), (self.cx + hw, self.cy - hd),
            (self.cx + hw, self.cy + hd), (self.cx - hw, self.cy + hd)
        ]
        for (lx, ly) in corners:
            ax.plot([lx, lx], [ly, ly], [0, self.top_z], color='black', lw=2)

        # 3. 标签
        ax.text(self.cx, self.cy, self.top_z + 0.1, f"Shelf {self.id}",
                color='black', fontsize=10, ha='center', fontweight='bold')