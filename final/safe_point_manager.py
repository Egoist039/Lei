import numpy as np


class SafePointManager:
    def __init__(self, scene, robot, planner):
        self.scene = scene
        self.robot = robot
        self.planner = planner

        # 定义坐标
        self.safe_points_map = {
            'Front': np.array([0.25, 0.0, 0.45]),
            'Back': np.array([-0.25, 0.0, 0.45])
        }
        self.safe_qs_map = {}

        # [删除] self._visualize()  <-- 这一行删掉

    # [删除] def _visualize(self): ... <-- 整个函数删掉

    def solve_all_iks(self):
        print("  [SafePointManager] Solving IK for safe points...")
        for name, pos in self.safe_points_map.items():
            # 计算基座朝向作为种子
            base_angle = np.arctan2(pos[1], pos[0])
            seed = [base_angle, 0.8, 2.0]

            q = self._solve_ik_elbow_up(pos, seed)
            if q is not None:
                self.safe_qs_map[name] = q
            else:
                print(f"  [ERROR] Could not solve IK for Safe Point: {name}")

        if not self.safe_qs_map:
            raise RuntimeError("No valid safe points found!")

    def _solve_ik_elbow_up(self, pos, seed_q):
        seeds = [seed_q, [0, 1.0, 2.0], [np.pi / 2, 1.0, 2.0], [-np.pi / 2, 1.0, 2.0]]

        for s in seeds:
            try:
                q = self.robot.inverse_kinematics(pos, s, max_iter=80, tol=0.015)
                # 校验位置精度
                if np.linalg.norm(self.robot.forward_kinematics(q)[-1] - pos) > 0.05: continue
                # 校验碰撞
                if self.planner._is_colliding(q, self.scene.global_obstacles, self.robot): continue

                # 校验姿态 (提肘)
                fk = self.robot.forward_kinematics(q)
                if fk[2][2] > fk[3][2] and fk[2][2] > fk[1][2]:  # z_elbow > z_wrist & shoulder
                    return q
            except:
                continue
        return None

    def get_closest_safe_config(self, target_pos_xyz):
        best_name = None
        min_dist = float('inf')
        best_q = None

        for name, q_safe in self.safe_qs_map.items():
            safe_pos_xyz = self.safe_points_map[name]
            dist = np.linalg.norm(target_pos_xyz - safe_pos_xyz)
            if dist < min_dist:
                min_dist = dist
                best_name = name
                best_q = q_safe

        return best_name, best_q