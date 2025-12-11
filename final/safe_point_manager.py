import numpy as np

# [Reference]
# Heuristic Constraint (Elbow-Up) for safe grasping
# Strategy: Siciliano (2009) - Kinematic Redundancy Resolution
class SafeManager:
    def __init__(self, scene, robot, planner):
        self.scene = scene
        self.robot = robot
        self.planner = planner
        self.pts = {
            'Front': np.array([0.25, 0.0, 0.45]),
            'Back': np.array([-0.25, 0.0, 0.45]) }
        self.qs = {}
        self._draw()

    def _draw(self):
        for n, p in self.pts.items():
            self.scene.ax.plot([p[0]], [p[1]], [p[2]], 'gx', markersize=8, label=f'Safe {n}')

    # [Reference]
    # Multi-start IK Strategy (Random/Fixed Restart) to avoid local minima
    # Inspired by: MoveIt! KDL Kinematics Plugin
    def solve_iks(self):
        print("  [SafeMgr] Solving IKs...")
        for n, pos in self.pts.items():
            ang = np.arctan2(pos[1], pos[0])
            seed = [ang, 0.8, 2.0]
            q = self.ik_elbow_up(pos, seed)
            if q is not None: self.qs[n] = q
            else: print(f"  [Error] IK fail: {n}")
        if not self.qs: raise RuntimeError("No safe points!")

    def ik_elbow_up(self, pos, seed):
        seeds = [seed, [0, 1.0, 2.0], [np.pi/2, 1.0, 2.0], [-np.pi/2, 1.0, 2.0]]
        for s in seeds:
            try:
                q = self.robot.ik(pos, s, max_iter=80, tol=0.015)
                # 精度
                if np.linalg.norm(self.robot.fk(q)[-1] - pos) > 0.05: continue
                # 碰撞
                if self.planner.is_collide(q, self.scene.obs, self.robot): continue
                # Elbow up
                fk = self.robot.fk(q)
                zs, ze, zw = fk[1][2], fk[2][2], fk[3][2]
                if ze > zw and ze > zs: return q
            except: continue
        return None

    def get_nearest(self, pos):
        best_n, min_d, best_q = None, float('inf'), None
        for n, q in self.qs.items():
            d = np.linalg.norm(pos - self.pts[n])
            if d < min_d:
                min_d = d
                best_n = n
                best_q = q
        return best_n, best_q