import numpy as np
import random

# [Reference]
# RRT-Connect Logic adapted from: PythonRobotics (Atsushi Sakai et al.)
# https://github.com/AtsushiSakai/PythonRobotics
# Algorithm based on: Kuffner & LaValle (2000) "RRT-connect"

class JointRRTPlanner:
    ...
class RRTPlanner:
    class Node:
        def __init__(self, q, parent=None):
            self.q = np.array(q)
            self.parent = parent

    def __init__(self, step=0.1, max_iter=10000, limits=None):
        self.step = step
        self.max_iter = max_iter
        if limits is None:
            self.limits = [(-np.pi, np.pi), (0, np.pi), (0, np.pi)]
        else:
            self.limits = limits


    def _wait(self, pos, steps):
        return np.tile(pos, (steps, 1))

        # [Reference]
        # S-Curve Velocity Profile (Cosine Interpolation)
        # Theory: Craig (2005) "Introduction to Robotics", Chapter 7
    def _interp_path(self, wpts, total_steps):
        # S-Curve
        wpts = np.array(wpts)
        if len(wpts) < 2: return self._wait(wpts[0], total_steps)

        dists = np.linalg.norm(wpts[1:] - wpts[:-1], axis=1)
        cum_dist = np.cumsum(np.r_[0, dists])
        total_len = cum_dist[-1]

        if total_len < 1e-6: return self._wait(wpts[0], total_steps)

        t = np.linspace(0, 1, total_steps)
        s = 10 * t ** 3 - 15 * t ** 4 + 6 * t ** 5
        target_dists = s * total_len

        traj = np.zeros((total_steps, 3))
        idx = np.searchsorted(cum_dist, target_dists, side='right') - 1
        idx = np.clip(idx, 0, len(wpts) - 2)

        for i, k in enumerate(idx):
            seg_len = dists[k]
            curr_dist = target_dists[i]
            ratio = (curr_dist - cum_dist[k]) / seg_len if seg_len > 1e-6 else 0
            traj[i] = wpts[k] + (wpts[k + 1] - wpts[k]) * ratio
        return traj


    #  RRT
    def dist(self, q1, q2):
        return np.linalg.norm(q1 - q2)

        # [Reference]
        # Collision detection via link discretization
        # Logic based on: LaValle (2006) "Planning Algorithms", Chapter 5.3
        # Implementation style reference: PythonRobotics

    def is_collide(self, q, obs, robot):
        fk = robot.fk(q)
        pts = [fk[1], fk[2], fk[3], fk[4]]  # p1, p2, wrist, tcp
        if fk[2][2] < 0.05 or fk[3][2] < 0.02 or fk[4][2] < 0.02: return True
        for i in range(len(pts) - 1):
            if self._seg_collide(pts[i], pts[i + 1], obs): return True
        return False

    def _seg_collide(self, p1, p2, obs):
        vec = p2 - p1
        d = np.linalg.norm(vec)
        steps = max(2, int(d / 0.02))
        margin = 0.03

        for i in range(steps + 1):
            curr = p1 + vec * (i / steps)
            x, y, z = curr
            for box in obs:
                cx, cy, cz, w, d_box, h = box
                # AABB collision
                if (cx - w / 2 - margin <= x <= cx + w / 2 + margin) and \
                        (cy - d_box / 2 - margin <= y <= cy + d_box / 2 + margin) and \
                        (cz - h / 2 - margin <= z <= cz + h / 2 + margin):
                    return True
        return False

    def check_path(self, q1, q2, obs, robot):
        d = self.dist(q1, q2)
        steps = max(2, int(d / 0.05))
        vec = q2 - q1
        for i in range(1, steps + 1):
            q = q1 + vec * (i / steps)
            if self.is_collide(q, obs, robot): return False
        return True

    def get_ik(self, robot, target, seed, obs=None, check=True, max_iter=80, tol=0.015):
        try:
            q = robot.ik(target, seed, max_iter=max_iter, tol=tol)
            if np.linalg.norm(robot.fk(q)[-1] - target) > 0.05: return None
            if check:
                if self.is_collide(q, obs, robot): return None
            return q
        except:
            return None

    def gen_linear_path(self, q1, q2, steps, hold=0):
        path = []
        if q1 is None or q2 is None: return path
        for i in range(steps):
            t = i / steps
            fac = 0.5 * (1 - np.cos(t * np.pi))
            q = q1 + (q2 - q1) * fac
            path.append(q)
        for _ in range(hold):
            path.append(q2)
        return path

    def gen_smart_path(self, q1, q2, obs, robot):
        if np.linalg.norm(q1 - q2) < 0.1:
            return self.gen_linear_path(q1, q2, steps=50)

        # print(f"  [Plan] RRT searching...")
        wpts = None
        for _ in range(3):
            wpts = self.plan(q1, q2, obs, robot)
            if wpts is not None: break

        if wpts is None:
            print("  [Plan] RRT failed, linear fallback.")
            return self.gen_linear_path(q1, q2, steps=800)
        else:
            # print(f"  [Plan] Success: {len(wpts)} nodes.")
            return list(self._interp_path(wpts, 1000))

    def _extend(self, tree, q_rnd, obs, robot):
        node_near = min(tree, key=lambda n: self.dist(n.q, q_rnd))
        vec = q_rnd - node_near.q
        d = np.linalg.norm(vec)

        if d <= self.step:
            q_new, status = q_rnd, 'REACHED'
        else:
            q_new, status = node_near.q + vec / d * self.step, 'ADVANCED'

        if self.check_path(node_near.q, q_new, obs, robot):
            tree.append(self.Node(q_new, parent=node_near))
            return status, tree[-1]
        return 'TRAPPED', None

    def _connect(self, tree, q_tgt, obs, robot):
        curr = q_tgt
        while True:
            status, node = self._extend(tree, curr, obs, robot)
            if status == 'TRAPPED': return 'TRAPPED', None
            if status == 'REACHED': return 'REACHED', node

    def plan(self, q_start, q_end, obs, robot):
        if self.is_collide(q_start, obs, robot) or self.is_collide(q_end, obs, robot): return None
        if self.check_path(q_start, q_end, obs, robot): return np.vstack([q_start, q_end])

        tree_a, tree_b = [self.Node(q_start)], [self.Node(q_end)]
        a_is_start = True

        for i in range(self.max_iter):
            rnd = np.array([random.uniform(b[0], b[1]) for b in self.limits])
            status, node_a = self._extend(tree_a, rnd, obs, robot)

            if status != 'TRAPPED':
                c_stat, node_b = self._connect(tree_b, node_a.q, obs, robot)
                if c_stat == 'REACHED':
                    return self._build_path(node_a, node_b, a_is_start)

            tree_a, tree_b = tree_b, tree_a
            a_is_start = not a_is_start
        return None

    def _build_path(self, n_a, n_b, a_is_start):
        p_a, curr = [], n_a
        while curr: p_a.append(curr.q); curr = curr.parent
        p_a = p_a[::-1]

        p_b, curr = [], n_b
        while curr: p_b.append(curr.q); curr = curr.parent

        return np.vstack([p_a, p_b]) if a_is_start else np.vstack([p_b, p_a])


class TaskScheduler:
    def __init__(self, robot, planner, safe_mgr, scene):
        self.robot = robot
        self.planner = planner
        self.safe = safe_mgr
        self.scene = scene
        self.traj, self.phases, self.modes = [], [], []

    def plan_task(self, shelf_pick, l_pick, shelf_place, l_place, obj_size=0.04):
        self.traj, self.phases, self.modes = [], [], []

        p_pick_raw = shelf_pick.get_target(l_pick, obj_size)
        p_place_raw = shelf_place.get_target(l_place, obj_size)
        p_pick, p_pick_ent = shelf_pick.get_approach(p_pick_raw)
        p_place, p_place_ent = shelf_place.get_approach(p_place_raw)

        self.safe.solve_iks()
        iks = self._solve_keys(p_pick, p_pick_ent, p_place, p_place_ent)
        if iks is None: return None

        q_safe_1, q_pick_ent, q_pick, q_safe_2, q_place_ent, q_place = iks

        print(f"[Task] Path: Safe -> Pick -> Safe -> Place")

        self._add(q_safe_1, q_pick_ent, 1, 1, smart=True)
        self._add(q_pick_ent, q_pick, 1, 1, linear=True, pause=10)
        self._add(q_pick, q_pick_ent, 2, 1, linear=True)
        self._add(q_pick_ent, q_safe_2, 2, 1, smart=True)
        self._add(q_safe_2, q_place_ent, 2, 1, smart=True)
        self._add(q_place_ent, q_place, 2, 1, linear=True, pause=10)
        self._add(q_place, q_place_ent, 3, 1, linear=True)
        self._add(q_place_ent, q_safe_2, 3, 1, smart=True)

        return np.array(self.traj), self.phases, self.modes, (p_pick, p_place)

    def _solve_keys(self, p_pick, p_pick_ent, p_place, p_place_ent):
        _, q_safe_1 = self.safe.get_nearest(p_pick_ent)
        q_pick_ent = self._try_ik(p_pick_ent, q_safe_1, check=True)
        q_pick = self._try_ik(p_pick, q_pick_ent, check=False)  # 抓取末端不检碰撞

        _, q_safe_2 = self.safe.get_nearest(p_place_ent)
        q_place_ent = self._try_ik(p_place_ent, q_safe_2, check=True)
        q_place = self._try_ik(p_place, q_place_ent, check=False)

        if any(x is None for x in [q_pick_ent, q_pick, q_place_ent, q_place]):
            return None
        return q_safe_1, q_pick_ent, q_pick, q_safe_2, q_place_ent, q_place

    def _try_ik(self, pos, seed, check=True):
        if seed is None: return None
        # elbow-up
        q = self.safe.ik_elbow_up(pos, seed)
        if q is None:
            q = self.planner.get_ik(self.robot, pos, seed, self.scene.obs, check=check)
        return q

    def _add(self, q1, q2, phase, mode, smart=False, linear=False, pause=0):
        if smart:
            path = self.planner.gen_smart_path(q1, q2, self.scene.obs, self.robot)
        elif linear:
            path = self.planner.gen_linear_path(q1, q2, steps=200, hold=pause)
        else:
            return

        for q in path:
            self.traj.append(q)
            self.phases.append(phase)
            self.modes.append(mode)