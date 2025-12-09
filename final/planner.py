import numpy as np
import random
from utils import generate_continuous_path_from_waypoints


class JointRRTPlanner:
    class Node:
        def __init__(self, q, parent=None):
            self.q = np.array(q)
            self.parent = parent

    def __init__(self, step_size=0.1, max_iter=10000, joint_limits=None):
        self.step_size = step_size
        self.max_iter = max_iter
        if joint_limits is None:
            self.joint_limits = [(-np.pi, np.pi), (0, np.pi), (0, np.pi)]
        else:
            self.joint_limits = joint_limits

    def _get_dist(self, q1, q2):
        return np.linalg.norm(q1 - q2)

    def _is_colliding(self, q, obstacle_list, robot):
        fk = robot.forward_kinematics(q)
        pts = [fk[1], fk[2], fk[3], fk[4]]
        if fk[2][2] < 0.05 or fk[3][2] < 0.02 or fk[4][2] < 0.02: return True
        for i in range(len(pts) - 1):
            if self._check_segment_collision(pts[i], pts[i + 1], obstacle_list): return True
        return False

    def _check_segment_collision(self, p_start, p_end, obstacle_list):
        vec = p_end - p_start
        dist = np.linalg.norm(vec)
        steps = max(2, int(dist / 0.02))
        margin = 0.03
        for i in range(steps + 1):
            p = p_start + vec * (i / steps)
            x, y, z = p
            for box in obstacle_list:
                cx, cy, cz, w, d, h = box
                if (cx - w / 2 - margin <= x <= cx + w / 2 + margin) and \
                        (cy - d / 2 - margin <= y <= cy + d / 2 + margin) and \
                        (cz - h / 2 - margin <= z <= cz + h / 2 + margin):
                    return True
        return False

    def _check_path_validity(self, q_from, q_to, obstacle_list, robot):
        dist = self._get_dist(q_from, q_to)
        steps = max(2, int(dist / 0.05))
        vec = q_to - q_from
        for i in range(1, steps + 1):
            q_test = q_from + vec * (i / steps)
            if self._is_colliding(q_test, obstacle_list, robot): return False
        return True

    def compute_ik(self, robot, target_pos, seed_q, obstacle_list=None, check_collision=True, max_iter=80, tol=0.015):
        try:
            q = robot.inverse_kinematics(target_pos, seed_q, max_iter=max_iter, tol=tol)
            if np.linalg.norm(robot.forward_kinematics(q)[-1] - target_pos) > 0.05: return None
            if check_collision:
                if self._is_colliding(q, obstacle_list, robot): return None
            return q
        except:
            return None

    def generate_linear_path(self, q_start, q_end, steps, hold_steps=0):
        """
        生成路径：使用余弦插值 (S-Curve) 替代纯线性插值
        公式: factor = 0.5 * (1 - cos(t * pi))
        """
        path = []
        if q_start is None or q_end is None: return path

        for i in range(steps):
            # 1. 归一化时间 t: 0.0 -> 1.0
            t = i / steps

            # 2. 余弦插值核心公式
            # t=0 -> cos(0)=1  -> factor=0
            # t=1 -> cos(pi)=-1 -> factor=1
            factor = 0.5 * (1 - np.cos(t * np.pi))

            # 3. 计算位置
            q_new = q_start + (q_end - q_start) * factor
            path.append(q_new)

        for _ in range(hold_steps):
            path.append(q_end)

        return path

    def generate_smart_path(self, q_start, q_end, obstacle_list, robot):
        if np.linalg.norm(q_start - q_end) < 0.1:
            return self.generate_linear_path(q_start, q_end, steps=50)

        print(f"  [Planner] Running Smart RRT...")
        path_waypoints = None
        for _ in range(3):
            path_waypoints = self.plan(q_start, q_end, obstacle_list, robot)
            if path_waypoints is not None: break

        if path_waypoints is None:
            print("  [Planner] RRT Failed. Using Linear Fallback.")
            return self.generate_linear_path(q_start, q_end, steps=800)
        else:
            print(f"  [Planner] RRT Success ({len(path_waypoints)} waypoints).")
            return list(generate_continuous_path_from_waypoints(path_waypoints, 1000))

    def _extend(self, tree, q_rand, obstacle_list, robot):
        nearest_node = min(tree, key=lambda n: self._get_dist(n.q, q_rand))
        vec = q_rand - nearest_node.q
        dist = np.linalg.norm(vec)
        if dist <= self.step_size:
            q_new, status = q_rand, 'REACHED'
        else:
            q_new, status = nearest_node.q + vec / dist * self.step_size, 'ADVANCED'

        if self._check_path_validity(nearest_node.q, q_new, obstacle_list, robot):
            tree.append(self.Node(q_new, parent=nearest_node))
            return status, tree[-1]
        return 'TRAPPED', None

    def _connect(self, tree, q_target, obstacle_list, robot):
        q_curr = q_target
        while True:
            status, new_node = self._extend(tree, q_curr, obstacle_list, robot)
            if status == 'TRAPPED': return 'TRAPPED', None
            if status == 'REACHED': return 'REACHED', new_node

    def plan(self, start_q, end_q, obstacle_list, robot):
        if self._is_colliding(start_q, obstacle_list, robot) or self._is_colliding(end_q, obstacle_list,
                                                                                   robot): return None
        if self._check_path_validity(start_q, end_q, obstacle_list, robot): return np.vstack([start_q, end_q])

        tree_a, tree_b = [self.Node(start_q)], [self.Node(end_q)]
        tree_a_is_start = True

        for i in range(self.max_iter):
            rand_q = np.array([random.uniform(b[0], b[1]) for b in self.joint_limits])
            status, new_node_a = self._extend(tree_a, rand_q, obstacle_list, robot)
            if status != 'TRAPPED':
                connect_status, new_node_b = self._connect(tree_b, new_node_a.q, obstacle_list, robot)
                if connect_status == 'REACHED':
                    return self._generate_path(new_node_a, new_node_b, tree_a_is_start)
            tree_a, tree_b = tree_b, tree_a
            tree_a_is_start = not tree_a_is_start
        return None

    def _generate_path(self, node_a, node_b, a_is_start):
        path_a, curr = [], node_a
        while curr: path_a.append(curr.q); curr = curr.parent
        path_a = path_a[::-1]
        path_b, curr = [], node_b
        while curr: path_b.append(curr.q); curr = curr.parent
        return np.vstack([path_a, path_b]) if a_is_start else np.vstack([path_b, path_a])


class TaskPlanner:
    def __init__(self, robot, planner_algo, safe_manager, scene):
        self.robot = robot
        self.planner = planner_algo
        self.safe_manager = safe_manager
        self.scene = scene
        self.full_q_traj, self.full_phases, self.full_modes = [], [], []

    def plan_pick_and_place(self, pick_shelf, pick_layer, place_shelf, place_layer, obj_size=0.04):
        self.full_q_traj, self.full_phases, self.full_modes = [], [], []
        raw_p_pick = pick_shelf.get_target_pos(pick_layer, obj_size)
        raw_p_place = place_shelf.get_target_pos(place_layer, obj_size)
        p_pick, p_pick_ent = pick_shelf.calculate_approach_points(raw_p_pick)
        p_place, p_place_ent = place_shelf.calculate_approach_points(raw_p_place)
        self.safe_manager.solve_all_iks()
        iks = self._solve_all_keyframes(p_pick, p_pick_ent, p_place, p_place_ent)
        if iks is None: return None

        q_safe_start, q_pick_ent, q_pick, q_safe_end, q_place_ent, q_place = iks

        print(f"[TaskPlanner] Path: Safe -> Pick -> Safe -> Place")


        self._add_segment(q_safe_start, q_pick_ent, 1, 1, smart=True)
        self._add_segment(q_pick_ent, q_pick, 1, 1, linear=True, pause=10)
        self._add_segment(q_pick, q_pick_ent, 2, 1, linear=True)
        self._add_segment(q_pick_ent, q_safe_end, 2, 1, smart=True)
        self._add_segment(q_safe_end, q_place_ent, 2, 1, smart=True)
        self._add_segment(q_place_ent, q_place, 2, 1, linear=True, pause=10)
        self._add_segment(q_place, q_place_ent, 3, 1, linear=True)
        self._add_segment(q_place_ent, q_safe_end, 3, 1, smart=True)

        return np.array(self.full_q_traj), self.full_phases, self.full_modes, (p_pick, p_place)

    def _solve_all_keyframes(self, p_pick, p_pick_ent, p_place, p_place_ent):
        _, q_safe_start = self.safe_manager.get_closest_safe_config(p_pick_ent)
        q_pick_ent = self._try_ik(p_pick_ent, q_safe_start, safe_check=True)
        q_pick = self._try_ik(p_pick, q_pick_ent, safe_check=False)  # 末端容忍碰撞
        _, q_safe_end = self.safe_manager.get_closest_safe_config(p_place_ent)
        q_place_ent = self._try_ik(p_place_ent, q_safe_end, safe_check=True)
        q_place = self._try_ik(p_place, q_place_ent, safe_check=False)

        if any(x is None for x in [q_pick_ent, q_pick, q_place_ent, q_place]):
            return None
        return q_safe_start, q_pick_ent, q_pick, q_safe_end, q_place_ent, q_place

    def _try_ik(self, pos, seed, safe_check=True):

        if seed is None: return None
        # 优先用 SafeManager 的 Elbow-Up 约束
        q = self.safe_manager._solve_ik_elbow_up(pos, seed)
        if q is None:
            q = self.planner.compute_ik(self.robot, pos, seed, self.scene.global_obstacles, check_collision=safe_check)
        return q

    def _add_segment(self, q_start, q_end, phase, mode, smart=False, linear=False, pause=0):
        if smart:
            path = self.planner.generate_smart_path(q_start, q_end, self.scene.global_obstacles, self.robot)
        elif linear:
            path = self.planner.generate_linear_path(q_start, q_end, steps=200, hold_steps=pause)
        else:
            return
        for q in path:
            self.full_q_traj.append(q)
            self.full_phases.append(phase)
            self.full_modes.append(mode)