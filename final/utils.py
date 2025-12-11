import numpy as np

# def limit_vector(vector, max_val):
#     """ 矢量限幅 """
#     norm = np.linalg.norm(vector)
#     return vector * (max_val / norm) if norm > max_val else vector

def wait_at(pos, steps):
    """ 生成原地等待的轨迹点 """
    return np.tile(pos, (steps, 1))

# def generate_s_curve_trajectory(p_start, p_end, steps):
#     """ 生成 S 型平滑轨迹 (五次多项式插值) """
#     t = np.linspace(0, 1, steps)
#     s = 10 * t ** 3 - 15 * t ** 4 + 6 * t ** 5
#     traj = np.zeros((steps, 3))
#     for i in range(3):
#         traj[:, i] = p_start[i] + (p_end[i] - p_start[i]) * s
#     return traj

def generate_continuous_path_from_waypoints(waypoints, total_steps):

    waypoints = np.array(waypoints)
    if len(waypoints) < 2: return wait_at(waypoints[0], total_steps)

    dists = np.linalg.norm(waypoints[1:] - waypoints[:-1], axis=1)
    cum_dist = np.cumsum(np.r_[0, dists])
    total_len = cum_dist[-1]

    if total_len < 1e-6: return wait_at(waypoints[0], total_steps)

    t = np.linspace(0, 1, total_steps)
    s = 10 * t ** 3 - 15 * t ** 4 + 6 * t ** 5
    target_dists = s * total_len

    traj = np.zeros((total_steps, 3))
    indices = np.searchsorted(cum_dist, target_dists, side='right') - 1
    indices = np.clip(indices, 0, len(waypoints) - 2)

    for i, idx in enumerate(indices):
        segment_len = dists[idx]
        current_dist = target_dists[i]
        ratio = (current_dist - cum_dist[idx]) / segment_len if segment_len > 1e-6 else 0
        traj[i] = waypoints[idx] + (waypoints[idx + 1] - waypoints[idx]) * ratio
    return traj