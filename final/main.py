import numpy as np
import traceback

# 引入模块
from scene import Scene
from shelf import Shelf
from robot import Robot3DoF_Spatial, TorquePIDController
from planner import JointRRTPlanner, TaskPlanner
from simulation import Simulation
from monitor import PerformanceMonitor
from user_interface import UserInterface
from safe_point_manager import SafePointManager


def setup_scene():
    scene = Scene()
    dist_y, dist_x_gap = 0.55, 0.16
    shelves = {
        'A': Shelf('A', (dist_x_gap, dist_y), num_layers=3),
        'B': Shelf('B', (-dist_x_gap, dist_y), num_layers=3),
        'C': Shelf('C', (dist_x_gap, -dist_y), num_layers=3),
        'D': Shelf('D', (-dist_x_gap, -dist_y), num_layers=3)
    }
    for s in shelves.values(): scene.add_shelf(s)
    scene.draw_static_elements()
    return scene, shelves


def execute_simulation_dynamics(robot, scene, monitor, traj_data, task_points):
    q_traj, phases, modes = traj_data
    if len(q_traj) == 0: return

    print(f"[Main] Starting Dynamics Simulation (PD + Gravity Comp)...")

    Kp_vec = np.array([185.1, 118.0, 155.3])
    Ki_vec = np.array([6.0, 12.9, 11.3])
    Kd_vec = np.array([10.3, 7.4, 1.0])
    # Kp_vec = np.array([182.8, 471.9, 121.1])
    # Ki_vec = np.array([3.2, 11.5, 2.6])
    # Kd_vec = np.array([15.2, 32.3, 1.0])


    pid = TorquePIDController(Kp=Kp_vec, Ki=Ki_vec, Kd=Kd_vec, dt=0.01,
                              max_torque=robot.max_torque)

    dt = 0.01
    q = q_traj[0].copy()
    dq = np.zeros(3)

    history = {'q': [], 'ee': [], 'phase': [], 'wrist_mode': []}

    try:
        for i, target_q in enumerate(q_traj):
            t_now = i * dt

            tau_pd = pid.update(target_q, q, dq)
            tau_gravity = robot.get_gravity_torque(q)
            tau_cmd = tau_pd + tau_gravity
            ddq = robot.forward_dynamics(q, dq, tau_cmd)

            dq += ddq * dt
            q += dq * dt

            #
            fk = robot.forward_kinematics(q)
            history['q'].append(q.copy())
            history['ee'].append(fk[-1].copy())
            history['phase'].append(phases[i])
            history['wrist_mode'].append(modes[i])

            # 记录详细数据用于绘图
            monitor.log(t_now,
                        robot.forward_kinematics(target_q)[-1], fk[-1],  # 笛卡尔
                        target_q, q, tau_cmd)  # 关节 & 力矩

    except Exception:
        traceback.print_exc()

    # 播放动画
    if history['q']:
        print("[Main] Playing Animation...")
        sim = Simulation(scene, robot, history, {'p_pick': task_points[0], 'p_place': task_points[1]})
        sim.run(interval=10)
        monitor.plot()  #


def main():

    try:
        ui = UserInterface(valid_shelves=['A', 'B', 'C', 'D'], layers_range=range(1, 4))
        task_cfg = ui.get_user_task()
    except:
        task_cfg = {'pick': {'id': 'A', 'layer': 2}, 'place': {'id': 'C', 'layer': 1}}

    # 2. 初始化
    robot = Robot3DoF_Spatial()
    planner_algo = JointRRTPlanner(step_size=0.08, max_iter=15000)
    monitor = PerformanceMonitor()
    scene, shelves_db = setup_scene()
    safe_manager = SafePointManager(scene, robot, planner_algo)

    # 3. 规划
    task_planner = TaskPlanner(robot, planner_algo, safe_manager, scene)
    result = task_planner.plan_pick_and_place(
        pick_shelf=shelves_db[task_cfg['pick']['id']],
        pick_layer=task_cfg['pick']['layer'],
        place_shelf=shelves_db[task_cfg['place']['id']],
        place_layer=task_cfg['place']['layer']
    )

    if result is None:
        print("[Main] Planning failed.")
        return

    full_traj, phases, modes, task_points = result
    vis_pts = [robot.forward_kinematics(q)[-1] for q in full_traj]
    scene.draw_trajectory(np.array(vis_pts), color='lime')
    execute_simulation_dynamics(robot, scene, monitor, (full_traj, phases, modes), task_points)


if __name__ == "__main__":
    main()