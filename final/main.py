import numpy as np
import traceback

# 引入模块
from scene import Scene
from shelf import Shelf
# 确保 robot 模块里有 Robot3DoF_Spatial 和 TorquePIDController
from robot import Robot3DoF_Spatial, TorquePIDController
from planner import JointRRTPlanner, TaskPlanner
from simulation import Simulation
from monitor import PerformanceMonitor
from user_interface import UserInterface
from safe_point_manager import SafePointManager


def setup_scene():
    """ 场景配置 """
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

    # === 1. PID 参数调整 (纯 PD 控制) ===
    # Kp: 提供定位刚度 (可以适当大一点)
    # Ki: 设为 0.0 (不再需要积分项！)
    # Kd: 提供阻尼，防止震荡Kp_vec = np.array([198.8, 372.6, 43.2])
    # Ki_vec = np.array([3.2, 33.5, 18.7])
    # Kd_vec = np.array([9.8, 9.5, 1.4])

    Kp_vec = np.array([185.1, 118.0, 155.3])
    Ki_vec = np.array([6.0, 12.9, 11.3])
    Kd_vec = np.array([10.3, 7.4, 1.0])


    pid = TorquePIDController(Kp=Kp_vec, Ki=Ki_vec, Kd=Kd_vec, dt=0.01,
                              max_torque=robot.max_torque)

    dt = 0.01
    q = q_traj[0].copy()
    dq = np.zeros(3)

    history = {'q': [], 'ee': [], 'phase': [], 'wrist_mode': []}

    try:
        for i, target_q in enumerate(q_traj):
            t_now = i * dt

            # --- 步骤 A: 计算 PD 输出 ---
            # 此时 tau_pd 只负责"去哪"，不负责"抗重力"
            tau_pd = pid.update(target_q, q, dq)

            # --- 步骤 B: 计算重力补偿 (Feedforward) ---
            # 问 robot: "要在当前角度保持静止，需要多少力矩？"
            tau_gravity = robot.get_gravity_torque(q)

            # --- 步骤 C: 总力矩叠加 ---
            # 总力矩 = PD修正力 + 抗重力支撑力
            tau_cmd = tau_pd + tau_gravity

            # --- 步骤 D: 物理引擎推演 ---
            # 注意: robot.forward_dynamics 内部是 tau_applied - G
            # 所以我们传入 (tau_pd + G)，物理引擎里减去 G，
            # 剩下的净力矩正好就是 tau_pd，用来驱动运动 (F=ma)
            ddq = robot.forward_dynamics(q, dq, tau_cmd)

            dq += ddq * dt
            q += dq * dt

            # --- Step 4: 数据记录 ---
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
        monitor.plot()  # 将绘制三个关节的详细图表


def main():
    # ... (保持原有的 main 逻辑不变，只需确保调用 execute_simulation_dynamics) ...
    # 1. 获取任务
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

    # 可视化规划路径
    vis_pts = [robot.forward_kinematics(q)[-1] for q in full_traj]
    scene.draw_trajectory(np.array(vis_pts), color='lime')

    # 4. 执行 (动力学)
    execute_simulation_dynamics(robot, scene, monitor, (full_traj, phases, modes), task_points)


if __name__ == "__main__":
    main()