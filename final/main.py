import numpy as np
import traceback

from scene import Scene
from shelf import Shelf
from robot import Robot, PID
from planner import RRTPlanner, TaskScheduler
from simulation import Sim
from monitor import Monitor
from user_interface import UI
from safe_point_manager import SafeManager


def init_env():
    s = Scene()
    dy, dx = 0.55, 0.16
    shelves = {
        'A': Shelf('A', (dx, dy)),
        'B': Shelf('B', (-dx, dy)),
        'C': Shelf('C', (dx, -dy)),
        'D': Shelf('D', (-dx, -dy))
    }
    for v in shelves.values(): s.add_shelf(v)
    s.draw_static()
    return s, shelves


def run_dyn(robot, scene, mon, traj, pts):
    qs, phs, mds = traj
    if len(qs) == 0: return

    print(f"[Main] Sim Dynamics...")

    # 优化前
    # kp = np.array([150, 30, 10])
    # ki = np.array([150, 30, 10])
    # kd = np.array([150, 30, 10])
    # 优化后
    kp = np.array([185.1, 118.0, 155.3])
    ki = np.array([6.0, 12.9, 11.3])
    kd = np.array([10.3, 7.4, 1.0])

    pid = PID(kp, ki, kd, 0.01, robot.max_tau)
    q = qs[0].copy()
    dq = np.zeros(3)

    hist = {'q': [], 'ee': [], 'phase': [], 'mode': []}

    try:
        for i, ref in enumerate(qs):
            t = i * 0.01
            tau_pid = pid.update(ref, q, dq)
            tau_g = robot.grav_tau(q)
            tau = tau_pid + tau_g

            ddq = robot.dynamics(q, dq, tau)
            dq += ddq * 0.01
            q += dq * 0.01

            fk = robot.fk(q)
            hist['q'].append(q.copy())
            hist['ee'].append(fk[-1].copy())
            hist['phase'].append(phs[i])
            hist['mode'].append(mds[i])

            mon.log(t, robot.fk(ref)[-1], fk[-1], ref, q, tau)

    except:
        traceback.print_exc()

    if hist['q']:
        print("[Main] Animation...")
        sim = Sim(scene, robot, hist, {'pick': pts[0], 'place': pts[1]})
        sim.run(10)
        mon.plot()


def main():
    try:
        ui = UI(['A', 'B', 'C', 'D'], range(1, 4))
        cfg = ui.get_task()
    except:
        cfg = {'pick': {'id': 'A', 'layer': 2}, 'place': {'id': 'C', 'layer': 1}}

    bot = Robot()
    rrt = RRTPlanner(step=0.08, max_iter=15000)
    mon = Monitor()
    scene, db = init_env()
    safe = SafeManager(scene, bot, rrt)

    sched = TaskScheduler(bot, rrt, safe, scene)
    res = sched.plan_task(
        db[cfg['pick']['id']], cfg['pick']['layer'],
        db[cfg['place']['id']], cfg['place']['layer']
    )

    if res is None:
        print("[Main] Failed.")
        return

    full_q, phs, mds, pts = res
    vis = [bot.fk(q)[-1] for q in full_q]
    scene.draw_path(np.array(vis), 'lime')
    run_dyn(bot, scene, mon, (full_q, phs, mds), pts)


if __name__ == "__main__":
    main()