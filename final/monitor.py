import matplotlib.pyplot as plt
import numpy as np


class PerformanceMonitor:
    def __init__(self):
        self.time_log = []
        self.pos_error_log = []
        self.joint_targets = []
        self.joint_actuals = []
        self.torques = []

    def log(self, t, target_pos, current_pos, target_q, current_q, torque):

        err = np.linalg.norm(target_pos - current_pos)
        self.pos_error_log.append(err)
        self.time_log.append(t)
        self.joint_targets.append(target_q.copy())
        self.joint_actuals.append(current_q.copy())
        self.torques.append(torque.copy())

    def plot(self):

        if not self.time_log:
            print("[Monitor] No data.")
            return

        joint_targets = np.array(self.joint_targets)
        joint_actuals = np.array(self.joint_actuals)
        torques = np.array(self.torques)

        # 创建 3x1 子图
        fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
        fig.suptitle("Joint Space PID Dynamics Response", fontsize=16)

        joint_names = ['Joint 1 (Base)', 'Joint 2 (Shoulder)', 'Joint 3 (Elbow)']

        for i in range(3):
            ax1 = axes[i]
            # 左轴：角度 (Position)
            l1, = ax1.plot(self.time_log, joint_targets[:, i], 'r--', label='Target (rad)', alpha=0.8)
            l2, = ax1.plot(self.time_log, joint_actuals[:, i], 'b-', label='Actual (rad)', lw=2)
            ax1.set_ylabel(f"{joint_names[i]}\nAngle (rad)", fontsize=12)
            ax1.grid(True, linestyle=':', alpha=0.6)

            # 右轴：力矩 (Torque)
            ax2 = ax1.twinx()
            l3, = ax2.plot(self.time_log, torques[:, i], 'g-', label='Torque (Nm)', alpha=0.3, lw=1)
            ax2.set_ylabel("Torque (Nm)", color='green', fontsize=12)
            ax2.tick_params(axis='y', labelcolor='green')

            # 合并图例
            lines = [l1, l2, l3]
            ax1.legend(lines, [l.get_label() for l in lines], loc='upper right')

        axes[-1].set_xlabel("Time (s)", fontsize=12)
        plt.tight_layout()
        plt.show()

        # 可选：如果你还想看笛卡尔误差
        # plt.figure(figsize=(8, 4))
        # plt.plot(self.time_log, self.pos_error_log, 'k-')
        # plt.title("Cartesian Tracking Error")
        # plt.xlabel("Time (s)")
        # plt.ylabel("Error (m)")
        # plt.grid(True)
        # plt.show()