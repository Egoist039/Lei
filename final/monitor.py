import matplotlib.pyplot as plt
import numpy as np


class Monitor:
    def __init__(self):
        self.t = []
        self.err = []
        self.q_ref = []
        self.q_act = []
        self.tau = []

    def log(self, t, p_ref, p_act, q_ref, q_act, tau):
        e = np.linalg.norm(p_ref - p_act)
        self.err.append(e)
        self.t.append(t)
        self.q_ref.append(q_ref.copy())
        self.q_act.append(q_act.copy())
        self.tau.append(tau.copy())

    def plot(self):
        if not self.t: return

        qr = np.array(self.q_ref)
        qa = np.array(self.q_act)
        ts = np.array(self.tau)

        fig, ax = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
        names = ['Base', 'Shoulder', 'Elbow']

        for i in range(3):
            a1 = ax[i]
            l1, = a1.plot(self.t, qr[:, i], 'r--', label='Ref')
            l2, = a1.plot(self.t, qa[:, i], 'b-', label='Act')
            a1.set_ylabel(f"{names[i]} (rad)")
            a1.grid(True, ls=':', alpha=0.6)

            a2 = a1.twinx()
            l3, = a2.plot(self.t, ts[:, i], 'g-', alpha=0.3)
            a2.set_ylabel("Nm", color='green')

            if i == 0: a1.legend([l1, l2, l3], ['Ref', 'Act', 'Torque'])

        ax[-1].set_xlabel("Time (s)")
        plt.tight_layout()
        plt.show()