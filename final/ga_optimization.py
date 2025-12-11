import numpy as np
import random
from robot import Robot, PID


class GA:
    def __init__(self, robot, n_pop=50, gens=20):
        self.bot = robot
        self.n = n_pop
        self.gens = gens
        self.rate = 0.1

        # Ranges: Kp, Ki, Kd
        self.lim_p = [(10, 200), (50, 600), (10, 300)]
        self.lim_i = [(0, 10), (0, 50), (0, 20)]
        self.lim_d = [(1, 20), (5, 50), (1, 30)]

    def rand_gene(self):
        g = []
        for i in range(3):
            g.append(random.uniform(*self.lim_p[i]))
            g.append(random.uniform(*self.lim_i[i]))
            g.append(random.uniform(*self.lim_d[i]))
        return np.array(g)

    def decode(self, g):
        kp = np.array([g[0], g[3], g[6]])
        ki = np.array([g[1], g[4], g[7]])
        kd = np.array([g[2], g[5], g[8]])
        return kp, ki, kd

    def calc_fit(self, g):
        kp, ki, kd = self.decode(g)

        # Init controller
        pid = PID(kp, ki, kd, 0.01, self.bot.max_tau)

        q = np.zeros(3)
        dq = np.zeros(3)
        tgt = np.array([0.5, 0.8, -0.5])

        err_sum = 0.0
        effort = 0.0

        try:
            for i in range(150):
                ref = tgt if i > 10 else q

                tau = pid.update(ref, q, dq) + self.bot.grav_tau(q)
                ddq = self.bot.dynamics(q, dq, tau)

                dq += ddq * 0.01
                q += dq * 0.01

                # J3 weight higher
                e = np.abs(ref - q)
                err_sum += np.sum(e * [1.0, 1.0, 3.0])
                effort += np.sum(np.abs(tau)) * 0.001

                if np.max(np.abs(q)) > 10: return 1e9

        except:
            return 1e9

        return err_sum + effort

    def select(self, pop, scores):
        parents = []
        for _ in range(self.n):
            # Tournament 3
            idxs = random.sample(range(self.n), 3)
            best = min(idxs, key=lambda i: scores[i])
            parents.append(pop[best])
        return np.array(parents)

    def cross(self, parents):
        kids = []
        for i in range(0, self.n, 2):
            p1, p2 = parents[i], parents[(i + 1) % self.n]
            cut = random.randint(1, 8)
            kids.append(np.concatenate((p1[:cut], p2[cut:])))
            kids.append(np.concatenate((p2[:cut], p1[cut:])))
        return np.array(kids)

    def mutate(self, pop):
        for i in range(self.n):
            if random.random() < self.rate:
                idx = random.randint(0, 8)
                # Map index to limit
                j = idx // 3
                t = idx % 3
                if t == 0:
                    b = self.lim_p[j]
                elif t == 1:
                    b = self.lim_i[j]
                else:
                    b = self.lim_d[j]

                val = pop[i][idx]
                delta = (random.random() - 0.5) * 0.4 * val
                pop[i][idx] = np.clip(val + delta, b[0], b[1])
        return pop

    def run(self):
        print(f"--- GA Start (N={self.n}) ---")
        pop = np.array([self.rand_gene() for _ in range(self.n)])

        best_s = float('inf')
        best_g = None

        for i in range(self.gens):
            scores = [self.calc_fit(p) for p in pop]
            min_s = min(scores)

            if min_s < best_s:
                best_s = min_s
                best_g = pop[scores.index(min_s)].copy()

            print(f"Gen {i + 1}: Best={min_s:.4f}")

            parents = self.select(pop, scores)
            kids = self.cross(parents)
            pop = self.mutate(kids)
            pop[0] = best_g  # Elitism

        kp, ki, kd = self.decode(best_g)
        print("\n=== Result ===")
        print(f"Kp = {np.round(kp, 1).tolist()}")
        print(f"Ki = {np.round(ki, 1).tolist()}")
        print(f"Kd = {np.round(kd, 1).tolist()}")
        return kp, ki, kd


if __name__ == "__main__":
    bot = Robot()
    ga = GA(bot, n_pop=50, gens=50)
    ga.run()