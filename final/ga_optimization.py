import numpy as np
import random
import copy
import time

# 引入你的机器人和控制器
from robot import Robot3DoF_Spatial, TorquePIDController


class PIDGeneticOptimizer:
    def __init__(self, robot, pop_size=50, generations=20, mutation_rate=0.1):
        """
        :param robot: 机器人实例
        :param pop_size: 种群大小 (每一代有多少个个体)
        :param generations: 进化代数
        :param mutation_rate: 变异概率
        """
        self.robot = robot
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate

        # 搜索范围 [min, max]
        # 结构: [Joint1_PID, Joint2_PID, Joint3_PID]
        self.bounds = {
            'Kp': [(10, 200), (50, 600), (10, 300)],
            'Ki': [(0, 10), (0, 50), (0, 20)],
            'Kd': [(1, 20), (5, 50), (1, 30)]
        }

        # 基因长度: 3个关节 * 3个参数 = 9个基因
        self.gene_length = 9

    def _create_individual(self):
        """ 创建一个随机个体 """
        genes = []
        for i in range(3):
            kp = random.uniform(*self.bounds['Kp'][i])
            ki = random.uniform(*self.bounds['Ki'][i])
            kd = random.uniform(*self.bounds['Kd'][i])
            genes.extend([kp, ki, kd])
        return np.array(genes)

    def _decode_genes(self, genes):
        """ 解码基因 """
        Kp_vec = np.array([genes[0], genes[3], genes[6]])
        Ki_vec = np.array([genes[1], genes[4], genes[7]])
        Kd_vec = np.array([genes[2], genes[5], genes[8]])
        return Kp_vec, Ki_vec, Kd_vec

    def fitness_function(self, genes):
        """
        适应度函数: 赋予关节3最高权重
        """
        Kp, Ki, Kd = self._decode_genes(genes)

        # 1. 初始化控制器
        pid = TorquePIDController(Kp, Ki, Kd, dt=0.01, max_torque=self.robot.max_torque)

        # 2. 定义测试任务 (Step Response)
        q_start = np.zeros(3)
        q_target = np.array([0.5, 0.8, -0.5])

        steps = 150
        dt = 0.01

        q = q_start.copy()
        dq = np.zeros(3)
        total_error = 0.0
        total_effort = 0.0

        try:
            for i in range(steps):
                current_target = q_start if i < 10 else q_target

                # --- 控制回路 ---
                tau_pid = pid.update(current_target, q, dq)
                tau_g = self.robot.get_gravity_torque(q)
                tau_cmd = tau_pid + tau_g

                # --- 物理引擎 ---
                ddq = self.robot.forward_dynamics(q, dq, tau_cmd)
                dq += ddq * dt
                q += dq * dt

                # --- 计算代价 ---
                err = np.abs(current_target - q)

                # [核心修改] 权重分配: [Joint1, Joint2, Joint3]
                # 之前是 [1.0, 2.0, 1.0]，现在给 Joint 3 最高权重
                weights = np.array([1.0, 1.0, 3.0])

                total_error += np.sum(err * weights)

                # 力矩惩罚 (防止过度震荡)
                total_effort += np.sum(np.abs(tau_cmd)) * 0.001

                if np.max(np.abs(q)) > 10.0:
                    return 1e9

        except Exception:
            return 1e9

        score = total_error + total_effort
        return score

    def selection(self, population, scores):
        parents = []
        for _ in range(self.pop_size):
            candidates_idx = random.sample(range(self.pop_size), 3)
            best_idx = min(candidates_idx, key=lambda idx: scores[idx])
            parents.append(population[best_idx])
        return np.array(parents)

    def crossover(self, parents):
        offspring = []
        for i in range(0, self.pop_size, 2):
            p1 = parents[i]
            p2 = parents[(i + 1) % self.pop_size]
            point = random.randint(1, self.gene_length - 1)
            c1 = np.concatenate((p1[:point], p2[point:]))
            c2 = np.concatenate((p2[:point], p1[point:]))
            offspring.extend([c1, c2])
        return np.array(offspring)

    def mutation(self, offspring):
        for i in range(self.pop_size):
            if random.random() < self.mutation_rate:
                gene_idx = random.randint(0, self.gene_length - 1)
                joint_idx = gene_idx // 3
                param_idx = gene_idx % 3
                param_names = ['Kp', 'Ki', 'Kd']
                key = param_names[param_idx]
                bounds = self.bounds[key][joint_idx]

                current_val = offspring[i][gene_idx]
                delta = (random.random() - 0.5) * 0.4 * current_val
                new_val = np.clip(current_val + delta, bounds[0], bounds[1])
                offspring[i][gene_idx] = new_val
        return offspring

    def run(self):
        print(f"=== Starting Genetic Algorithm (High Weight on Joint 3) ===")
        population = np.array([self._create_individual() for _ in range(self.pop_size)])
        best_overall_score = float('inf')
        best_overall_genes = None

        for gen in range(self.generations):
            scores = [self.fitness_function(ind) for ind in population]
            best_gen_score = min(scores)

            if best_gen_score < best_overall_score:
                best_overall_score = best_gen_score
                best_overall_genes = population[scores.index(best_gen_score)].copy()

            print(f"Gen {gen + 1}/{self.generations} | Best Score: {best_gen_score:.4f}")

            parents = self.selection(population, scores)
            offspring = self.crossover(parents)
            population = self.mutation(offspring)
            population[0] = best_overall_genes

        Kp, Ki, Kd = self._decode_genes(best_overall_genes)
        print("\nRecommended Parameters:")
        print(f"Kp_vec = np.array([{Kp[0]:.1f}, {Kp[1]:.1f}, {Kp[2]:.1f}])")
        print(f"Ki_vec = np.array([{Ki[0]:.1f}, {Ki[1]:.1f}, {Ki[2]:.1f}])")
        print(f"Kd_vec = np.array([{Kd[0]:.1f}, {Kd[1]:.1f}, {Kd[2]:.1f}])")
        return Kp, Ki, Kd


if __name__ == "__main__":
    robot = Robot3DoF_Spatial()
    optimizer = PIDGeneticOptimizer(robot, pop_size=50, generations=40)
    optimizer.run()