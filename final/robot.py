import numpy as np


class Robot3DoF_Spatial:
    def __init__(self):
        self.l1 = 0.5  # base
        self.l2 = 0.35
        self.l3 = 0.3
        self.m2 = 2.0
        self.m3 = 1.5
        self.g = 9.81

        self.r_c2 = np.array([self.l2 / 2, 0, 0])
        self.r_c3 = np.array([self.l3 / 2, 0, 0])

        #  I = 1/12*m*L^2)
        self.I1 = np.diag([0.01, 0.01, 0.1])
        self.I2 = np.diag([0.01, (1 / 12) * self.m2 * self.l2 ** 2, (1 / 12) * self.m2 * self.l2 ** 2])
        self.I3 = np.diag([0.01, (1 / 12) * self.m3 * self.l3 ** 2, (1 / 12) * self.m3 * self.l3 ** 2])

        self.max_torque = 80.0  # 增大限制以对抗重力
        self.max_vel = 3.0
        self.gripper_len = self.l3 *0.1

    def forward_kinematics(self, theta):

        t1, t2, t3 = theta
        p0 = np.array([0, 0, 0])
        p1 = np.array([0, 0, self.l1])
        r2 = self.l2 * np.cos(t2)
        z2 = self.l2 * np.sin(t2)
        r_wrist = r2 + self.l3 * np.cos(t2 + t3)
        z_wrist = z2 + self.l3 * np.sin(t2 + t3)

        # gripper
        r_tcp = r_wrist + self.gripper_len
        z_tcp = z_wrist

        p2 = np.array([
            r2 * np.cos(t1),
            r2 * np.sin(t1),
            self.l1 + z2
        ])

        p3_wrist = np.array([
            r_wrist * np.cos(t1),
            r_wrist * np.sin(t1),
            self.l1 + z_wrist
        ])

        p_tcp = np.array([
            r_tcp * np.cos(t1),
            r_tcp * np.sin(t1),
            self.l1 + z_tcp
        ])

        return [p0, p1, p2, p3_wrist, p_tcp]

    def get_jacobian(self, theta):

        t1, t2, t3 = theta
        c1, s1 = np.cos(t1), np.sin(t1)
        c2, s2 = np.cos(t2), np.sin(t2)
        c23, s23 = np.cos(t2 + t3), np.sin(t2 + t3)
        # r_gripper
        r_total = self.l2 * c2 + self.l3 * c23 + self.gripper_len

        J = np.zeros((3, 3))

        # r_wrist = r2 + self.l3 * np.cos(t2 + t3)
        # z_wrist = z2 + self.l3 * np.sin(t2 + t3)
        # r_tcp = r_wrist + self.gripper_len
        # z_tcp = z_wrist
        # p_tcp = np.array([
        #     r_tcp * np.cos(t1),
        #     r_tcp * np.sin(t1),
        #     self.l1 + z_tcp
        # ])

        # dx
        J[0, :] = [
            -s1 * r_total,
            c1 * (-self.l2 * s2 - self.l3 * s23),
            c1 * (-self.l3 * s23)
        ]
        # dy
        J[1, :] = [
            c1 * r_total,
            s1 * (-self.l2 * s2 - self.l3 * s23),
            s1 * (-self.l3 * s23)
        ]
        # dz
        J[2, :] = [
            0,
            self.l2 * c2 + self.l3 * c23,
            self.l3 * c23
        ]

        return J

    def inverse_kinematics(self, target_pos, initial_q, max_iter=100, tol=1e-4):
        q = np.array(initial_q, dtype=float)
        for _ in range(max_iter):
            current_pos = self.forward_kinematics(q)[-1]
            error = target_pos - current_pos
            if np.linalg.norm(error) < tol: return q

            J = self.get_jacobian(q)
            # 阻尼最小二乘
            dq = np.linalg.inv(J @ J.T + 0.0025 * np.eye(3)) @ error
            dq = J.T @ dq
            q += dq
        return q

    def compute_mass_matrix(self, q):

        t1, t2, t3 = q
        M = np.zeros((3, 3))

        # 旋转矩阵计算
        c1, s1 = np.cos(t1), np.sin(t1)
        c2, s2 = np.cos(t2), np.sin(t2)
        c3, s3 = np.cos(t3), np.sin(t3)
        r1 = np.array([[c1, -s1, 0], [s1, c1, 0], [0, 0, 1]])
        ry2 = np.array([[c2, 0, s2], [0, 1, 0], [-s2, 0, c2]])
        r2 = r1 @ ry2
        ry3 = np.array([[c3, 0, s3], [0, 1, 0], [-s3, 0, c3]])
        r3 = r2 @ ry3

        jw1 = np.zeros((3, 3));
        jw1[2, 0] = 1
        M += jw1.T @ self.I1 @ jw1


        p_j1 = np.array([0, 0, self.l1])
        p_c2 = p_j1 + r2 @ self.r_c2

        jv2 = np.zeros((3, 3));
        jw2 = np.zeros((3, 3))
        z0 = np.array([0, 0, 1])
        y1 = np.array([-s1, c1, 0])  # 旋转后的轴

        jv2[:, 0] = np.cross(z0, p_c2);
        jw2[:, 0] = z0
        jv2[:, 1] = np.cross(y1, p_c2 - p_j1);
        jw2[:, 1] = y1

        M += self.m2 * jv2.T @ jv2 + jw2.T @ (r2 @ self.I2 @ r2.T) @ jw2

        # 3. 连杆3 (小臂) 贡献
        p_j2 = p_j1 + r2 @ np.array([self.l2, 0, 0])
        p_c3 = p_j2 + r3 @ self.r_c3

        jv3 = np.zeros((3, 3));
        jw3 = np.zeros((3, 3))
        y2 = r2 @ np.array([0, 1, 0])

        jv3[:, 0] = np.cross(z0, p_c3);
        jw3[:, 0] = z0
        jv3[:, 1] = np.cross(y1, p_c3 - p_j1);
        jw3[:, 1] = y1
        jv3[:, 2] = np.cross(y2, p_c3 - p_j2);
        jw3[:, 2] = y2

        M += self.m3 * jv3.T @ jv3 + jw3.T @ (r3 @ self.I3 @ r3.T) @ jw3

        return M

    def get_gravity_torque(self, q):

        t1, t2, t3 = q
        c2 = np.cos(t2)
        c23 = np.cos(t2 + t3)
        tau3 = self.m3 * self.g * (self.l3 / 2 * c23)

        tau2 = (self.m2 * self.g * self.l2 / 2 * c2) + \
               (self.m3 * self.g * (self.l2 * c2 + self.l3 / 2 * c23))

        return np.array([0.0, tau2, tau3])

    def compute_coriolis_matrix(self, q, dq):
        """
        利用数值差分法计算科氏力矩阵 C(q, dq)
        基于 Christoffel 符号公式
        """
        n = len(q)
        epsilon = 1e-6  # 差分步长
        M = self.compute_mass_matrix(q)
        C = np.zeros((n, n))

        # 1. 计算 M 对 q 的偏导数矩阵 dM/dqi
        # dM_dq[i] 存储的是 dM / dqi (形状 3x3)
        dM_dq = []
        for i in range(n):
            q_plus = q.copy()
            q_plus[i] += epsilon
            M_plus = self.compute_mass_matrix(q_plus)

            q_minus = q.copy()
            q_minus[i] -= epsilon
            M_minus = self.compute_mass_matrix(q_minus)

            dM_dqi = (M_plus - M_minus) / (2 * epsilon)
            dM_dq.append(dM_dqi)

        # 2. 计算 Christoffel 符号并构建 C 矩阵
        # C_kj = sum( c_ijk * dq_i )
        # c_ijk = 0.5 * (dM_kj/dqi + dM_ki/dqj - dM_ij/dqk)
        for k in range(n):
            for j in range(n):
                c_kj = 0.0
                for i in range(n):
                    # 提取偏导项
                    dMkj_dqi = dM_dq[i][k, j]
                    dMki_dqj = dM_dq[j][k, i]
                    dMij_dqk = dM_dq[k][i, j]

                    # Christoffel 符号公式
                    c_ijk = 0.5 * (dMkj_dqi + dMki_dqj - dMij_dqk)

                    # 累加得到 C 矩阵元素
                    c_kj += c_ijk * dq[i]

                C[k, j] = c_kj

        return C

    def forward_dynamics(self, q, dq, tau_applied):
        """
        [更新版] 核心动力学方程 F = ma
        M(q)ddq + C(q,dq)dq + G(q) + Friction = Tau
        """
        # 1. 计算各项动力学矩阵
        M = self.compute_mass_matrix(q)
        G = self.get_gravity_torque(q)
        C_mat = self.compute_coriolis_matrix(q, dq)  # [新增] 计算 C 矩阵

        # 2. 计算科氏力项力矩 Vector (C * dq)
        tau_coriolis = C_mat @ dq

        # 3. 摩擦力模型
        friction = (0.5 * dq) + (0.2 * np.tanh(dq * 10))

        # 4. 动力学平衡方程
        # M * ddq = Tau_applied - C*dq - G - Friction
        rhs = tau_applied - tau_coriolis - G - friction

        # 5. 求解加速度
        # 为了防止 M 矩阵奇异导致报错，通常加一个极小的正则化项(可选)
        # M_reg = M + 1e-6 * np.eye(3)
        ddq = np.linalg.solve(M, rhs)

        return ddq



class TorquePIDController:
    def __init__(self, Kp, Ki, Kd, dt, max_torque):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.dt = dt
        self.max_torque = max_torque
        self.integral = np.zeros(3)

    def update(self, target_q, current_q, current_dq):

        error = target_q - current_q
        P = self.Kp * error
        # Anti-Windup
        self.integral += error * self.dt
        self.integral = np.clip(self.integral, -50.0, 50.0)  # 允许更大的积分
        I = self.Ki * self.integral
        D = self.Kd * (0 - current_dq)
        torque = P + I + D
        norm = np.linalg.norm(torque)
        if norm > self.max_torque:
            torque = torque / norm * self.max_torque

        return torque