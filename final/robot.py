import numpy as np


# ==========================================
# 1. 机械臂模型类 (升级为动力学模型)
# ==========================================
class Robot3DoF_Spatial:
    def __init__(self):
        # --- 运动学参数 (保持不变) ---
        self.l1 = 0.5  # 基座
        self.l2 = 0.35  # 大臂
        self.l3 = 0.3  # 小臂

        # --- [新增] 动力学参数 (质量与惯性) ---
        self.m2 = 2.0  # 大臂质量 (kg)
        self.m3 = 1.5  # 小臂质量 (kg)
        self.g = 9.81  # 重力加速度

        # 质心位置 (假设位于连杆几何中心)
        self.r_c2 = np.array([self.l2 / 2, 0, 0])
        self.r_c3 = np.array([self.l3 / 2, 0, 0])

        # 简化的惯性张量 (假设细杆 I = 1/12*m*L^2)
        # 这是一个 3x3 对角矩阵
        self.I1 = np.diag([0.01, 0.01, 0.1])
        self.I2 = np.diag([0.01, (1 / 12) * self.m2 * self.l2 ** 2, (1 / 12) * self.m2 * self.l2 ** 2])
        self.I3 = np.diag([0.01, (1 / 12) * self.m3 * self.l3 ** 2, (1 / 12) * self.m3 * self.l3 ** 2])

        # 物理限制
        self.max_torque = 80.0  # 增大限制以对抗重力
        self.max_vel = 3.0

        # 抓取工具长度 (用于绘图)
        self.gripper_len = self.l3 *0.1

    # ... (保留 forward_kinematics, inverse_kinematics, get_jacobian 方法，此处省略以节省篇幅) ...
    # 请务必保留原有的 FK, IK, Jacobian 代码！

    def forward_kinematics(self, theta):
        """
        [简化版 FK]
        假设抓夹始终水平，TCP 只是在 Wrist 的基础上水平延伸 gripper_len
        """
        t1, t2, t3 = theta

        p0 = np.array([0, 0, 0])
        p1 = np.array([0, 0, self.l1])

        # --- 1. 计算手腕 (Wrist) ---
        r2 = self.l2 * np.cos(t2)
        z2 = self.l2 * np.sin(t2)

        r_wrist = r2 + self.l3 * np.cos(t2 + t3)
        z_wrist = z2 + self.l3 * np.sin(t2 + t3)

        # --- 2. 计算 TCP (抓夹中心) ---
        # 简化逻辑：直接在半径上加 gripper_len，高度 z 不变
        r_tcp = r_wrist + self.gripper_len
        z_tcp = z_wrist  # 水平延伸，高度即为手腕高度

        # --- 3. 转换坐标 ---
        # Elbow
        p2 = np.array([
            r2 * np.cos(t1),
            r2 * np.sin(t1),
            self.l1 + z2
        ])

        # Wrist (灰色小球)
        p3_wrist = np.array([
            r_wrist * np.cos(t1),
            r_wrist * np.sin(t1),
            self.l1 + z_wrist
        ])

        # TCP (抓夹中心)
        p_tcp = np.array([
            r_tcp * np.cos(t1),
            r_tcp * np.sin(t1),
            self.l1 + z_tcp
        ])

        return [p0, p1, p2, p3_wrist, p_tcp]

    def get_jacobian(self, theta):
        """
        [简化版 Jacobian]
        针对水平抓夹的 TCP 雅可比
        """
        t1, t2, t3 = theta
        c1, s1 = np.cos(t1), np.sin(t1)
        c2, s2 = np.cos(t2), np.sin(t2)
        c23, s23 = np.cos(t2 + t3), np.sin(t2 + t3)

        # 当前的总水平半径 (含抓夹)
        r_total = self.l2 * c2 + self.l3 * c23 + self.gripper_len

        J = np.zeros((3, 3))

        # === 核心逻辑 ===
        # 对于 d/dt1 (基座旋转)：半径变大了 (r_total)，线速度变大
        # 对于 d/dt2, d/dt3 (俯仰运动)：抓夹只是跟着手腕平移，没有额外的旋转力臂效应

        # dx/dtheta
        J[0, :] = [
            -s1 * r_total,  # d/dt1 (受抓夹长度影响)
            c1 * (-self.l2 * s2 - self.l3 * s23),  # d/dt2 (抓夹水平平移，导数不变)
            c1 * (-self.l3 * s23)  # d/dt3 (抓夹水平平移，导数不变)
        ]

        # dy/dtheta
        J[1, :] = [
            c1 * r_total,  # d/dt1
            s1 * (-self.l2 * s2 - self.l3 * s23),  # d/dt2
            s1 * (-self.l3 * s23)  # d/dt3
        ]

        # dz/dtheta (高度 z 与 t1 无关，且抓夹水平延伸不影响高度变化率)
        J[2, :] = [
            0,  # d/dt1
            self.l2 * c2 + self.l3 * c23,  # d/dt2
            self.l3 * c23  # d/dt3
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
        """ [新增] 计算 3x3 耦合质量矩阵 M(q) """
        t1, t2, t3 = q
        M = np.zeros((3, 3))

        # 旋转矩阵计算
        c1, s1 = np.cos(t1), np.sin(t1)
        R1 = np.array([[c1, -s1, 0], [s1, c1, 0], [0, 0, 1]])

        c2, s2 = np.cos(t2), np.sin(t2)
        Ry2 = np.array([[c2, 0, s2], [0, 1, 0], [-s2, 0, c2]])
        R2 = R1 @ Ry2

        c3, s3 = np.cos(t3), np.sin(t3)
        Ry3 = np.array([[c3, 0, s3], [0, 1, 0], [-s3, 0, c3]])
        R3 = R2 @ Ry3

        # 1. 连杆1 (基座) 贡献
        Jw1 = np.zeros((3, 3));
        Jw1[2, 0] = 1  # 只有Z轴旋转
        M += Jw1.T @ self.I1 @ Jw1

        # 2. 连杆2 (大臂) 贡献
        p_j1 = np.array([0, 0, self.l1])
        p_c2 = p_j1 + R2 @ self.r_c2

        Jv2 = np.zeros((3, 3));
        Jw2 = np.zeros((3, 3))
        z0 = np.array([0, 0, 1])
        y1 = np.array([-s1, c1, 0])  # 旋转后的轴

        Jv2[:, 0] = np.cross(z0, p_c2);
        Jw2[:, 0] = z0
        Jv2[:, 1] = np.cross(y1, p_c2 - p_j1);
        Jw2[:, 1] = y1

        M += self.m2 * Jv2.T @ Jv2 + Jw2.T @ (R2 @ self.I2 @ R2.T) @ Jw2

        # 3. 连杆3 (小臂) 贡献
        p_j2 = p_j1 + R2 @ np.array([self.l2, 0, 0])
        p_c3 = p_j2 + R3 @ self.r_c3

        Jv3 = np.zeros((3, 3));
        Jw3 = np.zeros((3, 3))
        y2 = R2 @ np.array([0, 1, 0])

        Jv3[:, 0] = np.cross(z0, p_c3);
        Jw3[:, 0] = z0
        Jv3[:, 1] = np.cross(y1, p_c3 - p_j1);
        Jw3[:, 1] = y1
        Jv3[:, 2] = np.cross(y2, p_c3 - p_j2);
        Jw3[:, 2] = y2

        M += self.m3 * Jv3.T @ Jv3 + Jw3.T @ (R3 @ self.I3 @ R3.T) @ Jw3

        return M

    def get_gravity_torque(self, q):
        """ [新增] 计算重力矩 G(q) """
        t1, t2, t3 = q
        # 简化模型：只考虑 θ2, θ3 受到重力 (假设 θ2=0 垂直向上)
        # 注意：这里的三角函数取决于你的坐标系定义。
        # 假设 t2=0 是竖直向上，则重力矩与 sin(t2) 成正比 (力臂水平投影)
        # 假设 t2=0 是水平向前，则重力矩与 cos(t2) 成正比

        # 基于你的 FK: z = l*sin(t) => t=0 水平 => cos(t) 是水平力臂
        c2 = np.cos(t2)
        c23 = np.cos(t2 + t3)

        # 关节3力矩 (只受m3影响)
        tau3 = self.m3 * self.g * (self.l3 / 2 * c23)

        # 关节2力矩 (受m2和m3影响)
        tau2 = (self.m2 * self.g * self.l2 / 2 * c2) + \
               (self.m3 * self.g * (self.l2 * c2 + self.l3 / 2 * c23))

        return np.array([0.0, tau2, tau3])

    def forward_dynamics(self, q, dq, tau_applied):
        """
        [新增] 核心动力学方程 F = ma
        M(q) * ddq + C + G + Friction = Tau
        """
        M = self.compute_mass_matrix(q)
        G = self.get_gravity_torque(q)

        # === 摩擦力模型 ===
        # 1. 粘性阻尼 (Viscous): 0.5 * dq
        # 2. 库伦摩擦 (Coulomb): 0.2 * tanh(dq) (平滑版sign)
        # 你可以根据需要注释掉这一行来测试无摩擦情况
        friction = (0.5 * dq) + (0.2 * np.tanh(dq * 10))

        # 动力学平衡方程: Tau_net = Tau_motor - G - Friction
        rhs = tau_applied - G - friction

        # 求解加速度 ddq
        ddq = np.linalg.solve(M, rhs)
        return ddq


# ==========================================
# 2. 力矩控制器 (Torque Controller)
# ==========================================
class TorquePIDController:
    def __init__(self, Kp, Ki, Kd, dt, max_torque):
        self.Kp = Kp  # 现在代表刚度 (Stiffness)
        self.Ki = Ki  # 消除稳态误差 (抗重力)
        self.Kd = Kd  # 阻尼 (Damping)
        self.dt = dt
        self.max_torque = max_torque
        self.integral = np.zeros(3)

    def update(self, target_q, current_q, current_dq):
        # 1. 误差计算
        error = target_q - current_q

        # 2. P项
        P = self.Kp * error

        # 3. I项 (带抗饱和 Anti-Windup)
        self.integral += error * self.dt
        self.integral = np.clip(self.integral, -50.0, 50.0)  # 允许更大的积分
        I = self.Ki * self.integral

        # 4. D项 (使用 -Velocity 代替 d(Error)/dt 以减少噪声)
        D = self.Kd * (0 - current_dq)

        # 5. 总力矩
        torque = P + I + D

        # 6. 限幅
        norm = np.linalg.norm(torque)
        if norm > self.max_torque:
            torque = torque / norm * self.max_torque

        return torque