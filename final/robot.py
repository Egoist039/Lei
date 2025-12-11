import numpy as np


class Robot:
    def __init__(self):
        self.l1 = 0.5
        self.l2 = 0.35
        self.l3 = 0.3
        self.m2 = 2.0
        self.m3 = 1.5
        self.g = 9.81
        self.r_c2 = np.array([self.l2 / 2, 0, 0])
        self.r_c3 = np.array([self.l3 / 2, 0, 0])
        self.I1 = np.diag([0.01, 0.01, 0.1])
        self.I2 = np.diag([0.01, (1 / 12) * self.m2 * self.l2 ** 2, (1 / 12) * self.m2 * self.l2 ** 2])
        self.I3 = np.diag([0.01, (1 / 12) * self.m3 * self.l3 ** 2, (1 / 12) * self.m3 * self.l3 ** 2])
        self.max_tau = 80.0
        self.max_vel = 3.0
        self.grip_len = self.l3 * 0.1

    def fk(self, q):
        t1, t2, t3 = q
        p0 = np.array([0, 0, 0])
        p1 = np.array([0, 0, self.l1])
        r2 = self.l2 * np.cos(t2)
        z2 = self.l2 * np.sin(t2)
        r_w = r2 + self.l3 * np.cos(t2 + t3)
        z_w = z2 + self.l3 * np.sin(t2 + t3)
        # gripper
        r_tcp = r_w + self.grip_len
        z_tcp = z_w
        p2 = np.array([r2 * np.cos(t1), r2 * np.sin(t1), self.l1 + z2])
        p_w = np.array([r_w * np.cos(t1), r_w * np.sin(t1), self.l1 + z_w])
        p_tcp = np.array([r_tcp * np.cos(t1), r_tcp * np.sin(t1), self.l1 + z_tcp])

        return [p0, p1, p2, p_w, p_tcp]

    def jacobian(self, q):
        t1, t2, t3 = q
        c1, s1 = np.cos(t1), np.sin(t1)
        c2, s2 = np.cos(t2), np.sin(t2)
        c23, s23 = np.cos(t2 + t3), np.sin(t2 + t3)

        r_tot = self.l2 * c2 + self.l3 * c23 + self.grip_len

        J = np.zeros((3, 3))
        # dx
        J[0, :] = [-s1 * r_tot, c1 * (-self.l2 * s2 - self.l3 * s23), c1 * (-self.l3 * s23)]
        # dy
        J[1, :] = [c1 * r_tot, s1 * (-self.l2 * s2 - self.l3 * s23), s1 * (-self.l3 * s23)]
        # dz
        J[2, :] = [0, self.l2 * c2 + self.l3 * c23, self.l3 * c23]
        return J

    # [Reference]
    # Inverse Kinematics using Damped Least Squares (DLS) to handle singularities
    # Method: dq = inv(J*J.T + lambda*I) * J * e
    # Source: Buss (2004) & Corke (2017) "Robotics, Vision and Control"
    def ik(self, target, seed, max_iter=100, tol=1e-4):
        q = np.array(seed, dtype=float)
        for _ in range(max_iter):
            curr = self.fk(q)[-1]
            err = target - curr
            if np.linalg.norm(err) < tol: return q

            J = self.jacobian(q)
            # DLS 0.0025
            dq = np.linalg.inv(J @ J.T + 0.0025 * np.eye(3)) @ err
            q += J.T @ dq
        return q

    def mass_mat(self, q):
        t1, t2, t3 = q
        M = np.zeros((3, 3))

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
        y1 = np.array([-s1, c1, 0])

        jv2[:, 0] = np.cross(z0, p_c2);
        jw2[:, 0] = z0
        jv2[:, 1] = np.cross(y1, p_c2 - p_j1);
        jw2[:, 1] = y1

        M += self.m2 * jv2.T @ jv2 + jw2.T @ (r2 @ self.I2 @ r2.T) @ jw2

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

    def grav_tau(self, q):
        t1, t2, t3 = q
        c2 = np.cos(t2)
        c23 = np.cos(t2 + t3)

        tau3 = self.m3 * self.g * (self.l3 / 2 * c23)
        tau2 = (self.m2 * self.g * self.l2 / 2 * c2) + \
               (self.m3 * self.g * (self.l2 * c2 + self.l3 / 2 * c23))
        return np.array([0.0, tau2, tau3])

    # [Reference]
    # Numerical Differentiation for Coriolis Matrix
    # Approach: First Kind Christoffel Symbols via Finite Difference
    # Source: Modern Robotics Code Library (Lynch & Park)
    # https://github.com/NxRLab/ModernRobotics
    def coriolis(self, q, dq):
        n = len(q)
        eps = 1e-6
        M = self.mass_mat(q)
        C = np.zeros((n, n))
        dM_dq = []

        for i in range(n):
            q_p = q.copy();
            q_p[i] += eps
            M_p = self.mass_mat(q_p)
            q_m = q.copy();
            q_m[i] -= eps
            M_m = self.mass_mat(q_m)
            dM_dq.append((M_p - M_m) / (2 * eps))

        for k in range(n):
            for j in range(n):
                c_kj = 0.0
                for i in range(n):
                    term = 0.5 * (dM_dq[i][k, j] + dM_dq[j][k, i] - dM_dq[k][i, j])
                    c_kj += term * dq[i]
                C[k, j] = c_kj
        return C

    def dynamics(self, q, dq, tau):
        # M ddq + C dq + G + Fric = Tau
        M = self.mass_mat(q)
        G = self.grav_tau(q)
        C = self.coriolis(q, dq)

        tau_c = C @ dq

        # [Reference]
        # Friction Model: Viscous + Smoothed Coulomb (tanh)
        # Source: Siciliano (2009) "Robotics: Modelling, Planning and Control"
        fric = (0.5 * dq) + (0.2 * np.tanh(dq * 10))

        rhs = tau - tau_c - G - fric
        ddq = np.linalg.solve(M, rhs)
        return ddq

# [Reference]
# PID Control with Gravity Feedforward
# Source: Previous Coursework & Undergraduate Control Theory
class PID:
    def __init__(self, P, I, D, dt, max_tau):
        self.Kp = P
        self.Ki = I
        self.Kd = D
        self.dt = dt
        self.max_tau = max_tau
        self.integ = np.zeros(3)

    def update(self, target, curr, d_curr):
        err = target - curr
        p_term = self.Kp * err
        self.integ += err * self.dt
        self.integ = np.clip(self.integ, -50.0, 50.0)
        i_term = self.Ki * self.integ
        d_term = self.Kd * (0 - d_curr)

        out = p_term + i_term + d_term
        norm = np.linalg.norm(out)
        if norm > self.max_tau:
            out = out / norm * self.max_tau
        return out