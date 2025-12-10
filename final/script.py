import sympy as sp


def generate_full_analytical_code():
    print("⏳ 正在进行全矩阵符号推导，请稍候...")

    # 1. 定义符号
    q1, q2, q3 = sp.symbols('t1 t2 t3', real=True)
    dq1, dq2, dq3 = sp.symbols('dt1 dt2 dt3', real=True)
    q = sp.Matrix([q1, q2, q3])
    dq = sp.Matrix([dq1, dq2, dq3])

    # 物理参数
    l1, l2, l3 = sp.symbols('self.l1 self.l2 self.l3', real=True, positive=True)
    m2, m3 = sp.symbols('self.m2 self.m3', real=True, positive=True)

    # === [修复部分] ===
    # 使用 sp.Symbol 单独定义，防止被逗号分割
    I2xx = sp.Symbol('self.I2[0,0]', real=True)
    I2yy = sp.Symbol('self.I2[1,1]', real=True)
    I2zz = sp.Symbol('self.I2[2,2]', real=True)

    I3xx = sp.Symbol('self.I3[0,0]', real=True)
    I3yy = sp.Symbol('self.I3[1,1]', real=True)
    I3zz = sp.Symbol('self.I3[2,2]', real=True)

    I2_loc = sp.diag(I2xx, I2yy, I2zz)
    I3_loc = sp.diag(I3xx, I3yy, I3zz)

    # 质心位置 (对应 robot.py: r_c2=[l2/2,0,0])
    rc2 = sp.Matrix([l2 / 2, 0, 0])
    rc3 = sp.Matrix([l3 / 2, 0, 0])

    # 2. 运动学
    c1, s1 = sp.cos(q1), sp.sin(q1)
    R1 = sp.Matrix([[c1, -s1, 0], [s1, c1, 0], [0, 0, 1]])

    c2, s2 = sp.cos(q2), sp.sin(q2)
    Ry2 = sp.Matrix([[c2, 0, s2], [0, 1, 0], [-s2, 0, c2]])
    R2 = R1 * Ry2

    c3, s3 = sp.cos(q3), sp.sin(q3)
    Ry3 = sp.Matrix([[c3, 0, s3], [0, 1, 0], [-s3, 0, c3]])
    R3 = R2 * Ry3

    # 雅可比
    z0 = sp.Matrix([0, 0, 1])
    y1 = R1.col(1)
    y2 = R2.col(1)

    p_j1 = sp.Matrix([0, 0, l1])
    p_c2 = p_j1 + R2 * rc2
    p_j2 = p_j1 + R2 * sp.Matrix([l2, 0, 0])
    p_c3 = p_j2 + R3 * rc3

    # 简化：只计算平动动能 + 关键的转动动能
    # Jv
    Jv2 = sp.Matrix.hstack(z0.cross(p_c2), y1.cross(p_c2 - p_j1), sp.zeros(3, 1))
    Jv3 = sp.Matrix.hstack(z0.cross(p_c3), y1.cross(p_c3 - p_j1), y2.cross(p_c3 - p_j2))

    # Jw
    Jw2 = sp.Matrix.hstack(z0, y1, sp.zeros(3, 1))
    Jw3 = sp.Matrix.hstack(z0, y1, y2)

    # 3. 能量与质量矩阵 M
    # I_world
    I2_w = R2 * I2_loc * R2.T
    I3_w = R3 * I3_loc * R3.T

    # K = 0.5 * v.T * M * v
    M_val = m2 * (Jv2.T * Jv2) + m3 * (Jv3.T * Jv3) + (Jw2.T * I2_w * Jw2) + (Jw3.T * I3_w * Jw3)
    M = sp.simplify(M_val)

    # 4. Christoffel 符号求 C 矩阵
    n = 3

    print("🚀 正在生成代码...")
    print("-" * 60)
    print("def compute_coriolis_matrix_analytical(self, q, dq):")
    print("    \"\"\"由 SymPy 自动生成的解析解 (包含转动惯量)\"\"\"")
    print("    t1, t2, t3 = q")
    print("    dt1, dt2, dt3 = dq")
    print("    import numpy as np")
    print("    sin = np.sin")
    print("    cos = np.cos")
    print("")

    # 遍历计算每一个非零元素
    for k in range(n):
        for j in range(n):
            term_sum = 0
            for i in range(n):
                # c_ijk
                c_ijk = 0.5 * (sp.diff(M[k, j], q[i]) + sp.diff(M[k, i], q[j]) - sp.diff(M[i, j], q[k]))
                term_sum += c_ijk * dq[i]

            # 化简并打印
            final_term = sp.simplify(term_sum)

            if final_term == 0:
                print(f"    C{k}{j} = 0.0")
            else:
                # 转换成 python 代码格式
                code_str = str(final_term)
                # 替换 python 无法识别的符号
                code_str = code_str.replace("sin", "np.sin").replace("cos", "np.cos")
                print(f"    C{k}{j} = {code_str}")

    print("")
    print("    C = np.array([")
    print("        [C00, C01, C02],")
    print("        [C10, C11, C12],")
    print("        [C20, C21, C22]")
    print("    ])")
    print("    return C")
    print("-" * 60)


if __name__ == "__main__":
    generate_full_analytical_code()


    # def compute_coriolis_matrix(self, q, dq):
    #     """
    #     [解析法] 计算科氏力与离心力矩阵 C(q, dq)
    #     代码由 SymPy 符号推导自动生成，针对当前 Robot3DoF_Spatial 构型优化。
    #     效率极高，精确包含转动惯量影响。
    #     """
    #     # 1. 解包变量
    #     t1, t2, t3 = q
    #     dt1, dt2, dt3 = dq
    #
    #     # 2. 引入 numpy (如果文件顶部已经 import numpy as np，这里其实可以省略)
    #     # 为了防止命名空间污染，直接使用 np.sin / np.cos
    #
    #     # 3. 直接代入解析公式
    #     # -------------------------------------------------------------------------
    #     # Row 1 (对应 C[0, :])
    #     # -------------------------------------------------------------------------
    #     C00 = -dt2 * (-1.0 * self.I2[0, 0] * np.sin(2 * t2) + 1.0 * self.I2[2, 2] * np.sin(2 * t2) - 1.0 * self.I3[
    #         0, 0] * np.sin(2 * t2 + 2 * t3) + 1.0 * self.I3[2, 2] * np.sin(
    #         2 * t2 + 2 * t3) + 0.25 * self.l2 ** 2 * self.m2 * np.sin(2 * t2) + 0.25 * self.m3 * (
    #                               4 * self.l2 ** 2 * np.sin(2 * t2) + 4 * self.l2 * self.l3 * np.sin(
    #                           2 * t2 + t3) + self.l3 ** 2 * np.sin(2 * t2 + 2 * t3))) / 2 - dt3 * (
    #                       -1.0 * self.I3[0, 0] * np.cos(t2 + t3) + 1.0 * self.I3[2, 2] * np.cos(
    #                   t2 + t3) + 0.25 * self.l3 * self.m3 * (
    #                                   2 * self.l2 * np.cos(t2) + self.l3 * np.cos(t2 + t3))) * np.sin(t2 + t3)
    #
    #     C01 = -dt1 * (-0.5 * self.I2[0, 0] * np.sin(2 * t2) + 0.5 * self.I2[2, 2] * np.sin(2 * t2) - 0.5 * self.I3[
    #         0, 0] * np.sin(2 * t2 + 2 * t3) + 0.5 * self.I3[2, 2] * np.sin(
    #         2 * t2 + 2 * t3) + 0.125 * self.l2 ** 2 * self.m2 * np.sin(2 * t2) + 0.25 * self.m3 * (
    #                               2 * self.l2 ** 2 * np.sin(2 * t2) + 2 * self.l2 * self.l3 * np.sin(
    #                           2 * t2 + t3) + self.l3 ** 2 * np.sin(2 * t2 + 2 * t3) / 2))
    #
    #     C02 = -dt1 * (-1.0 * self.I3[0, 0] * np.cos(t2 + t3) + 1.0 * self.I3[2, 2] * np.cos(
    #         t2 + t3) + 0.25 * self.l3 * self.m3 * (2 * self.l2 * np.cos(t2) + self.l3 * np.cos(t2 + t3))) * np.sin(
    #         t2 + t3)
    #
    #     # -------------------------------------------------------------------------
    #     # Row 2 (对应 C[1, :])
    #     # -------------------------------------------------------------------------
    #     C10 = dt1 * (-0.5 * self.I2[0, 0] * np.sin(2 * t2) + 0.5 * self.I2[2, 2] * np.sin(2 * t2) - 0.5 * self.I3[
    #         0, 0] * np.sin(2 * t2 + 2 * t3) + 0.5 * self.I3[2, 2] * np.sin(
    #         2 * t2 + 2 * t3) + 0.125 * self.l2 ** 2 * self.m2 * np.sin(2 * t2) + 0.25 * self.m3 * (
    #                              2 * self.l2 ** 2 * np.sin(2 * t2) + 2 * self.l2 * self.l3 * np.sin(
    #                          2 * t2 + t3) + self.l3 ** 2 * np.sin(2 * t2 + 2 * t3) / 2))
    #
    #     C11 = -0.5 * dt3 * self.l2 * self.l3 * self.m3 * np.sin(t3)
    #
    #     C12 = 0.5 * self.l2 * self.l3 * self.m3 * (-dt2 - dt3) * np.sin(t3)
    #
    #     # -------------------------------------------------------------------------
    #     # Row 3 (对应 C[2, :])
    #     # -------------------------------------------------------------------------
    #     C20 = dt1 * (-1.0 * self.I3[0, 0] * np.cos(t2 + t3) + 1.0 * self.I3[2, 2] * np.cos(
    #         t2 + t3) + 0.25 * self.l3 * self.m3 * (2 * self.l2 * np.cos(t2) + self.l3 * np.cos(t2 + t3))) * np.sin(
    #         t2 + t3)
    #
    #     C21 = 0.5 * dt2 * self.l2 * self.l3 * self.m3 * np.sin(t3)
    #
    #     C22 = 0.0
    #
    #     # 4. 组装矩阵
    #     C = np.array([
    #         [C00, C01, C02],
    #         [C10, C11, C12],
    #         [C20, C21, C22]
    #     ])
    #
    #     return C