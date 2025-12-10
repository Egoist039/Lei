import sympy as sp
import numpy as np


def generate_analytical_code():
    # 1. 定义符号
    q1, q2, q3 = sp.symbols('q[0] q[1] q[2]', real=True)
    dq1, dq2, dq3 = sp.symbols('dq[0] dq[1] dq[2]', real=True)

    # 物理参数符号 (对应你的 robot.py)
    l1, l2, l3 = sp.symbols('self.l1 self.l2 self.l3', real=True)
    m2, m3 = sp.symbols('self.m2 self.m3', real=True)

    # 为了简化，我们假设简单的惯量 (如果需要精确对应 robot.py 的数值矩阵，公式会极长)
    # 这里我们演示结构。实际上 robot.py 里用的是数值计算 M。
    # 我们可以用符号推导 M 对 q 的偏导数。

    print("正在进行符号推导，请稍候...")

    # 定义旋转矩阵 (符号化)
    c1, s1 = sp.cos(q1), sp.sin(q1)
    c2, s2 = sp.cos(q2), sp.sin(q2)
    c3, s3 = sp.cos(q3), sp.sin(q3)

    R1 = sp.Matrix([[c1, -s1, 0], [s1, c1, 0], [0, 0, 1]])
    Ry2 = sp.Matrix([[c2, 0, s2], [0, 1, 0], [-s2, 0, c2]])
    R2 = R1 * Ry2
    Ry3 = sp.Matrix([[c3, 0, s3], [0, 1, 0], [-s3, 0, c3]])
    R3 = R2 * Ry3

    # 质心位置 (符号化)
    # 注意：robot.py 里 r_c2 = [l2/2, 0, 0]
    p_j1 = sp.Matrix([0, 0, l1])
    p_c2 = p_j1 + R2 * sp.Matrix([l2 / 2, 0, 0])

    p_j2 = p_j1 + R2 * sp.Matrix([l2, 0, 0])
    p_c3 = p_j2 + R3 * sp.Matrix([l3 / 2, 0, 0])

    # 雅可比 (符号化)
    z0 = sp.Matrix([0, 0, 1])
    y1 = R1.col(1)  # R1 的第2列是旋转后的Y轴
    y2 = R2.col(1)

    # Jv2
    Jv2_1 = z0.cross(p_c2)
    Jv2_2 = y1.cross(p_c2 - p_j1)
    Jv2_3 = sp.Matrix([0, 0, 0])
    Jv2 = sp.Matrix.hstack(Jv2_1, Jv2_2, Jv2_3)

    # Jv3
    Jv3_1 = z0.cross(p_c3)
    Jv3_2 = y1.cross(p_c3 - p_j1)
    Jv3_3 = y2.cross(p_c3 - p_j2)
    Jv3 = sp.Matrix.hstack(Jv3_1, Jv3_2, Jv3_3)

    # 简化动能计算：只考虑质点动能 (平动)
    # 因为加上转动惯量后公式太长，但结构是一样的
    # K = 0.5 * m * v.T * v
    M_trans = m2 * Jv2.T * Jv2 + m3 * Jv3.T * Jv3
    M = sp.simplify(M_trans)

    # === 核心：解析法求 Christoffel 符号 ===
    q = [q1, q2, q3]
    dq_vec = [dq1, dq2, dq3]
    n = 3
    C = sp.zeros(n, n)

    for k in range(n):
        for j in range(n):
            val = 0
            for i in range(n):
                # c_ijk 公式
                term = 0.5 * (sp.diff(M[k, j], q[i]) + sp.diff(M[k, i], q[j]) - sp.diff(M[i, j], q[k]))
                val += term * dq_vec[i]
            C[k, j] = sp.simplify(val)

    print("\n=== 生成的解析解代码 (Coriolis Matrix) ===\n")
    print("def compute_coriolis_analytical(self, q, dq):")
    print("    q0, q1, q2 = q")  # 注意 robot.py 用的是 t1,t2,t3
    print("    dq0, dq1, dq2 = dq")
    print("    s1, c1 = np.sin(q0), np.cos(q0)")  # q0=t1
    print("    s2, c2 = np.sin(q1), np.cos(q1)")  # q1=t2
    print("    s3, c3 = np.sin(q2), np.cos(q2)")  # q2=t3
    print("    s23 = np.sin(q1 + q2)")
    print("    c23 = np.cos(q1 + q2)")

    for i in range(3):
        for j in range(3):
            expr = C[i, j]
            # 简单的字符串替换，把 sympy 格式转成 python 格式
            code = str(expr).replace("sin", "np.sin").replace("cos", "np.cos")
            print(f"    C{i}{j} = {code}")

    print("\n    C = np.array([")
    print("        [C00, C01, C02],")
    print("        [C10, C11, C12],")
    print("        [C20, C21, C22]")
    print("    ])")
    print("    return C")


if __name__ == "__main__":
    generate_analytical_code()