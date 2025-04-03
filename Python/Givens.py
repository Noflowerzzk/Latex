import numpy as np

def givens_rotation(A):
    """ 使用 Givens 变换计算矩阵 A 的 QR 分解 """
    m, n = A.shape
    Q = np.eye(m)
    R = A.copy().astype(float)

    for j in range(n):
        for i in range(m - 1, j, -1):  # 从底部向上遍历行
            if R[i, j] != 0:  # 只在非零元素时进行旋转
                a, b = R[j, j], R[i, j]
                r = np.hypot(a, b)  # 计算 sqrt(a^2 + b^2) 避免溢出
                c, s = a / r, -b / r  # 计算旋转矩阵的参数
                
                # 构造 Givens 旋转矩阵
                G = np.eye(m)
                G[[j, i], [j, i]] = c
                G[j, i], G[i, j] = -s, s

                R = G @ R
                Q = Q @ G.T

    return Q, R

# 测试
A = np.array([[2, 2, 1],
              [0, 2, 2],
              [2, 1, 2]], dtype=float)

Q, R = givens_rotation(A)

print("Q:\n", Q)
print("R:\n", R)
