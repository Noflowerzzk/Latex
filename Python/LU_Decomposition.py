import numpy as np

def lu_decomposition(A):
    """ 对方阵 A 进行 LU 分解，返回 L 和 U """
    n = len(A)
    L = np.eye(n)
    U = A.astype(float).copy()
    
    for i in range(n):
        for j in range(i+1, n):
            if U[i, i] == 0:
                raise ValueError("不存在 LU 分解！")
            factor = U[j, i] / U[i, i]
            L[j, i] = factor  # 存储乘数
            U[j, i:] -= factor * U[i, i:]  # 消元
            
    return L, U
142
A = np.array([[2, -1, 1],
              [3, 3, 9],
              [3, 3, 5]], dtype=float)
L, U = lu_decomposition(A)

print("L:")
print(L)
print("U:")
print(U)
