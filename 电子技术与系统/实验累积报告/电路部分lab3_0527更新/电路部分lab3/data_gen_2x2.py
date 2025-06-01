import numpy as np

count = 1  # 需要生成满足条件的矩阵组数量

while count > 0:
    # 随机生成两个 2x2 矩阵，元素范围为 0~15
    A = np.random.randint(0, 16, (2, 2))
    B = np.random.randint(0, 16, (2, 2))

    # 计算矩阵乘法
    C = np.matmul(A, B)

    # 检查结果是否在 0~255 之间
    if not np.all(C <= 255):
        continue

    # 打印结果
    count -= 1
    print("Matrix A:")
    print(A)
    print("\nMatrix B:")
    print(B)
    print("\nMatrix C = A x B:")
    print(C)
    print("\nFlattened C =", C.flatten().tolist())

    # 初始化 2x8 扩展矩阵，全为 0
    extended_A = np.zeros((2, 8), dtype=int)

    # 把 A 的每一行，依次右移 0, 1, 2 位，填入 extended 矩阵
    for i in range(2):
        extended_A[i, i:i+2] = A[i]

    print("\nextended_A:")
    print(extended_A)

    # 每列拼接三行的值成 12位二进制数（每个元素用4位表示）
    concat_A = []
    for col in range(8):
        bits = ''.join(f'{extended_A[row][col]:04b}' for row in range(2))  # 高位在上
        concat_A.append(int(bits, 2))

    print("\nconcat_A:")
    print("0: " + ' '.join(str(x) for x in concat_A))

    # 初始化 2x8 扩展矩阵，全为 0
    extended_B = np.zeros((2, 8), dtype=int)

    # 转置 B
    B_T = B.T

    # 把 B_T 的每一行，依次右移 0, 1, 2 位，填入 extended 矩阵
    for i in range(2):
        extended_B[i, i:i+2] = B_T[i]

    print("\nextended_B:")
    print(extended_B)

    # 每列拼接三行的值成 12位二进制数（每个元素用4位表示）
    concat_B = []
    for col in range(8):
        bits = ''.join(f'{extended_B[row][col]:04b}' for row in range(2))  # 高位在上
        concat_B.append(int(bits, 2))

    print("\nconcat_B:")
    print("0: " + ' '.join(str(x) for x in concat_B))
