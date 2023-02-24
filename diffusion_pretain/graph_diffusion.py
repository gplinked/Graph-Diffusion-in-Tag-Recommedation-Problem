import scipy.sparse as sp
import numpy as np


def gdc(A: sp.csr_matrix, alpha: float, eps: float):
    N = A.shape[0]

    # 自循环I+A
    A_loop = sp.eye(N) + A

    # 生成对称转移矩阵
    D_loop_vec = A_loop.sum(0).A1
    D_loop_vec_invsqrt = 1 / np.sqrt(D_loop_vec)
    # 对角化矩阵
    D_loop_invsqrt = sp.diags(D_loop_vec_invsqrt)
    # 转移矩阵为（I+D)^-1/2 (I+A)（I+D)^-1/2
    T_sym = D_loop_invsqrt @ A_loop @ D_loop_invsqrt

    # 图扩散，alpha为PPR预定义参数 S=alpha∑((1-a)T)^k
    S = alpha * sp.linalg.inv(sp.eye(N) - (1 - alpha) * T_sym)

    # 用eps阈值进行稀疏化
    S_tilde = S.multiply(S >= eps)

    # 形成列归一化转移矩阵
    D_tilde_vec = S_tilde.sum(0).A1
    T_S = S_tilde / D_tilde_vec

    return T_S


