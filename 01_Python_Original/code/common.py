# common.py
# 存放通用的数值算法：追赶法（TDMA）求解三对角方程组
import numpy as np


def tdma(a, b, c, d):
    """
    求解三对角方程组 a[i]x[i-1] + b[i]x[i] + c[i]x[i+1] = d[i]
    输入：
        a: 下对角线，长度为 n（a[0] 通常为 0）
        b: 主对角线，长度为 n
        c: 上对角线，长度为 n（c[n-1] 通常为 0）
        d: 右端常数项，长度为 n
    输出：
        x: 解向量
    """
    n = len(d)
    cp = np.zeros(n)
    dp = np.zeros(n)

    # 向前消元
    cp[0] = c[0] / b[0]
    dp[0] = d[0] / b[0]
    for i in range(1, n):
        m = b[i] - a[i] * cp[i - 1]
        if i < n - 1:
            cp[i] = c[i] / m
        else:
            cp[i] = 0.0
        dp[i] = (d[i] - a[i] * dp[i - 1]) / m

    # 向后回代
    x = np.zeros(n)
    x[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]

    return x