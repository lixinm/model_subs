import numpy as np
from scipy.integrate import dblquad

# ========== 1. 定义效用函数（有限 n 期） ==========
def u_new(w, r, n, pN):
    """
    买新: U_new = w*(1-r^n)/(1-r) - pN
    若 r 很接近 1，则近似为 n*w - pN
    """
    if abs(r - 1.0) < 1e-12:
        return n * w - pN
    else:
        return (w * (1 - r**n)) / (1 - r) - pN

def u_used(w, r, n, pU, aU):
    """
    买二手: U_used = aU*w*(1-r^n)/(1-r) - pU
    """
    if abs(r - 1.0) < 1e-12:
        return n * aU * w - pU
    else:
        return (aU * w * (1 - r**n)) / (1 - r) - pU

def u_sub(w, r, n, pSub):
    """
    订阅: U_sub = (w - pSub)*(1-r^n)/(1-r)
    """
    if abs(r - 1.0) < 1e-12:
        return n * (w - pSub)
    else:
        return ((w - pSub) * (1 - r**n)) / (1 - r)

# ========== 2. 定义积分的 integrand(指示函数) ==========
def integrand_new(r, w, n, pN, pU, pSub, aU):
    """
    对给定 (w, r)，若 "买新" 效用 >= "买二手" & "订阅" 效用，则返回1，否则0
    注意: dblquad 调用时, 积分次序是 integrand(x, y).
    这里我们将 (r, w) 映射为 (x, y).
    也可以反过来, 只要上下限对应即可.
    """
    un = u_new(w, r, n, pN)
    uu = u_used(w, r, n, pU, aU)
    us = u_sub(w, r, n, pSub)
    return 1.0 if (un >= uu and un >= us) else 0.0

def integrand_used(r, w, n, pN, pU, pSub, aU):
    """
    对给定 (w, r)，若 "买二手" 效用 >= 其它两项，则返回1，否则0
    """
    un = u_new(w, r, n, pN)
    uu = u_used(w, r, n, pU, aU)
    us = u_sub(w, r, n, pSub)
    return 1.0 if (uu >= un and uu >= us) else 0.0

def integrand_sub(r, w, n, pN, pU, pSub, aU):
    """
    对给定 (w, r)，若 "订阅" 效用 >= 其它两项，则返回1，否则0
    """
    un = u_new(w, r, n, pN)
    uu = u_used(w, r, n, pU, aU)
    us = u_sub(w, r, n, pSub)
    return 1.0 if (us >= un and us >= uu) else 0.0

# ========== 3. 参数 & 积分计算 ==========

# 你可以根据需要修改这些参数:
n     = 5        # 考虑 5 期
pN    = 0.3      # 新品一次性成本
pU    = 0.1      # 二手一次性成本
pSub  = 0.05     # 订阅每期成本
aU    = 0.8      # 二手折扣系数, 0<aU<=1

# dblquad 的函数签名: dblquad(func, y_min, y_max, x_min, x_max),
# 其中 func(x, y). 在下例中，我们令 x 对应 w, y 对应 r.
# 所以先对 w 积分, 再对 r 积分. 你也可以交换顺序.

DemandNew,  errN  = dblquad(
    lambda w, r: integrand_new(r, w, n, pN, pU, pSub, aU),
    0, 1,  # r in [0,1]
    lambda r: 0, lambda r: 1  # w in [0,1]
)

DemandUsed, errU = dblquad(
    lambda w, r: integrand_used(r, w, n, pN, pU, pSub, aU),
    0, 1,
    lambda r: 0, lambda r: 1
)

DemandSub,  errS  = dblquad(
    lambda w, r: integrand_sub(r, w, n, pN, pU, pSub, aU),
    0, 1,
    lambda r: 0, lambda r: 1
)

print("Demand (New, Used, Sub) = ", DemandNew, DemandUsed, DemandSub)
print("Error estimates = ", errN, errU, errS)