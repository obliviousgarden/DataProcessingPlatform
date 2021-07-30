import constant
import math

constant.e = 1.602176634 * 1e-19
e = constant.e
constant.h = 6.62607015 * 1e-34
h = constant.h
constant.h_bar = h/(2*math.pi)
h_bar = constant.h_bar
constant.m_e = 9.1093837015 * 1e-31
m_e = constant.m_e
constant.miu_0 = math.pi * 4e-7
miu_0 = constant.miu_0
constant.k_B = 1.38064852e-23
k_B = constant.k_B
constant.g_L = 1
g_L = constant.g_L
constant.g_S = 2.0023193043768
g_S = constant.g_S
constant.miu_B = e*h_bar/(2*m_e)
miu_B = constant.miu_B


def Lande_g_Factor(s, l, j):
    factor_s = s * (s + 1)
    factor_l = l * (l + 1)
    factor_j = j * (j + 1)
    g_j = g_L * (factor_j - factor_s + factor_l) / (2 * factor_j) + g_S * (factor_j + factor_s - factor_l) / (
                2 * factor_j)
    return g_j
