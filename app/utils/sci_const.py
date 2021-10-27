import constant
import math
import numpy as np

constant.e = 1.602176634 * 1e-19
e = constant.e
constant.h = 6.62607015 * 1e-34
h = constant.h
constant.h_bar = h / (2 * math.pi)
h_bar = constant.h_bar
constant.m_e = 9.1093837015 * 1e-31
m_e = constant.m_e
constant.miu_0 = math.pi * 4e-7
miu_0 = constant.miu_0
constant.epsilon_0 = 8.8541878128 * 1e-12
epsilon_0 = constant.epsilon_0
constant.c = np.reciprocal(np.sqrt(np.multiply(epsilon_0, miu_0)))
c = constant.c
constant.k_B = 1.38064852e-23
k_B = constant.k_B
constant.g_L = 1
g_L = constant.g_L
constant.g_S = 2.0023193043768
g_S = constant.g_S
constant.miu_B = e * h_bar / (2 * m_e)
miu_B = constant.miu_B
constant.magnetization_unit_list = ["Tesla(T)", "Gauss(G)", "kiloGauss(kG)", "Oersted(Oe)", "kiloOersted(kOe)", "A/m",
                                    "emu/cm^3"]
magnetization_unit_list = constant.magnetization_unit_list
constant.magnetization_si_conversion_factor_dict = {'Tesla(T)>>>A/m': 1.e4 / (4 * np.pi),
                                                    'Gauss(G)>>>A/m': 1.e3 / (4 * np.pi),
                                                    'kiloGauss(kG)>>>A/m': 1. / (4 * np.pi),
                                                    'Oersted(Oe)>>>A/m': 1.e3 / (4 * np.pi),
                                                    'kiloOersted(kOe)>>>A/m': 1. / (4 * np.pi),
                                                    'A/m>>>A/m': 1.,
                                                    'emu/cm^3>>>A/m': 1.e3}
magnetization_si_conversion_factor_dict = constant.magnetization_si_conversion_factor_dict


def Lande_g_Factor(s, l, j):
    factor_s = s * (s + 1)
    factor_l = l * (l + 1)
    factor_j = j * (j + 1)
    g_j = g_L * (factor_j - factor_s + factor_l) / (2 * factor_j) + g_S * (factor_j + factor_s - factor_l) / (
            2 * factor_j)
    return g_j


def n_to_epsilon(n, k):
    epsilon = np.square(np.add(n, np.multiply(k, 1.j)))
    epsilon_1 = np.real(epsilon)
    epsilon_2 = np.imag(epsilon)
    return epsilon_1, epsilon_2

