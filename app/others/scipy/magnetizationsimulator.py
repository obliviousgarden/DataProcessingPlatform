from app.others.scipy.abstract_simulator import AbstractSimulator
import numpy as np
from app.utils.sci_const import MIU_B, MIU_0, K_B


class MagnetizationSimulator(AbstractSimulator):
    def __init__(self, model, file_path, p0=None, bounds=()):
        # 模型 1:Jiles-Atherton,模型 2:Brillouin,模型 3:Takacs,模型 4:Langevin
        super().__init__(model, file_path, p0, bounds)

    def get_data(self):
        H_list = []
        M_list = []
        # 获取数据的部分
        return H_list, M_list

    def simulate(self):
        print('开始模拟')
        H_raw, M_raw = self.get_data()
        print('获取数据完毕', H_raw, M_raw)
        # TODO：1 拆分磁化曲线的上升和下降
        # TODO：fitcurve
        # TODO：总结结果并且返回

        pass


def func_Jiles_Atherton(H, M, delta, alpha, a, c, k, Ms, Man):
    # H外加磁场；Delta外加磁场变化系数；Alpha有效磁场强度参数；a磁耦合系数；c可逆磁化系数；k非可逆损耗系数；Ms饱和磁化强度；Man滞回磁化强度

    # 不可逆磁化强度的微分(对dH)
    D_Mirr = np.divide(np.subtract(Man, M),
                       np.subtract(delta * k * (1 - c),
                                   np.multiply(alpha, np.subtract(Man, M))))
    # 等效磁场强度
    He = np.add(H, np.multiply(alpha, M))
    # 滞回磁化强度的微分(对dHe)
    D_Man = np.multiply(Ms / a,
                        1
                        - np.square(np.reciprocal(np.tanh(np.divide(He, a))))
                        + np.square(np.divide(a, He)))
    # 总磁化强度的微分(对H)
    D_M = np.divide(np.add(np.multiply(c, D_Man),
                           np.multiply(1 - c, D_Mirr)),
                    1 - alpha * c)
    # TODO:数值计算M！！！
    return D_M


def func_Brillouin(H, N, g=2.0, T=298.0):
    J = 1 / (2 * N)
    # Zeeman能量
    x = np.multiply((g * MIU_B * MIU_0 * J) / (K_B * T), H)
    # Brillouin方程
    B_J = np.subtract(np.multiply(1 + N, np.reciprocal(np.tanh(np.multiply(1 + N, x)))),
                      np.multiply(N, np.reciprocal(np.tanh(np.multiply(N, x)))))
    # 磁化强度
    M = np.multiply(N * g * MIU_B * J, B_J)
    return M


def func_Takacs(H, N, g=2.0, T=298.0):
    J = 1 / (2 * N)
    # Zeeman能量
    x = np.multiply((g * MIU_B * MIU_0 * J) / (K_B * T), H)
    a = (0.5 * (1 + 2 * J) * (1 - 0.055)) / (2 * J * (J - 0.27)) + 0.1 / (np.square(J))
    b = 0.8
    B_J = np.reciprocal(
        np.divide(np.multiply(a * np.square(J), x),
                  np.subtract(1, b * np.square(x))))
    # 磁化强度
    M = np.multiply(N * g * MIU_B * J, B_J)
    return M


def func_Langevin(H, Ms):
    # Langevin 方程
    L = np.subtract(np.reciprocal(np.tanh(H)), np.reciprocal(H))
    # 磁化强度
    M = np.multiply(Ms, L)
    return M


if __name__ == "__main__":
    # TODO:测试func是否得到正确的结果
    pass
