import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, least_squares
from scipy.integrate import odeint
from app.utils import sci_const


class MagnetizationSimulator:
    def __init__(self, model, file_path, p0=None, bounds=()):
        self.model = model
        self.file_path = file_path
        # 一个初始尝试值的数组
        self.p0 = p0
        # 一个参数的边间turple，左边数组下界，右边数组上届
        self.bounds = bounds
        # 回线起始于1象限
        self.begin_first_quadrant = True

    def open_file(self):
        try:
            with open(self.file_path, 'r', encoding="utf-8") as file:
                lines = file.readlines()
        except UnicodeDecodeError:
            with open(self.file_path, 'r', encoding="shift_jis") as file:
                lines = file.readlines()
        return lines

    def get_data(self):
        h_list = []  # 统一用SI单位 A/m
        m_list = []  # 统一用SI单位 A/m
        lines = self.open_file()
        name = lines[2].split(',')[1]
        # length = float(lines[11].split(',')[1])  # 单位 mm
        # width = float(lines[12].split(',')[1])  # 单位 mm
        # thickness = float(lines[13].split(',')[1])  # 单位 A
        length = 5.0  # 单位 mm
        width = 5.0  # 单位 mm
        thickness = 11395.73333  # 单位 A #S070

        volume = length * width * thickness * 1e-10  # 单位 cm^3

        for line_index in range(28, lines.__len__()):
            # print(line_index, ":", lines[line_index])
            h, m = map(float, lines[line_index].split(','))
            h_list.append(h)
            m_list.append(m)
        # 单位转换 [Oe,emu] 转 [A/m,A/m]
        h_list = np.multiply(h_list, 1e3 / (4 * np.pi))
        m_list = np.multiply(np.divide(m_list, volume), 1e3)
        # 调整数据：1切割正向和负向回线2把负向回线逆时针旋转180度3每一项取平均值合并
        # 1 切割
        # 最高级别注意！！！ 这里需要进行2次切割：
        # 1次是时间梯度也就是按照磁场增减方向
        # 1次是磁场大小的正负切割，如果同时包含了正负的磁场那么在接近于0附近的时候ODEINT函数内部计算必定不收敛
        # 这里定义一个有4个数组的元组，按照顺序分别代表 H下降，H正 H下降，H负 H上升，H负 H上升，H正
        trend_break_index = 0
        sign_break_index = []
        for i in range(h_list.size):
            if (h_list[0] > h_list[1] and h_list[i] < h_list[i + 1]) or (
                    h_list[0] < h_list[1] and h_list[i] > h_list[i + 1]):
                trend_break_index = i
                break
        for i in range(h_list.size):
            if len(sign_break_index) == 2:
                break
            elif h_list[i] * h_list[i + 2] < 0:
                sign_break_index.append(i + 1)
        h1_list = []
        h2_list = []
        h3_list = []
        h4_list = []
        m1_list = []
        m2_list = []
        m3_list = []
        m4_list = []
        if sign_break_index[0] < trend_break_index < sign_break_index[1]:
            # 检查break index 的有效性
            h1_list = h_list[1:sign_break_index[0]]
            h2_list = h_list[sign_break_index[0] + 1:trend_break_index]
            h3_list = h_list[trend_break_index + 1:sign_break_index[1]]
            h4_list = h_list[sign_break_index[1] + 1:]
            m1_list = m_list[1:sign_break_index[0]]
            m2_list = m_list[sign_break_index[0] + 1:trend_break_index]
            m3_list = m_list[trend_break_index + 1:sign_break_index[1]]
            m4_list = m_list[sign_break_index[1] + 1:]
        else:
            print('BREAK INDEX 错误L:trend_break_index={0},sign_break_index_1={1},sign_break_index_2={2}'.format(
                trend_break_index, sign_break_index[0], sign_break_index[1]))
        # 2 旋转

        # 翻转第2和第4的横坐标，保证所有曲线的起点都是从m_max
        # 注意：曲线是有方向的切忌翻转，翻转后第2支和第4支全部都不收敛
        # h2_list = h2_list[::-1]
        # h4_list = h4_list[::-1]
        # m2_list = m2_list[::-1]
        # m4_list = m4_list[::-1]

        if np.mean(h1_list) > 0:
            # 第1象限开始的曲线
            self.begin_first_quadrant = True
            h2_list = np.multiply(h2_list, -1)
            h3_list = np.multiply(h3_list, -1)
            m2_list = np.multiply(m2_list, -1)
            m3_list = np.multiply(m3_list, -1)
        else:
            self.begin_first_quadrant = False
            # 第3象限开始的曲线
            h1_list = np.multiply(h1_list, -1)
            h4_list = np.multiply(h4_list, -1)
            m1_list = np.multiply(m1_list, -1)
            m4_list = np.multiply(m4_list, -1)
        # 3 拼装
        m_0_list = np.array([m1_list[0], m2_list[0], m3_list[0], m4_list[0]])
        break_index_list = np.array(
            [h1_list.size, h1_list.size + h2_list.size, h1_list.size + h2_list.size + h3_list.size,
             h1_list.size + h2_list.size + h3_list.size + h4_list.size])
        h_list = np.concatenate([h1_list, h2_list, h3_list, h4_list])
        m_list = np.concatenate([m1_list, m2_list, m3_list, m4_list])

        return h_list, m_list, m_0_list, break_index_list

    def simulate(self, g_j=1.33, temp=300.):
        print('开始模拟')
        h_raw, m_raw, m_0_list, break_index_list = self.get_data()
        print('获取数据完毕', h_raw, m_raw)
        # 模型 1:Jiles-Atherton, 模型 2:Brillouin, 模型 3:Langevin, 模型 4:Takacs
        if self.model == 4:
            print('Takecs模型暂时不可用')
            return {'popt': [], 'pcov': [], 'perr': []}
        popt, pcov = curve_fit(
            {1: Jiles_Atherton_Model(m_0_list=m_0_list, break_index_list=break_index_list).func_Jiles_Atherton,
             2: Brillouin_Model(g_j=g_j, temp=temp).func_Brillouin,
             3: Langevin_Model(g_j=g_j, temp=temp).func_Langevin,
             4: Takacs_Model(g_j=g_j, temp=temp, break_index_list=break_index_list).func_Takacs}.get(self.model)
            , h_raw
            , m_raw
            , p0=self.p0
            , bounds=self.bounds
            , method='trf')

        perr = np.sqrt(np.diag(pcov))
        print(popt, pcov, perr)
        # 作图
        # h, a, alpha, c, m_s, k
        m_cal = []
        if self.model == 1:
            m_cal = Jiles_Atherton_Model(m_0_list=m_0_list, break_index_list=break_index_list).func_Jiles_Atherton(
                h_raw, popt[0], popt[1], popt[2], popt[3], popt[4])

        elif self.model == 2:
            m_cal = Brillouin_Model(g_j=g_j, temp=temp).func_Brillouin(h_raw, popt[0], popt[1])

        elif self.model == 3:
            m_cal = Langevin_Model(g_j=g_j, temp=temp).func_Langevin(h_raw, popt[0], popt[1])

        elif self.model == 4:
            # m_cal = Takacs_Model(g_j=g_j,temp=temp).func_Takacs(h_raw, popt[0], popt[1])
            m_cal = Takacs_Model(g_j=g_j, temp=temp, break_index_list=break_index_list).func_Takacs(h_raw, popt[0],
                                                                                                    popt[1])

        ax1 = plt.subplot(2, 1, 1)
        ax2 = plt.subplot(2, 1, 2)
        plt.sca(ax1)
        # 把4支各自分割旋转恢复回去
        if self.begin_first_quadrant:
            # 第1象限开始的曲线
            h_raw[break_index_list[0]:break_index_list[1]] = np.multiply(h_raw[break_index_list[0]:break_index_list[1]],
                                                                         -1)
            h_raw[break_index_list[1]:break_index_list[2]] = np.multiply(h_raw[break_index_list[1]:break_index_list[2]],
                                                                         -1)
            m_raw[break_index_list[0]:break_index_list[1]] = np.multiply(m_raw[break_index_list[0]:break_index_list[1]],
                                                                         -1)
            m_raw[break_index_list[1]:break_index_list[2]] = np.multiply(m_raw[break_index_list[1]:break_index_list[2]],
                                                                         -1)
            m_cal[break_index_list[0]:break_index_list[1]] = np.multiply(m_cal[break_index_list[0]:break_index_list[1]],
                                                                         -1)
            m_cal[break_index_list[1]:break_index_list[2]] = np.multiply(m_cal[break_index_list[1]:break_index_list[2]],
                                                                         -1)
        else:
            # 第3象限开始的曲线
            h_raw[:break_index_list[0]] = np.multiply(h_raw[:break_index_list[0]], -1)
            h_raw[break_index_list[2]:] = np.multiply(h_raw[break_index_list[2]:], -1)
            m_raw[:break_index_list[0]] = np.multiply(m_raw[:break_index_list[0]], -1)
            m_raw[break_index_list[2]:] = np.multiply(m_raw[break_index_list[2]:], -1)
            m_cal[:break_index_list[0]] = np.multiply(m_cal[:break_index_list[0]], -1)
            m_cal[break_index_list[2]:] = np.multiply(m_cal[break_index_list[2]:], -1)
        plt.plot(h_raw, m_raw)
        plt.plot(h_raw, m_cal)
        plt.sca(ax2)
        plt.plot(h_raw, np.gradient(m_raw))
        plt.plot(h_raw, np.gradient(m_cal))
        plt.show()

        return {'popt': popt, 'pcov': pcov, 'perr': perr}


class Jiles_Atherton_Model:
    def __init__(self, m_0_list, break_index_list):
        self.m_0_list = m_0_list
        self.break_index_list = break_index_list

    def func_Jiles_Atherton(self, h, a, alpha, c, m_s, k):
        # 参数含义:
        # h External Magnetic Field
        # a Magnetic Coupling Coefficient
        # alpha Quanties Interdomain Coupling in the Magnetic Material
        # c Reversible Magnetization Coefficient, c=1 means Paramagnetism
        # m_s Saturation Magnetization
        # k Irreversible Loss Coefficient
        print("a={0},alpha={1}.c={2},m_s={3},k={4},m_0_list={5}".format(a, alpha, c, m_s, k, self.m_0_list))

        # 分别对4支函数进行计算
        # 第1支 下降支
        (m1_result, m1_info_dict) = odeint(self.func_dm_dh,
                                           self.m_0_list[0],
                                           h[0:self.break_index_list[0]],
                                           args=(a, alpha, c, m_s, k, -1),
                                           printmessg=True,
                                           full_output=True)
        # 第2支 上升支
        (m2_result, m2_info_dict) = odeint(self.func_dm_dh,
                                           self.m_0_list[1],
                                           h[self.break_index_list[0]:self.break_index_list[1]],
                                           args=(a, alpha, c, m_s, k, 1),
                                           printmessg=True,
                                           full_output=True)
        # 第3支 下降支
        (m3_result, m3_info_dict) = odeint(self.func_dm_dh,
                                           self.m_0_list[2],
                                           h[self.break_index_list[1]:self.break_index_list[2]],
                                           args=(a, alpha, c, m_s, k, -1),
                                           printmessg=True,
                                           full_output=True)
        # 第4支 上升支
        (m4_result, m4_info_dict) = odeint(self.func_dm_dh,
                                           self.m_0_list[3],
                                           h[self.break_index_list[2]:],
                                           args=(a, alpha, c, m_s, k, 1),
                                           printmessg=True,
                                           full_output=True)
        # 注意这里的m是一个数组的数组是一个二维数组必须进行降维
        m1_list = list(np.concatenate(m1_result.reshape((-1, 1), order="F")))
        m2_list = list(np.concatenate(m2_result.reshape((-1, 1), order="F")))
        m3_list = list(np.concatenate(m3_result.reshape((-1, 1), order="F")))
        m4_list = list(np.concatenate(m4_result.reshape((-1, 1), order="F")))

        print(np.max(m1_list))
        print(np.max(m2_list))
        print(np.max(m3_list))
        print(np.max(m4_list))

        m_result = np.concatenate([m1_list, m2_list, m3_list, m4_list])
        return m_result

    def func_dm_dh(m, h, a, alpha, c, m_s, k, delta):
        # dm/dh= (分子1+分子2)/分母
        # 这个方程的详细说明：这是一个用来描述磁场H和磁化强度M的关系公式
        # 磁化强度分为两部分可逆M_rev和不可逆M_irr
        # M = M_irr + M_rev
        # 其中M_rev = c*(M_an-M_irr) c=1时，则为顺磁 [所以c代表了M中顺磁的比例100%时 c=1]
        #  (注意：M_an是非滞后磁化c=1意味着M = M_an 也就是说磁化强度全是非滞后磁化，即顺磁)
        # 所以 M = c*M_an + (1-c)*M_irr
        # 其中M_an是由M_S和系数a直接确定的，a代表了磁耦合的强度
        # 其中M_irr是由微分方程得到的dM_irr/dH，最后演化为了这里的微分方程dM/dH

        # 分母
        denominator = 1 - alpha * c
        # 分子1
        term_1 = np.divide(np.add(h, np.multiply(alpha, m)), a)
        numerator_1 = np.multiply(c * m_s / a,
                                  np.add(
                                      np.subtract(1,
                                                  np.power(
                                                      np.reciprocal(np.tanh(term_1)),
                                                      2)),
                                      np.power(term_1, 2)
                                  ))
        # 分子2
        h_e = np.add(h, np.multiply(alpha, m))
        m_an = np.multiply(m_s,
                           np.subtract(
                               np.reciprocal(np.tanh(np.divide(h_e, a))),
                               np.divide(a, h_e)
                           ))
        term_2 = np.subtract(m_an, m)
        numerator_2 = np.multiply(1 - c, np.divide(term_2,
                                                   np.subtract(
                                                       delta * k * (1 - c),
                                                       np.multiply(alpha, term_2)
                                                   )))
        dm_dh = np.divide(np.add(numerator_1, numerator_2), denominator)
        return dm_dh


class Brillouin_Model:
    def __init__(self, g_j, temp):
        self.g_j = g_j
        self.temp = temp

    def func_Brillouin(self, h, m_s, j):
        # 参数含义:
        # h External Magnetic Field
        # m_s Saturation Magnetization
        # j Total Angular Momentum
        # temp Temperature
        print("g_j={0},temp={1},m_s={2},j={3}".format(self.g_j, self.temp, m_s, j))
        b = np.multiply(sci_const.miu_0, h)
        x = np.divide(np.multiply(self.g_j * sci_const.miu_B, b), sci_const.k_B * self.temp)
        b_j = np.subtract(np.multiply((2 * j + 1) / (2 * j),
                                      np.reciprocal(np.tanh(np.multiply((2 * j + 1) / 2, x)))),
                          np.multiply(1 / (2 * j),
                                      np.reciprocal(np.tanh(np.multiply(1 / 2, x)))))
        m = np.multiply(m_s, b_j)
        return m


class Langevin_Model:
    def __init__(self, g_j, temp):
        self.g_j = g_j
        self.temp = temp

    def func_Langevin(self, h, m_s, j):
        # 参数含义:
        # h External Magnetic Field
        # m_s Saturation Magnetization
        # j Total Angular Momentum
        # temp Temperature
        print("g_j={0},temp={1},m_s={2},j={3}".format(self.g_j, self.temp, m_s, j))
        b = np.multiply(sci_const.miu_0, h)
        x = np.divide(np.multiply(self.g_j * sci_const.miu_B, b), sci_const.k_B * self.temp)
        l_x = np.subtract(np.reciprocal(np.tanh(np.multiply(j, x))),
                          np.reciprocal(np.multiply(j, x)))
        m = np.multiply(m_s, l_x)
        return m


class Takacs_Model:
    def __init__(self, g_j, temp, break_index_list):
        self.g_j = g_j
        self.temp = temp
        self.break_index_list = break_index_list

    def func_Takacs(self, h, m_s, j):
        # 参数含义:
        # h External Magnetic Field
        # m_s Saturation Magnetization
        # j Total Angular Momentum
        # temp Temperature
        print("g_j={0},temp={1},m_s={2},j={3}".format(self.g_j, self.temp, m_s, j))

        b = np.multiply(sci_const.miu_0, h)
        x = np.divide(np.multiply(self.g_j * sci_const.miu_B, b), sci_const.k_B * self.temp)
        const_a = (0.5 * (1 + 2 * j) * (1 - 0.055)) / (2 * j * (j - 0.27)) + 0.1 / (j * j)
        const_b = 0.8
        b_j = np.reciprocal(np.divide(np.multiply(const_a * j * j, x),
                                      np.subtract(1,
                                                  np.multiply(const_b, np.multiply(x, x)))))
        m = np.multiply(m_s, b_j)
        return m


if __name__ == '__main__':
    #  J-A Model
    # a, alpha, c, m_s, k, delta
    # a=76434.33625750794,alpha=0.17716254982042273.c=0.0002219543796726332,m_s=647678.7831977553,k=2319.3791455301916
    # magnetization_simulator = MagnetizationSimulator(
    #     model=1,
    #     file_path="D:\\PycharmProjects\\AIProcessingPlatform\\app\\ui\\V210705112208.Csv",
    #     p0=[76434, 0.17716254982042273, 0.0002219543796726332, 647678, 2319],
    #     bounds=([1e3, 0, 0.00000001, 1e4, 1e3], [1e5, 0.3, 0.01, 1e7, 1e5])
    # )

    # B Model
    # m_s, j
    g_j = sci_const.Lande_g_Factor(3 / 2, 3, 9 / 2)  # Co2+ ion
    magnetization_simulator = MagnetizationSimulator(
        model=2,
        file_path="D:\\PycharmProjects\\AIProcessingPlatform\\app\\ui\\V210705112208.Csv",
        p0=[6e6, 4500],
        bounds=([1e4, 1], [1e8, 1e10])
    )
    magnetization_simulator.simulate(g_j=g_j, temp=300)

    # L Model
    # m_s
    # g_j = sci_const.Lande_g_Factor(3/2,3,9/2) # Co2+ ion
    # magnetization_simulator = MagnetizationSimulator(
    #     model=3,
    #     file_path="D:\\PycharmProjects\\AIProcessingPlatform\\app\\ui\\V210705112208.Csv",
    #     p0=[6e6,4500],
    #     bounds=([1e4,1], [1e8, 1e10])
    # )
    # magnetization_simulator.simulate(g_j=g_j,temp=300)

    # T Model (主要适用于x不为零且有一定大小的情况下，暂时从前端封住self.model不准为4)
    # m_s, j
    # g_j = sci_const.Lande_g_Factor(3/2,3,9/2) # Co2+ ion
    # magnetization_simulator = MagnetizationSimulator(
    #     model=4,
    #     file_path="D:\\PycharmProjects\\AIProcessingPlatform\\app\\ui\\V210705112208.Csv",
    #     p0=[1e7,0.5],
    #     bounds=([1e4,0.1], [1e20, 100])
    # )
    # magnetization_simulator.simulate(g_j=g_j,temp=300)
