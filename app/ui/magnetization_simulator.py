import time

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, least_squares
from scipy.integrate import odeint, quad
from scipy.stats import lognorm
from app.utils import sci_const
from app.utils.science_unit import ScienceUnit,science_unit_convert
from app.utils.science_base import PhysicalQuantity
from scipy import interpolate
from joblib import Parallel, delayed
# from numba import cfunc


class MagnetizationSimulator:
    def __init__(self,model:int,bg_file_path:str,g_j:float,temp:float,distribution_flag:bool,first_pos_info_tuple=(),p0=[],bounds=(),size_dict={}):
        self.model = model
        self.bg_file_path = bg_file_path
        # 一个初始尝试值的数组
        self.p0 = p0
        # 一个参数的边间tuple，左边数组下界，右边数组上届
        self.bounds = bounds
        # 回线起始于1象限
        self.begin_first_quadrant = True
        # 接受并解析第1个数据点的位置信息及其单位信息first_pos_info_tuple
        ((self.H_start_row,self.H_start_col,self.H_start_unit),(self.M_start_row,self.M_start_col,self.M_start_unit)) = first_pos_info_tuple
        self.g_j = g_j
        self.temp = temp
        self.distribution_flag = distribution_flag
        self.size_dict = size_dict
        print("Instantiation: model:{}, bg_file_path:{}, p0={}, bounds={}, g_j={}, temp={}, distribution_flag={}, first_pos_info=H[{},{}]({})&M[{},{}]({}), size_dict={}."
              .format(self.model,self.bg_file_path,self.p0,self.bounds, self.g_j, self.temp, self.distribution_flag,self.H_start_row,self.H_start_col,self.H_start_unit,self.M_start_row,self.M_start_col,self.M_start_unit,size_dict))

    def open_file(self,path):
        try:
            with open(path, 'r', encoding="utf-8") as file:
                lines = file.readlines()
        except UnicodeDecodeError:
            with open(path, 'r', encoding="shift_jis") as file:
                lines = file.readlines()
        return lines

    def get_data(self, file_name:str, file_path:str):
        h_list = []
        bg_m_list = []
        m_list = []
        bg_lines = self.open_file(path = self.bg_file_path)
        lines = self.open_file(path = file_path)
        length = self.size_dict[file_name][0]  # 单位 mm
        width = self.size_dict[file_name][1]  # 单位 mm
        thickness = self.size_dict[file_name][2]  # 单位 nm #S070

        volume = length * width * thickness * 1e-9  # 单位 cm^3
        # 有的文件会出现有H没有对应M的情况
        if self.H_start_row != self.M_start_row:
            print("WARNING: self.H_start_row != self.M_start_row.")

        for line_index in range(int(self.H_start_row)-1, lines.__len__()):
            # print(line_index, ":", lines[line_index])
            try:
                bg_data = bg_lines[line_index].replace('\n','').replace('\t',',').split(',')
                data = lines[line_index].replace('\n','').replace('\t',',').split(',')
                h_list.append(float(data[int(self.H_start_col)-1]))
                bg_m_list.append(float(bg_data[int(self.M_start_col)-1]))
                m_list.append(float(data[int(self.M_start_col)-1]))
            except IndexError:
                print("IndexError: list index out of range. It is normal for data from DENJIKEN.")
                break
        # 去背底
        if m_list.__len__() > bg_m_list.__len__():
            bg_m_list = bg_m_list + (m_list.__len__()-bg_m_list.__len__())*[bg_m_list[-1]]
        m_list = np.subtract(m_list,bg_m_list[:m_list.__len__()])
        # Z轴校准(尽量让曲线是中心对称的)
        m_list = np.subtract(m_list,np.average(m_list))

        # 单位转换 [Oe,emu] 转 [A/m,A/m]
        print(self.H_start_unit)
        print(ScienceUnit.get_from_description_with_symbol_bracket(self.H_start_unit))
        h_list = science_unit_convert(from_list=h_list,from_unit=ScienceUnit.Magnetization.Oe.value,to_unit=ScienceUnit.Magnetization.A_m_1.value)
        if self.M_start_unit == "*Magn.Moment(emu)":
            m_list = np.divide(m_list, volume)
            self.M_start_unit = "emu/cm^3"
            m_list = science_unit_convert(from_list=m_list,from_unit=ScienceUnit.get_from_symbol(self.M_start_unit),to_unit=ScienceUnit.Magnetization.A_m_1.value)
        else:
            m_list = science_unit_convert(from_list=m_list,from_unit=ScienceUnit.get_from_symbol(self.M_start_unit),to_unit=ScienceUnit.Magnetization.A_m_1.value)

        # 调整数据：1切割正向和负向回线2把负向回线逆时针旋转180度3每一项取平均值合并
        # 1 切割
        # 最高级别注意！！！ 这里需要进行2次切割：
        # 1次是时间梯度也就是按照磁场增减方向
        # 1次是磁场大小的正负切割，如果同时包含了正负的磁场那么在接近于0附近的时候ODEINT函数内部计算必定不收敛
        # 这里定义一个有4个数组的元组，按照顺序分别代表 H下降，H正 H下降，H负 H上升，H负 H上升，H正
        trend_break_index = 0
        sign_break_index = []

        # FIXME: 此处变为了取中间位置的值，如果来回的横轴密度不同则有可能出问题
        for i in range(h_list.size-4):
            # 这里后方需要进行两次判断的理由是因为实际的H控制有一定的误差，不是一直单调变化的
            if (h_list[0] > h_list[1] and h_list[i] < h_list[i + 1] and h_list[i + 1] < h_list[i + 2] and h_list[i + 2] < h_list[i + 3]) or (
                    h_list[0] < h_list[1] and h_list[i] > h_list[i + 1] and  h_list[i + 1] > h_list[i + 2] and h_list[i + 2] > h_list[i + 3]):
                trend_break_index = i
                break
        # if trend_break_index == 0:
        #     trend_break_index = int(h_list.size/2)
        # 注意！此处判断H方向翻转的条件：i+1为0 i和i+2符号相反 OR i+1不为0 i和i+1符号相反
        # 注意！此处判断H方向翻转的条件：i+1为0 i和i+2符号相反 OR i+1不为0 i和i+1符号相反
        for i in range(h_list.size-3):
            if len(sign_break_index) == 2:
                break
            elif (h_list[i] * h_list[i + 2] < 0 and h_list[i + 1] == 0) or (h_list[i + 1] * h_list[i + 2] < 0):
                sign_break_index.append(i + 1)
        # if len(sign_break_index) < 2:
        #     sign_break_index = [int(h_list.size/4),int(3*h_list.size/4)]
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

        return h_list, m_list, m_0_list, break_index_list, length, width, thickness

    def simulate(self,file_name:str, file_path:str):
        print('开始模拟')
        h_raw, m_raw, m_0_list, break_index_list, length, width, thickness = self.get_data(file_name,file_path)
        print('获取数据完毕', h_raw, m_raw)
        m_max = np.max(m_raw)
        n = 1
        # 模型 1:Jiles-Atherton, 模型 2:Brillouin, 模型 3:Langevin, 模型 4:Takacs, 模型 5:Langevin-2
        if self.model == 4:
            print('Takecs模型暂时不可用')
            return {'popt': [], 'pcov': [], 'perr': []}
        elif self.model == 5:
            n = self.p0.pop()
            self.bounds[0].pop()
            self.bounds[1].pop()
            print(n)

        popt, pcov = curve_fit(
            {1: Jiles_Atherton_Model(m_0_list=m_0_list, break_index_list=break_index_list).func_Jiles_Atherton,
             2: Brillouin_Model(g_j=self.g_j, temp=self.temp, distribution=self.distribution_flag).func_Brillouin,
             3: Langevin_Model(g_j=self.g_j, temp=self.temp, distribution=self.distribution_flag).func_Langevin,
             4: Takacs_Model(g_j=self.g_j, temp=self.temp, break_index_list=break_index_list).func_Takacs,
             5: Langevin_Model_2(g_j=self.g_j, temp=self.temp, m_max=m_max, n=n).func_Langevin_2}.get(self.model)
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
            if self.distribution_flag:
                m_cal = Brillouin_Model(g_j=self.g_j, temp=self.temp, distribution=self.distribution_flag).func_Brillouin(h_raw, popt[0], popt[1], popt[2])
            else:
                m_cal = Brillouin_Model(g_j=self.g_j, temp=self.temp, distribution=self.distribution_flag).func_Brillouin(h_raw, popt[0], popt[1])
        elif self.model == 3:
            if self.distribution_flag:
                m_cal = Langevin_Model(g_j=self.g_j, temp=self.temp, distribution=self.distribution_flag).func_Langevin(h_raw, popt[0], popt[1], popt[2])
            else:
                m_cal = Langevin_Model(g_j=self.g_j, temp=self.temp, distribution=self.distribution_flag).func_Langevin(h_raw, popt[0], popt[1])
        elif self.model == 4:
            # 模型暂时被封住了
            m_cal = Takacs_Model(g_j=self.g_j, temp=self.temp, break_index_list=break_index_list).func_Takacs(h_raw, popt[0], popt[1])
        elif self.model == 5:
            m_cal = Langevin_Model_2(g_j=self.g_j, temp=self.temp, m_max=m_max, n=n).func_Langevin_2(h_raw, popt[0], popt[1])

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

        if self.distribution_flag:
            if self.model == 5:
                sigma = popt[0]
            else:
                sigma = popt[2]
        else:
            sigma = 0.

        return {
            'popt': popt,
            'pcov': pcov,
            'perr': perr,
            'l': PhysicalQuantity('l',ScienceUnit.Length.m.value, [length/1.e3]),
            'w': PhysicalQuantity('w',ScienceUnit.Length.m.value, [width/1.e3]),
            't': PhysicalQuantity('t',ScienceUnit.Length.m.value, [thickness/1.e9]),
            'h_raw': PhysicalQuantity('h_raw', ScienceUnit.Magnetization.A_m_1.value, h_raw),
            'm_raw': PhysicalQuantity('m_raw', ScienceUnit.Magnetization.A_m_1.value, m_raw),
            'm_cal': PhysicalQuantity('m_cal', ScienceUnit.Magnetization.A_m_1.value, m_cal),
            'sigma': PhysicalQuantity('sigma',ScienceUnit.Dimensionless.DN.value, [sigma])
        }


class Jiles_Atherton_Model:
    def __init__(self, m_0_list, break_index_list):
        self.m_0_list = m_0_list
        self.break_index_list = break_index_list

    def func_Jiles_Atherton(self, h, m_s, a, k, alpha, c):
        # 参数含义:
        # h External Magnetic Field
        # a Magnetic Coupling Coefficient
        # alpha Quanties Interdomain Coupling in the Magnetic Material
        # c Reversible Magnetization Coefficient, c=1 means Paramagnetism
        # m_s Saturation Magnetization
        # k Irreversible Loss Coefficient
        print("ms={0},a={1}.k={2},alpha={3},c={4},m_0_list={5}".format(m_s, a, k, alpha, c, self.m_0_list))

        # 分别对4支函数进行计算
        # 第1支 下降支
        (m1_result, m1_info_dict) = odeint(self.func_dm_dh,
                                           self.m_0_list[0],
                                           h[0:self.break_index_list[0]],
                                           args=(m_s, alpha, c, a, k, -1),
                                           printmessg=True,
                                           full_output=True)
        # 第2支 上升支
        (m2_result, m2_info_dict) = odeint(self.func_dm_dh,
                                           self.m_0_list[1],
                                           h[self.break_index_list[0]:self.break_index_list[1]],
                                           args=(m_s, alpha, c, a, k, 1),
                                           printmessg=True,
                                           full_output=True)
        # 第3支 下降支
        (m3_result, m3_info_dict) = odeint(self.func_dm_dh,
                                           self.m_0_list[2],
                                           h[self.break_index_list[1]:self.break_index_list[2]],
                                           args=(m_s, alpha, c, a, k, -1),
                                           printmessg=True,
                                           full_output=True)
        # 第4支 上升支
        (m4_result, m4_info_dict) = odeint(self.func_dm_dh,
                                           self.m_0_list[3],
                                           h[self.break_index_list[2]:],
                                           args=(m_s, alpha, c, a, k, 1),
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

    def func_dm_dh(self, m, h, m_s, alpha, c, a, k, delta):
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
    def __init__(self, g_j, temp, distribution=False):
        self.g_j = g_j
        self.temp = temp
        self.distribution = distribution

    def func_Brillouin(self, h, m_s, j, sigma = 0.0):
        # 参数含义:
        # h External Magnetic Field
        # m_s Saturation Magnetization
        # j Total Angular Momentum
        # temp Temperature
        D_m = np.power(self.g_j*sci_const.miu_B*j*6./(np.pi*m_s),1/3)
        print("g_j={0},temp={1},m_s={2},j={3},distribution={4},D_m={5} nm,sigma={6}".format(self.g_j, self.temp, m_s, j,
                                                                                            self.distribution,D_m*1.e9, sigma))
        if self.distribution:
            # 这里比较特殊，所以h被遍历
            start = time.time()
            m = Parallel(n_jobs=4)(
                delayed(self.func_integral)(var_h, m_s, j, sigma)
                for var_h in h)
            end = time.time()
            print(str(end - start) + 's')
            return m
        else:
            b = np.multiply(sci_const.miu_0, h)
            x = np.divide(np.multiply(self.g_j * sci_const.miu_B, b), sci_const.k_B * self.temp)
            b_j = np.subtract(np.multiply((2 * j + 1) / (2 * j),
                                          np.reciprocal(np.tanh(np.multiply((2 * j + 1) / 2, x)))),
                              np.multiply(1 / (2 * j),
                                          np.reciprocal(np.tanh(np.multiply(1 / 2, x)))))
            m = np.multiply(m_s, b_j)
        return m

    def func_integral(self, var_h, m_s, j, sigma):
        (var_m, abserr) = quad(self.func_Brillouin_y, 0, 2, args=(var_h, m_s, j, sigma), limlst=3)
        return var_m

    def func_Brillouin_y(self, y, h, m_s, j, sigma):
        # j -> j*y^3
        j_y = np.multiply(j, np.power(y, 3))
        b = np.multiply(sci_const.miu_0, h)
        x = np.divide(np.multiply(self.g_j * sci_const.miu_B, b), sci_const.k_B * self.temp)
        b_j = np.subtract(np.multiply((2 * j_y + 1) / (2 * j_y),
                                      np.reciprocal(np.tanh(np.multiply((2 * j_y + 1) / 2, x)))),
                          np.multiply(1 / (2 * j_y),
                                      np.reciprocal(np.tanh(np.multiply(1 / 2, x)))))
        d_m = np.power(self.g_j*sci_const.miu_B*j*6./(np.pi*m_s), 1/3)
        # f_y = lognorm.pdf(y, sigma, loc=0, scale= d_m)
        f_y = lognorm.pdf(y, sigma)
        dm = np.multiply(np.multiply(m_s, b_j), f_y)
        return dm


class Langevin_Model:
    def __init__(self, g_j, temp, distribution=False):
        self.g_j = g_j
        self.temp = temp
        self.distribution = distribution

    def func_Langevin(self, h, m_s, j, sigma = 0.0):
        # 参数含义:
        # h External Magnetic Field
        # m_s Saturation Magnetization
        # j Total Angular Momentum
        # temp Temperature
        # sigma Standard Deviation of the Reduced Diameter Distribution (OPTIONAL)
        print(time.asctime(time.localtime(time.time())))
        D_m = np.power(self.g_j*sci_const.miu_B*j*6./(np.pi*m_s),1/3)
        print("g_j={0},temp={1},m_s={2},j={3},distribution={4},D_m={5} m,sigma={6}".format(self.g_j, self.temp, m_s, j,
                                                                                 self.distribution,D_m, sigma))
        if self.distribution:
            # 这里比较特殊，所有h被遍历
            start = time.time()
            m = []
            # 此处对把h降低到一定数量，降低循环次数加快算法
            sampling_frequency = 10 # 每10个元素取一个样
            h_sampling = []
            for i in range(0,int(h.size/sampling_frequency)):
                h_sampling.append(h[1+i*sampling_frequency])
            if h[0] >= 0:
                h_sampling.insert(0,np.max(h))
                h_sampling.append(np.min(h))
            else:
                h_sampling.insert(0,np.min(h))
                h_sampling.append(np.max(h))
            m_sampling = []
            for var_h in h_sampling:
                (var_m,abserr) = quad(self.func_Langevin_y,0,np.inf,args=(var_h,m_s,j,sigma),limlst=3)
                # (var_m,abserr) = quad(self.func_Langevin_y,0,np.inf,args=(var_h,m_s,j,sigma))
                m_sampling.append(var_m)
            # 插值用函数
            interpolate_func = interpolate.interp1d(h_sampling,m_sampling,kind="slinear")
            m = interpolate_func(h)
            # 此处重新填充到h的本身数量
            # m = Parallel(n_jobs=4)(
            #     delayed(self.func_integral)(var_h, m_s, j, sigma)
            #     for var_h in h)
            # for var_h in h:
            #     var_m = cfunc("float64(float64)")(self.func_Langevin_y)(var_h, m_s, j, sigma)
            #     m.append(var_m)
            end = time.time()
            print(str(end - start) + 's')
            return m
        else:
            b = np.multiply(sci_const.miu_0, h)
            x = np.divide(np.multiply(self.g_j * sci_const.miu_B, b), sci_const.k_B * self.temp)
            l_x = np.subtract(np.reciprocal(np.tanh(np.multiply(j, x))),
                              np.reciprocal(np.multiply(j, x)))
            m = np.multiply(m_s, l_x)
            return m

    # def func_integral(self, var_h, m_s, j, sigma):
    #     # (var_m, abserr) = quad(self.func_Langevin_y, 0, np.inf, args=(var_h, m_s, j, sigma), limlst=3)
    #     (var_m, abserr) = quad(self.func_Langevin_y, 0, 2, args=(var_h, m_s, j, sigma), limlst=3)
    #     return var_m

    def func_Langevin_y(self, y, h, m_s, j, sigma):
        # j -> j*y^3
        j_y = np.multiply(j, np.power(y, 3))
        b = np.multiply(sci_const.miu_0, h)
        x = np.divide(np.multiply(self.g_j * sci_const.miu_B, b), sci_const.k_B * self.temp)
        l_x = np.subtract(np.reciprocal(np.tanh(np.multiply(j_y, x))),
                          np.reciprocal(np.multiply(j_y, x)))
        # d_m = np.power(self.g_j*sci_const.miu_B*j*6./(np.pi*m_s), 1/3)
        # f_y = lognorm.pdf(y, sigma, loc=0, scale=d_m)
        f_y = lognorm.pdf(y, sigma)  # 这里已经进行了归一化，loc和scale使用默认的参数即可
        dm = np.multiply(np.multiply(m_s, l_x), f_y)
        return dm


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


class Langevin_Model_2:
    def __init__(self, g_j, temp, m_max, n):
        self.g_j = g_j
        self.temp = temp
        self.m_max = m_max
        self.n = n

    def func_Langevin_2(self, h, sigma, dm):
        # 参数含义:
        # h External Magnetic Field
        # m_s Saturation Magnetization
        # j Total Angular Momentum
        # temp Temperature
        # sigma Standard Deviation of the Diameter Distribution (OPTIONAL)
        # Dm Average Value of the Diameter Distribution (OPTIONAL)
        # print(time.asctime(time.localtime(time.time())))

        print("g_j={0},temp={1},M_max={2}A/m,N={3},sigma={4},Dm={5}m.".format(self.g_j, self.temp, self.m_max, self.n, sigma, dm))
        # 此处和Model-3的积分计算不同，此处是累加
        # 通过Dm的上下限和细度N 划分出d的1×N的矩阵
        # d = np.linspace(1.e-14,1.e-8,self.n).reshape(1,self.n)
        # x = np.divide(d,dm)
        # lndf_x = lognorm.pdf(x, np.log(sigma))
        # lndf_d = np.multiply(x,lndf_x)
        # d = np.linspace(10.,100.,self.n).reshape(1,self.n)
        # x = np.divide(d,dm)
        # lndf_x = lognorm.pdf(x, sigma)
        # lndf_d = np.multiply(x,lndf_x)
        d = np.linspace(1.e-10,1.e-8,self.n).reshape(1,self.n)
        x = np.divide(d,dm)
        s = np.log(sigma)
        lndf_x = lognorm.pdf(x,s)
        lndf_d = np.multiply(x,lndf_x)
        # M_Co = 1349486.25  # A/m, calculated value, g_j = 1.60
        # m_Co = 1450697.72  # A/m, observed value, g_j = 1.72
        # m_Co = 145069.772  # A/m, observed value, g_j = 1.72
        m_Co = self.m_max
        alpha_term_1 = np.divide(np.multiply(m_Co,h),np.multiply(sci_const.k_B,self.temp)).reshape(np.size(h),1)
        # for i in range(self.n):
        #     alpha_term_2_i = np.multiply(4.*np.pi/3.,np.power(np.divide(d[i],2),3))
        #     alpha_i = np.multiply(alpha_term_1, alpha_term_2_i)
        alpha_term_2 = np.multiply(4.*np.pi/3.,np.power(np.divide(d,2.),3.))
        alpha = np.multiply(alpha_term_1, alpha_term_2)
        l_alpha = np.subtract(np.reciprocal(np.tanh(alpha)),
                              np.reciprocal(alpha))
        m_term_1 = np.multiply(4.*np.pi/3.,np.power(np.divide(d,2.),3.))
        m_term_2 = np.multiply(lndf_d,l_alpha)
        m = np.multiply(m_Co,np.multiply(m_term_1,m_term_2))
        m_sum = np.sum(np.transpose(m), axis=0)
        magnification = self.m_max / np.max(m_sum)
        m_sum = np.multiply(m_sum, magnification)
        return m_sum



if __name__ == '__main__':
    # a=[[1,2,3],[2,2,2]]
    # print(np.sum(a,axis=0))

    # x = [0,1,2,3,4,5,6,7,8,9,10]
    # f = lognorm.pdf(x, 0.2, loc=0, scale= 5)
    # print(x,f)
    # print(np.vsplit(test_h_list,3))
    # print(np.dsplit(test_h_list,3))
    #  J-A Model
    # ms, a, k, alpha, c, delta
    # a=76434.33625750794,alpha=0.17716254982042273.c=0.0002219543796726332,m_s=647678.7831977553,k=2319.3791455301916
    # magnetization_simulator = MagnetizationSimulator(
    #     model=1,
    #     file_path="D:\\PycharmProjects\\AIProcessingPlatform\\app\\ui\\V210705112208.Csv",
    #     p0=[647678,76434, 2319, 0.17716254982042273, 0.0002219543796726332],
    #     bounds=([1e4, 1e3, 1e3, 0, 0.00000001], [1e7, 1e5, 1e5, 0.3, 0.01])
    # )
    # magnetization_simulator.simulate()

    # B Model
    # m_s, j, sigma(OPTIONAL)
    # g_j = sci_const.Lande_g_Factor(3 / 2, 3, 9 / 2)  # Co2+ ion
    # magnetization_simulator = MagnetizationSimulator(
    #     model=2,
    #     file_path="D:\\PycharmProjects\\AIProcessingPlatform\\app\\ui\\V210705112208.Csv",
    #     p0=[6e6, 4500],
    #     bounds=([1e4, 1], [1e8, 1e10])
    # )
    # magnetization_simulator.simulate(g_j=g_j, temp=300)

    # L Model
    # m_s, j, sigma(OPTIONAL)
    # g_j = sci_const.Lande_g_Factor(3 / 2, 3, 9 / 2)  # Co2+ ion
    # magnetization_simulator = MagnetizationSimulator(
    #     model=3,
    #     file_path="D:\\PycharmProjects\\AIProcessingPlatform\\app\\ui\\V210705112208.Csv",
    #     p0=[646085, 5913.0, 0.27535552365222604],
    #     bounds=([1e4, 1, 0.0], [1e8, 1e10, 1.0])
    # )
    # magnetization_simulator.simulate(g_j=g_j, temp=300, distribution=True)

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

    # L Model (new)
    # m_s, j, sigma(OPTIONAL)
    # g_j = sci_const.Lande_g_Factor(3 / 2, 3, 9 / 2)  # Co2+ ion
    my_simulator = MagnetizationSimulator(model=5,
                                          first_pos_info_tuple=((88, 10, 'Oersted(Oe)'), (88, 11, '*Magn.Moment(emu)')),
                                          bg_file_path='D:\\PycharmProjects\\AIProcessingPlatform\\app\\ui\\datafortest\\#S328',
                                          size_dict={'#S328': [5.0, 5.0, 1000.0]},
                                          p0=[1.2, 8.e-9, 200],
                                          bounds=([1.0, 1.e-10, 10], [2.0, 1.e-8, 1000]),
                                          g_j=1.3341064347922666,
                                          temp=300,
                                          distribution_flag=True)
    result_dict = my_simulator.simulate('#S328',
                                                 'D:\\PycharmProjects\\AIProcessingPlatform\\app\\ui\\datafortest\\BG20220321')
    print("111")

