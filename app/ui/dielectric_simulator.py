import numpy as np
from scipy.optimize import curve_fit
from app.utils import sci_const
from app.utils.science_unit import ScienceUnit,science_unit_convert
from app.utils.science_base import PhysicalQuantity


class DielectricSimulator:
    def __init__(self, model, first_pos_info_tuple=(), info_dict=None, fixed_param_dict=None, p0=None, bounds=()):
        if fixed_param_dict is None:
            fixed_param_dict = {}
        if info_dict is None:
            info_dict = {}
        self.model = model
        # 一个初始尝试值的数组
        self.p0 = p0
        # 一个参数的边间turple，左边数组下界，右边数组上届
        self.bounds = bounds
        # 接受并解析第1个数据点的位置信息及其单位信息first_pos_info_tuple
        ((self.f_start_row,self.f_start_col,self.f_start_unit),(self.Cp_start_row,self.Cp_start_col,self.Cp_start_unit)) = first_pos_info_tuple
        # 接受信息字典size_dict
        self.info_dict = info_dict
        # 接受被固定的参数信息fixed_param_dict
        self.fixed_param_dict = fixed_param_dict

    def open_file(self,path):
        try:
            with open(path, 'r', encoding="utf-8") as file:
                lines = file.readlines()
        except UnicodeDecodeError:
            with open(path, 'r', encoding="shift_jis") as file:
                lines = file.readlines()
        return lines

    def get_data(self,file_name:str, file_path:str):
        freq_list = []
        epsilon_list = []
        lines = self.open_file(path = file_path)
        thickness = self.info_dict[file_name][0]  # m
        area = self.info_dict[file_name][1]  # m^2
        h = self.info_dict[file_name][2]  # Oe
        co = self.info_dict[file_name][3]  # at.%
        dcb = self.info_dict[file_name][4]  # V
        osc = self.info_dict[file_name][5]  # V
        cc = self.info_dict[file_name][6]  # 1
        f_list = []
        cp_list = []
        if self.f_start_row != self.Cp_start_row:
            print("WARNING: self.f_start_row != self.Cp_start_row.")

        for line_index in range(int(self.f_start_row)-1, lines.__len__()):
            data = lines[line_index].replace('\n','').replace('\t',',').split(',')
            f_list.append(float(data[int(self.f_start_col)-1]))
            cp_list.append(float(data[int(self.Cp_start_col)-1]))
        freq_list = f_list
        # epsilon_r = Cp * thickness / (epsilon_0 * area)
        epsilon_list = np.multiply(np.divide(np.multiply(cp_list,thickness),sci_const.epsilon_0*area),cc)
        # 接下来用这种方法来切割掉曲线前端可能会出现的噪声，噪声的判断是数据二阶导数大于0的情况
        d2_epsilon_list = np.gradient(np.gradient(epsilon_list))
        start_pos = 0
        for i in range(d2_epsilon_list.size):
            if d2_epsilon_list[i] >= 1:
                start_pos = i
        print('横轴切割完毕开始点是{}'.format(start_pos))
        return freq_list[start_pos:], epsilon_list[start_pos:], thickness, area, h, co, dcb, osc, cc

    def simulate(self,file_name:str, file_path:str):
        print('开始模拟')
        freq_raw, epsilon_raw, thickness, area, h, co, dcb, osc, cc = self.get_data(file_name,file_path)
        print('获取数据完毕', freq_raw, epsilon_raw)
        # 模型 1:Havriliak–Negami,模型 2:Cole-Cole,模型 3:Cole–Davidson,模型 4:Debye
        # 这里比较特殊，需要对被固定的参数进行操作
        popt, pcov = curve_fit({1: Havriliak_Negami_Model(fixed_param_dict=self.fixed_param_dict).func_Havriliak_Negami,
                                2: Cole_Cole_Model(fixed_param_dict=self.fixed_param_dict).func_Cole_Cole,
                                3: Cole_Davidson_Model(fixed_param_dict=self.fixed_param_dict).func_Cole_Davidson,
                                4: Debye_Model(fixed_param_dict=self.fixed_param_dict).func_Debye}.get(self.model)
                               , freq_raw
                               , epsilon_raw
                               , p0=self.p0
                               , bounds=self.bounds)
        perr = np.sqrt(np.diag(pcov))
        print(popt,pcov,perr)
        epsilon_cal = []
        if self.model == 1:
            epsilon_cal = Havriliak_Negami_Model(fixed_param_dict=self.fixed_param_dict).func_Havriliak_Negami(freq_raw, popt.tolist())
        elif self.model == 2:
            epsilon_cal = Cole_Cole_Model(fixed_param_dict=self.fixed_param_dict).func_Cole_Cole(freq_raw, popt.tolist())
        elif self.model == 3:
            epsilon_cal = Cole_Davidson_Model(fixed_param_dict=self.fixed_param_dict).func_Cole_Davidson(freq_raw, popt.tolist())
        elif self.model ==4:
            epsilon_cal = Debye_Model(fixed_param_dict=self.fixed_param_dict).func_Debye(freq_raw, popt.tolist())
        else:
            print('ERROR. Unknown Model.')
        return {'popt': popt.tolist(),
                'pcov': pcov.tolist(),
                'perr': perr.tolist(),
                't': PhysicalQuantity('t',ScienceUnit.Length.m.value,[thickness]),
                'A': PhysicalQuantity('A',ScienceUnit.Area.m2.value,[area]),
                'H': PhysicalQuantity('H',ScienceUnit.Magnetization.Oe.value,[h]),
                'Co': PhysicalQuantity('Co',ScienceUnit.AtomicContent.at.value,[co]),
                'DCB': PhysicalQuantity('DCB',ScienceUnit.Voltage.V.value,[dcb]),
                'OSC': PhysicalQuantity('OSC',ScienceUnit.Voltage.V.value,[osc]),
                'C.C.': PhysicalQuantity('C.C.',ScienceUnit.Dimensionless.DN.value,[cc]),
                'freq_raw': PhysicalQuantity('freq_raw',ScienceUnit.Frequency.Hz.value,freq_raw),
                'epsilon_raw': PhysicalQuantity('epsilon_raw',ScienceUnit.Dimensionless.DN.value,epsilon_raw),
                'epsilon_cal': PhysicalQuantity('epsilon_cal',ScienceUnit.Dimensionless.DN.value,epsilon_cal)
                }


class Havriliak_Negami_Model:
    def __init__(self,fixed_param_dict:dict):
        self.fixed_param_dict = fixed_param_dict

    def func_Havriliak_Negami(self, freq, *args):
        if isinstance(args[0],list):
            arg_list = list(args[0])
        else:
            arg_list = list(args)
        if 'alpha' in self.fixed_param_dict.keys():
            alpha = self.fixed_param_dict['alpha']
        else:
            alpha = arg_list.pop(0)
        if 'beta' in self.fixed_param_dict.keys():
            beta = self.fixed_param_dict['beta']
        else:
            beta = arg_list.pop(0)
        if 'tau' in self.fixed_param_dict.keys():
            tau = self.fixed_param_dict['tau']
        else:
            tau = arg_list.pop(0)
        if 'epsilon_inf' in self.fixed_param_dict.keys():
            epsilon_inf = self.fixed_param_dict['epsilon_inf']
        else:
            epsilon_inf = arg_list.pop(0)
        if 'delta_epsilon' in self.fixed_param_dict.keys():
            delta_epsilon = self.fixed_param_dict['delta_epsilon']
        else:
            delta_epsilon = arg_list.pop(0)

        omega = np.multiply(freq, 2.0 * np.pi)
        # v1 = (omega*tau)^alpha
        v1 = np.power(np.multiply(omega, tau), alpha)
        # print('v1', v1)
        # p1 = alpha*pi/2
        p1 = alpha * np.pi / 2
        # print('p1', p1)
        # v2 = [v1*sin(p1)]/[1+v1*cos(p1)]
        v2 = np.divide(np.multiply(v1, np.sin(p1)), 1 + np.multiply(v1, np.cos(p1)))
        # print('v2', v2)
        # phi = arctan(v2)
        phi = np.arctan(v2)
        # print('phi', phi)
        # B = cos(beta*phi)
        B = np.cos(np.multiply(beta, phi))
        # print('B', B)
        # v3 = 1+2*v1*cos(p1)+v1^2
        v3 = 1.0 + np.multiply(v1, 2.0 * np.cos(p1)) + np.square(v1)
        # print('v3', v3)
        # A = (v3)^(-beta/2)
        A = np.power(v3, -beta / 2.0)
        # print('A', A)
        # epsilon = epsilon_inf + delta_epsilon*A*B
        epsilon = epsilon_inf + np.multiply(np.multiply(A, B), delta_epsilon)
        print(epsilon.shape)
        return epsilon


class Cole_Cole_Model:
    def __init__(self,fixed_param_dict:dict):
        self.fixed_param_dict = fixed_param_dict

    def func_Cole_Cole(self, freq, *args):
        if isinstance(args[0],list):
            arg_list = list(args[0])
        else:
            arg_list = list(args)
        if 'alpha' in self.fixed_param_dict.keys():
            alpha = self.fixed_param_dict['alpha']
        else:
            alpha = arg_list.pop(0)
        if 'tau' in self.fixed_param_dict.keys():
            tau = self.fixed_param_dict['tau']
        else:
            tau = arg_list.pop(0)
        if 'epsilon_inf' in self.fixed_param_dict.keys():
            epsilon_inf = self.fixed_param_dict['epsilon_inf']
        else:
            epsilon_inf = arg_list.pop(0)
        if 'delta_epsilon' in self.fixed_param_dict.keys():
            delta_epsilon = self.fixed_param_dict['delta_epsilon']
        else:
            delta_epsilon = arg_list.pop(0)

        omega = np.multiply(freq, 2.0 * np.pi)
        # v1 = (omega*tau)^alpha
        v1 = np.power(np.multiply(omega, tau), alpha)
        # p1 = alpha*pi/2
        p1 = alpha * np.pi / 2
        # A = 1+v1*cos(p1)
        A = 1.0 + np.multiply(v1, np.cos(p1))
        # B = 1+2*v1*cos(p1)+v1^2
        B = 1.0 + 2.0 * v1 * np.cos(p1) + np.square(v1)
        # epsilon = epsilon_inf + delta_epsilon*A/B
        epsilon = epsilon_inf + np.multiply(np.divide(A, B), delta_epsilon)
        return epsilon


class Cole_Davidson_Model:
    def __init__(self,fixed_param_dict:dict):
        self.fixed_param_dict = fixed_param_dict

    def func_Cole_Davidson(self, freq, *args):
        if isinstance(args[0],list):
            arg_list = list(args[0])
        else:
            arg_list = list(args)
        if 'beta' in self.fixed_param_dict.keys():
            beta = self.fixed_param_dict['beta']
        else:
            beta = arg_list.pop(0)
        if 'tau' in self.fixed_param_dict.keys():
            tau = self.fixed_param_dict['tau']
        else:
            tau = arg_list.pop(0)
        if 'epsilon_inf' in self.fixed_param_dict.keys():
            epsilon_inf = self.fixed_param_dict['epsilon_inf']
        else:
            epsilon_inf = arg_list.pop(0)
        if 'delta_epsilon' in self.fixed_param_dict.keys():
            delta_epsilon = self.fixed_param_dict['delta_epsilon']
        else:
            delta_epsilon = arg_list.pop(0)

        omega = np.multiply(freq, 2.0 * np.pi)
        # v1 = omega*tau
        v1 = np.multiply(omega, tau)
        # phi = arctan(omega*tau)
        phi = np.arctan(np.multiply(omega, tau))
        # A = (1+v1^2)^(beta/2)
        A = np.power(1 + np.square(v1), -beta / 2)
        # B = cos(beta*phi)
        B = np.cos(np.multiply(beta, phi))
        # epsilon = epsilon_inf + delta_epsilon*A*B
        epsilon = epsilon_inf + np.multiply(np.multiply(A, B), delta_epsilon)
        return epsilon

class Debye_Model:
    def __init__(self,fixed_param_dict:dict):
        self.fixed_param_dict = fixed_param_dict

    def func_Debye(self, freq, *args):
        if isinstance(args[0],list):
            arg_list = list(args[0])
        else:
            arg_list = list(args)
        if 'tau' in self.fixed_param_dict.keys():
            tau = self.fixed_param_dict['tau']
        else:
            tau = arg_list.pop(0)
        if 'epsilon_inf' in self.fixed_param_dict.keys():
            epsilon_inf = self.fixed_param_dict['epsilon_inf']
        else:
            epsilon_inf = arg_list.pop(0)
        if 'delta_epsilon' in self.fixed_param_dict.keys():
            delta_epsilon = self.fixed_param_dict['delta_epsilon']
        else:
            delta_epsilon = arg_list.pop(0)

        omega = np.multiply(freq, 2.0 * np.pi)
        # v1 = omega*tau
        v1 = np.multiply(omega, tau)
        # A = 1
        A = 1.0
        # B = 1+v1*2
        B = 1.0 + np.square(v1)
        # epsilon = epsilon_inf + delta_epsilon*A/B
        epsilon = epsilon_inf + np.multiply(np.divide(A, B), delta_epsilon)
        return epsilon
