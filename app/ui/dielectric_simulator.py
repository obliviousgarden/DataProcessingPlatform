import numpy as np
from scipy.optimize import curve_fit


class DielectricSimulator:
    def __init__(self, model, file_path, p0=None, bounds=()):
        self.model = model
        self.file_path = file_path
        # 一个初始尝试值的数组
        self.p0 = p0
        # 一个参数的边间turple，左边数组下界，右边数组上届
        self.bounds = bounds

    def get_data(self):
        freq_list = []
        epsilon_list = []
        with open(self.file_path, 'r') as file:
            lines = file.readlines()
            thickness = float(lines[4])
            electrode_area = float(lines[5])
            for line_index in range(19, lines.__len__()):
                freq, loss, c = map(float, lines[line_index].replace('\n', '').split('\t'))
                freq_list.append(freq)
                epsilon_list.append(c * thickness * 10000000000 / (8.854 * electrode_area))
        # 接下来用这种方法来切割掉曲线前端可能会出现的噪声，噪声的判断是数据二阶导数大于0的情况
        d2_epsilon_list = np.gradient(np.gradient(epsilon_list))
        start_pos = 0
        for i in range(d2_epsilon_list.size):
            if d2_epsilon_list[i] >= 1:
                start_pos = i
        print('横轴切割完毕开始点是{}'.format(start_pos))
        return freq_list[start_pos:], epsilon_list[start_pos:]

    def simulate(self):
        print('开始模拟')
        freq_raw, epsilon_raw = self.get_data()
        print('获取数据完毕', freq_raw, epsilon_raw)
        # 模型 1:Havriliak–Negami,模型 2:Cole-Cole,模型 3:Cole–Davidson,模型 4:Debye
        popt, pcov = curve_fit({1: func_Havriliak_Negami, 2: func_Cole_Cole, 3: func_Cole_Davidson,
                                4: func_Debye}.get(self.model)
                               , freq_raw
                               , epsilon_raw
                               , p0=self.p0
                               , bounds=self.bounds)
        perr = np.sqrt(np.diag(pcov))
        return {'popt': popt, 'pcov': pcov, 'perr': perr}


def func_Havriliak_Negami(freq, alpha, beta, tau, epsilon_inf, delta_epsilon):
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
    return epsilon


def func_Cole_Cole(freq, alpha, tau, epsilon_inf, delta_epsilon):
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


def func_Cole_Davidson(freq, beta, tau, epsilon_inf, delta_epsilon):
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


def func_Debye(freq, tau, epsilon_inf, delta_epsilon):
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
