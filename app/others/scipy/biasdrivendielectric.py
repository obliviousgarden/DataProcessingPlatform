import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.misc import derivative

class DataLoader():
    def __init__(self):
        print('数据初始化中...')
        self.curDir = os.path.abspath(os.path.dirname(__file__))
        self.rootDir = self.curDir[:self.curDir.find("app\\") + len("app\\")]
        self.rawDir = self.rootDir + 'dataset\\biasdrivendielectric\\raw\\'

    def get_data(self, dc_bias):
        file_name_list = os.listdir(self.rawDir)
        freq_list = []
        epsilon_list = []
        for file_name in file_name_list:
            if dc_bias == float(file_name.split('-DCB')[1]):
                with open(self.rawDir + file_name, 'r') as file:
                    lines = file.readlines()
                    thickness = float(lines[4])
                    electrode_area = float(lines[5])
                    for line_index in range(19, lines.__len__()):
                        freq, loss, c = map(float, lines[line_index].replace('\n', '').split('\t'))
                        freq_list.append(freq)
                        epsilon_list.append(c * thickness * 10000000000 / (8.854 * electrode_area))
                break
        # # START 排除可能出现的第一个弛豫的代码
        # log_freq_list = np.log10(freq_list)
        # fig = plt.figure()
        # plt.ion()
        # plt.show()
        # ax = fig.add_subplot(2, 2, 1)
        # ax.scatter(log_freq_list, epsilon_list, label='epsilon(Log_freq)')
        # ax.legend()
        # d_epsilon_list = np.gradient(epsilon_list)
        # ax2 = fig.add_subplot(2, 2, 2)
        # ax2.plot(log_freq_list,d_epsilon_list,label='D_epsilon(Log_freq)')
        # ax2.legend()
        # print('d',d_epsilon_list)
        # d2_epsilon_list= np.gradient(d_epsilon_list)
        # ax3 = fig.add_subplot(2, 2, 3)
        # ax3.plot(log_freq_list, d2_epsilon_list, label='D2_epsilon(Log_freq)')
        # print('d2',d2_epsilon_list)
        # start_pos = 0
        # for i in range(d2_epsilon_list.size):
        #     if d2_epsilon_list[i] >= 1:
        #         start_pos = i
        # ax4 = fig.add_subplot(2, 2, 4)
        # ax4.plot(log_freq_list[start_pos:], epsilon_list[start_pos:], label='D_epsilon(Log_freq)_Cut')
        # # END


        return freq_list, epsilon_list


def main(dc_bias):
    data_loader = DataLoader()
    freq_raw, epsilon_raw = np.array(data_loader.get_data(dc_bias), dtype='float32')
    omega_raw = np.multiply(freq_raw, 2.0 * np.pi)
    fig = plt.figure()
    plt.ion()
    plt.show()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xscale('log')
    ax.scatter(freq_raw, epsilon_raw, label='epsilon(freq)_Raw')
    ax.legend()
    alpha_est = 0.95
    beta_est = 0.85
    tau_est = 1e-6
    epsilon_inf_est = np.min(epsilon_raw)
    delta_epsilon_est = np.max(epsilon_raw) - epsilon_inf_est
    popt, pcov = curve_fit(func, freq_raw, epsilon_raw
                           , p0=[alpha_est, beta_est, tau_est, epsilon_inf_est, delta_epsilon_est]
                           , bounds=([0, 0, 1e-8, 0, 0], [1, 1, 1e-5, 50, np.inf]))
    perr = np.sqrt(np.diag(pcov))
    print('结果参数popt', popt)
    print('参数的协方差pcov', pcov)
    print('参数的标准差perr', perr)

    epsilon_pred = func(freq_raw, popt[0], popt[1], popt[2], popt[3], popt[4])
    ax.plot(freq_raw, epsilon_pred, 'r-', label='epsilon(freq)_Pred')
    ax.legend()


def func(freq, alpha, beta, tau, epsilon_inf, delta_epsilon):
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


if __name__ == '__main__':
    main(1.4)
