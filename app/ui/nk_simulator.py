import math

import numpy as np
from app.utils.science_unit_converter import ScienceUnitConverter
from app.utils.science_unit import ScienceUnit, Unit
from app.utils.science_plot import SciencePlot, SciencePlotData
from scipy.optimize import curve_fit
from app.utils import sci_const


class NKSimulator:
    def __init__(self, refractive_index_model: int, first_pos_info_tuple=(), wl_bound=(), p0=None, bounds=()):
        # 拟合模型
        self.refractive_index_model = refractive_index_model
        # 单位和第一个数据点为位置信息
        (self.wl_unit, self.start_row, self.wl_start_rol, self.n_start_rol, self.k_start_rol) = first_pos_info_tuple
        # 波长范围
        self.wl_bound = wl_bound
        self.wl_lower_bound = 0.
        self.wl_upper_bound = 0.
        self.lower_index = 0
        self.upper_index = 0
        self.bound_valid = True
        # 拟合参数的初值的数组
        self.p0 = p0
        # 拟合参数的上下边界
        self.bounds = bounds

    def open_file(self, path):
        try:
            with open(path, 'r', encoding="utf-8") as file:
                lines = file.readlines()
        except UnicodeDecodeError:
            with open(path, 'r', encoding="shift_jis") as file:
                lines = file.readlines()
        return lines

    def get_data(self, file_name: str, file_path: str):
        wl_list = []  # 波长 / um
        n_list = []  # 折射率
        k_list = []  # 消光系数
        # 1 读入数据
        lines = self.open_file(path=file_path)
        for line_index in range(int(self.start_row) - 1, lines.__len__()):
            data = lines[line_index].replace('\n', '').split(',')
            wl_list.append(float(data[int(self.wl_start_rol) - 1]))
            n_list.append(float(data[int(self.n_start_rol) - 1]))
            k_list.append(float(data[int(self.k_start_rol) - 1]))
        # 2 波长单位变换到um
        from_unit = ScienceUnit.Wavelength.um
        to_unit = ScienceUnit.Wavelength.um
        if from_unit != to_unit:
            # X轴的单位变换
            for data in wl_list:
                data = ScienceUnitConverter.convert(from_unit_class=from_unit.__class__,
                                                    to_unit_class=to_unit.__class__, from_unit=from_unit.value,
                                                    to_unit=to_unit.value, value=data)

        # 3 wl_bound单位变换,截取需要数据
        wl_unit_name = str(self.wl_bound[0])
        # TODO: 通过名称获取枚举的函数
        wl_unit = ScienceUnit.Wavelength.nm

        self.wl_lower_bound = ScienceUnitConverter.convert(from_unit_class=wl_unit.__class__,
                                                           to_unit_class=to_unit.__class__, from_unit=wl_unit.value,
                                                           to_unit=to_unit.value, value=self.wl_bound[1])
        self.wl_upper_bound = ScienceUnitConverter.convert(from_unit_class=wl_unit.__class__,
                                                           to_unit_class=to_unit.__class__, from_unit=wl_unit.value,
                                                           to_unit=to_unit.value, value=self.wl_bound[2])

        for index in range(wl_list.__len__()):
            if wl_list[index] < self.wl_lower_bound <= wl_list[index + 1]:
                self.lower_index = index + 1
            elif wl_list[index] <= self.wl_upper_bound < wl_list[index + 1]:
                self.upper_index = index + 1
            else:
                pass
        print("Interception Index: ({},{})".format(self.lower_index, self.upper_index))
        self.bound_valid = True
        if self.bound_valid:
            return wl_list[self.lower_index:self.upper_index], n_list[self.lower_index:self.upper_index], k_list[self.lower_index:self.upper_index]
        else:
            return wl_list, n_list, k_list

    def simulate(self, file_name: str, file_path: str):
        print('开始模拟')
        wl_raw, n_raw, k_raw = self.get_data(file_name, file_path)
        complex_n_raw = np.hstack([n_raw, k_raw])
        print('获取数据完毕', wl_raw, n_raw, k_raw)
        # 模型 1: Cauthy, 模型 2: Sellmeier-3, 模型 3: Sellmeier-2
        refractive_model = RefractiveIndexModel()
        popt, pcov = curve_fit({1: refractive_model.model_cauthy,
                                2: refractive_model.model_sellmeier_2,
                                3: refractive_model.model_sellmeier_3}.get(self.refractive_index_model),
                               wl_raw,
                               complex_n_raw,
                               p0=self.p0,
                               bounds=self.bounds)
        perr = np.sqrt(np.diag(pcov))
        print('模拟结束:\n popt:{};\n pcov:{};\n perr:{}'.format(popt, pcov, perr))

        wl_cal = np.linspace(self.wl_lower_bound, self.wl_upper_bound, 100)
        result_cal = []
        if self.refractive_index_model == 1:
            result_cal = refractive_model.model_cauthy(wl_cal,popt[0],popt[1])
        elif self.refractive_index_model == 2:
            result_cal = refractive_model.model_sellmeier_2(wl_cal,popt[0],popt[1],popt[2],popt[3])
        elif self.refractive_index_model == 3:
            # popt = [0.67805894, 0.37140533, 3.3485284, 0.05628989, 0.10801027, 39.906666]
            result_cal = refractive_model.model_sellmeier_3(wl_cal,popt[0],popt[1],popt[2],popt[3],popt[4],popt[5])
        else:
            print("Unknown Refractive Index Model.")
        n_cal = result_cal[0:wl_cal.size]
        k_cal = result_cal[wl_cal.size:]

        plot_data = SciencePlotData()
        plot_data.add_figure_info(figure_title='Figure n on wl', x_label='wl', y_label='n')
        plot_data.add_plot_data(figure_title='Figure n on wl', x_data=wl_cal, y_data=n_cal, y_legend='n_exp')
        if self.bound_valid:
            plot_data.add_plot_data(figure_title='Figure n on wl', x_data=wl_raw[self.lower_index:self.upper_index], y_data=n_raw[self.lower_index:self.upper_index], y_legend='n_cal')
        else:
            plot_data.add_plot_data(figure_title='Figure n on wl', x_data=wl_raw, y_data=n_raw, y_legend='n_cal')

        plot_data.add_figure_info(figure_title='Figure k on wl', x_label='wl', y_label='k')
        plot_data.add_plot_data(figure_title='Figure k on wl', x_data=wl_cal, y_data=k_cal, y_legend='k_exp')
        if self.bound_valid:
            plot_data.add_plot_data(figure_title='Figure k on wl', x_data=wl_raw[self.lower_index:self.upper_index], y_data=k_raw[self.lower_index:self.upper_index], y_legend='k_cal')
        else:
            plot_data.add_plot_data(figure_title='Figure k on wl', x_data=wl_raw, y_data=k_raw, y_legend='k_cal')
        SciencePlot.sci_plot(plot_data)

        return {'popt': popt, 'pcov': pcov, 'perr': perr}


class RefractiveIndexModel:
    # 波长是um
    def __init__(self):
        print('RefractiveIndexModel.__init__')

    def model_cauthy(self, wl, a, b):
        n = np.add(a, np.divide(b, np.square(wl)))
        # Cauthy模型是不考虑k的 所以k是和n shape相同的0矩阵
        k = np.zeros(n.shape)
        result = np.hstack([n, k])
        return result

    def model_sellmeier_2(self, wl, b1, b2, c1, c2):
        wl2 = np.square(wl)
        term_1 = np.divide(np.divide(
            np.multiply(b1, wl2),
            np.subtract(wl2, np.square(c1))))
        term_2 = np.divide(np.divide(
            np.multiply(b2, wl2),
            np.subtract(wl2, np.square(c2))))
        # 此处特地乘上了 1 + 0 * j 否则np.sqrt负数结果是NaN而不是复数
        complex_n = np.sqrt(np.multiply(np.add(1, np.add(term_1, term_2)), 1 + 0j))
        n = np.real(complex_n)
        k = np.imag(complex_n)
        result = np.hstack([n, k])
        return result


    def model_sellmeier_3(self, wl, b1, b2, b3, c1, c2, c3):
        wl2 = np.square(wl)
        term_1 = np.divide(np.multiply(b1, wl2),
                           np.subtract(wl2, np.square(c1)))
        term_2 = np.divide(np.multiply(b2, wl2),
                           np.subtract(wl2, np.square(c2)))
        term_3 = np.divide(np.multiply(b3, wl2),
                           np.subtract(wl2, np.square(c3)))
        # 此处特地乘上了 1 + 0 * j 否则np.sqrt负数结果是NaN而不是复数
        complex_n = np.sqrt(np.multiply(np.add(1, np.add(term_1, np.add(term_2, term_3))), 1 + 0j))
        n = np.real(complex_n)
        k = np.imag(complex_n)
        result = np.hstack([n, k])
        return result


if __name__ == '__main__':
    file_path_m = "D:\\PycharmProjects\\AIProcessingPlatform\\app\\ui\\Co.csv"
    file_path_d = "D:\\PycharmProjects\\AIProcessingPlatform\\app\\ui\\SrF2.csv"
    first_pos_info_tuple = ('um', 3, 1, 4, 5)
    wl_bound = ('nm', 405., 1550.)

    # Refractive Index Models:
    # 1. Cauthy: n = A + B/λ^2
    # 2. Sellmeier-2: n^2 = 1 + B_1*λ^2/(λ^2-C_1^2) + B_2*λ^2/(λ^2-C_2^2)
    # 3. Sellmeier-3: n^2 = 1 + B_1*λ^2/(λ^2-C_1^2) + B_2*λ^2/(λ^2-C_2^2) + B_3*λ^2/(λ^2-C_3^2)

    # Model 3 Sellmeier
    # B_1,B_2,B_3,C_1,C_2,C_3
    p0 = [0.67805894, 0.37140533, 3.3485284, 0.05628989, 0.10801027, 39.906666]
    bounds = ([0.1, 0.1, 0.1, 0.01, 0.01, 0.01],
              [5, 5, 5, 50, 50, 50])

    simulator = NKSimulator(
        refractive_index_model=3,
        first_pos_info_tuple=first_pos_info_tuple,
        wl_bound=wl_bound,
        p0=p0,
        bounds=bounds
    )
    simulator.simulate(file_name="SrF2", file_path=file_path_d)
