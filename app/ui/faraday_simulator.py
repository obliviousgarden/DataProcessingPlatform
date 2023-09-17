import numpy as np
from scipy.optimize import curve_fit
from app.utils import sci_const
from app.ui.nk_simulator import NKSimulator
from app.utils.science_unit_converter import ScienceUnitConverter
from app.utils.science_unit import ScienceUnit, Unit, science_unit_convert
from app.utils.science_plot import SciencePlot, SciencePlotData
from scipy import interpolate
import csv


# 折射率的模拟已经被解耦出去，这里只有等效介质的模拟不过其中调用了折射率模拟的模块

class FaradaySimulator:
    def __init__(self, faraday_model: int, effective_medium_model: int, first_pos_info_tuple_dict=dict, wl_bound=tuple, p0=None, bounds=()):
        # Faraday计算模型
        self.faraday_model = faraday_model
        # 等效介质模型
        self.effective_medium_model = effective_medium_model

        # 接受并解析第1个数据点的位置信息及其单位信息
        (self.wl_unit_m, self.start_row_m, self.wl_start_rol_m, self.n_start_rol_m,self.k_start_rol_m) = first_pos_info_tuple_dict.get('metal').get('nk')
        (self.wl_unit_epsilon_1_prime_m, self.start_row_epsilon_1_prime_m, self.wl_start_rol_epsilon_1_prime_m, self.e_start_rol_epsilon_1_prime_m) = first_pos_info_tuple_dict.get('metal').get('epsilon_1_prime')
        (self.wl_unit_epsilon_2_prime_m, self.start_row_epsilon_2_prime_m, self.wl_start_rol_epsilon_2_prime_m, self.e_start_rol_epsilon_2_prime_m) = first_pos_info_tuple_dict.get('metal').get('epsilon_2_prime')
        (self.wl_unit_d, self.start_row_d, self.wl_start_rol_d, self.n_start_rol_d,self.k_start_rol_d) = first_pos_info_tuple_dict.get('dielectric').get('nk')
        (self.wl_unit_epsilon_1_prime_d, self.start_row_epsilon_1_prime_d, self.wl_start_rol_epsilon_1_prime_d, self.e_start_rol_epsilon_1_prime_d) = first_pos_info_tuple_dict.get('dielectric').get('epsilon_1_prime')
        (self.wl_unit_epsilon_2_prime_d, self.start_row_epsilon_2_prime_d, self.wl_start_rol_epsilon_2_prime_d, self.e_start_rol_epsilon_2_prime_d) = first_pos_info_tuple_dict.get('dielectric').get('epsilon_2_prime')
        (self.wl_unit_theta_on_wl, self.start_row_theta_on_wl, self.wl_start_rol,
         self.theta_start_rol) = first_pos_info_tuple_dict.get('theta_on_wl')
        # 频率/波长范围
        (self.wl_bound_unit, self.wl_lower_bound, self.wl_upper_bound) = wl_bound
        self.bound_valid = True
        # 一个初始尝试值的数组
        self.p0 = p0
        # 一个参数的边间tuple，左边数组下界，右边数组上届
        self.bounds = bounds

    def open_file(self, path):
        try:
            with open(path, 'r', encoding="utf-8") as file:
                lines = file.readlines()
        except UnicodeDecodeError:
            with open(path, 'r', encoding="shift_jis") as file:
                lines = file.readlines()
        return lines

    def iterate_data(self, file_path: str, start_row: int, x_start_col: int, y_start_col_list: list):
        x_list = []
        y_list_list = []
        for y_index in range(y_start_col_list.__len__()):
            y_list_list.append([])
        null_index_list = []
        lines = self.open_file(path=file_path)
        for line_index in range(int(start_row) - 1, lines.__len__()):
            data = lines[line_index].replace('\n', '').split(',')
            x_list.append(float(data[int(x_start_col) - 1]))
            for y_index in range(y_start_col_list.__len__()):
                y = data[int(y_start_col_list[y_index]) - 1]
                if y == 'NULL':
                    null_index_list.append(line_index-(int(start_row) - 1))
                    y_list_list[y_index].append(np.inf)
                else:
                    y_list_list[y_index].append(float(data[int(y_start_col_list[y_index]) - 1]))
        if null_index_list.__len__() != 0:
            for counter,null_index in enumerate(null_index_list):
                null_index = null_index - counter
                x_list.pop(null_index)
                for y_index in range(y_start_col_list.__len__()):
                    y_list_list[y_index].pop(null_index)

        return x_list, y_list_list

    def get_data(self, file_path_dict: dict):
        wl_list_m, [n_list_m, k_list_m] = self.iterate_data(file_path=file_path_dict.get('metal').get('nk'),
                                                            start_row=self.start_row_m,
                                                            x_start_col=self.wl_start_rol_m,
                                                            y_start_col_list=[self.n_start_rol_m, self.k_start_rol_m])
        wl_list_epsilon_1_prime_m, [e_list_epsilon_1_prime_m] = self.iterate_data(file_path=file_path_dict.get('metal').get('epsilon_1_prime'),
                                                            start_row=self.start_row_epsilon_1_prime_m,
                                                            x_start_col=self.wl_start_rol_epsilon_1_prime_m,
                                                            y_start_col_list=[self.e_start_rol_epsilon_1_prime_m])
        wl_list_epsilon_2_prime_m, [e_list_epsilon_2_prime_m] = self.iterate_data(file_path=file_path_dict.get('metal').get('epsilon_2_prime'),
                                                            start_row=self.start_row_epsilon_2_prime_m,
                                                            x_start_col=self.wl_start_rol_epsilon_2_prime_m,
                                                            y_start_col_list=[self.e_start_rol_epsilon_2_prime_m])

        wl_list_d, [n_list_d, k_list_d] = self.iterate_data(file_path=file_path_dict.get('dielectric').get('nk'),
                                                            start_row=self.start_row_d,
                                                            x_start_col=self.wl_start_rol_d,
                                                            y_start_col_list=[self.n_start_rol_d, self.k_start_rol_d])
        wl_list_epsilon_1_prime_d, [e_list_epsilon_1_prime_d] = self.iterate_data(file_path=file_path_dict.get('dielectric').get('epsilon_1_prime'),
                                                                                  start_row=self.start_row_epsilon_1_prime_d,
                                                                                  x_start_col=self.wl_start_rol_epsilon_1_prime_d,
                                                                                  y_start_col_list=[self.e_start_rol_epsilon_1_prime_d])
        wl_list_epsilon_2_prime_d, [e_list_epsilon_2_prime_d] = self.iterate_data(file_path=file_path_dict.get('dielectric').get('epsilon_2_prime'),
                                                                                  start_row=self.start_row_epsilon_2_prime_d,
                                                                                  x_start_col=self.wl_start_rol_epsilon_2_prime_d,
                                                                                  y_start_col_list=[self.e_start_rol_epsilon_2_prime_d])

        wl_list, [theta_list] = self.iterate_data(file_path=file_path_dict.get('theta_on_wl'),
                                                  start_row=self.start_row_theta_on_wl,
                                                  x_start_col=self.wl_start_rol,
                                                  y_start_col_list=[self.theta_start_rol])
        # 2 波长的单位变换到um
        wl_to_unit = ScienceUnit.Length.um.value
        wl_from_unit_m = ScienceUnit.get_from_symbol(self.wl_unit_m)
        wl_from_unit_epsilon_1_prime_m = ScienceUnit.get_from_symbol(self.wl_unit_epsilon_1_prime_m)
        wl_from_unit_epsilon_2_prime_m = ScienceUnit.get_from_symbol(self.wl_unit_epsilon_2_prime_m)

        wl_from_unit_d = ScienceUnit.get_from_symbol(self.wl_unit_d)
        wl_from_unit_epsilon_1_prime_d = ScienceUnit.get_from_symbol(self.wl_unit_epsilon_1_prime_d)
        wl_from_unit_epsilon_2_prime_d = ScienceUnit.get_from_symbol(self.wl_unit_epsilon_2_prime_d)

        wl_from_unit_theta_on_wl = ScienceUnit.get_from_symbol(self.wl_unit_theta_on_wl)
        if wl_from_unit_m.get_description() != wl_to_unit.get_description():
            wl_list_m = science_unit_convert(from_list=wl_list_m, from_unit=wl_from_unit_m,to_unit=wl_to_unit)
        if wl_from_unit_epsilon_1_prime_m.get_description() != wl_to_unit.get_description():
            wl_list_epsilon_1_prime_m = science_unit_convert(from_list=wl_list_epsilon_1_prime_m, from_unit=wl_from_unit_epsilon_1_prime_m,to_unit=wl_to_unit)
        if wl_from_unit_epsilon_2_prime_m.get_description() != wl_to_unit.get_description():
            wl_list_epsilon_2_prime_m = science_unit_convert(from_list=wl_list_epsilon_2_prime_m, from_unit=wl_from_unit_epsilon_2_prime_m,to_unit=wl_to_unit)
        if wl_from_unit_d.get_description() != wl_to_unit.get_description():
            wl_list_d = science_unit_convert(from_list=wl_list_d, from_unit=wl_from_unit_d,to_unit=wl_to_unit)
        if wl_from_unit_epsilon_1_prime_d.get_description() != wl_to_unit.get_description():
            wl_list_epsilon_1_prime_d = science_unit_convert(from_list=wl_list_epsilon_1_prime_d, from_unit=wl_from_unit_epsilon_1_prime_d,to_unit=wl_to_unit)
        if wl_from_unit_epsilon_2_prime_d.get_description() != wl_to_unit.get_description():
            wl_list_epsilon_2_prime_d = science_unit_convert(from_list=wl_list_epsilon_2_prime_d, from_unit=wl_from_unit_epsilon_2_prime_d,to_unit=wl_to_unit)
        if wl_from_unit_theta_on_wl.get_description() != wl_to_unit.get_description():
            wl_list = science_unit_convert(from_list=wl_list, from_unit=wl_from_unit_theta_on_wl,to_unit=wl_to_unit)
        # 3 wl_bound单位变换,截取需要数据
        wl_from_bound_unit = ScienceUnit.get_from_symbol(self.wl_bound_unit)
        [self.wl_lower_bound] = science_unit_convert(from_list=[self.wl_lower_bound], from_unit=wl_from_bound_unit,to_unit=wl_to_unit)
        [self.wl_upper_bound] = science_unit_convert(from_list=[self.wl_upper_bound], from_unit=wl_from_bound_unit,to_unit=wl_to_unit)


        wl_lower_index,wl_upper_index = self.found_bound_index(wl_list,self.wl_lower_bound,self.wl_upper_bound)
        wl_lower_index_m,wl_upper_index_m = self.found_bound_index(wl_list_m,self.wl_lower_bound,self.wl_upper_bound)
        wl_lower_index_epsilon_1_prime_m,wl_upper_index_epsilon_1_prime_m = self.found_bound_index(wl_list_epsilon_1_prime_m,self.wl_lower_bound,self.wl_upper_bound)
        wl_lower_index_epsilon_2_prime_m,wl_upper_index_epsilon_2_prime_m = self.found_bound_index(wl_list_epsilon_2_prime_m,self.wl_lower_bound,self.wl_upper_bound)
        wl_lower_index_d,wl_upper_index_d = self.found_bound_index(wl_list_d,self.wl_lower_bound,self.wl_upper_bound)
        wl_lower_index_epsilon_1_prime_d,wl_upper_index_epsilon_1_prime_d = self.found_bound_index(wl_list_epsilon_1_prime_d,self.wl_lower_bound,self.wl_upper_bound)
        wl_lower_index_epsilon_2_prime_d,wl_upper_index_epsilon_2_prime_d = self.found_bound_index(wl_list_epsilon_2_prime_d,self.wl_lower_bound,self.wl_upper_bound)
        self.bound_valid = True
        if self.bound_valid:
            # 注意：list[a:b]是不包含b索引对应的内容的所以需要+1
            return (wl_list_m[wl_lower_index_m:wl_upper_index_m+1], n_list_m[wl_lower_index_m:wl_upper_index_m+1], k_list_m[wl_lower_index_m:wl_upper_index_m+1]), \
                   (wl_list_epsilon_1_prime_m[wl_lower_index_epsilon_1_prime_m:wl_upper_index_epsilon_1_prime_m+1], e_list_epsilon_1_prime_m[wl_lower_index_epsilon_1_prime_m:wl_upper_index_epsilon_1_prime_m+1]), \
                   (wl_list_epsilon_2_prime_m[wl_lower_index_epsilon_2_prime_m:wl_upper_index_epsilon_2_prime_m+1], e_list_epsilon_2_prime_m[wl_lower_index_epsilon_2_prime_m:wl_upper_index_epsilon_2_prime_m+1]), \
                   (wl_list_d[wl_lower_index_d:wl_upper_index_d+1], n_list_d[wl_lower_index_d:wl_upper_index_d+1], k_list_d[wl_lower_index_d:wl_upper_index_d+1]), \
                   (wl_list_epsilon_1_prime_d[wl_lower_index_epsilon_1_prime_d:wl_upper_index_epsilon_1_prime_d+1], e_list_epsilon_1_prime_d[wl_lower_index_epsilon_1_prime_d:wl_upper_index_epsilon_1_prime_d+1]), \
                   (wl_list_epsilon_2_prime_d[wl_lower_index_epsilon_2_prime_d:wl_upper_index_epsilon_2_prime_d+1], e_list_epsilon_2_prime_d[wl_lower_index_epsilon_2_prime_d:wl_upper_index_epsilon_2_prime_d+1]), \
                   (wl_list[wl_lower_index:wl_upper_index+1], theta_list[wl_lower_index:wl_upper_index+1])
        else:
            return (wl_list_m, n_list_m, k_list_m), \
                   (wl_list_epsilon_1_prime_m, e_list_epsilon_1_prime_m), \
                   (wl_list_epsilon_2_prime_m, e_list_epsilon_2_prime_m), \
                   (wl_list_d, n_list_d, k_list_d), \
                   (wl_list_epsilon_1_prime_d, e_list_epsilon_1_prime_d), \
                   (wl_list_epsilon_2_prime_d, e_list_epsilon_2_prime_d), \
                   (wl_list, theta_list)

    @staticmethod
    def found_bound_index(data_list:list,lower_bound:float,upper_bound:float):
        lower_index = 0
        upper_index = data_list.__len__()-1
        for index in range(data_list.__len__()-1):
            if index == 0 and data_list[index] == lower_bound:
                lower_index = index
            elif data_list[index] < lower_bound <= data_list[index + 1]:
                lower_index = index
            elif data_list[index] < upper_bound <= data_list[index + 1]:
                upper_index = index + 1
            else:
                pass
        return lower_index,upper_index

    def simulate(self, file_name_dict: dict, file_path_dict: dict):
        # ROUTE A:
        # 1 实验复折射率M,D-->计算复折射率M,D
        # 2 计算复折射率M,D-->计算复介电常数M,D
        # 3 计算复介电常数M,D-->有效介质近似-->计算复介电常数EFF
        # 4 计算复介电常数EFF-->计算FARADAY
        print('开始模拟')
        (wl_raw_m, n_raw_m, k_raw_m), (wl_raw_epsilon_1_prime_m, epsilon_1_prime_raw_m), (wl_raw_epsilon_2_prime_m, epsilon_2_prime_raw_m), (wl_raw_d, n_raw_d, k_raw_d), (wl_raw_epsilon_1_prime_d, epsilon_1_prime_raw_d), (wl_raw_epsilon_2_prime_d, epsilon_2_prime_raw_d), (wl_raw, theta_raw) = self.get_data(file_path_dict=file_path_dict)
        print('获取数据完成')
        print('Metal:[wl(um),n,k]\n{}\n'.format(np.dstack((wl_raw_m,n_raw_m,k_raw_m))))
        print('Dielectric:[wl(um),n,k]\n{}\n'.format(np.dstack((wl_raw_d,n_raw_d,k_raw_d))))
        print('Theta_F on Wavelength:[wl(um),theta_F]\n{}\n'.format(np.dstack((wl_raw,theta_raw))))
        print('开始复折射率转复介电常数epsilon_xx')
        epsilon_1_raw_m,epsilon_2_raw_m = sci_const.n_to_epsilon(n_raw_m,k_raw_m) # epsilon_xx[ Co ] =  epsilon_1_m + i * epsilon_2_m
        epsilon_1_raw_d,epsilon_2_raw_d = sci_const.n_to_epsilon(n_raw_d,k_raw_d) # epsilon_xx[ SrF2 ] =  epsilon_1_d + i * epsilon_2_d
        print('获取复介电常数epsilon_xx完成')
        print('Metal:[wl(um),epsilon_1,epsilon_2]\n{}\n'.format(np.dstack((wl_raw_m,epsilon_1_raw_m,epsilon_2_raw_m))))
        print('Metal:[wl(um),epsilon_1_prime]\n{}\n'.format(np.dstack((wl_raw_epsilon_1_prime_m,epsilon_1_prime_raw_m))))
        print('Metal:[wl(um),epsilon_2_prime]\n{}\n'.format(np.dstack((wl_raw_epsilon_2_prime_m,epsilon_2_prime_raw_m))))
        print('Dielectric:[wl(um),epsilon_1,epsilon_2]\n{}\n'.format(np.dstack((wl_raw_d,epsilon_1_raw_d,epsilon_2_raw_d))))
        print('Dielectric:[wl(um),epsilon_1_prime]\n{}\n'.format(np.dstack((wl_raw_epsilon_1_prime_d,epsilon_1_prime_raw_d))))
        print('Dielectric:[wl(um),epsilon_2_prime]\n{}\n'.format(np.dstack((wl_raw_epsilon_2_prime_d,epsilon_2_prime_raw_d))))
        print('由于Theta_F on Wavelength点数，所以开始在wl_bound的范围上插值')
        wl_list = np.linspace(self.wl_lower_bound,self.wl_upper_bound,100)
        # 插值方式 "nearest","zero","slinear","quadratic","cubic" slinear > cubic > quadratic
        interpolate_func = interpolate.interp1d(wl_raw,theta_raw,kind="slinear")
        interpolate_func_epsilon_1_m = interpolate.interp1d(wl_raw_m,epsilon_1_raw_m,kind="slinear")
        interpolate_func_epsilon_2_m = interpolate.interp1d(wl_raw_m,epsilon_2_raw_m,kind="slinear")
        interpolate_func_epsilon_1_prime_m = interpolate.interp1d(wl_raw_epsilon_1_prime_m,epsilon_1_prime_raw_m,kind="slinear")
        interpolate_func_epsilon_2_prime_m = interpolate.interp1d(wl_raw_epsilon_2_prime_m,epsilon_2_prime_raw_m,kind="slinear")
        interpolate_func_epsilon_1_d = interpolate.interp1d(wl_raw_d,epsilon_1_raw_d,kind="slinear")
        interpolate_func_epsilon_2_d = interpolate.interp1d(wl_raw_d,epsilon_2_raw_d,kind="slinear")
        interpolate_func_epsilon_1_prime_d = interpolate.interp1d(wl_raw_epsilon_1_prime_d,epsilon_1_prime_raw_d,kind="slinear")
        interpolate_func_epsilon_2_prime_d = interpolate.interp1d(wl_raw_epsilon_2_prime_d,epsilon_2_prime_raw_d,kind="slinear")
        theta_list = interpolate_func(wl_list)
        epsilon_1_list_m = interpolate_func_epsilon_1_m(wl_list)
        epsilon_2_list_m = interpolate_func_epsilon_2_m(wl_list)
        epsilon_prime_1_list_m = interpolate_func_epsilon_1_prime_m(wl_list)
        epsilon_prime_2_list_m = interpolate_func_epsilon_2_prime_m(wl_list)
        epsilon_1_list_d = interpolate_func_epsilon_1_d(wl_list)
        epsilon_2_list_d = interpolate_func_epsilon_2_d(wl_list)
        epsilon_prime_1_list_d = interpolate_func_epsilon_1_prime_d(wl_list)
        epsilon_prime_2_list_d = interpolate_func_epsilon_2_prime_d(wl_list)

        if False:
            plot_data = SciencePlotData()

            plot_data.add_figure_info(figure_title='Metal: Epsilon 1 on Wavelength', x_label='Wavelength(um)', y_label='Epsilon 1')
            plot_data.add_plot_data(figure_title='Metal: Epsilon 1 on Wavelength', x_data=wl_raw_m, y_data=epsilon_1_raw_m, y_legend='raw')
            plot_data.add_plot_data(figure_title='Metal: Epsilon 1 on Wavelength', x_data=wl_list, y_data=epsilon_1_list_m, y_legend='interpolate')
            plot_data.add_figure_info(figure_title='Metal: Epsilon 2 on Wavelength', x_label='Wavelength(um)', y_label='Epsilon 2')
            plot_data.add_plot_data(figure_title='Metal: Epsilon 2 on Wavelength', x_data=wl_raw_m, y_data=epsilon_2_raw_m, y_legend='raw')
            plot_data.add_plot_data(figure_title='Metal: Epsilon 2 on Wavelength', x_data=wl_list, y_data=epsilon_2_list_m, y_legend='interpolate')

            plot_data.add_figure_info(figure_title='Metal: Epsilon Prime 1 on Wavelength', x_label='Wavelength(um)', y_label='Epsilon Prime 1')
            plot_data.add_plot_data(figure_title='Metal: Epsilon Prime 1 on Wavelength', x_data=wl_raw_epsilon_1_prime_m, y_data=wl_raw_epsilon_1_prime_m, y_legend='raw')
            plot_data.add_plot_data(figure_title='Metal: Epsilon Prime 1 on Wavelength', x_data=wl_list, y_data=epsilon_prime_1_list_m, y_legend='interpolate')
            plot_data.add_figure_info(figure_title='Metal: Epsilon Prime 2 on Wavelength', x_label='Wavelength(um)', y_label='Epsilon Prime 2')
            plot_data.add_plot_data(figure_title='Metal: Epsilon Prime 2 on Wavelength', x_data=wl_raw_epsilon_2_prime_m, y_data=wl_raw_epsilon_2_prime_m, y_legend='raw')
            plot_data.add_plot_data(figure_title='Metal: Epsilon Prime 2 on Wavelength', x_data=wl_list, y_data=epsilon_prime_2_list_m, y_legend='interpolate')

            plot_data.add_figure_info(figure_title='Dielectric: Epsilon 1 on Wavelength', x_label='Wavelength(um)', y_label='Epsilon 1')
            plot_data.add_plot_data(figure_title='Dielectric: Epsilon 1 on Wavelength', x_data=wl_raw_d, y_data=epsilon_1_raw_d, y_legend='raw')
            plot_data.add_plot_data(figure_title='Dielectric: Epsilon 1 on Wavelength', x_data=wl_list, y_data=epsilon_1_list_d, y_legend='interpolate')
            plot_data.add_figure_info(figure_title='Dielectric: Epsilon 2 on Wavelength', x_label='Wavelength(um)', y_label='Epsilon 2')
            plot_data.add_plot_data(figure_title='Dielectric: Epsilon 2 on Wavelength', x_data=wl_raw_d, y_data=epsilon_2_raw_d, y_legend='raw')
            plot_data.add_plot_data(figure_title='Dielectric: Epsilon 2 on Wavelength', x_data=wl_list, y_data=epsilon_2_list_d, y_legend='interpolate')

            plot_data.add_figure_info(figure_title='Dielectric: Epsilon Prime 1 on Wavelength', x_label='Wavelength(um)', y_label='Epsilon Prime 1')
            plot_data.add_plot_data(figure_title='Dielectric: Epsilon Prime 1 on Wavelength', x_data=wl_raw_epsilon_1_prime_d, y_data=wl_raw_epsilon_1_prime_d, y_legend='raw')
            plot_data.add_plot_data(figure_title='Dielectric: Epsilon Prime 1 on Wavelength', x_data=wl_list, y_data=epsilon_prime_1_list_d, y_legend='interpolate')
            plot_data.add_figure_info(figure_title='Dielectric: Epsilon Prime 2 on Wavelength', x_label='Wavelength(um)', y_label='Epsilon Prime 2')
            plot_data.add_plot_data(figure_title='Dielectric: Epsilon Prime 2 on Wavelength', x_data=wl_raw_epsilon_2_prime_d, y_data=wl_raw_epsilon_2_prime_d, y_legend='raw')
            plot_data.add_plot_data(figure_title='Dielectric: Epsilon Prime 2 on Wavelength', x_data=wl_list, y_data=epsilon_prime_2_list_d, y_legend='interpolate')

            SciencePlot.sci_plot(plot_data)


        print('插值结束')
        print('开始利用FARADAY模型进行拟合')
        epsilon_xx_list_m = np.add(epsilon_1_list_m,np.multiply(epsilon_2_list_m,1j))
        epsilon_xy_list_m = np.add(epsilon_prime_1_list_m,np.multiply(epsilon_prime_2_list_m,1j))
        epsilon_xx_list_d = np.add(epsilon_1_list_d,np.multiply(epsilon_2_list_d,1j))
        epsilon_xy_list_d = np.add(epsilon_prime_1_list_d,np.multiply(epsilon_prime_2_list_d,1j))
        ema_model = EffectiveMediumModel(epsilon_xx_list_m,epsilon_xy_list_m,epsilon_xx_list_d,epsilon_xy_list_d)

        faraday_model = FaradayModel({1: ema_model.maxwell_garnett,
                                      2: ema_model.bruggeman,
                                      3: ema_model.belyaev}.get(self.effective_medium_model))
        # 模型 1: Maxwell Garnett Equation, 模型 2: Bruggeman's Model, 模型 3: Belyaev's Model
        popt, pcov = curve_fit({1: faraday_model.nk,
                                2: faraday_model.epsilon,
                                3: faraday_model.epsilon_xx}.get(self.faraday_model),
                               wl_list,
                               theta_list,
                               p0=self.p0,
                               bounds=self.bounds)
        perr = np.sqrt(np.diag(pcov))
        if self.faraday_model == 2:
            theta_cal = faraday_model.epsilon(wl_list,popt[0])
            theta_Co = faraday_model.epsilon(wl_list,1.)
            theta_SrF2 = faraday_model.epsilon(wl_list,0.)
        elif self.faraday_model == 3:
            theta_cal = faraday_model.epsilon_xx(wl_list,popt[0])
            theta_Co = faraday_model.epsilon_xx(wl_list,1.)
            theta_SrF2 = faraday_model.epsilon_xx(wl_list,0.)
        if True:
            plot_data = SciencePlotData()
            plot_data.add_figure_info(figure_title='Faraday on Wavelength', x_label='Wavelength(um)', y_label='Theta(deg./um)')
            plot_data.add_plot_data(figure_title='Faraday on Wavelength', x_data=wl_raw, y_data=theta_raw, y_legend='Co-Sr-F,Raw.')
            plot_data.add_plot_data(figure_title='Faraday on Wavelength', x_data=wl_list, y_data=theta_list, y_legend='Co-Sr-F,Exp.')
            plot_data.add_plot_data(figure_title='Faraday on Wavelength', x_data=wl_list, y_data=theta_cal, y_legend='Co-Sr-F,Cal.,c_m='+str(popt[0]))
            # plot_data.add_plot_data(figure_title='Faraday on Wavelength', x_data=wl_list, y_data=theta_Co, y_legend='Co,Cal.,c_m='+str(1.0))
            # plot_data.add_plot_data(figure_title='Faraday on Wavelength', x_data=wl_list, y_data=theta_SrF2, y_legend='SrF2,Cal.,c_m='+str(0.0))
            SciencePlot.sci_plot(plot_data)

        print('模拟完成')
        print('开始写入文件')
        file_name_write = file_name_dict.get('theta_on_wl')
        file_path_write = "D:\\PycharmProjects\\AIProcessingPlatform\\app\\ui\\output\\" + file_name_write + ".csv"
        with open(file_path_write,'w',newline='') as f:
            csv_writer = csv.writer(f)
            first_line = ['c_m',str(popt[0])]
            heads_1 = ['wl_raw(um)','theta_raw(deg./um)']
            heads_2 = ['wl_raw(um)','theta_list(deg./um)','theta_cal(deg./um)']
            rows_1 = zip(wl_raw,theta_raw)
            rows_2 = zip(wl_list,theta_list,theta_cal)
            csv_writer.writerow(first_line)
            csv_writer.writerow(heads_1)
            for row in rows_1:
                csv_writer.writerow(row)
            csv_writer.writerow(heads_2)
            for row in rows_2:
                csv_writer.writerow(row)

        print('写入文件完成')


        # ROUTE B:
        # 1 实验复折射率M,D-->计算复折射率M,D
        # 2,3 计算复折射率M,D-->有效介质近似(需要自己设计)-->计算复折射率EFF
        # 4 计算复折射率EFF-->计算FARADAY


class EffectiveMediumModel:
    def __init__(self,epsilon_xx_list_m,epsilon_xy_list_m,epsilon_xx_list_d,epsilon_xy_list_d):
        print('EffectiveMediumModel.__init__')
        self.epsilon_xx_m = epsilon_xx_list_m
        self.epsilon_xy_m = epsilon_xy_list_m
        self.epsilon_xx_d = epsilon_xx_list_d
        self.epsilon_xy_d = epsilon_xy_list_d

    def maxwell_garnett(self, c_m):
        term_xx_1 = np.multiply(3.*c_m,np.subtract(self.epsilon_xx_m,self.epsilon_xx_d))
        term_xx_2 = np.multiply(1.-c_m,self.epsilon_xx_m)
        term_xx_3 = np.multiply(2.+c_m,self.epsilon_xx_d)
        epsilon_xx_eff = np.multiply(self.epsilon_xx_d,np.add(1.,np.divide(term_xx_1,np.add(term_xx_2,term_xx_3))))

        term_xy_1 = np.multiply(3.*c_m,np.subtract(self.epsilon_xy_m,self.epsilon_xy_d))
        term_xy_2 = np.multiply(1.-c_m,self.epsilon_xy_m)
        term_xy_3 = np.multiply(2.+c_m,self.epsilon_xy_d)
        epsilon_xy_eff = np.multiply(self.epsilon_xy_d,np.add(1.,np.divide(term_xy_1,np.add(term_xy_2,term_xy_3))))
        if False:
            x = np.linspace(0, 1, epsilon_xx_eff.__len__())
            plot_data = SciencePlotData()

            plot_data.add_figure_info(figure_title='Real[Epsilon_XX]', x_label='x', y_label='Epsilon')
            plot_data.add_plot_data(figure_title='Real[Epsilon_XX]', x_data=x, y_data=np.real(self.epsilon_xx_m), y_legend='Metal')
            plot_data.add_plot_data(figure_title='Real[Epsilon_XX]', x_data=x, y_data=np.real(self.epsilon_xx_d), y_legend='Dielectric')
            plot_data.add_plot_data(figure_title='Real[Epsilon_XX]', x_data=x, y_data=np.real(epsilon_xx_eff), y_legend='Effective')
            plot_data.add_figure_info(figure_title='Real[Epsilon_XY]', x_label='x', y_label='Epsilon')
            plot_data.add_plot_data(figure_title='Real[Epsilon_XY]', x_data=x, y_data=np.real(self.epsilon_xy_m), y_legend='Metal')
            # plot_data.add_plot_data(figure_title='Real[Epsilon_XY]', x_data=x, y_data=np.real(self.epsilon_xy_d), y_legend='Dielectric')
            plot_data.add_plot_data(figure_title='Real[Epsilon_XY]', x_data=x, y_data=np.real(epsilon_xy_eff), y_legend='Effective')

            plot_data.add_figure_info(figure_title='Imag[Epsilon_XX]', x_label='x', y_label='Epsilon')
            plot_data.add_plot_data(figure_title='Imag[Epsilon_XX]', x_data=x, y_data=np.imag(self.epsilon_xx_m), y_legend='Metal')
            plot_data.add_plot_data(figure_title='Imag[Epsilon_XX]', x_data=x, y_data=np.imag(self.epsilon_xx_d), y_legend='Dielectric')
            plot_data.add_plot_data(figure_title='Imag[Epsilon_XX]', x_data=x, y_data=np.imag(epsilon_xx_eff), y_legend='Effective')
            plot_data.add_figure_info(figure_title='Imag[Epsilon_XY]', x_label='x', y_label='Epsilon')
            plot_data.add_plot_data(figure_title='Imag[Epsilon_XY]', x_data=x, y_data=np.imag(self.epsilon_xy_m), y_legend='Metal')
            # plot_data.add_plot_data(figure_title='Imag[Epsilon_XY]', x_data=x, y_data=np.imag(self.epsilon_xy_d), y_legend='Dielectric')
            plot_data.add_plot_data(figure_title='Imag[Epsilon_XY]', x_data=x, y_data=np.imag(epsilon_xy_eff), y_legend='Effective')

            SciencePlot.sci_plot(plot_data)
        return [epsilon_xx_eff,epsilon_xy_eff]

    def bruggeman(self, c_m):
        h_b = np.subtract(np.multiply(2 - 3 * c_m, self.epsilon_xx_d),
                          np.multiply(1 - 3 * c_m, self.epsilon_xx_m))
        term_1 = np.sqrt(np.add(np.square(h_b),
                                np.multiply(8., np.multiply(self.epsilon_xx_m, self.epsilon_xx_d))))
        epsilon_eff = np.divide(np.add(h_b,
                                       term_1),
                                4.)
        return [epsilon_eff,[]]

    def belyaev(self, c_m, wl, a, miu_m):
        omega = np.divide(2 * np.pi, wl)
        k_m = np.multiply(np.sqrt(self.epsilon_xx_m, miu_m),
                          np.divide(omega, sci_const.c))
        x = np.multiply(k_m, a)
        term_x = np.subtract(1, np.multiply(x, np.reciprocal(np.tan(x))))
        j_x = np.multiply(2,
                          np.divide(term_x,
                                    np.subtract(np.square(x), term_x)))
        epsilon_m_times_j = np.multiply(self.epsilon_xx_m, j_x)
        h_b = np.subtract(np.multiply(2 - 3 * c_m, self.epsilon_xx_d),
                          np.multiply(1 - 3 * c_m, epsilon_m_times_j))
        term_1 = np.sqrt(np.add(np.square(h_b),
                                np.multiply(8., np.multiply(epsilon_m_times_j, self.epsilon_xx_d))))
        epsilon_eff = np.divide(np.add(h_b,
                                       term_1),
                                4.)
        return [epsilon_eff,[]]


class FaradayModel:
    def __init__(self,effective_medium_func):
        self.effective_medium_func = effective_medium_func


    def nk(self, wl, c_m):
        # TODO:
        pass

    def epsilon(self, wl, c_m):
        [epsilon_xx_eff,epsilon_xy_eff] = self.effective_medium_func(c_m)
        term_1 = np.divide(np.pi,wl) # um*-1
        term_2 = np.imag(np.divide(epsilon_xy_eff,np.sqrt(epsilon_xx_eff)))
        theta_f_rad = np.multiply(term_1,term_2) # rad./um
        theta_f_deg = np.multiply(theta_f_rad,180./np.pi) # deg./um
        theta_f = theta_f_deg

        # term_1 = np.divide(np.pi,wl)
        # term_2 = np.add(epsilon_xx_eff,np.multiply(epsilon_xy_eff,1.j))
        # term_3 = np.subtract(epsilon_xx_eff,np.multiply(epsilon_xy_eff,1.j))
        # # delta_n = np.imag(np.divide(epsilon_xy_eff,np.sqrt(epsilon_xx_eff)))
        # # delta_n = np.subtract(np.sqrt(term_2),np.sqrt(term_3))
        # delta_n = np.subtract(np.sqrt(term_3),np.sqrt(term_2))
        # complex_theta_f = np.multiply(np.multiply(delta_n,term_1),180./np.pi)
        # # complex_theta_f = np.multiply(np.multiply(delta_n,term_1),1.)
        # real_theta_f = np.real(complex_theta_f)  # deg./um
        # # imag_theta_f = np.imag(complex_theta_f)  # deg./um

        return theta_f

    def epsilon_xx(self, wl, c_m):
        [epsilon_xx_eff,epsilon_xy_eff] = self.effective_medium_func(c_m)
        term_1 = np.divide(np.pi,wl)
        epsilon_xx = np.real(epsilon_xx_eff)
        epsilon_xy = np.multiply(np.imag(epsilon_xx_eff),-1.j)
        term_2 = np.divide(epsilon_xy,np.sqrt(epsilon_xx))
        theta_f_rad = np.multiply(1.j,np.multiply(term_1,term_2))  # complex theta_f rad./um
        theta_f_deg = np.multiply(theta_f_rad,180./np.pi)  # complex theta_f deg./um
        theta_f = np.real(theta_f_deg)
        if False:
            plot_data = SciencePlotData()
            plot_data.add_figure_info(figure_title='Faraday on Wavelength', x_label='Wavelength(um)', y_label='Theta(deg./um)')
            plot_data.add_plot_data(figure_title='Faraday on Wavelength', x_data=wl, y_data=theta_f, y_legend='Effective,c_m='+str(c_m))
            SciencePlot.sci_plot(plot_data)
        return theta_f



if __name__ == '__main__':
    # Faraday Models:
    # 1. NK
    # 2. Epsilon
    # 3. Epsilon_xx

    # Effective Medium Models:
    # 1. Maxwell Garnett Equation
    # 2. Bruggeman's Model
    # 3. Belyaev's Model

    file_path_nk_m = "D:\\PycharmProjects\\AIProcessingPlatform\\app\\ui\\Co_1.csv" # Complex[ epsilon_xx ] of Co
    file_path_epsilon_1_prime_m = "D:\\PycharmProjects\\AIProcessingPlatform\\app\\ui\\Co-epsilon_1'.csv" # Re[ epsilon_xy ] of Co
    file_path_epsilon_2_prime_m = "D:\\PycharmProjects\\AIProcessingPlatform\\app\\ui\\Co-epsilon_2'.csv" # Im[ epsilon_xy ] of Co
    file_path_nk_d = "D:\\PycharmProjects\\AIProcessingPlatform\\app\\ui\\SrF2.csv" # Complex[ epsilon_xx ] of SrF2
    file_path_epsilon_1_prime_d = "D:\\PycharmProjects\\AIProcessingPlatform\\app\\ui\SrF2-epsilon_1'.csv" # Re[ epsilon_xy ] of SrF2
    file_path_epsilon_2_prime_d = "D:\\PycharmProjects\\AIProcessingPlatform\\app\\ui\\SrF2-epsilon_2'.csv" # Im[ epsilon_xy ] of SrF2

    first_pos_info_tuple_nk_m = ('um', 3, 1, 4, 5)
    first_pos_info_tuple_nk_epsilon_1_prime_m = ('um', 2, 3, 4)
    first_pos_info_tuple_nk_epsilon_2_prime_m = ('um', 2, 3, 4)
    first_pos_info_tuple_nk_d = ('um', 3, 1, 4, 5)
    first_pos_info_tuple_nk_epsilon_1_prime_d = ('um', 2, 1, 2)
    first_pos_info_tuple_nk_epsilon_2_prime_d = ('um', 2, 1, 2)
    wl_bound = ('nm', 405., 1550.)

    file_path_theta_on_wl = "D:\\PycharmProjects\\AIProcessingPlatform\\app\\ui\\theta-wavelength.csv"
    first_pos_info_tuple_theta_on_wl = ('nm', 4, 1, 16)
    # FIXME:暂时使用以后删除
    info_dict = {'11.40':2,
                 '14.89':3,
                 '17.51':4,
                 '23.21':5,
                 '29.01':6,
                 '37.99':10,
                 '40.97':11,
                 '62.33':14,
                 '71.57':15,
                 '75.64':16,
                 '82.83':17}

    # c_m(vol%), d(um)
    # p0 = [0.4718]
    p0 = [0.5]
    bounds = ([0.1],
              [1.])

    # FIXME:暂时这样以后把循环放开
    for key,value in info_dict.items():
        first_pos_info_tuple_theta_on_wl = ('nm', 4, 1, value)
        faraday_simulator = FaradaySimulator(
            faraday_model=2,
            effective_medium_model=1,
            first_pos_info_tuple_dict={
                'metal': {
                    'nk':first_pos_info_tuple_nk_m,
                    'epsilon_1_prime':first_pos_info_tuple_nk_epsilon_1_prime_m,
                    'epsilon_2_prime':first_pos_info_tuple_nk_epsilon_2_prime_m},
                'dielectric': {
                    'nk':first_pos_info_tuple_nk_d,
                    'epsilon_1_prime':first_pos_info_tuple_nk_epsilon_1_prime_d,
                    'epsilon_2_prime':first_pos_info_tuple_nk_epsilon_2_prime_d},
                'theta_on_wl': first_pos_info_tuple_theta_on_wl
            },
            wl_bound=wl_bound,
            p0=p0,
            bounds=bounds
        )

        faraday_simulator.simulate(file_name_dict={'metal': 'Co',
                                                   'dielectric': 'SrF2',
                                                   'theta_on_wl': 'Co-SrF2-'+key+'at.%'},
                                   file_path_dict={
                                       'metal': {
                                           'nk':file_path_nk_m,
                                           'epsilon_1_prime':file_path_epsilon_1_prime_m,
                                           'epsilon_2_prime':file_path_epsilon_2_prime_m},
                                       'dielectric': {
                                           'nk':file_path_nk_d,
                                           'epsilon_1_prime':file_path_epsilon_1_prime_d,
                                           'epsilon_2_prime':file_path_epsilon_2_prime_d},
                                       'theta_on_wl': file_path_theta_on_wl})
