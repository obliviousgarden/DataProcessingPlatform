import matplotlib
from PyQt5 import QtCore, QtGui, QtWidgets
import numpy as np
import os
import matplotlib.pyplot as plt
import random

from PyQt5.QtWidgets import QApplication

matplotlib.use("Qt5Agg")  # 声明使用QT5
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from dielectric_simulator import DielectricSimulator, func_Havriliak_Negami, \
    func_Cole_Cole, func_Cole_Davidson, func_Debye


class ModalDielectric(object):

    def __init__(self,parent):
        self.parent = parent
        # 模型 1:Havriliak–Negami,模型 2:Cole-Cole,模型 3:Cole–Davidson,模型 4:Debye
        # 默认模型为1:Havriliak–Negami
        self.model = 1

        self.alpha_min = 0.0
        self.alpha_max = 1.0
        self.beta_min = 0.0
        self.beta_max = 1.0
        self.tau_min = 1e-9
        self.tau_max = 1e-3
        self.epsiloninf_min = 0.0
        self.epsiloninf_max = 50.0
        self.deltaepsilon_min = 100.0
        self.deltaepsilon_max = 1000.0

        self.alpha = self.alpha_max
        self.beta = self.beta_max
        self.tau = self.tau_min
        self.epsiloninf = self.epsiloninf_min
        self.deltaepsilon = self.deltaepsilon_min

        self.file_path = []
        self.file_name = []
        self.list_view_file_model = QtCore.QStringListModel()
        self.list_view_file_model.setStringList(self.file_name)
        self.list_view_results_model = QtGui.QStandardItemModel()
        # file_name为键
        self.result_dict = {}
        # plot相关(需要考虑下修改，这里实际上是对UI的初始化)
        self.dialog_plot = QtWidgets.QDialog()
        self.tab_plot = QtWidgets.QTabWidget(self.dialog_plot)
        self.tab_plot.setGeometry(QtCore.QRect(10, 0, 1061, 678))
        self.tab_plot_epsilon = QtWidgets.QWidget()
        self.tab_plot_delta_epsilon = QtWidgets.QWidget()
        self.tab_plot.addTab(self.tab_plot_epsilon,"")
        self.tab_plot.addTab(self.tab_plot_delta_epsilon,"")
        _translate = QtCore.QCoreApplication.translate
        self.tab_plot.setTabText(self.tab_plot.indexOf(self.tab_plot_epsilon), _translate("MainWindow", "Epsilon"))
        self.tab_plot.setTabText(self.tab_plot.indexOf(self.tab_plot_delta_epsilon), _translate("MainWindow", "Delta_Epsilon"))

        self.layout_plot_epsilon = QtWidgets.QVBoxLayout()
        self.layout_plot_delta_epsilon = QtWidgets.QVBoxLayout()
        self.fig_epsilon = plt.Figure()
        self.canvas_epsilon = FigureCanvas(self.fig_epsilon)
        self.fig_delta_epsilon = plt.Figure()
        self.canvas_delta_epsilon = FigureCanvas(self.fig_delta_epsilon)

    def setupUi(self):
        # 初始化参数UI
        self.parent.Slider_alpha.setValue((self.alpha - self.alpha_min) / (self.alpha_max - self.alpha_min) * 100.0)
        self.parent.Slider_beta.setValue((self.beta - self.beta_min) / (self.beta_max - self.beta_min) * 100.0)
        self.parent.Slider_tau.setValue(
            (np.log10(self.tau) - np.log10(self.tau_min)) / (np.log10(self.tau_max / self.tau_min)) * 100.0)
        self.parent.Slider_epsiloninf.setValue(
            (self.epsiloninf - self.epsiloninf_min) / (self.epsiloninf_max - self.epsiloninf_min) * 100.0)
        self.parent.Slider_deltaepsilon.setValue(
            (self.deltaepsilon - self.deltaepsilon_min) / (self.deltaepsilon_max - self.deltaepsilon_min) * 100.0)
        self.parent.lcdNumber_alpha.display(str(self.alpha))
        self.parent.lcdNumber_beta.display(str(self.beta))
        self.parent.lcdNumber_tau.display('%.1e' % self.tau)
        self.parent.lcdNumber_epsiloninf.display(str(self.epsiloninf))
        self.parent.lcdNumber_deltaepsilon.display(str(self.deltaepsilon))

        # 设定RadioButton
        self.parent.RadioButton_hnmodel.clicked.connect(self.on_RadioButton_clicked)
        self.parent.RadioButton_ccmodel.clicked.connect(self.on_RadioButton_clicked)
        self.parent.RadioButton_cdmodel.clicked.connect(self.on_RadioButton_clicked)
        self.parent.RadioButton_dmodel.clicked.connect(self.on_RadioButton_clicked)
        # 设定PushButton
        self.parent.PushButton_file.clicked.connect(self.on_PushButton_file_clicked)
        self.parent.PushButton_dir.clicked.connect(self.on_PushButton_dir_clicked)
        self.parent.PushButton_simulate.clicked.connect(self.on_PushButton_simulate_clicked)
        self.parent.PushButton_plot.clicked.connect(self.on_PushButton_plot_clicked)
        self.parent.PushButton_clearAll.clicked.connect(self.on_PushButton_clearAll_clicked)
        # 设定ListView内的Model
        self.parent.ListView_file.setModel(self.list_view_file_model)
        self.parent.ListView_results.setModel(self.list_view_results_model)
        # 禁止编辑
        self.parent.ListView_file.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.parent.ListView_results.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        # 设定Slider
        self.parent.Slider_alpha.valueChanged.connect(self.on_Slider_alpha_valueChanged)
        self.parent.Slider_beta.valueChanged.connect(self.on_Slider_beta_valueChanged)
        self.parent.Slider_tau.valueChanged.connect(self.on_Slider_tau_valueChanged)
        self.parent.Slider_epsiloninf.valueChanged.connect(self.on_Slider_epsiloninf_valueChanged)
        self.parent.Slider_deltaepsilon.valueChanged.connect(self.on_Slider_deltaepsilon_valueChanged)
        # 初始化figure和cavas
        self.dialog_plot.setWindowTitle('Plot results - Dielectric Simulation')
        self.layout_plot_epsilon.addWidget(self.canvas_epsilon)
        self.tab_plot_epsilon.setLayout(self.layout_plot_epsilon)
        self.layout_plot_delta_epsilon.addWidget(self.canvas_delta_epsilon)
        self.tab_plot_delta_epsilon.setLayout(self.layout_plot_delta_epsilon)
        # Checkbox
        self.parent.CheckBox_All.clicked.connect(self.on_CheckBox_All_clicked)
        self.parent.CheckBox_reference.clicked.connect(self.on_CheckBox_reference_clicked)

        # 初始化没有数据所以直接禁用
        self.parent.CheckBox_All.setEnabled(False)
        self.parent.PushButton_plot.setEnabled(False)
        self.parent.CheckBox_reference.setEnabled(False)
        self.parent.ComboBox_reference.setEnabled(False)

    def update_tau_range(self, tau_min, tau_max):
        print("更新Tau范围,结束")
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.tau = self.tau_min
        self.parent.Slider_tau.setValue(
            (np.log10(self.tau) - np.log10(self.tau_min)) / (np.log10(self.tau_max / self.tau_min)) * 100.0)
        self.parent.lcdNumber_tau.display('%.1e' % self.tau)

    def update_epsilon_inf_range(self, epsilon_inf_min, epsilon_inf_max):
        print("更新epsilon_inf范围,结束")
        self.epsiloninf_min = epsilon_inf_min
        self.epsiloninf_max = epsilon_inf_max
        self.epsiloninf = self.epsiloninf_min
        self.parent.Slider_epsiloninf.setValue(
            (self.epsiloninf - self.epsiloninf_min) / (self.epsiloninf_max - self.epsiloninf_min) * 100.0)
        self.parent.lcdNumber_epsiloninf.display(str(self.epsiloninf))

    def update_delta_epsilon_range(self, delta_epsilon_min, delta_epsilon_max):
        print("更新delta_epsilon范围,结束")
        self.deltaepsilon_min = delta_epsilon_min
        self.deltaepsilon_max = delta_epsilon_max
        self.deltaepsilon = self.deltaepsilon_min
        self.parent.Slider_deltaepsilon.setValue(
            (self.deltaepsilon - self.deltaepsilon_min) / (self.deltaepsilon_max - self.deltaepsilon_min) * 100.0)
        self.parent.lcdNumber_deltaepsilon.display(str(self.deltaepsilon))

    def on_RadioButton_clicked(self):
        if self.parent.RadioButton_hnmodel.isChecked():
            self.model = 1
            self.parent.Slider_alpha.setEnabled(True)
            self.parent.Slider_beta.setEnabled(True)
        elif self.parent.RadioButton_ccmodel.isChecked():
            self.model = 2
            self.parent.Slider_alpha.setEnabled(True)
            self.parent.Slider_beta.setEnabled(False)
            self.parent.Slider_beta.setValue(100)
        elif self.parent.RadioButton_cdmodel.isChecked():
            self.model = 3
            self.parent.Slider_alpha.setEnabled(False)
            self.parent.Slider_beta.setEnabled(True)
            self.parent.Slider_alpha.setValue(100)
        else:
            self.model = 4
            self.parent.Slider_alpha.setEnabled(False)
            self.parent.Slider_beta.setEnabled(False)
            self.parent.Slider_alpha.setValue(100)
            self.parent.Slider_beta.setValue(100)

    def on_PushButton_file_clicked(self):
        self.file_name = []
        self.file_path = []
        file_name, file_type = QtWidgets.QFileDialog.getOpenFileNames()
        print('file:', file_name)
        self.file_path = file_name
        for i in range(file_name.__len__()):
            self.file_name.append(file_name[i].split('/')[-1])
        self.list_view_file_model.setStringList(self.file_name)
        print(self.file_path)
        print(self.file_name)

    def on_PushButton_dir_clicked(self):
        self.file_name = []
        self.file_path = []
        dir_ = QtWidgets.QFileDialog.getExistingDirectory()
        print('dir:', dir_)
        if dir_ != '':
            self.file_name = os.listdir(dir_)
            for i in range(self.file_name.__len__()):
                self.file_path.append(dir_ + '/' + self.file_name[i])
            self.list_view_file_model.setStringList(self.file_name)
        print(self.file_path)
        print(self.file_name)

    def on_Slider_alpha_valueChanged(self):
        self.alpha = self.alpha_min + (self.alpha_max - self.alpha_min) * self.parent.Slider_alpha.value() / 100.0
        self.parent.lcdNumber_alpha.display(str(self.alpha))

    def on_Slider_beta_valueChanged(self):
        self.beta = self.beta_min + (self.beta_max - self.beta_min) * self.parent.Slider_beta.value() / 100.0
        self.parent.lcdNumber_beta.display(str(self.beta))

    def on_Slider_tau_valueChanged(self):
        self.tau = np.power(10, np.log10(self.tau_min) + (
                np.log10(self.tau_max) - np.log10(self.tau_min)) * self.parent.Slider_tau.value() / 100.0)
        self.parent.lcdNumber_tau.display('%.1e' % self.tau)

    def on_Slider_epsiloninf_valueChanged(self):
        self.epsiloninf = self.epsiloninf_min + (
                self.epsiloninf_max - self.epsiloninf_min) * self.parent.Slider_epsiloninf.value() / 100.0
        self.parent.lcdNumber_epsiloninf.display(str(self.epsiloninf))

    def on_Slider_deltaepsilon_valueChanged(self):
        self.deltaepsilon = self.deltaepsilon_min + (
                self.deltaepsilon_max - self.deltaepsilon_min) * self.parent.Slider_deltaepsilon.value() / 100.0
        self.parent.lcdNumber_deltaepsilon.display(str(self.deltaepsilon))

    def on_PushButton_simulate_clicked(self):
        # 清空列表
        self.on_PushButton_clearAll_clicked()
        # 记得清空一下结果字典
        self.result_dict = {}
        # 全选取消
        self.parent.CheckBox_All.setCheckState(QtCore.Qt.Unchecked)
        # 这里需要根据model构造p0和bounds参数
        p0 = []
        bounds = ([], [])
        if self.model == 1:
            p0.append(self.alpha)
            p0.append(self.beta)
            bounds[0].append(self.alpha_min)
            bounds[0].append(self.beta_min)
            bounds[1].append(self.alpha_max)
            bounds[1].append(self.beta_max)
        elif self.model == 2:
            p0.append(self.alpha)
            bounds[0].append(self.alpha_min)
            bounds[1].append(self.alpha_max)
        elif self.model == 3:
            p0.append(self.beta)
            bounds[0].append(self.beta_min)
            bounds[1].append(self.beta_max)
        else:
            pass
        p0.extend([self.tau, self.epsiloninf, self.deltaepsilon])
        bounds[0].extend([self.tau_min, self.epsiloninf_min, self.deltaepsilon_min])
        bounds[1].extend([self.tau_max, self.epsiloninf_max, self.deltaepsilon_max])
        print('初始值', p0)
        print('边界', bounds)
        self.list_view_results_model.clear()
        for i in range(self.file_path.__len__()):
            print(i, self.file_name[i])
            self.result_dict[self.file_name[i]] = DielectricSimulator(self.model, self.file_path[i],
                                                                      p0=p0,
                                                                      bounds=bounds).simulate()
            result = self.result_dict[self.file_name[i]]['popt']

            item = QtGui.QStandardItem(" #{}\nAlpha={},Beta={}Tau={}\nEpsilon_Inf={}Delta_Epsilon={}"
                                       .format(self.file_name[i]
                                               , {1: result[0], 2: result[0], 3: 1.0, 4: 1.0}[self.model]
                                               , {1: result[1], 2: 1.0, 3: result[0], 4: 1.0}[self.model]
                                               , result[-3], result[-2], result[-1]))
            item.setCheckable(True)
            self.list_view_results_model.appendRow(item)
        print('结果', self.result_dict)
        # 拟合完成，可以选择背底，以及绘图了
        self.parent.CheckBox_All.setEnabled(True)
        self.parent.PushButton_plot.setEnabled(True)
        if self.list_view_results_model.rowCount() > 1:
            self.parent.CheckBox_reference.setEnabled(True)
            # 加入combobox的每一项
            for key in self.result_dict:
                self.parent.ComboBox_reference.addItem(key)

    def on_CheckBox_All_clicked(self):
        check_state = QtCore.Qt.Checked if self.parent.CheckBox_All.isChecked() else QtCore.Qt.Unchecked

        for i in range(self.list_view_results_model.rowCount()):
            self.list_view_results_model.item(i).setCheckState(check_state)

    def on_CheckBox_reference_clicked(self):
        self.parent.ComboBox_reference.setEnabled(True if self.parent.CheckBox_reference.isChecked() else False)

    def on_PushButton_plot_clicked(self):
        plot_data_dict = {}
        for i in range(self.list_view_results_model.rowCount()):
            item = self.list_view_results_model.item(i)
            if item.isCheckable() and item.checkState() == QtCore.Qt.Checked:
                plot_data_dict[item.text().split('\n')[0].replace(' #', '')] = {}
        for i in range(self.file_name.__len__()):
            if self.file_name[i] in plot_data_dict.keys():
                freq_list, epsilon_raw_list = DielectricSimulator(self.model, self.file_path[i]).get_data()
                plot_data = plot_data_dict[self.file_name[i]]
                param_list = self.result_dict[self.file_name[i]]['popt']
                plot_data['freq'] = freq_list
                plot_data['epsilon_raw'] = epsilon_raw_list
                if self.model == 1:
                    plot_data['epsilon'] = func_Havriliak_Negami(freq_list, alpha=param_list[0], beta=param_list[1],
                                                                 tau=param_list[2], epsilon_inf=param_list[3],
                                                                 delta_epsilon=param_list[4])
                elif self.model == 2:
                    plot_data['epsilon'] = func_Cole_Cole(freq_list, alpha=param_list[0], tau=param_list[1],
                                                          epsilon_inf=param_list[2], delta_epsilon=param_list[3])
                elif self.model == 3:
                    plot_data['epsilon'] = func_Cole_Davidson(freq_list, beta=param_list[0], tau=param_list[1],
                                                              epsilon_inf=param_list[2], delta_epsilon=param_list[3])
                elif self.model == 4:
                    plot_data['epsilon'] = func_Debye(freq_list, tau=param_list[0], epsilon_inf=param_list[1],
                                                      delta_epsilon=param_list[2])
                else:
                    pass
        print('PLOT数据', plot_data_dict)
        # 判断是否有delta图
        delta_epsilon_exist = True if self.parent.CheckBox_reference.isChecked() else False
        delta_epsilon_ref_file_name = self.parent.ComboBox_reference.currentText() if delta_epsilon_exist else ''
        # 数据组装获取完成，开始绘图
        self.fig_epsilon.clear()
        ax_epsilon = self.fig_epsilon.add_subplot(111)
        ax_epsilon.cla()
        ax_epsilon.set_xscale('log')

        self.fig_delta_epsilon.clear()
        ax_delta_epsilon = self.fig_delta_epsilon.add_subplot(111)
        ax_delta_epsilon.cla()
        ax_delta_epsilon.set_xscale('log')

        # 循环取出数据绘图
        for key in plot_data_dict.keys():
            color_scatter = '#'
            color_plot = '#'
            for _ in range(6):
                random_num = random.randint(0, 7)
                color_scatter += ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F'][
                    random_num * 2]
                color_plot += ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F'][random_num]
            print(color_scatter, color_plot)
            ax_epsilon.scatter(plot_data_dict[key]['freq'], plot_data_dict[key]['epsilon_raw'], c=color_scatter,
                               label='#' + key + '_epsilon_raw')
            ax_epsilon.plot(plot_data_dict[key]['freq'], plot_data_dict[key]['epsilon'], c=color_plot,
                            label='#' + key + '_epsilon')
            # 判断是否有ref,踢出ref自身
            if delta_epsilon_exist:
                print('Test!',key,delta_epsilon_ref_file_name,key==delta_epsilon_ref_file_name,key is delta_epsilon_ref_file_name)
                if key != delta_epsilon_ref_file_name:
                    # 注意: 这里有一个非常关键的问题之前由于epsilon被剪切过，这里会大小不一，那么同样需要切除ref的对应部分，否则矩阵不能相减
                    # 注意：数据剪切部分一定在前端低频部分
                    plot_data_delta_epsilon_raw = np.multiply(
                        np.divide(
                            np.subtract(
                                plot_data_dict[key]['epsilon_raw'],
                                plot_data_dict[delta_epsilon_ref_file_name]['epsilon_raw'][-len(plot_data_dict[key]['epsilon_raw']):]
                            ),
                            plot_data_dict[delta_epsilon_ref_file_name]['epsilon_raw'][-len(plot_data_dict[key]['epsilon_raw']):]
                        ),
                        100)
                    plot_data_delta_epsilon = np.multiply(
                        np.divide(
                            np.subtract(
                                plot_data_dict[key]['epsilon'],
                                plot_data_dict[delta_epsilon_ref_file_name]['epsilon'][-len(plot_data_dict[key]['epsilon']):]
                            ),
                            plot_data_dict[delta_epsilon_ref_file_name]['epsilon'][-len(plot_data_dict[key]['epsilon_raw']):]
                        ),
                        100)
                    ax_delta_epsilon.scatter(plot_data_dict[key]['freq'],
                                             plot_data_delta_epsilon_raw,
                                             c=color_scatter,label='#' + key + '_delta_epsilon_raw')
                    ax_delta_epsilon.plot(plot_data_dict[key]['freq'], plot_data_delta_epsilon, c=color_plot,
                                    label='#' + key + '_delta_epsilon')
                    print('TEST!!!!',key)

        ax_epsilon.legend(loc='upper right')
        self.canvas_epsilon.draw()
        if delta_epsilon_exist:
            ax_delta_epsilon.legend(loc='upper right')
            self.canvas_delta_epsilon.draw()
        self.dialog_plot.open()

    def save_results_as(self):
        # TODO:仿照M进行改写
        file_path, file_type = QtWidgets.QFileDialog.getSaveFileName(filter="Text Files (*.txt);;CSV (*.csv)")
        print(file_path)
        if file_type == 'Text Files (*.txt)':
            with open(file_path, 'w') as file:
                file.write(
                    "Dielectric Relaxation Simulation Results by Simulator(v1.0alpha), developed by Cheng WANG\n")
                file.write("Model:\t" + {1: 'Havriliak_Negami Model', 2: 'Cole_Cole Model', 3: 'Cole_Davidson Model',
                                         4: 'Debye Model'}.get(self.model) + '\n')
                file.write("#\tAlpha\tBeta\tTau\tEpsilon_Inf\tDelta_Epsilon\n")
                for key in self.result_dict.keys():
                    popt = self.result_dict[key]['popt']
                    print(popt)
                    file.write(key + '\t')
                    if self.model == 1:
                        file.write(str(popt[0]) + '\t' + str(popt[1]) + '\t')
                    elif self.model == 2:
                        file.write(str(popt[0]) + '\t1.0\t')
                    elif self.model == 3:
                        file.write('1.0\t' + str(popt[0]) + '\t')
                    elif self.model == 4:
                        file.write('1.0\t1.0\t')
                    else:
                        pass
                    file.write(str(popt[-3]) + '\t' + str(popt[-2]) + '\t' + str(popt[-1]) + '\n')
        elif file_type == 'CSV (*.csv)':
            print(file_path, file_type)
        else:
            print("save_results_as,unknown file_type.")

    def on_PushButton_clearAll_clicked(self):
        self.list_view_results_model.clear()
        self.parent.ComboBox_reference.clear()
        self.parent.CheckBox_reference.setCheckState(QtCore.Qt.Unchecked)
        self.parent.ComboBox_reference.setEnabled(False)
        print('结果已被清空')

