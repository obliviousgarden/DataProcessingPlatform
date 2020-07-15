import matplotlib
from PyQt5 import QtCore, QtGui, QtWidgets
from simulator_ui import Ui_MainWindow
import numpy as np
import os, sys
import matplotlib.pyplot as plt
import random

matplotlib.use("Qt5Agg")  # 声明使用QT5
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from dielectricrelaxationsimulator import DielectricRelaxationSimulator, func_Havriliak_Negami, \
    func_Cole_Cole, func_Cole_Davidson, func_Debye


class Simulator(Ui_MainWindow):
    def __init__(self, parent=None):
        super(Simulator, self).__init__()
        # 模型 1:Havriliak–Negami,模型 2:Cole-Cole,模型 3:Cole–Davidson,模型 4:Debye
        self.model = 1

        self.alpha_min = 0.0
        self.alpha_max = 1.0
        self.beta_min = 0.0
        self.beta_max = 1.0
        self.tau_min = 1e-10
        self.tau_max = 1e-5
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
        # plot相关
        self.dialog_plot = QtWidgets.QDialog()
        self.layout_plot = QtWidgets.QVBoxLayout()
        self.fig = plt.Figure()
        self.canvas = FigureCanvas(self.fig)

    def setupUi(self, MainWindow):
        # 继承调用父类的
        Ui_MainWindow.setupUi(self, MainWindow)
        # 初始化状态数据
        self.Slider_alpha.setValue((self.alpha - self.alpha_min) / (self.alpha_max - self.alpha_min) * 100.0)
        self.Slider_beta.setValue((self.beta - self.beta_min) / (self.beta_max - self.beta_min) * 100.0)
        self.Slider_tau.setValue(
            (np.log10(self.tau) - np.log10(self.tau_min)) / (np.log10(self.tau_max / self.tau_min)) * 100.0)
        self.Slider_epsiloninf.setValue(
            (self.epsiloninf - self.epsiloninf_min) / (self.epsiloninf_max - self.epsiloninf_min) * 100.0)
        self.Slider_deltaepsilon.setValue(
            (self.deltaepsilon - self.deltaepsilon_min) / (self.deltaepsilon_max - self.deltaepsilon_min) * 100.0)
        self.lcdNumber_alpha.display(str(self.alpha))
        self.lcdNumber_beta.display(str(self.beta))
        self.lcdNumber_tau.display('%.1e' % self.tau)
        self.lcdNumber_epsiloninf.display(str(self.epsiloninf))
        self.lcdNumber_deltaepsilon.display(str(self.deltaepsilon))
        # 设定RadioButton
        self.RadioButton_hnmodel.clicked.connect(self.on_RadioButton_clicked)
        self.RadioButton_ccmodel.clicked.connect(self.on_RadioButton_clicked)
        self.RadioButton_cdmodel.clicked.connect(self.on_RadioButton_clicked)
        self.RadioButton_dmodel.clicked.connect(self.on_RadioButton_clicked)
        # 设定PushButton
        self.PushButton_file.clicked.connect(self.on_PushButton_file_clicked)
        self.PushButton_dir.clicked.connect(self.on_PushButton_dir_clicked)
        self.PushButton_simulate.clicked.connect(self.on_PushButton_simulate_clicked)
        self.PushButton_plot.clicked.connect(self.on_PushButton_plot_clicked)
        # 设定ListView内的Model
        self.ListView_file.setModel(self.list_view_file_model)
        self.ListView_results.setModel(self.list_view_results_model)
        # 禁止编辑
        self.ListView_file.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.ListView_results.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        # 设定Slider
        self.Slider_alpha.valueChanged.connect(self.on_Slider_alpha_valueChanged)
        self.Slider_beta.valueChanged.connect(self.on_Slider_beta_valueChanged)
        self.Slider_tau.valueChanged.connect(self.on_Slider_tau_valueChanged)
        self.Slider_epsiloninf.valueChanged.connect(self.on_Slider_epsiloninf_valueChanged)
        self.Slider_deltaepsilon.valueChanged.connect(self.on_Slider_deltaepsilon_valueChanged)
        # 初始化figure和cavas
        self.dialog_plot.setWindowTitle('Plot results')
        self.layout_plot.addWidget(self.canvas)
        self.dialog_plot.setLayout(self.layout_plot)
        # 菜单功能
        self.actionQuit.triggered.connect(self.on_Action_quit)
        self.actionSaveResultsAs.triggered.connect(self.on_Action_save_results_as)
        self.actionParameters_Range.triggered.connect(self.on_Action_parameters_range)
        # 全选Checkbox
        self.CheckBox_All.clicked.connect(self.on_CheckBox_All_clicked)

    def on_RadioButton_clicked(self):
        if self.RadioButton_hnmodel.isChecked():
            self.model = 1
            self.Slider_alpha.setEnabled(True)
            self.Slider_beta.setEnabled(True)
        elif self.RadioButton_ccmodel.isChecked():
            self.model = 2
            self.Slider_alpha.setEnabled(True)
            self.Slider_beta.setEnabled(False)
            self.Slider_beta.setValue(100)
        elif self.RadioButton_cdmodel.isChecked():
            self.model = 3
            self.Slider_alpha.setEnabled(False)
            self.Slider_beta.setEnabled(True)
            self.Slider_alpha.setValue(100)
        else:
            self.model = 4
            self.Slider_alpha.setEnabled(False)
            self.Slider_beta.setEnabled(False)
            self.Slider_alpha.setValue(100)
            self.Slider_beta.setValue(100)

    def on_PushButton_file_clicked(self):
        self.file_name = []
        self.file_path = []
        file_name, file_type = QtWidgets.QFileDialog.getOpenFileNames()
        print('file:',file_name)
        self.file_path = file_name
        for i in range(file_name.__len__()):
            self.file_name.append(file_name[i].split('/')[-1])
        self.list_view_file_model.setStringList(self.file_name)
        print(self.file_path)
        print(self.file_name)

    def on_PushButton_dir_clicked(self):
        self.file_name = []
        self.file_path = []
        dir = QtWidgets.QFileDialog.getExistingDirectory()
        print('dir:',dir)
        if dir is not '':
            self.file_name = os.listdir(dir)
            for i in range(self.file_name.__len__()):
                self.file_path.append(dir + '/' + self.file_name[i])
            self.list_view_file_model.setStringList(self.file_name)
            print(self.file_path)
            print(self.file_name)

    def on_Slider_alpha_valueChanged(self):
        self.alpha = self.alpha_min + (self.alpha_max - self.alpha_min) * self.Slider_alpha.value() / 100.0
        self.lcdNumber_alpha.display(str(self.alpha))

    def on_Slider_beta_valueChanged(self):
        self.beta = self.beta_min + (self.beta_max - self.beta_min) * self.Slider_beta.value() / 100.0
        self.lcdNumber_beta.display(str(self.beta))

    def on_Slider_tau_valueChanged(self):
        self.tau = np.power(10, np.log10(self.tau_min) + (
                np.log10(self.tau_max) - np.log10(self.tau_min)) * self.Slider_tau.value() / 100.0)
        self.lcdNumber_tau.display('%.1e' % self.tau)

    def on_Slider_epsiloninf_valueChanged(self):
        self.epsiloninf = self.epsiloninf_min + (
                self.epsiloninf_max - self.epsiloninf_min) * self.Slider_epsiloninf.value() / 100.0
        self.lcdNumber_epsiloninf.display(str(self.epsiloninf))

    def on_Slider_deltaepsilon_valueChanged(self):
        self.deltaepsilon = self.deltaepsilon_min + (
                self.deltaepsilon_max - self.deltaepsilon_min) * self.Slider_deltaepsilon.value() / 100.0
        self.lcdNumber_deltaepsilon.display(str(self.deltaepsilon))

    def on_PushButton_simulate_clicked(self):
        # 记得清空一下结果字典
        self.result_dict = {}
        # 全选取消
        self.CheckBox_All.setCheckState(QtCore.Qt.Unchecked)
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
            self.result_dict[self.file_name[i]] = DielectricRelaxationSimulator(self.model, self.file_path[i],
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

    def on_CheckBox_All_clicked(self):
        check_state = QtCore.Qt.Checked if self.CheckBox_All.isChecked() else QtCore.Qt.Unchecked
        for i in range(self.list_view_results_model.rowCount()):
            self.list_view_results_model.item(i).setCheckState(check_state)

    def on_PushButton_plot_clicked(self):
        plot_data_dict = {}
        for i in range(self.list_view_results_model.rowCount()):
            item = self.list_view_results_model.item(i)
            if item.isCheckable() and item.checkState() == QtCore.Qt.Checked:
                plot_data_dict[item.text().split('\n')[0].replace(' #', '')] = {}
        for i in range(self.file_name.__len__()):
            if self.file_name[i] in plot_data_dict.keys():
                freq_list, epsilon_raw_list = DielectricRelaxationSimulator(self.model, self.file_path[i]).get_data()
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
        # 数据组装获取完成，开始绘图
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.cla()
        ax.set_xscale('log')
        for key in plot_data_dict.keys():
            color_scatter = '#'
            color_plot = '#'
            for _ in range(6):
                random_num = random.randint(0, 7)
                color_scatter += ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F'][
                    random_num * 2]
                color_plot += ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F'][random_num]
            print(color_scatter, color_plot)
            ax.scatter(plot_data_dict[key]['freq'], plot_data_dict[key]['epsilon_raw'], c=color_scatter,
                       label='#' + key + '_epsilon_raw')
            ax.plot(plot_data_dict[key]['freq'], plot_data_dict[key]['epsilon'], c=color_plot,
                    label='#' + key + '_epsilon')
        ax.legend(loc='upper right')
        self.canvas.draw()
        self.dialog_plot.open()

    def on_Action_quit(self):
        sys.exit(0)
        pass

    def on_Action_save_results_as(self):
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
            pass

    def on_Action_parameters_range(self):
        # TODO:设置参数的上下限
        pass


if __name__ == "__main__":
    # 必须添加应用程序，同时不要忘了sys.argv参数
    app = QtWidgets.QApplication(sys.argv)
    # 主窗口
    mainWindow = QtWidgets.QMainWindow()
    # 固定主窗口尺寸
    mainWindow.setFixedSize(mainWindow.width(), mainWindow.height())
    ui = Simulator()
    ui.setupUi(mainWindow)
    mainWindow.show()  # show（）显示主窗口

    # 软件正常退出
    sys.exit(app.exec_())
