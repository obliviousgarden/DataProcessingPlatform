import matplotlib
from PyQt5 import QtCore, QtGui, QtWidgets
from simulator_ui import Ui_MainWindow
import numpy as np
import os, sys
import matplotlib.pyplot as plt
import random
from app.utils import sci_const

matplotlib.use("Qt5Agg")  # 声明使用QT5
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from magnetization_simulator import MagnetizationSimulator

class ModalMagnetization(object):

    def __init__(self,parent):
        self.parent = parent
        # 模型 1:Jiles-Atherton,模型 2:Brillouin,模型 3:Langevin,模型 4:Takacs
        # 默认模型为1:Jiles-Atherton
        self.model = 1
        # 饱和磁化强度(A/m)-LOG 10 kOe~795775A/m
        self.ms_min = 10000.
        self.ms_max = 1000000.
        # 磁耦合系数(J/(Tm^3))
        self.a_min = 1e3
        self.a_max = 1e5
        # 不可逆损耗系数(??)
        self.k_min = 1e3
        self.k_max = 1e5
        # 磁畴耦合系数(??)
        self.alfa_min = 0.0
        self.alfa_max = 1.0
        # 可逆磁化系数(1)
        self.c_min = 0.0
        self.c_max = 1.0
        # 轨道量子数，是自然数(1)
        self.j_min = 1e3
        self.j_max = 1e5
        # 颗粒尺寸分布的标准差(1)
        self.sigma_min = 0.0
        self.sigma_max = 1.0
        # 轨道量子数的朗德因子(1)
        self.g_j = sci_const.Lande_g_Factor(3 / 2, 3, 9 / 2)  # Co2+ ion
        # 温度(K)
        self.temp = 300
        # 初始化参数
        self.ms = self.ms_max
        self.a = self.a_max
        self.k = self.k_max
        self.alfa = self.alfa_min
        self.c = self.c_min
        self.j = self.j_min
        self.sigma = self.sigma_min
        # 录入文件相关系数和自定义UI组件
        self.file_path = []
        self.file_name = []
        self.list_view_file_model = QtCore.QStringListModel()
        self.list_view_file_model.setStringList(self.file_name)
        # 结果的自定义UI组件
        self.list_view_results_model = QtGui.QStandardItemModel()
        # 结果字典，键是文件名
        self.result_dict = {}
        # plot相关的弹窗的UI及其初始化,用tabUI预留出可能出现的多张图片
        self.dialog_plot_M = QtWidgets.QDialog()
        self.tab_plot_M = QtWidgets.QTabWidget(self.dialog_plot_M)
        self.tab_plot_M.setGeometry(QtCore.QRect(10, 0, 1061, 678))
        self.tab_plot_M_magnetization = QtWidgets.QWidget()
        self.tab_plot_M.addTab(self.tab_plot_M_magnetization, "")
        _translate = QtCore.QCoreApplication.translate
        self.tab_plot_M.setTabText(self.tab_plot_M.indexOf(self.tab_plot_M_magnetization), _translate("MainWindow", "Magnetization Curves"))

        self.layout_plot_M_magnetization = QtWidgets.QVBoxLayout()
        self.fig_M_magnetization = plt.Figure()
        self.canvas_M_magnetization = FigureCanvas(self.fig_M_magnetization)

    def setupUi(self):
        # 初始化参数UI
        self.parent.Slider_ms.setValue(
            (np.log10(self.ms) - np.log10(self.ms_min)) / (np.log10(self.ms_max / self.ms_min)) * 100.0)
        self.parent.Slider_a.setValue(
            (np.log10(self.a) - np.log10(self.a_min)) / (np.log10(self.a_max / self.a_min)) * 100.0)
        self.parent.Slider_k.setValue(
            (np.log10(self.k) - np.log10(self.k_min)) / (np.log10(self.k_max / self.k_min)) * 100.0)
        self.parent.Slider_alfa.setValue((self.alfa - self.alfa_min) / (self.alfa_max - self.alfa_min) * 100.0)
        self.parent.Slider_c.setValue((self.c - self.c_min) / (self.c_max - self.c_min) * 100.0)
        self.parent.Slider_j.setValue(
            (np.log10(self.j) - np.log10(self.j_min)) / (np.log10(self.j_max / self.j_min)) * 100.0)
        self.parent.Slider_sigma.setValue((self.sigma - self.sigma_min) / (self.sigma_max - self.sigma_min) * 100.0)

        self.parent.lcdNumber_ms.display('%.1e' % self.ms)
        self.parent.lcdNumber_a.display('%.1e' % self.a)
        self.parent.lcdNumber_k.display('%.1e' % self.k)
        self.parent.lcdNumber_alfa.display(str(self.alfa))
        self.parent.lcdNumber_c.display(str(self.c))
        self.parent.lcdNumber_j.display('%.1e' % self.j)
        self.parent.lcdNumber_sigma.display(str(self.sigma))

        # 设定RadioButton,选择模型
        self.parent.RadioButton_jamodel.clicked.connect(self.on_RadioButton_clicked)
        self.parent.RadioButton_brmodel.clicked.connect(self.on_RadioButton_clicked)
        self.parent.RadioButton_lamodel.clicked.connect(self.on_RadioButton_clicked)
        self.parent.RadioButton_tamodel.clicked.connect(self.on_RadioButton_clicked)
        self.parent.RadioButton_jamodel.click()
        # 设定sigma的ComboBox,判定是否考虑尺寸分布
        self.parent.ComboBox_sigma.clicked.connect(self.on_ComboBox_sigma_clicked)
        # 设定PushButton
        self.parent.PushButton_step1_M_file.clicked.connect(self.on_PushButton_step1_M_file_clicked)
        self.parent.PushButton_step1_M_dir.clicked.connect(self.on_PushButton_step1_M_dir_clicked)
        self.parent.PushButton_simulate_M.clicked.connect(self.on_PushButton_simulate_M_clicked)
        self.parent.PushButton_plot_M.clicked.connect(self.on_PushButton_plot_M_clicked)
        self.parent.PushButton_clearall_M.clicked.connect(self.on_PushButton_clearall_M_clicked)
        # 设定ListView内的Model，并禁止编辑
        self.parent.ListView_results_M.setModel(self.list_view_results_model)
        self.parent.ListView_results_M.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        # 设定Slider
        self.parent.Slider_ms.valueChanged.connect(self.on_Slider_ms_valueChanged)
        self.parent.Slider_a.valueChanged.connect(self.on_Slider_a_valueChanged)
        self.parent.Slider_k.valueChanged.connect(self.on_Slider_k_valueChanged)
        self.parent.Slider_alfa.valueChanged.connect(self.on_Slider_alfa_valueChanged)
        self.parent.Slider_c.valueChanged.connect(self.on_Slider_c_valueChanged)
        self.parent.Slider_j.valueChanged.connect(self.on_Slider_j_valueChanged)
        self.parent.Slider_sigma.valueChanged.connect(self.on_Slider_sigma_valueChanged)
        # 初始化figure和cavas
        self.dialog_plot_M.setWindowTitle('Plot results - Magnetization Simulation')
        self.layout_plot_M_magnetization.addWidget(self.canvas_M_magnetization)
        self.tab_plot_M_magnetization.setLayout(self.layout_plot_M_magnetization)
        # Checkbox
        self.parent.CheckBox_bg_results_M.clicked.connect(self.on_CheckBox_bg_results_M_clicked)
        self.parent.CheckBox_All_results_M.clicked.connect(self.CheckBox_All_results_M_clicked)
        # 初始化没有数据所以直接禁用
        self.parent.CheckBox_All_results_M.setEnabled(False)
        self.parent.PushButton_plot_M.setEnabled(False)
        self.parent.CheckBox_bg_results_M.setEnabled(False)
        self.parent.ComboBox_bg_results_M.setEnabled(False)
        # 设定LineEdit，并且禁止数字外的输入
        self.parent.LineEdit_step1_M_H_row.textChanged.connect(self.on_LineEdit_step1_M_H_row_textChanged)
        self.parent.LineEdit_step1_M_H_col.textChanged.connect(self.on_LineEdit_step1_M_H_col_textChanged)
        self.parent.LineEdit_step1_M_M_row.textChanged.connect(self.on_LineEdit_step1_M_M_row_textChanged)
        self.parent.LineEdit_step1_M_M_col.textChanged.connect(self.on_LineEdit_step1_M_M_col_textChanged)
        self.parent.LineEdit_step1_M_width.textChanged.connect(self.on_LineEdit_step1_M_width_textChanged)
        self.parent.LineEdit_step1_M_length.textChanged.connect(self.on_LineEdit_step1_M_length_textChanged)
        self.parent.LineEdit_step1_M_thickness.textChanged.connect(self.on_LineEdit_step1_M_thickness_textChanged)
        # 为单位的ComboBox添加选项
        for magnetization_unit in sci_const.magnetization_unit_list:
            self.parent.ComboBoxFormLayout_step1_M_H_unit.addItem(magnetization_unit)
            self.parent.ComboBoxFormLayout_step1_M_M_unit.addItem(magnetization_unit)
        # 磁化这里会有这样的一个东西，选择emu的情况下 尺寸的相关数据是必填的
        self.parent.ComboBoxFormLayout_step1_M_M_unit.addItem("*Magn.Moment(emu)")
        self.parent.ComboBoxFormLayout_step1_M_H_unit.setCurrentIndex(3)
        self.parent.ComboBoxFormLayout_step1_M_M_unit.setCurrentIndex(sci_const.magnetization_unit_list.__len__())
        # 特殊单位emu的情况下，对尺寸相关组件的操作
        self.parent.ComboBoxFormLayout_step1_M_M_unit.currentIndexChanged.connect(self.on_ComboBoxFormLayout_step1_M_H_unit_textChanged)

    def on_ComboBoxFormLayout_step1_M_H_unit_textChanged(self, current_index):
        if current_index == sci_const.magnetization_unit_list.__len__():
            self.parent.LineEdit_step1_M_width.setEnabled(True)
            self.parent.LineEdit_step1_M_length.setEnabled(True)
            self.parent.LineEdit_step1_M_thickness.setEnabled(True)
        else:
            self.parent.LineEdit_step1_M_width.setEnabled(False)
            self.parent.LineEdit_step1_M_length.setEnabled(False)
            self.parent.LineEdit_step1_M_thickness.setEnabled(False)

    def on_LineEdit_step1_M_H_row_textChanged(self,line_input):
        self.on_LineEdit_textChanged_number(self.parent.LineEdit_step1_M_H_row,line_input)

    def on_LineEdit_step1_M_H_col_textChanged(self,line_input):
        self.on_LineEdit_textChanged_number(self.parent.LineEdit_step1_M_H_col,line_input)

    def on_LineEdit_step1_M_M_row_textChanged(self,line_input):
        self.on_LineEdit_textChanged_number(self.parent.LineEdit_step1_M_M_row,line_input)

    def on_LineEdit_step1_M_M_col_textChanged(self,line_input):
        self.on_LineEdit_textChanged_number(self.parent.LineEdit_step1_M_M_col,line_input)

    def on_LineEdit_step1_M_width_textChanged(self,line_input):
        self.on_LineEdit_textChanged_number(self.parent.LineEdit_step1_M_width,line_input)

    def on_LineEdit_step1_M_length_textChanged(self,line_input):
        self.on_LineEdit_textChanged_number(self.parent.LineEdit_step1_M_length,line_input)

    def on_LineEdit_step1_M_thickness_textChanged(self,line_input):
        self.on_LineEdit_textChanged_number(self.parent.LineEdit_step1_M_thickness,line_input)

    def on_LineEdit_textChanged_number(self,object_LineEdit_number:QtWidgets.QLineEdit, line_input):
        try:
            float(line_input)
        except ValueError:
            line_input = ''.join(filter(lambda x: x in '.0123456789', line_input))
        object_LineEdit_number.setText(line_input)

    def on_RadioButton_clicked(self):
        if self.parent.RadioButton_jamodel.isChecked():
            self.model = 1
            # 更新公式的图片资源
            self.parent.Label_fomula_M.setStyleSheet("image: url(:/Formula/source/formula_m_ja.png)")
            self.parent.Slider_a.setEnabled(True)
            self.parent.Slider_a.setValue(100)
            self.parent.Slider_k.setEnabled(True)
            self.parent.Slider_k.setValue(100)
            self.parent.Slider_alfa.setEnabled(True)
            self.parent.Slider_alfa.setValue(0)
            self.parent.Slider_c.setEnabled(True)
            self.parent.Slider_c.setValue(0)
            self.parent.Slider_j.setEnabled(False)
            self.parent.Slider_sigma.setEnabled(False)
            self.parent.ComboBox_sigma.setEnabled(False)
        elif self.parent.RadioButton_brmodel.isChecked():
            self.model = 2
            # 更新公式的图片资源
            self.parent.Label_fomula_M.setStyleSheet("image: url(:/Formula/source/formula_m_b.png)")
            self.parent.Slider_a.setEnabled(False)
            self.parent.Slider_k.setEnabled(False)
            self.parent.Slider_alfa.setEnabled(False)
            self.parent.Slider_c.setEnabled(False)
            self.parent.Slider_j.setEnabled(True)
            self.parent.Slider_j.setValue(100)
            self.parent.ComboBox_sigma.setEnabled(True)
            self.parent.ComboBox_sigma.setChecked(True)
            self.parent.Slider_sigma.setEnabled(True)
            self.parent.Slider_sigma.setValue(100)

        elif self.parent.RadioButton_lamodel.isChecked():
            self.model = 3
            # 更新公式的图片资源
            self.parent.Label_fomula_M.setStyleSheet("image: url(:/Formula/source/formula_m_l.png)")
            self.parent.Slider_a.setEnabled(False)
            self.parent.Slider_k.setEnabled(False)
            self.parent.Slider_alfa.setEnabled(False)
            self.parent.Slider_c.setEnabled(False)
            self.parent.Slider_j.setEnabled(True)
            self.parent.Slider_j.setValue(100)
            self.parent.ComboBox_sigma.setEnabled(True)
            self.parent.ComboBox_sigma.setChecked(True)
            self.parent.Slider_sigma.setEnabled(True)
            self.parent.Slider_sigma.setValue(100)
        else:
            self.model = 4
            # 更新公式的图片资源
            self.parent.Label_fomula_M.setStyleSheet("image: url(:/Formula/source/formula_m_t.png)")
            self.parent.Slider_a.setEnabled(False)
            self.parent.Slider_k.setEnabled(False)
            self.parent.Slider_alfa.setEnabled(False)
            self.parent.Slider_c.setEnabled(False)
            self.parent.Slider_j.setEnabled(True)
            self.parent.Slider_j.setValue(100)
            self.parent.ComboBox_sigma.setEnabled(True)
            self.parent.ComboBox_sigma.setChecked(True)
            self.parent.Slider_sigma.setEnabled(True)
            self.parent.Slider_sigma.setValue(100)

    def on_ComboBox_sigma_clicked(self):
        if self.parent.ComboBox_sigma.isChecked():
            self.parent.Slider_sigma.setEnabled(True)
        else:
            self.parent.Slider_sigma.setEnabled(False)

    def on_PushButton_step1_M_file_clicked(self):
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

    def on_PushButton_step1_M_dir_clicked(self):
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

    def on_Slider_ms_valueChanged(self):
        self.ms = np.power(10, np.log10(self.ms_min) + (
                np.log10(self.ms_max) - np.log10(self.ms_min)) * self.parent.Slider_ms.value() / 100.0)
        self.parent.lcdNumber_ms.display('%.1e' % self.ms)

    def on_Slider_a_valueChanged(self):
        self.a = np.power(10, np.log10(self.a_min) + (
                np.log10(self.a_max) - np.log10(self.a_min)) * self.parent.Slider_a.value() / 100.0)
        self.parent.lcdNumber_a.display('%.1e' % self.a)

    def on_Slider_k_valueChanged(self):
        self.k = np.power(10, np.log10(self.k_min) + (
                np.log10(self.k_max) - np.log10(self.k_min)) * self.parent.Slider_k.value() / 100.0)
        self.parent.lcdNumber_k.display('%.1e' % self.k)

    def on_Slider_alfa_valueChanged(self):
        self.alfa = self.alfa_min + (self.alfa_max - self.alfa_min) * self.parent.Slider_alfa.value() / 100.0
        self.parent.lcdNumber_alfa.display(str(self.alfa))

    def on_Slider_c_valueChanged(self):
        self.c = self.c_min + (self.c_max - self.c_min) * self.parent.Slider_c.value() / 100.0
        self.parent.lcdNumber_c.display(str(self.c))

    def on_Slider_j_valueChanged(self):
        self.j = np.power(10, np.log10(self.j_min) + (
                np.log10(self.j_max) - np.log10(self.j_min)) * self.parent.Slider_j.value() / 100.0)
        self.parent.lcdNumber_j.display('%.1e' % self.j)

    def on_Slider_sigma_valueChanged(self):
        self.sigma = self.sigma_min + (self.sigma_max - self.sigma_min) * self.parent.Slider_sigma.value() / 100.0
        self.parent.lcdNumber_sigma.display(str(self.sigma))


    def on_PushButton_simulate_M_clicked(self):
        # 清空列表
        self.on_PushButton_clearall_M_clicked()
        # 记得清空一下结果字典
        self.result_dict = {}
        # 全选取消
        self.parent.CheckBox_All_results_M.setCheckState(QtCore.Qt.Unchecked)
        # 这里需要根据model构造p0和bounds参数
        p0 = []
        bounds = ([], [])
        # ms
        p0.append(self.ms)
        bounds[0].append(self.ms_min)
        bounds[1].append(self.ms_max)
        if self.model == 1:
            # alpha, c, a, k,
            p0.extend([self.alfa,self.c,self.a,self.k])
            bounds[0].extend([self.alfa_min,self.c_min,self.a_min,self.k_min])
            bounds[1].extend([self.alfa_max,self.c_max,self.a_max,self.k_max])
        else:
            # j,sigma(OPTIONAL)
            p0.extend([self.j,self.sigma])
            bounds[0].extend([self.j_min,self.sigma_min])
            bounds[1].extend([self.j_max,self.sigma_max])
            if self.model == 4:
                print('Tacks Model is temporarily blocked.')
        print('初始值', p0)
        print('边界', bounds)
        # 清空结果的listview
        self.list_view_results_model.clear()
        for i in range(self.file_path.__len__()):
            print(i,self.file_name[i])
            # 需要的额外参数：M以及H 的 第一个数据位置 和 单位，试料的尺寸 长宽高
            first_pos_info_tuple = (
                (float(self.parent.LineEdit_step1_M_H_row.text()),
                 float(self.parent.LineEdit_step1_M_H_col.text()),
                 str(self.parent.ComboBoxFormLayout_step1_M_H_unit.currentText())),
                (float(self.parent.LineEdit_step1_M_M_row.text()),
                 float(self.parent.LineEdit_step1_M_M_col.text()),
                 str(self.parent.ComboBoxFormLayout_step1_M_M_unit.currentText()))
            )
            distribution_flag = self.parent.ComboBox_sigma.isChecked()
            print("g_j={0},temp={1},first_pos_info={2},distribution_flag={3}".format(self.g_j, self.temp, first_pos_info_tuple, distribution_flag))
            self.result_dict[self.file_name[i]] = MagnetizationSimulator(self.model, self.file_path[i],
                                                                         p0=p0,
                                                                         bounds=bounds,
                                                                         info_tuple=first_pos_info_tuple).simulate(g_j=self.g_j,
                                                                                                                   temp=self.temp,
                                                                                                                   distribution=distribution_flag)
            result = self.result_dict[self.file_name[i]]['popt']
            # TODO:

    def on_PushButton_plot_M_clicked(self):
        pass
    def on_PushButton_clearall_M_clicked(self):
        pass
    def on_CheckBox_bg_results_M_clicked(self):
        pass
    def CheckBox_All_results_M_clicked(self):
        pass
