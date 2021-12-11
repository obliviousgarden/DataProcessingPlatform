import matplotlib
from PyQt5 import QtCore, QtGui, QtWidgets
import numpy as np
import os
import matplotlib.pyplot as plt
import random
import re
from app.utils.science_data import ScienceData,ScienceFileType,ScienceWriter
from app.utils.science_base import PhysicalQuantity
from app.utils.science_unit import ScienceUnit
from app.utils.science_plot import SciencePlot,SciencePlotData


from PyQt5.QtWidgets import QApplication

matplotlib.use("Qt5Agg")  # 声明使用QT5
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from dielectric_simulator import DielectricSimulator


class ModalDielectric(object):

    def __init__(self,parent):
        self.parent = parent
        # 模型 1:Havriliak–Negami,模型 2:Cole-Cole,模型 3:Cole–Davidson,模型 4:Debye
        # 默认模型为1:Havriliak–Negami
        self.model = 1
        self.device = None
        self.dependent = None

        self.alpha_min = 0.0
        self.alpha_max = 1.0
        self.alpha_fix = False
        self.beta_min = 0.0
        self.beta_max = 1.0
        self.beta_fix = False
        self.tau_min = 1e-9
        self.tau_max = 1e-3
        self.tau_fix = False
        self.epsilon_inf_min = 0.0
        self.epsilon_inf_max = 50.0
        self.epsilon_inf_fix = False
        self.delta_epsilon_min = 100.0
        self.delta_epsilon_max = 1000.0
        self.delta_epsilon_fix = False

        self.alpha = self.alpha_max
        self.beta = self.beta_max
        self.tau = self.tau_min
        self.epsilon_inf = self.epsilon_inf_min
        self.delta_epsilon = self.delta_epsilon_min

        self.fixed_param_dict = {}

        self.file_path = []
        self.file_name = []
        self.table_view_file_model = QtGui.QStandardItemModel()
        self.resetTableViewHeaderItems(tableviewmodel=self.table_view_file_model)
        self.list_view_results_model = QtGui.QStandardItemModel()
        # file_name为键
        self.result_dict = {} # 存放结果
        self.dependent_dict = {} # 存放dependent功能需要的数据
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

        self.frequency_unit_list = ScienceUnit.get_unit_list_by_classification(ScienceUnit.Frequency)
        self.capacitance_unit_list = ScienceUnit.get_unit_list_by_classification(ScienceUnit.Capacitance)

    def setupUi(self):
        pass
        # 初始化参数UI
        self.parent.Slider_alpha.setValue((self.alpha - self.alpha_min) / (self.alpha_max - self.alpha_min) * 100.0)
        self.parent.Slider_beta.setValue((self.beta - self.beta_min) / (self.beta_max - self.beta_min) * 100.0)
        self.parent.Slider_tau.setValue(
            (np.log10(self.tau) - np.log10(self.tau_min)) / (np.log10(self.tau_max / self.tau_min)) * 100.0)
        self.parent.Slider_epsiloninf.setValue(
            (self.epsilon_inf - self.epsilon_inf_min) / (self.epsilon_inf_max - self.epsilon_inf_min) * 100.0)
        self.parent.Slider_deltaepsilon.setValue(
            (self.delta_epsilon - self.delta_epsilon_min) / (self.delta_epsilon_max - self.delta_epsilon_min) * 100.0)

        self.parent.lcdNumber_alpha.display(str(self.alpha))
        self.parent.lcdNumber_beta.display(str(self.beta))
        self.parent.lcdNumber_tau.display('%.1e' % self.tau)
        self.parent.lcdNumber_epsiloninf.display(str(self.epsilon_inf))
        self.parent.lcdNumber_deltaepsilon.display(str(self.delta_epsilon))

        # 设定RadioButton,选择模型
        self.parent.RadioButton_hnmodel.clicked.connect(self.on_RadioButton_model_clicked)
        self.parent.RadioButton_ccmodel.clicked.connect(self.on_RadioButton_model_clicked)
        self.parent.RadioButton_cdmodel.clicked.connect(self.on_RadioButton_model_clicked)
        self.parent.RadioButton_dmodel.clicked.connect(self.on_RadioButton_model_clicked)
        # 设定RadioButton,选择设备
        self.parent.RadioButton_device_impedance.clicked.connect(self.on_RadioButton_device_clicked)
        self.parent.RadioButton_device_lcr.clicked.connect(self.on_RadioButton_device_clicked)
        self.parent.RadioButton_other.clicked.connect(self.on_RadioButton_device_clicked)
        self.parent.RadioButton_device_impedance.click()
        # 设定RadioButton,选择依赖性
        self.parent.RadioButton_dependent_H.clicked.connect(self.on_RadioButton_dependent_clicked)
        self.parent.RadioButton_dependent_Co.clicked.connect(self.on_RadioButton_dependent_clicked)
        self.parent.RadioButton_dependent_DCB.clicked.connect(self.on_RadioButton_dependent_clicked)


        # # 设定PushButton
        self.parent.PushButton_file.clicked.connect(self.on_PushButton_file_clicked)
        self.parent.PushButton_dir.clicked.connect(self.on_PushButton_dir_clicked)
        self.parent.PushButton_simulate.clicked.connect(self.on_PushButton_simulate_clicked)
        self.parent.PushButton_plot.clicked.connect(self.on_PushButton_plot_clicked)
        self.parent.PushButton_clearAll.clicked.connect(self.on_PushButton_clearAll_clicked)
        # 设定TableView内的Model，并禁止编辑
        self.parent.TableView_file.setModel(self.table_view_file_model)
        self.parent.TableView_file_M.setFont(QtGui.QFont("Times", 12, QtGui.QFont.Black))
        self.parent.TableView_file_M.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        # self.parent.TableView_file_M.doubleClicked.connect(self.on_TableView_file_doubleClicked)

        self.parent.ListView_results.setModel(self.list_view_results_model)
        self.parent.ListView_results.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)

        # 设定Slider
        self.parent.Slider_alpha.valueChanged.connect(self.on_Slider_alpha_valueChanged)
        self.parent.Slider_beta.valueChanged.connect(self.on_Slider_beta_valueChanged)
        self.parent.Slider_tau.valueChanged.connect(self.on_Slider_tau_valueChanged)
        self.parent.Slider_epsiloninf.valueChanged.connect(self.on_Slider_epsilon_inf_valueChanged)
        self.parent.Slider_deltaepsilon.valueChanged.connect(self.on_Slider_delta_epsilon_valueChanged)

        # # 初始化figure和cavas
        self.dialog_plot.setWindowTitle('Plot results - Dielectric Simulation')
        self.layout_plot_epsilon.addWidget(self.canvas_epsilon)
        self.tab_plot_epsilon.setLayout(self.layout_plot_epsilon)
        self.layout_plot_delta_epsilon.addWidget(self.canvas_delta_epsilon)
        self.tab_plot_delta_epsilon.setLayout(self.layout_plot_delta_epsilon)
        # Checkbox
        self.parent.GroupBox_dependent.clicked.connect(self.on_GroupBox_dependent_clicked)
        self.parent.CheckBox_All.clicked.connect(self.on_CheckBox_All_clicked)

        # 初始化没有数据所以直接禁用
        self.parent.CheckBox_All.setEnabled(False)
        self.parent.PushButton_plot.setEnabled(False)

        # 设定LineEdit，并且禁止数字外的输入
        self.parent.LineEdit_step1_f_row.textChanged.connect(self.on_LineEdit_step1_f_row_textChanged)
        self.parent.LineEdit_step1_f_col.textChanged.connect(self.on_LineEdit_step1_f_col_textChanged)
        self.parent.LineEdit_step1_Cp_row.textChanged.connect(self.on_LineEdit_step1_Cp_row_textChanged)
        self.parent.LineEdit_step1_Cp_col.textChanged.connect(self.on_LineEdit_step1_Cp_col_textChanged)

        # 为单位的ComboBox添加选项
        for frequency_unit in self.frequency_unit_list:
            self.parent.ComboBox_step1_f_unit.addItem(frequency_unit.get_description_with_symbol_bracket())
        for capacitance_unit in self.capacitance_unit_list:
            self.parent.ComboBox_step1_Cp_unit.addItem(capacitance_unit.get_description_with_symbol_bracket())

    def resetTableViewHeaderItems(self, tableviewmodel: QtGui.QStandardItemModel):
        table_header_item_0 = QtGui.QStandardItem('file_name')
        table_header_item_0.setFont(QtGui.QFont("Times", 12, QtGui.QFont.Black))
        table_header_item_1 = QtGui.QStandardItem('thickness(m)')
        table_header_item_1.setFont(QtGui.QFont("Times", 12, QtGui.QFont.Black))
        table_header_item_2 = QtGui.QStandardItem('area(m^2)')
        table_header_item_2.setFont(QtGui.QFont("Times", 12, QtGui.QFont.Black))
        table_header_item_3 = QtGui.QStandardItem('H(Oe)')
        table_header_item_3.setFont(QtGui.QFont("Times", 12, QtGui.QFont.Black))
        table_header_item_4 = QtGui.QStandardItem('x_Co(at.%)')
        table_header_item_4.setFont(QtGui.QFont("Times", 12, QtGui.QFont.Black))
        table_header_item_5 = QtGui.QStandardItem('DCB(V)')
        table_header_item_5.setFont(QtGui.QFont("Times", 12, QtGui.QFont.Black))
        table_header_item_6 = QtGui.QStandardItem('OSC(V)')
        table_header_item_6.setFont(QtGui.QFont("Times", 12, QtGui.QFont.Black))
        table_header_item_7 = QtGui.QStandardItem('C.C.(1)')
        table_header_item_7.setFont(QtGui.QFont("Times", 12, QtGui.QFont.Black))
        tableviewmodel.setHorizontalHeaderItem(0, table_header_item_0)
        tableviewmodel.setHorizontalHeaderItem(1, table_header_item_1)
        tableviewmodel.setHorizontalHeaderItem(2, table_header_item_2)
        tableviewmodel.setHorizontalHeaderItem(3, table_header_item_3)
        tableviewmodel.setHorizontalHeaderItem(4, table_header_item_4)
        tableviewmodel.setHorizontalHeaderItem(5, table_header_item_5)
        tableviewmodel.setHorizontalHeaderItem(6, table_header_item_6)
        tableviewmodel.setHorizontalHeaderItem(7, table_header_item_7)

    def on_LineEdit_step1_f_row_textChanged(self,line_input):
        self.on_LineEdit_textChanged_number(self.parent.LineEdit_step1_f_row, line_input)

    def on_LineEdit_step1_f_col_textChanged(self,line_input):
        self.on_LineEdit_textChanged_number(self.parent.LineEdit_step1_f_col, line_input)

    def on_LineEdit_step1_Cp_row_textChanged(self,line_input):
        self.on_LineEdit_textChanged_number(self.parent.LineEdit_step1_Cp_row, line_input)

    def on_LineEdit_step1_Cp_col_textChanged(self,line_input):
        self.on_LineEdit_textChanged_number(self.parent.LineEdit_step1_Cp_col, line_input)

    def on_LineEdit_textChanged_number(self, object_LineEdit_number: QtWidgets.QLineEdit, line_input):
        try:
            float(line_input)
        except ValueError:
            line_input = ''.join(filter(lambda x: x in '.0123456789', line_input))
        object_LineEdit_number.setText(line_input)

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
        self.epsilon_inf_min = epsilon_inf_min
        self.epsilon_inf_max = epsilon_inf_max
        self.epsilon_inf = self.epsilon_inf_min
        self.parent.Slider_epsiloninf.setValue(
            (self.epsilon_inf - self.epsilon_inf_min) / (self.epsilon_inf_max - self.epsilon_inf_min) * 100.0)
        self.parent.lcdNumber_epsiloninf.display(str(self.epsilon_inf))

    def update_delta_epsilon_range(self, delta_epsilon_min, delta_epsilon_max):
        print("更新delta_epsilon范围,结束")
        self.delta_epsilon_min = delta_epsilon_min
        self.delta_epsilon_max = delta_epsilon_max
        self.delta_epsilon = self.delta_epsilon_min
        self.parent.Slider_deltaepsilon.setValue(
            (self.delta_epsilon - self.delta_epsilon_min) / (self.delta_epsilon_max - self.delta_epsilon_min) * 100.0)
        self.parent.lcdNumber_deltaepsilon.display(str(self.delta_epsilon))

    def on_RadioButton_model_clicked(self):
        if self.parent.RadioButton_hnmodel.isChecked():
            self.model = 1
            self.parent.Slider_alpha.setEnabled(True)
            self.parent.Slider_beta.setEnabled(True)
            self.parent.CheckBox_alpha_fix.setEnabled(True)
            self.parent.CheckBox_beta_fix.setEnabled(True)
        elif self.parent.RadioButton_ccmodel.isChecked():
            self.model = 2
            self.parent.Slider_alpha.setEnabled(True)
            self.parent.Slider_beta.setEnabled(False)
            self.parent.Slider_beta.setValue(100)
            self.parent.CheckBox_alpha_fix.setEnabled(True)
            self.parent.CheckBox_beta_fix.setEnabled(False)
        elif self.parent.RadioButton_cdmodel.isChecked():
            self.model = 3
            self.parent.Slider_alpha.setEnabled(False)
            self.parent.Slider_beta.setEnabled(True)
            self.parent.Slider_alpha.setValue(100)
            self.parent.CheckBox_alpha_fix.setEnabled(False)
            self.parent.CheckBox_beta_fix.setEnabled(True)
        else:
            self.model = 4
            self.parent.Slider_alpha.setEnabled(False)
            self.parent.Slider_beta.setEnabled(False)
            self.parent.Slider_alpha.setValue(100)
            self.parent.Slider_beta.setValue(100)
            self.parent.CheckBox_alpha_fix.setEnabled(False)
            self.parent.CheckBox_beta_fix.setEnabled(False)

    def on_RadioButton_device_clicked(self):
        if self.parent.RadioButton_other.isChecked():
            # Device: Other
            self.parent.LineEdit_step1_f_row.setEnabled(True)
            self.parent.LineEdit_step1_f_col.setEnabled(True)
            self.parent.ComboBox_step1_f_unit.setEnabled(True)
            self.parent.LineEdit_step1_Cp_row.setEnabled(True)
            self.parent.LineEdit_step1_Cp_col.setEnabled(True)
            self.parent.ComboBox_step1_Cp_unit.setEnabled(True)
        else:
            self.parent.LineEdit_step1_f_row.setEnabled(False)
            self.parent.LineEdit_step1_f_col.setEnabled(False)
            self.parent.ComboBox_step1_f_unit.setEnabled(False)
            self.parent.LineEdit_step1_Cp_row.setEnabled(False)
            self.parent.LineEdit_step1_Cp_col.setEnabled(False)
            self.parent.ComboBox_step1_Cp_unit.setEnabled(False)
            if self.parent.RadioButton_device_impedance.isChecked():
                # Device: Impedance Analyzer (~120MHz)
                self.parent.LineEdit_step1_f_row.setText("20")
                self.parent.LineEdit_step1_f_col.setText("1")
                self.parent.ComboBox_step1_f_unit.setCurrentIndex(0)
                self.parent.LineEdit_step1_Cp_row.setText("20")
                self.parent.LineEdit_step1_Cp_col.setText("3")
                self.parent.ComboBox_step1_Cp_unit.setCurrentIndex(0)
            elif self.parent.RadioButton_device_lcr.isChecked():
                # Device: LCR Meter (~1MHz)
                self.parent.LineEdit_step1_f_row.setText("16")
                self.parent.LineEdit_step1_f_col.setText("1")
                self.parent.ComboBox_step1_f_unit.setCurrentIndex(0)
                self.parent.LineEdit_step1_Cp_row.setText("16")
                self.parent.LineEdit_step1_Cp_col.setText("2")
                self.parent.ComboBox_step1_Cp_unit.setCurrentIndex(0)
            else:
                print('ERROR. Unknown Device.')

    def on_RadioButton_dependent_clicked(self):
        if self.parent.RadioButton_dependent_H.isChecked():
            self.dependent = "H"
        elif self.parent.RadioButton_dependent_Co.isChecked():
            self.dependent = "Co"
        elif self.parent.RadioButton_dependent_DCB.isChecked():
            self.dependent = "DCB"
        else:
            print("Unknown Dependent.")
            self.dependent = None

    def on_PushButton_file_clicked(self):
        self.file_name = []
        self.file_path = []
        file_name, file_type = QtWidgets.QFileDialog.getOpenFileNames()
        print('file:', file_name)
        self.file_path = file_name
        self.table_view_file_model.clear()
        self.resetTableViewHeaderItems(self.table_view_file_model)
        for i in range(file_name.__len__()):
            self.file_name.append(file_name[i].split('/')[-1])
            file_name_item = QtGui.QStandardItem(self.file_name[i])
            file_name_item.setEditable(False)
            thickness_value = 1./1.e6 # 1000 nm
            area_value = 1./1.e6 # 1 mm^2
            with open(self.file_path[i], 'r') as file:
                lines = file.readlines()
                if self.parent.RadioButton_device_impedance.isChecked():
                    thickness_value = float(lines[4])/1e6  # um -> m
                    area_value = float(lines[5])/1e4  # cm^2 -> m^2
                elif self.parent.RadioButton_device_lcr.isChecked():
                    thickness_value = float(lines[4])  # m -> m
                    area_value = float(lines[5])  # m^2 -> m^2
                else:
                    # 未知设备的膜厚和电极面积的数值未知
                    pass
            thickness_item = QtGui.QStandardItem('%.5g' % thickness_value)
            thickness_item.setEditable(True)
            thickness_item.setSelectable(False)
            area_item = QtGui.QStandardItem('%.5g' % area_value)
            area_item.setEditable(True)
            area_item.setSelectable(False)
            if 'Oe' in self.file_name[i]:
                # 对文件名称的处理逻辑：匹配‘-’和‘Oe’之间的内容，用空格切断成字符串数组，去除空字符串的元素：如果最后1个元素包含数字，那么信息都在最后一个元素里处理，否则，最后一个纯字符是倍数，倒数第二个是纯数字的数
                res_list = re.findall(r'.*-(.*)Oe.*',self.file_name[i])[0].split(' ')
                while '' in res_list:
                    res_list.remove('')
                if re.match(r'\d',res_list[-1]):
                    if 'k' in res_list[-1]:
                        h_value = float(res_list[-1].replace('k',''))*1000
                    else:
                        h_value = float(res_list[-1].replace('k',''))
                elif res_list.__len__() > 1:
                    if res_list[-1] == 'k':
                        h_value = float(res_list[-2])*1000
                    else:
                        h_value = float(res_list[-2])
                else:
                    h_value = 0.
            else:
                h_value = 0.
            h_item = QtGui.QStandardItem('%.5g' % h_value)
            h_item.setEditable(True)
            h_item.setSelectable(False)
            if 'Co' in self.file_name[i]:
                res_list = re.findall(r'.*[- ]Co(.*).*',self.file_name[i])[0].split(' ')
                while '' in res_list:
                    res_list.remove('')
                print("result:{}".format(res_list))
                co_value = float(res_list[-1])
            else:
                co_value = 0.
            co_item = QtGui.QStandardItem('%.5g' % co_value)
            co_item.setEditable(True)
            co_item.setSelectable(False)
            if 'DCB' in self.file_name[i]:
                res_list = re.findall(r'.*[- ]DCB(.*).*',self.file_name[i])[0].split(' ')
                while '' in res_list:
                    res_list.remove('')
                print("result:{}".format(res_list))
                dcb_value = float(res_list[-1])
            else:
                dcb_value = 0.
            dcb_item = QtGui.QStandardItem('%.5g' % dcb_value)
            dcb_item.setEditable(True)
            dcb_item.setSelectable(False)
            if 'V' in self.file_name[i]:
                res_list = re.findall(r'.*-(.*)V.*',self.file_name[i])[0].split(' ')
                while '' in res_list:
                    res_list.remove('')
                if re.match(r'\d',res_list[-1]):
                    if 'k' in res_list[-1]:
                        ocs_value = float(res_list[-1].replace('k',''))*1000
                    else:
                        ocs_value = float(res_list[-1].replace('k',''))
                elif res_list.__len__() > 1:
                    if res_list[-1] == 'k':
                        ocs_value = float(res_list[-2])*1000
                    else:
                        ocs_value = float(res_list[-2])
                else:
                    ocs_value = 0.
            else:
                ocs_value = 0.
            ocs_item = QtGui.QStandardItem('%.5g' % ocs_value)
            ocs_item.setEditable(True)
            ocs_item.setSelectable(False)
            cc_item = QtGui.QStandardItem('1')
            cc_item.setEditable(True)
            cc_item.setSelectable(False)
            self.table_view_file_model.appendRow([file_name_item, thickness_item, area_item, h_item, co_item, dcb_item, ocs_item, cc_item])
        print(self.file_path)
        print(self.file_name)
        # 因为刚导入数据，结果解析相关部分全部封住
        self.list_view_results_model.clear()
        self.parent.CheckBox_All.setEnabled(False)
        self.parent.PushButton_plot.setEnabled(False)

    def on_PushButton_dir_clicked(self):
        self.file_name = []
        self.file_path = []
        dir_ = QtWidgets.QFileDialog.getExistingDirectory()
        print('dir:', dir_)
        if dir_ != '':
            self.table_view_file_model.clear()
            self.resetTableViewHeaderItems(self.table_view_file_model)
            self.file_name = os.listdir(dir_)
            for i in range(self.file_name.__len__()):
                self.file_path.append(dir_ + '/' + self.file_name[i])
                file_name_item = QtGui.QStandardItem(self.file_name[i])
                file_name_item.setEditable(False)
                thickness_value = 1./1.e6 # 1000 nm
                area_value = 1./1.e6 # 1 mm^2
                with open(self.file_path[i], 'r') as file:
                    lines = file.readlines()
                    if self.parent.RadioButton_device_impedance.isChecked():
                        thickness_value = float(lines[4])/1e6  # um -> m
                        area_value = float(lines[5])/1e4  # cm^2 -> m^2
                    elif self.parent.RadioButton_device_lcr.isChecked():
                        thickness_value = float(lines[4])  # m -> m
                        area_value = float(lines[5])  # m^2 -> m^2
                    else:
                        # 未知设备的膜厚和电极面积的数值未知
                        pass
                thickness_item = QtGui.QStandardItem('%.5g' % thickness_value)
                thickness_item.setEditable(True)
                thickness_item.setSelectable(False)
                area_item = QtGui.QStandardItem('%.5g' % area_value)
                area_item.setEditable(True)
                area_item.setSelectable(False)
                if 'Oe' in self.file_name[i]:
                    # 对文件名称的处理逻辑：匹配‘-’和‘Oe’之间的内容，用空格切断成字符串数组，去除空字符串的元素：如果最后1个元素包含数字，那么信息都在最后一个元素里处理，否则，最后一个纯字符是倍数，倒数第二个是纯数字的数
                    res_list = re.findall(r'.*-(.*)Oe.*',self.file_name[i])[0].split(' ')
                    while '' in res_list:
                        res_list.remove('')
                    if re.match(r'\d',res_list[-1]):
                        if 'k' in res_list[-1]:
                            h_value = float(res_list[-1].replace('k',''))*1000
                        else:
                            h_value = float(res_list[-1].replace('k',''))
                    elif res_list.__len__() > 1:
                        if res_list[-1] == 'k':
                            h_value = float(res_list[-2])*1000
                        else:
                            h_value = float(res_list[-2])
                    else:
                        h_value = 0.
                else:
                    h_value = 0.
                h_item = QtGui.QStandardItem('%.5g' % h_value)
                h_item.setEditable(True)
                h_item.setSelectable(False)
                if 'Co' in self.file_name[i]:
                    res_list = re.findall(r'.*[- ]Co(.*).*',self.file_name[i])[0].split(' ')
                    while '' in res_list:
                        res_list.remove('')
                    print("result:{}".format(res_list))
                    co_value = float(res_list[-1])
                else:
                    co_value = 0.
                co_item = QtGui.QStandardItem('%.5g' % co_value)
                co_item.setEditable(True)
                co_item.setSelectable(False)
                if 'DCB' in self.file_name[i]:
                    res_list = re.findall(r'.*[- ]DCB(.*).*',self.file_name[i])[0].split(' ')
                    while '' in res_list:
                        res_list.remove('')
                    print("result:{}".format(res_list))
                    dcb_value = float(res_list[-1])
                else:
                    dcb_value = 0.
                dcb_item = QtGui.QStandardItem('%.5g' % dcb_value)
                dcb_item.setEditable(True)
                dcb_item.setSelectable(False)
                if 'V' in self.file_name[i]:
                    res_list = re.findall(r'.*-(.*)V.*',self.file_name[i])[0].split(' ')
                    while '' in res_list:
                        res_list.remove('')
                    if re.match(r'\d',res_list[-1]):
                        if 'k' in res_list[-1]:
                            ocs_value = float(res_list[-1].replace('k',''))*1000
                        else:
                            ocs_value = float(res_list[-1].replace('k',''))
                    elif res_list.__len__() > 1:
                        if res_list[-1] == 'k':
                            ocs_value = float(res_list[-2])*1000
                        else:
                            ocs_value = float(res_list[-2])
                    else:
                        ocs_value = 0.
                else:
                    ocs_value = 0.
                ocs_item = QtGui.QStandardItem('%.5g' % ocs_value)
                ocs_item.setEditable(True)
                ocs_item.setSelectable(False)
                cc_item = QtGui.QStandardItem('1')
                cc_item.setEditable(True)
                cc_item.setSelectable(False)
                self.table_view_file_model.appendRow([file_name_item, thickness_item, area_item, h_item, co_item, dcb_item, ocs_item, cc_item])
        print(self.file_path)
        print(self.file_name)
        # 因为刚导入数据，结果解析相关部分全部封住
        self.list_view_results_model.clear()
        self.parent.CheckBox_All.setEnabled(False)
        self.parent.PushButton_plot.setEnabled(False)

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

    def on_Slider_epsilon_inf_valueChanged(self):
        self.epsilon_inf = self.epsilon_inf_min + (
                self.epsilon_inf_max - self.epsilon_inf_min) * self.parent.Slider_epsiloninf.value() / 100.0
        self.parent.lcdNumber_epsiloninf.display(str(self.epsilon_inf))

    def on_Slider_delta_epsilon_valueChanged(self):
        self.delta_epsilon = self.delta_epsilon_min + (
                self.delta_epsilon_max - self.delta_epsilon_min) * self.parent.Slider_deltaepsilon.value() / 100.0
        self.parent.lcdNumber_deltaepsilon.display(str(self.delta_epsilon))

    def on_PushButton_simulate_clicked(self):
        # 清空列表
        self.on_PushButton_clearAll_clicked()
        # 记得清空一下结果字典
        self.result_dict = {}
        # 全选取消
        self.parent.CheckBox_All.setCheckState(QtCore.Qt.Unchecked)
        # 这里需要根据model构造p0和bounds参数
        # 清空结果的listview
        self.list_view_results_model.clear()
        # 需要的额外参数：f以及Cp 的 第一个数据位置 和 单位，试料的膜厚和电极的面积
        first_pos_info_tuple = (
            (int(self.parent.LineEdit_step1_f_row.text()),
             int(self.parent.LineEdit_step1_f_col.text()),
             str(self.parent.ComboBox_step1_f_unit.currentText())),
            (int(self.parent.LineEdit_step1_Cp_row.text()),
             int(self.parent.LineEdit_step1_Cp_col.text()),
             str(self.parent.ComboBox_step1_Cp_unit.currentText()))
        )
        # 来自tableview的size info
        info_dict = {}
        for row in range(self.table_view_file_model.rowCount()):
            info_dict[self.table_view_file_model.item(row, 0).text()] = [
                float(self.table_view_file_model.item(row, 1).text()),
                float(self.table_view_file_model.item(row, 2).text()),
                float(self.table_view_file_model.item(row, 3).text()),
                float(self.table_view_file_model.item(row, 4).text()),
                float(self.table_view_file_model.item(row, 5).text()),
                float(self.table_view_file_model.item(row, 6).text()),
                float(self.table_view_file_model.item(row, 7).text()),
            ]
        # 这里需要根据model构造p0和bounds参数
        # 注意：由于有固定参数的功能，这里必须把被固定的参数剔除出p0和bounds，并加入到fixed_param_dict
        p0 = []
        bounds = ([], [])
        self.fixed_param_dict = {}
        if self.model == 1:
            if self.parent.CheckBox_alpha_fix.isChecked():
                self.fixed_param_dict['alpha'] = self.alpha
            else:
                p0.append(self.alpha)
                bounds[0].append(self.alpha_min)
                bounds[1].append(self.alpha_max)
            if self.parent.CheckBox_beta_fix.isChecked():
                self.fixed_param_dict['beta'] = self.beta
            else:
                p0.append(self.beta)
                bounds[0].append(self.beta_min)
                bounds[1].append(self.beta_max)
        elif self.model == 2:
            if self.parent.CheckBox_alpha_fix.isChecked():
                self.fixed_param_dict['alpha'] = self.alpha
            else:
                p0.append(self.alpha)
                bounds[0].append(self.alpha_min)
                bounds[1].append(self.alpha_max)
        elif self.model == 3:
            if self.parent.CheckBox_beta_fix.isChecked():
                self.fixed_param_dict['beta'] = self.beta
            else:
                p0.append(self.beta)
                bounds[0].append(self.beta_min)
                bounds[1].append(self.beta_max)
        else:
            pass

        if self.parent.CheckBox_tau_fix.isChecked():
            self.fixed_param_dict['tau'] = self.tau
        else:
            p0.append(self.tau)
            bounds[0].append(self.tau_min)
            bounds[1].append(self.tau_max)
        if self.parent.CheckBox_epsiloninf_fix.isChecked():
            self.fixed_param_dict['epsilon_inf'] = self.epsilon_inf
        else:
            p0.append(self.epsilon_inf)
            bounds[0].append(self.epsilon_inf_min)
            bounds[1].append(self.epsilon_inf_max)
        if self.parent.CheckBox_deltaepsilon_fix.isChecked():
            self.fixed_param_dict['delta_epsilon'] = self.delta_epsilon
        else:
            p0.append(self.delta_epsilon)
            bounds[0].append(self.delta_epsilon_min)
            bounds[1].append(self.delta_epsilon_max)
        print("first_pos_info_tuple={},info_dict={},fixed_param_dict={}".format(first_pos_info_tuple, info_dict, self.fixed_param_dict))
        print('初始值', p0)
        print('边界', bounds)

        for i in range(self.file_path.__len__()):
            print(i, self.file_name[i])
            my_simulator = DielectricSimulator(model=self.model,
                                               first_pos_info_tuple=first_pos_info_tuple,
                                               info_dict=info_dict,
                                               fixed_param_dict=self.fixed_param_dict,
                                               p0=p0, bounds=bounds)

            self.result_dict[self.file_name[i]] = my_simulator.simulate(self.file_name[i],self.file_path[i])
            param_dict = self.get_param_dict_from_popt(popt=self.result_dict[self.file_name[i]]['popt'],fixed_param_dict=self.fixed_param_dict)
            item = QtGui.QStandardItem(" {}, Model={},\nalpha={},beta={},tau={}\nepsilon_inf={}delta_epsilon={}"
                                       .format(self.file_name[i],
                                               str(self.model),
                                               param_dict.get('alpha'),
                                               param_dict.get('beta'),
                                               param_dict.get('tau'),
                                               param_dict.get('epsilon_inf'),
                                               param_dict.get('delta_epsilon')))
            item.setCheckable(True)
            self.list_view_results_model.appendRow(item)
        # 注意：由于去除了低频的噪声，导致矩阵的尺寸不同，这里需要再裁剪一次统一数据
        data_point_num = 0
        for file_name, result_content in self.result_dict.items():
            list_size = result_content['freq_raw'].get_data().__len__()
            if data_point_num == 0 or data_point_num > list_size:
                data_point_num = list_size
        for file_name, result_content in self.result_dict.items():
            result_content['freq_raw'] = PhysicalQuantity(result_content['freq_raw'].get_name(),result_content['freq_raw'].get_unit(),result_content['freq_raw'].get_data()[-data_point_num:])
            result_content['epsilon_raw'] = PhysicalQuantity(result_content['epsilon_raw'].get_name(),result_content['epsilon_raw'].get_unit(),result_content['epsilon_raw'].get_data()[-data_point_num:])
            result_content['epsilon_cal'] = PhysicalQuantity(result_content['epsilon_cal'].get_name(),result_content['epsilon_cal'].get_unit(),result_content['epsilon_cal'].get_data()[-data_point_num:])
        print('结果', self.result_dict)
        # 拟合完成，可以选择背底，以及绘图了
        self.parent.CheckBox_All.setEnabled(True)
        self.parent.PushButton_plot.setEnabled(True)

    def on_CheckBox_All_clicked(self):
        check_state = QtCore.Qt.Checked if self.parent.CheckBox_All.isChecked() else QtCore.Qt.Unchecked

        for i in range(self.list_view_results_model.rowCount()):
            self.list_view_results_model.item(i).setCheckState(check_state)

    def on_GroupBox_dependent_clicked(self):
        if self.parent.GroupBox_dependent.isChecked():
            self.parent.RadioButton_dependent_H.click()
        else:
            self.dependent = None

    def on_PushButton_plot_clicked(self):
        data_dict = self.data_process()
        science_plot_data = SciencePlotData()
        for row_index in range(self.list_view_results_model.rowCount()):
            item = self.list_view_results_model.item(row_index)
            if item.isCheckable() and item.checkState() == QtCore.Qt.Checked:
                print(item.text().split('\n')[0].split(',')[0])
                data_name = item.text().split('\n')[0].split(',')[0].replace(' ','')
                data_key = data_name + '-data'
                data_list = data_dict[data_key] # data_list = [H_raw,M_raw,H,M,......]
                science_plot_data.add_figure_info(figure_title=data_name,x_label='log(Frequency),f (Hz)',y_label='Relative permittivity,epsilon_r')
                science_plot_data.add_plot_data(figure_title=data_name,x_data=np.log10(data_list[0].get_data()),y_data=data_list[1].get_data(),y_legend='raw')
                science_plot_data.add_plot_data(figure_title=data_name,x_data=np.log10(data_list[0].get_data()),y_data=data_list[2].get_data(),y_legend='cal')
        SciencePlot.sci_plot(science_plot_data)
        if self.dependent is not None:
            science_plot_data_para = SciencePlotData()
            science_plot_absolute_data_data = SciencePlotData()
            science_plot_relative_data_data = SciencePlotData()
            para_list = data_dict['para-dependent-on-'+self.dependent]
            absolute_data_list = data_dict['absolute-data-dependent-on-'+self.dependent]
            relative_data_list = data_dict['relative-data-dependent-on-'+self.dependent]
            para_x_label = None
            para_x_data = None
            for para_content in para_list:
                if para_content.get_name() == self.dependent:
                    para_x_label = para_content.get_name_with_unit()
                    para_x_data = para_content.get_data()
            for para_content in para_list:
                if para_content.get_name() != self.dependent:
                    science_plot_data_para.add_figure_info(figure_title=para_content.get_name(),x_label=para_x_label,y_label=para_content.get_name_with_unit())
                    science_plot_data_para.add_plot_data(figure_title=para_content.get_name(),x_data=para_x_data,y_data=para_content.get_data(),y_legend=para_content.get_name())
            SciencePlot.sci_plot(science_plot_data_para)

            data_x_label = absolute_data_list[0].get_name()
            data_x_data = absolute_data_list[0].get_data()
            for index in range(int((absolute_data_list.__len__()-1)/2)):
                absolute_data_content_raw = absolute_data_list[1 + index*2]
                absolute_data_content_cal = absolute_data_list[2 + index*2]
                relative_data_content_raw = relative_data_list[1 + index*2]
                relative_data_content_cal = relative_data_list[2 + index*2]
                absolute_figure_title = 'absolute change' + absolute_data_content_raw.get_name()
                relative_figure_title = 'relative change' + relative_data_content_raw.get_name()
                science_plot_absolute_data_data.add_figure_info(figure_title=absolute_figure_title,x_label='LOG'+data_x_label,y_label=absolute_data_content_raw.get_name_with_unit())
                science_plot_absolute_data_data.add_plot_data(figure_title=absolute_figure_title,x_data=np.log10(data_x_data),y_data=absolute_data_content_raw.get_data(),y_legend=absolute_data_content_raw.get_name())
                science_plot_absolute_data_data.add_plot_data(figure_title=absolute_figure_title,x_data=np.log10(data_x_data),y_data=absolute_data_content_cal.get_data(),y_legend=absolute_data_content_cal.get_name())
                science_plot_relative_data_data.add_figure_info(figure_title=relative_figure_title,x_label='LOG'+data_x_label,y_label=relative_data_content_raw.get_name_with_unit())
                science_plot_relative_data_data.add_plot_data(figure_title=relative_figure_title,x_data=np.log10(data_x_data),y_data=relative_data_content_raw.get_data(),y_legend=relative_data_content_raw.get_name())
                science_plot_relative_data_data.add_plot_data(figure_title=relative_figure_title,x_data=np.log10(data_x_data),y_data=relative_data_content_cal.get_data(),y_legend=relative_data_content_cal.get_name())

            SciencePlot.sci_plot(science_plot_absolute_data_data)
            SciencePlot.sci_plot(science_plot_relative_data_data)


    def on_PushButton_clearAll_clicked(self):
        self.list_view_results_model.clear()
        print('结果已被清空')

    def save_results_as(self):
        filter_str = ScienceFileType.get_filter_str(ScienceFileType.XLSX, ScienceFileType.CSV, ScienceFileType.TXT, ScienceFileType.ALL)
        file_path, file_type = QtWidgets.QFileDialog.getSaveFileName(filter=filter_str)
        print("save_results_as:file_path:{},file_type:{}".format(file_path,file_type))
        if file_path != '':
            file_dic = file_path.rsplit('/',1)[0]
            file_name = file_path.rsplit('/',1)[1].split('.')[0]
            # 需要保存的文件有2个：1 数据表 2 特征参数表
            write_file_type = ScienceFileType.get_by_description(file_type)
            # 组装ScienceWriteData需要的数据
            # data_dict是一个2层字典，例：#S999_data-Magnetization(kG)-[1,2,3],#S999_para-Coercivity(Oe)-[10000]
            # 每一个样品的磁化的数据被分为两类：1. 用来画图的点坐标数据, 2. 由1的数据总结得到了磁性特征参数数据
            # 表1的表头包括:
            # 实验值： freq_raw 频率, epsilon_raw 相对介电率
            # 计算值： epsilon_cal 相对介电率
            # 表2的表头包括：
            # t 膜厚, A 电极面积, H 磁场大小, Co Co的含量,
            # DCB 偏置电压, OSC 交流电压, C.C. 补偿常数,
            # alpha 参数, beta 参数, tau 弛豫时间, epsilon_inf 高频介电率, delta_epsilon 介电率变化
            data_dict = self.data_process()
            write_data = ScienceData(file_dic=file_dic, file_name=file_name, file_type=write_file_type.value, data_dict=data_dict)
            ScienceWriter.write_file(write_data)

    def data_process(self):
        # result_dict转write_data,并且判断dependent,并进行计算和结果补充
        data_dict = {}
        print(self.result_dict)
        for sample_name, sample_content in self.result_dict.items():
            print(sample_name)
            name_para = sample_name+'-para'
            name_data = sample_name+'-data'
            # 表1的表头包括:
            # 实验值： freq_raw 频率, epsilon_raw 相对介电率
            # 计算值： epsilon_cal 相对介电率
            freq_raw = sample_content['freq_raw'].get_data() #Hz
            epsilon_raw = sample_content['epsilon_raw'].get_data() #1
            epsilon_cal = sample_content['epsilon_cal'].get_data() #1
            # 表2的表头包括：
            # t 膜厚, A 电极面积, H 磁场大小, Co Co的含量,
            # DCB 偏置电压, OSC 交流电压, C.C. 补偿常数,
            # alpha 参数, beta 参数, tau 弛豫时间, epsilon_inf 高频介电率, delta_epsilon 介电率变化
            t = sample_content['t'].get_data()[0] #m
            a = sample_content['A'].get_data()[0] #m^2
            h = sample_content['H'].get_data()[0] #Oe
            co = sample_content['Co'].get_data()[0] #at.%
            dcb = sample_content['DCB'].get_data()[0] #V
            osc = sample_content['OSC'].get_data()[0] #V
            cc = sample_content['C.C.'].get_data()[0] #1
            param_dict = self.get_param_dict_from_popt(popt=sample_content['popt'],fixed_param_dict=self.fixed_param_dict)
            alpha = param_dict.get('alpha')
            beta = param_dict.get('beta')
            tau = param_dict.get('tau')
            epsilon_inf = param_dict.get('epsilon_inf')
            delta_epsilon = param_dict.get('delta_epsilon')

            data_dict.update({
                name_para: [
                    PhysicalQuantity('t',ScienceUnit.Length.m.value,[t]),
                    PhysicalQuantity('A',ScienceUnit.Area.m2.value,[a]),
                    PhysicalQuantity('H',ScienceUnit.Magnetization.Oe.value,[h]),
                    PhysicalQuantity('Co',ScienceUnit.AtomicContent.at.value,[co]),
                    PhysicalQuantity('DCB',ScienceUnit.Voltage.V.value,[dcb]),
                    PhysicalQuantity('OSC',ScienceUnit.Voltage.V.value,[osc]),
                    PhysicalQuantity('C.C.',ScienceUnit.Dimensionless.DN.value,[cc]),
                    PhysicalQuantity('alpha',ScienceUnit.Dimensionless.DN.value,[alpha]),
                    PhysicalQuantity('beta',ScienceUnit.Dimensionless.DN.value,[beta]),
                    PhysicalQuantity('tau',ScienceUnit.Time.s.value,[tau]),
                    PhysicalQuantity('epsilon_inf',ScienceUnit.Dimensionless.DN.value,[epsilon_inf]),
                    PhysicalQuantity('delta_epsilon',ScienceUnit.Dimensionless.DN.value,[delta_epsilon])
                ],
                name_data: [
                    PhysicalQuantity('freq_raw',ScienceUnit.Frequency.Hz.value,freq_raw),
                    PhysicalQuantity('epsilon_raw',ScienceUnit.Dimensionless.DN.value,epsilon_raw),
                    PhysicalQuantity('epsilon_cal',ScienceUnit.Dimensionless.DN.value,epsilon_cal)
                ]
            })
        if self.dependent is not None:
            print('Dependent on {}'.format(self.dependent))
            name_dependent_para = 'para-dependent-on-' + self.dependent
            name_dependent_absolute_data = 'absolute-data-dependent-on-' + self.dependent
            name_dependent_relative_data = 'relative-data-dependent-on-' + self.dependent
            # 1 找到被参照的对象，对应参数的最小值，并通过此参数从小到大进行排序
            dependent_result_list = []
            for sample_name, sample_content in self.result_dict.items():
                content_dict = sample_content.copy()
                content_dict['sample_name'] = sample_name
                if self.dependent == 'H':
                    content_dict['dependent_physical_quantity'] = sample_content['H']
                    content_dict['dependent_value'] = sample_content['H'].get_data()[0]
                elif self.dependent == 'Co':
                    content_dict['dependent_physical_quantity'] = sample_content['Co']
                    content_dict['dependent_value'] = sample_content['Co'].get_data()[0]
                elif self.dependent == 'DCB':
                    content_dict['dependent_physical_quantity'] = sample_content.get('DCB')
                    content_dict['dependent_value'] = sample_content['DCB'].get_data()[0]
                else:
                    content_dict['dependent_physical_quantity'] = PhysicalQuantity('dependent on unknown',ScienceUnit.Unknown.unkn.value,[0.])
                    content_dict['dependent_value'] = 0.

                dependent_result_list.append(content_dict)
            dependent_result_list.sort(key=lambda dependent_result: dependent_result['dependent_value'])

            t_list = []
            a_list = []
            h_list = []
            co_list = []
            dcb_list = []
            osc_list = []
            cc_list = []
            alpha_list = []
            beta_list = []
            tau_list = []
            epsilon_inf_list = []
            delta_epsilon_list = []
            reference_result = dependent_result_list[0]
            dependent_absolute_change_list = []
            dependent_relative_change_list = []
            for index in range(dependent_result_list.__len__()):
                dependent_result = dependent_result_list[index]
                t_list.append(dependent_result['t'].get_data()[0])
                a_list.append(dependent_result['A'].get_data()[0])
                h_list.append(dependent_result['H'].get_data()[0])
                co_list.append(dependent_result['Co'].get_data()[0])
                dcb_list.append(dependent_result['DCB'].get_data()[0])
                osc_list.append(dependent_result['OSC'].get_data()[0])
                cc_list.append(dependent_result['C.C.'].get_data()[0])
                param_dict = self.get_param_dict_from_popt(popt=dependent_result['popt'],fixed_param_dict=self.fixed_param_dict)
                alpha_list.append(param_dict.get('alpha'))
                beta_list.append(param_dict.get('beta'))
                tau_list.append(param_dict.get('tau'))
                epsilon_inf_list.append(param_dict.get('epsilon_inf'))
                delta_epsilon_list.append(param_dict.get('delta_epsilon'))

                suffix_str = '_at_' + self.dependent + '='
                dependent_value_str = '%.5g' %  dependent_result['dependent_value']
                suffix_str = suffix_str + dependent_value_str + dependent_result['dependent_physical_quantity'].get_unit().get_symbol()
                if index == 0:
                    dependent_absolute_change_list.append(PhysicalQuantity('freq_raw'+suffix_str,
                                                                           ScienceUnit.Frequency.Hz.value,
                                                                           dependent_result['freq_raw'].get_data()))
                    dependent_absolute_change_list.append(PhysicalQuantity('ref_epsilon_raw'+suffix_str,
                                                                           ScienceUnit.Dimensionless.DN.value,
                                                                           dependent_result['epsilon_raw'].get_data()))
                    dependent_absolute_change_list.append(PhysicalQuantity('ref_epsilon_cal'+suffix_str,
                                                                           ScienceUnit.Dimensionless.DN.value,
                                                                           dependent_result['epsilon_cal'].get_data()))
                    dependent_relative_change_list.append(PhysicalQuantity('freq_raw'+suffix_str,
                                                                           ScienceUnit.Frequency.Hz.value,
                                                                           dependent_result['freq_raw'].get_data()))
                    dependent_relative_change_list.append(PhysicalQuantity('ref_epsilon_raw'+suffix_str,
                                                                           ScienceUnit.Dimensionless.DN.value,
                                                                           dependent_result['epsilon_raw'].get_data()))
                    dependent_relative_change_list.append(PhysicalQuantity('ref_epsilon_cal'+suffix_str,
                                                                           ScienceUnit.Dimensionless.DN.value,
                                                                           dependent_result['epsilon_cal'].get_data()))
                else:
                    dependent_absolute_change_list.append(PhysicalQuantity('delta_epsilon_raw'+suffix_str,
                                                                           ScienceUnit.Dimensionless.DN.value,
                                                                           np.subtract(dependent_result['epsilon_raw'].get_data(), reference_result['epsilon_raw'].get_data())))
                    dependent_absolute_change_list.append(PhysicalQuantity('delta_epsilon_cal'+suffix_str,
                                                                           ScienceUnit.Dimensionless.DN.value,
                                                                           np.subtract(dependent_result['epsilon_cal'].get_data(), reference_result['epsilon_cal'].get_data())))
                    dependent_relative_change_list.append(PhysicalQuantity('delta_epsilon_raw'+suffix_str,
                                                                           ScienceUnit.Dimensionless.percent.value,
                                                                           np.multiply(np.divide(np.subtract(dependent_result['epsilon_raw'].get_data(), reference_result['epsilon_raw'].get_data()), reference_result['epsilon_raw'].get_data()),100)))
                    dependent_relative_change_list.append(PhysicalQuantity('delta_epsilon_cal'+suffix_str,
                                                                           ScienceUnit.Dimensionless.percent.value,
                                                                           np.multiply(np.divide(np.subtract(dependent_result['epsilon_cal'].get_data(), reference_result['epsilon_cal'].get_data()), reference_result['epsilon_cal'].get_data()),100)))
            data_dict.update({
                name_dependent_para: [
                    PhysicalQuantity('t',ScienceUnit.Length.m.value,t_list),
                    PhysicalQuantity('A',ScienceUnit.Area.m2.value,a_list),
                    PhysicalQuantity('H',ScienceUnit.Magnetization.Oe.value,h_list),
                    PhysicalQuantity('Co',ScienceUnit.AtomicContent.at.value,co_list),
                    PhysicalQuantity('DCB',ScienceUnit.Voltage.V.value,dcb_list),
                    PhysicalQuantity('OSC',ScienceUnit.Voltage.V.value,osc_list),
                    PhysicalQuantity('C.C.',ScienceUnit.Dimensionless.DN.value,cc_list),
                    PhysicalQuantity('alpha',ScienceUnit.Dimensionless.DN.value,alpha_list),
                    PhysicalQuantity('beta',ScienceUnit.Dimensionless.DN.value,beta_list),
                    PhysicalQuantity('tau',ScienceUnit.Time.s.value,tau_list),
                    PhysicalQuantity('epsilon_inf',ScienceUnit.Dimensionless.DN.value,epsilon_inf_list),
                    PhysicalQuantity('delta_epsilon',ScienceUnit.Dimensionless.DN.value,delta_epsilon_list)
                ],
                name_dependent_absolute_data: dependent_absolute_change_list,
                name_dependent_relative_data: dependent_relative_change_list,
            })
        print(data_dict)
        return data_dict

    def get_param_dict_from_popt(self,popt:list,fixed_param_dict:dict):
        result = popt.copy()
        param_dict = {}
        if 'alpha' in fixed_param_dict.keys():
            param_dict['alpha'] = fixed_param_dict.get('alpha')
        elif self.model in [3,4]:
            param_dict['alpha'] = 1.
        else:
            param_dict['alpha'] = result.pop(0)
        if 'beta' in fixed_param_dict.keys():
            param_dict['beta'] = fixed_param_dict.get('beta')
        elif self.model in [2,4]:
            param_dict['beta'] = 1.
        else:
            param_dict['beta'] = result.pop(0)
        if 'tau' in fixed_param_dict.keys():
            param_dict['tau'] = fixed_param_dict.get('tau')
        else:
            param_dict['tau'] = result.pop(0)
        if 'epsilon_inf' in fixed_param_dict.keys():
            param_dict['epsilon_inf'] = fixed_param_dict.get('epsilon_inf')
        else:
            param_dict['epsilon_inf'] = result.pop(0)
        if 'delta_epsilon' in fixed_param_dict.keys():
            param_dict['delta_epsilon'] = fixed_param_dict.get('delta_epsilon')
        else:
            param_dict['delta_epsilon'] = result.pop(0)
        return param_dict


if __name__ == "__main__":
    test_str_list = [
        "#123-DCB1.11",
        "#123-4-DCB1.11",
        "#123-4 DCB1.11",
        "#123-4-DCB 1.11",
        "#123-4 DCB1.11",
    ]
    for test_str in test_str_list:
        # 对文件名称的处理逻辑：匹配‘-’和‘Oe’之间的内容，用空格切断成字符串数组，去除空字符串的元素：如果最后1个元素包含数字，那么信息都在最后一个元素里处理，否则，最后一个纯字符是倍数，倒数第二个是纯数字的数
        res_list = re.findall(r'.*[- ]DCB(.*).*',test_str)[0].split(' ')
        while '' in res_list:
            res_list.remove('')
        print("result:{}".format(res_list))
        # if re.match(r'\d',res_list[-1]):
        #     if 'k' in res_list[-1]:
        #         h_value = float(res_list[-1].replace('k',''))*1000
        #     else:
        #         h_value = float(res_list[-1].replace('k',''))
        # elif res_list.__len__() > 1:
        #     if res_list[-1] == 'k':
        #         h_value = float(res_list[-2])*1000
        #     else:
        #         h_value = float(res_list[-2])
        # else:
        #     h_value = 0
        #
        # print(h_value)

