import matplotlib
import numpy as np
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtCore import QRegExp
from PyQt5.QtGui import QRegExpValidator, QIntValidator, QDoubleValidator
from scipy.stats import lognorm

from simulator_ui import Ui_MainWindow
from params_range_setting_ui import Ui_ParametersRangeSettingDialog
from character_encoding_conversion_ui import Ui_CharacterEncodingConversionDialog
from magnetic_data_conversion_ui import Ui_MagneticDataConversionDialog
from lognormal_distribution_calculator_ui import Ui_LognormalDistributionCalculatorDialog
from app.utils.science_unit import ScienceUnit,science_unit_convert
from app.utils.science_data import ScienceData,ScienceFileType,ScienceWriter,ScienceReader
from app.utils.science_base import PhysicalQuantity
import os, sys
from modal_dielectric import ModalDielectric
from modal_faraday import ModalFaraday
from modal_magnetization import ModalMagnetization

matplotlib.use("Qt5Agg")  # 声明使用QT5


def on_Action_quit():
    sys.exit(0)
    pass


class Simulator(Ui_MainWindow):

    def __init__(self, parent=None):
        super(Simulator, self).__init__()
        # 注意：这里会把自己作为模块连接这个主类的引用传过去
        self.modal_dielectric = ModalDielectric(parent=self)
        self.modal_magnetization = ModalMagnetization(parent=self)
        self.modal_faraday = ModalFaraday(parent=self)

    def get_modal_dielectric(self):
        return self.modal_dielectric

    def get_modal_magnetization(self):
        return self.modal_magnetization

    def setupUi(self, MainWindow):
        # 继承调用父类的
        Ui_MainWindow.setupUi(self, MainWindow)
        # 菜单功能
        self.actionQuit.triggered.connect(on_Action_quit)
        self.actionSaveResultsAs.triggered.connect(self.on_Action_save_results_as)
        self.modal_dielectric.setupUi()
        self.modal_magnetization.setupUi()

    def on_Action_save_results_as(self):
        # 判断当前Tab的索引，调用对应的函数
        print("on_Action_save_results_as,tab.currentIndex()={}".format(self.tabs.currentIndex()))
        {0: self.modal_dielectric,
         1: self.modal_magnetization}.get(self.tabs.currentIndex()).save_results_as()


class ParametersRangeSettingDialog(Ui_ParametersRangeSettingDialog):
    def __init__(self, parent=None, simulator=None):
        super(ParametersRangeSettingDialog, self).__init__()
        self.simulator = simulator
        self.tau_double_validator = QDoubleValidator()
        self.tau_double_validator.setRange(0.0, 1.0)
        self.tau_double_validator.setNotation(QDoubleValidator.ScientificNotation)
        self.tau_double_validator.setDecimals(2)

        self.epsiloninf_double_validator = QDoubleValidator()
        self.epsiloninf_double_validator.setRange(0.0, 1000.0)
        self.epsiloninf_double_validator.setNotation(QDoubleValidator.StandardNotation)
        self.epsiloninf_double_validator.setDecimals(2)

        self.deltaepsilon_double_validator = QDoubleValidator()
        self.deltaepsilon_double_validator.setRange(0.0, 10000.0)
        self.deltaepsilon_double_validator.setNotation(QDoubleValidator.StandardNotation)
        self.deltaepsilon_double_validator.setDecimals(2)

        self.warning_box = QMessageBox(QMessageBox.Warning, 'WARNING', 'Check your inputs!!!')

    def setupUi(self, ParametersRangeSettingDialog):
        Ui_ParametersRangeSettingDialog.setupUi(self, ParametersRangeSettingDialog)
        self.pushButton_d_apply.clicked.connect(self.on_pushButton_d_apply)
        self.lineEdit_tau_from.setValidator(self.tau_double_validator)
        self.lineEdit_tau_to.setValidator(self.tau_double_validator)
        self.lineEdit_epsilon_inf_from.setValidator(self.epsiloninf_double_validator)
        self.lineEdit_epsilon_inf_to.setValidator(self.epsiloninf_double_validator)
        self.lineEdit_delta_epsilon_from.setValidator(self.deltaepsilon_double_validator)
        self.lineEdit_delta_epsilon_to.setValidator(self.deltaepsilon_double_validator)

    def on_pushButton_d_apply(self):
        error_flag = False
        if self.lineEdit_tau_from.text() == "" and self.lineEdit_tau_to.text() == "":
            pass
        elif 0.0 < float(self.lineEdit_tau_from.text()) < 1.0 and \
                0.0 < float(self.lineEdit_tau_to.text()) <= 1.0 and \
                float(self.lineEdit_tau_from.text()) < float(self.lineEdit_tau_to.text()):
            print("更新Tau范围,开始")
            self.simulator.get_modal_dielectric().update_tau_range(float(self.lineEdit_tau_from.text()),
                                                                    float(self.lineEdit_tau_to.text()))
        else:
            error_flag = True

        if self.lineEdit_epsilon_inf_from.text() == "" and self.lineEdit_epsilon_inf_to.text() == "":
            pass
        elif float(self.lineEdit_epsilon_inf_from.text()) < float(self.lineEdit_epsilon_inf_to.text()):
            print("更新epsilon_inf范围,开始")
            self.simulator.get_modal_dielectric().update_epsilon_inf_range(float(self.lineEdit_epsilon_inf_from.text()),
                                                                            float(self.lineEdit_tau_to.text()))

        else:
            error_flag = True

        if self.lineEdit_delta_epsilon_from.text() == "" and self.lineEdit_delta_epsilon_to.text() == "":
            pass
        elif float(self.lineEdit_delta_epsilon_from.text()) < float(self.lineEdit_delta_epsilon_to.text()):
            print("更新delta_epsilon范围,开始")
            self.simulator.get_modal_dielectric().update_delta_epsilon_range(
                float(self.lineEdit_delta_epsilon_from.text()), float(self.lineEdit_delta_epsilon_to.text()))

        else:
            error_flag = True

        if error_flag:
            self.warning_box.show()


class CharacterEncodingConversionDialog(Ui_CharacterEncodingConversionDialog):
    def __init__(self, parent=None, simulator=None):
        super(CharacterEncodingConversionDialog, self).__init__()
        self.simulator = simulator

    def setupUi(self, CharacterEncodingConversionDialog):
        Ui_CharacterEncodingConversionDialog.setupUi(self, CharacterEncodingConversionDialog)


class MagneticDataConversionDialog(Ui_MagneticDataConversionDialog):
    def __init__(self, parent=None, simulator=None):
        super(MagneticDataConversionDialog, self).__init__()
        self.simulator = simulator
        self.loaded_data = []
        self.converted_data = []
        self.is_si_to_cgs = True

    def setupUi(self, MagneticDataConversionDialog):
        Ui_MagneticDataConversionDialog.setupUi(self, MagneticDataConversionDialog)

        self.pushButton_mdc_file.clicked.connect(self.on_pushButton_mdc_file_clicked)
        self.pushButton_mdc_dir.clicked.connect(self.on_pushButton_mdc_dir_clicked)
        self.pushButton_mdc_conversion.clicked.connect(self.on_pushButton_mdc_conversion_clicked)

        self.comboBox_mdc_length_SI.addItems(['m', 'mm', 'um'])
        self.comboBox_mdc_length_CGS.addItems(['cm'])
        self.comboBox_mdc_volume_SI.addItems(['m^3', 'mm^3', 'um^3'])
        self.comboBox_mdc_volume_CGS.addItems(['cm^3'])
        self.comboBox_mdc_H_SI.addItems(['A/m'])
        self.comboBox_mdc_H_CGS.addItems(['Oe', 'kOe'])
        self.comboBox_mdc_M_SI.addItems(['A/m'])
        self.comboBox_mdc_M_CGS.addItems(['G', 'kG', 'emu/cm^3'])
        self.comboBox_mdc_BJ_SI.addItems(['T'])
        self.comboBox_mdc_BJ_CGS.addItems(['G', 'kG'])
        self.comboBox_mdc_miu_SI.addItems(['H/m'])
        self.comboBox_mdc_miu_CGS.addItems(['1'])

        self.comboBox_mdc_length_SI.setEnabled(False)
        self.comboBox_mdc_volume_SI.setEnabled(False)
        self.comboBox_mdc_H_SI.setEnabled(False)
        self.comboBox_mdc_M_SI.setEnabled(False)
        self.comboBox_mdc_BJ_SI.setEnabled(False)
        self.comboBox_mdc_miu_SI.setEnabled(False)
        self.comboBox_mdc_length_CGS.setEnabled(False)
        self.comboBox_mdc_volume_CGS.setEnabled(False)
        self.comboBox_mdc_H_CGS.setEnabled(False)
        self.comboBox_mdc_M_CGS.setEnabled(False)
        self.comboBox_mdc_BJ_CGS.setEnabled(False)
        self.comboBox_mdc_miu_CGS.setEnabled(False)

        self.pushButton_mdc_conversion.setEnabled(False)

    def on_pushButton_mdc_file_clicked(self):
        file_path_list, file_type_list = QtWidgets.QFileDialog.getOpenFileNames()
        print('file_path_list:', file_path_list)
        file_full_name_list = []
        for file_path in file_path_list:
            file_full_name_list.append(file_path.rsplit('/',1)[1])
        if file_path_list.__len__() != 0:
            self.load_data(file_dic=file_path_list[0].rsplit('/',1)[0], file_full_name_list=file_full_name_list)
            self.pushButton_mdc_conversion.setEnabled(True)

    def on_pushButton_mdc_dir_clicked(self):
        dir_ = QtWidgets.QFileDialog.getExistingDirectory()
        if dir_ != '':
            self.load_data(file_dic=dir_,file_full_name_list=os.listdir(dir_))
        self.pushButton_mdc_conversion.setEnabled(True)

    def load_data(self,file_dic:str, file_full_name_list:list):
        # loaded_data = {"file_name_1":{"sample_name_1_para":[PhysicalQuantity],"sample_name_1_data":[PhysicalQuantity]}}
        # 1. 检测文件类型TXT CSV XLS XLSX
        # 2. 装载数据
        # 3. 装载数据过程中，获取第一的文件内的信息，判断是SI转CGS还是CGS转SI，格式化前端
        self.loaded_data = []
        if file_full_name_list[0].split('.').__len__() == 1:
            print("ERROR: No extension name.")
            return
        self.loaded_data, self.is_si_to_cgs = ScienceReader.read_file(file_dic, file_full_name_list)
        # 前端的更新
        if self.is_si_to_cgs:
            self.label_mdc_info.setText("SI===>CGS")
            self.comboBox_mdc_length_SI.setEnabled(False)
            self.comboBox_mdc_volume_SI.setEnabled(False)
            self.comboBox_mdc_H_SI.setEnabled(False)
            self.comboBox_mdc_M_SI.setEnabled(False)
            self.comboBox_mdc_BJ_SI.setEnabled(False)
            self.comboBox_mdc_miu_SI.setEnabled(False)
            self.comboBox_mdc_length_CGS.setEnabled(True)
            self.comboBox_mdc_volume_CGS.setEnabled(True)
            self.comboBox_mdc_H_CGS.setEnabled(True)
            self.comboBox_mdc_M_CGS.setEnabled(True)
            self.comboBox_mdc_BJ_CGS.setEnabled(True)
            self.comboBox_mdc_miu_CGS.setEnabled(True)
        else:
            self.label_mdc_info.setText("SI<===CGS")
            self.comboBox_mdc_length_SI.setEnabled(True)
            self.comboBox_mdc_volume_SI.setEnabled(True)
            self.comboBox_mdc_H_SI.setEnabled(True)
            self.comboBox_mdc_M_SI.setEnabled(True)
            self.comboBox_mdc_BJ_SI.setEnabled(True)
            self.comboBox_mdc_miu_SI.setEnabled(True)
            self.comboBox_mdc_length_CGS.setEnabled(False)
            self.comboBox_mdc_volume_CGS.setEnabled(False)
            self.comboBox_mdc_H_CGS.setEnabled(False)
            self.comboBox_mdc_M_CGS.setEnabled(False)
            self.comboBox_mdc_BJ_CGS.setEnabled(False)
            self.comboBox_mdc_miu_CGS.setEnabled(False)
        print('loaded_data,FINISHED.')

    def on_pushButton_mdc_conversion_clicked(self):
        # 获取to_unit_dict目标单位
        length_unit = ScienceUnit.get_from_symbol(self.comboBox_mdc_length_CGS.currentText()) if self.is_si_to_cgs else ScienceUnit.get_from_symbol(self.comboBox_mdc_length_SI.currentText())
        volume_unit = ScienceUnit.get_from_symbol(self.comboBox_mdc_volume_CGS.currentText()) if self.is_si_to_cgs else ScienceUnit.get_from_symbol(self.comboBox_mdc_volume_SI.currentText())
        h_unit = ScienceUnit.get_from_symbol(self.comboBox_mdc_H_CGS.currentText()) if self.is_si_to_cgs else ScienceUnit.get_from_symbol(self.comboBox_mdc_H_SI.currentText())
        m_unit = ScienceUnit.get_from_symbol(self.comboBox_mdc_M_CGS.currentText()) if self.is_si_to_cgs else ScienceUnit.get_from_symbol(self.comboBox_mdc_M_SI.currentText())
        bj_unit = ScienceUnit.get_from_symbol(self.comboBox_mdc_BJ_CGS.currentText()) if self.is_si_to_cgs else ScienceUnit.get_from_symbol(self.comboBox_mdc_BJ_SI.currentText())
        miu_unit = ScienceUnit.get_from_symbol(self.comboBox_mdc_miu_CGS.currentText()) if self.is_si_to_cgs else ScienceUnit.get_from_symbol(self.comboBox_mdc_miu_SI.currentText())

        to_unit_dict = {
            'l': length_unit,
            'w': length_unit,
            't': length_unit,
            'V': volume_unit,
            'm': ScienceUnit.Mass.g.value,# 这里暂时没有开放给前端
            'Ms_cal': m_unit,
            'J_cal': ScienceUnit.Dimensionless.DN.value,
            'Hm': h_unit,
            'Hcb': h_unit,
            'Hcj': h_unit,
            'Ms': m_unit,
            'Mm': m_unit,
            'Mr': m_unit,
            'Bs': bj_unit,
            'Bm': bj_unit,
            'Br': bj_unit,
            'Js': bj_unit,
            'Jm': bj_unit,
            'Jr': bj_unit,
            'Hk': h_unit,
            'Q': ScienceUnit.Dimensionless.DN.value,
            'Dm': length_unit,
            'sigma': ScienceUnit.Dimensionless.DN.value,
            'H_raw': h_unit,
            'M_raw': m_unit,
            'H': h_unit,
            'M': m_unit,
            'B': bj_unit,
            'J': bj_unit,
            'χ': ScienceUnit.Dimensionless.DN.value, # FIXME:这里可能会有问题
            'μ': miu_unit,
            'μr': ScienceUnit.Dimensionless.DN.value,
        }
        # 对loaded_data进行单位转换
        self.converted_data = []
        for science_data in self.loaded_data:
            file_dic = science_data.get_file_dic()
            file_name = science_data.get_file_name()
            file_type = science_data.get_file_type()
            data_dict = {}
            for sheet_name, physical_quantity_list in science_data.get_data_dict().items():
                converted_physical_quantity_list = []
                for physical_quantity in physical_quantity_list:
                    converted_physical_quantity_list.append(PhysicalQuantity(name=physical_quantity.get_name(),
                                                                             unit=to_unit_dict.get(physical_quantity.get_name()),
                                                                             data=science_unit_convert(physical_quantity.get_data(),
                                                                                                       physical_quantity.get_unit(),
                                                                                                       to_unit_dict.get(physical_quantity.get_name()))))
                data_dict.update({sheet_name:converted_physical_quantity_list})
            file_name = file_name + '-CGS' if self.is_si_to_cgs else file_name + '-IS'
            self.converted_data.append(ScienceData(file_dic=file_dic, file_name=file_name, file_type=file_type, data_dict=data_dict))
        print('on_pushButton_mdc_conversion_clicked: CONVERSION FINISHED.')
        # 开始写入
        for science_data in self.converted_data:
            ScienceWriter.write_file(science_data)
        print('on_pushButton_mdc_conversion_clicked: WRITE FINISHED.')


class LognormalDistributionCalculatorDialog(Ui_LognormalDistributionCalculatorDialog):
    def __init__(self, parent=None, simulator=None):
        super(LognormalDistributionCalculatorDialog, self).__init__()
        self.simulator = simulator
        self.miu = 5.
        self.x_from = 0.
        self.x_to = 10.
        self.sigma = 0.5
        self.calculate_object = 'None'
        self.unit = ScienceUnit.Dimensionless.DN.value
        self.points = 100
        self.x_list = [1.]
        self.f_list = [1.]
        self.x_validator = QDoubleValidator(0,1000,3)
        self.sigma_validator = QDoubleValidator(0,1,3)
        self.int_validator = QIntValidator(10,1000)

    def setupUi(self, LognormalDistributionCalculatorDialog):
        Ui_LognormalDistributionCalculatorDialog.setupUi(self, LognormalDistributionCalculatorDialog)
        self.pushButton_ldc_calculate.clicked.connect(self.on_pushButton_ldc_calculate_clicked)
        self.pushButton_ldc_save.clicked.connect(self.on_pushButton_ldc_save_clicked)

        self.lineEdit_ldc_miu.setValidator(self.x_validator)
        self.lineEdit_ldc_from.setValidator(self.x_validator)
        self.lineEdit_ldc_to.setValidator(self.x_validator)
        self.lineEdit_ldc_sigma.setValidator(self.sigma_validator)
        self.lineEdit_ldc_points.setValidator(self.int_validator)

        self.lineEdit_ldc_miu.setText(str(self.miu))
        self.lineEdit_ldc_from.setText(str(self.x_from))
        self.lineEdit_ldc_to.setText(str(self.x_to))
        self.lineEdit_ldc_sigma.setText(str(self.sigma))
        self.lineEdit_ldc_points.setText(str(self.points))

        self.radioButton_ldc_dm.clicked.connect(self.on_radioButton_ldc_dm_clicked)
        self.radioButton_ldc_dm.click()

    def on_radioButton_ldc_dm_clicked(self):
        if self.radioButton_ldc_dm.isChecked():
            self.comboBox_ldc_units.addItems(['m', 'mm', 'um','nm','A'])
            self.comboBox_ldc_units.setCurrentText('nm')
            self.calculate_object = 'Dm'
            self.unit = ScienceUnit.get_from_symbol('nm')
        else:
            print("UNKNOWN LognormalDistributionCalculatorDialog on_radioButton_ldc_dm_clicked")

    def on_pushButton_ldc_calculate_clicked(self):
        # TODO：sigma是对于随机变量Y的，需要转化成X的sigma再进行计算才行
        self.miu = float(self.lineEdit_ldc_miu.text())
        self.x_from = float(self.lineEdit_ldc_from.text())
        self.x_to = float(self.lineEdit_ldc_to.text())
        self.sigma = float(self.lineEdit_ldc_sigma.text())
        self.unit = ScienceUnit.get_from_symbol(str(self.comboBox_ldc_units.currentText()))
        self.points = int(self.lineEdit_ldc_points.text())
        self.x_list = np.linspace(self.x_from,self.x_to,self.points)
        self.f_list = lognorm.pdf(self.x_list,self.sigma,loc=0,scale=self.miu)
        print("miu={},x_from={},x_to={},sigma={},unit={},x_list={},f_list={}.".format(self.miu,self.x_from,self.x_to,self.sigma,self.unit.get_symbol(),self.x_list,self.f_list))

    def on_pushButton_ldc_save_clicked(self):
        filter_str = ScienceFileType.get_filter_str(ScienceFileType.XLSX, ScienceFileType.CSV, ScienceFileType.TXT)
        file_path, file_type = QtWidgets.QFileDialog.getSaveFileName(filter=filter_str)
        print("LognormalDistributionCalculatorDialog.save_results_as:file_path:{},file_type:{}".format(file_path,file_type))
        file_dic = file_path.rsplit('/',1)[0]
        file_name = file_path.rsplit('/',1)[1].split('.')[0]
        write_file_type = ScienceFileType.get_by_description(file_type)
        data_dict = {
            "LOG-NORMAL Distribution":[
                PhysicalQuantity(self.calculate_object,self.unit,self.x_list),
                PhysicalQuantity('f_x',ScienceUnit.Dimensionless.DN.value,self.f_list)
        ]}
        ScienceWriter.write_file(ScienceData(file_dic=file_dic,file_name=file_name,file_type=write_file_type.value,data_dict=data_dict))
        print("LognormalDistributionCalculatorDialog.save FINISHED")


if __name__ == "__main__":
    # 必须添加应用程序，同时不要忘了sys.argv参数
    app = QtWidgets.QApplication(sys.argv)
    # 分别对窗体进行实例化
    mainWindow = QtWidgets.QMainWindow()
    parametersRangeSettingDialog = QtWidgets.QDialog()
    characterEncodingConversionDialog = QtWidgets.QDialog()
    magneticDataConversionDialog = QtWidgets.QDialog()
    lognormalDistributionCalculatorDialog = QtWidgets.QDialog()
    # 包装
    simulator = Simulator()
    parametersRangeSettingDialogWindow = ParametersRangeSettingDialog(simulator=simulator)
    characterEncodingConversionWindow = CharacterEncodingConversionDialog(simulator=simulator)
    magneticDataConversionWindow = MagneticDataConversionDialog(simulator=simulator)
    lognormalDistributionCalculatorWindow = LognormalDistributionCalculatorDialog(simulator=simulator)
    # 分别初始化UI
    simulator.setupUi(mainWindow)
    parametersRangeSettingDialogWindow.setupUi(parametersRangeSettingDialog)
    characterEncodingConversionWindow.setupUi(characterEncodingConversionDialog)
    magneticDataConversionWindow.setupUi(magneticDataConversionDialog)
    lognormalDistributionCalculatorWindow.setupUi(lognormalDistributionCalculatorDialog)
    # 连接窗体
    simulator.actionParameters_Range.triggered.connect(parametersRangeSettingDialog.show)
    simulator.actionUTF_8_Conversion.triggered.connect(characterEncodingConversionDialog.show)
    simulator.actionMagnetic_Data_Conversion_SI_CGS.triggered.connect(magneticDataConversionDialog.show)
    simulator.actionLOG_NORMAL_Distribution_Calculator.triggered.connect(lognormalDistributionCalculatorDialog.show)

    mainWindow.show()  # show（）显示主窗口

    # 软件正常退出
    sys.exit(app.exec_())
