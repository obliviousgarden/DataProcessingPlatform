import matplotlib
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtGui import QDoubleValidator
from PyQt5.QtWidgets import QMessageBox

from simulator_ui import Ui_MainWindow
from params_range_setting_ui import Ui_ParametersRangeSettingDialog
from character_encoding_conversion_ui import Ui_CharacterEncodingConversionDialog
import numpy as np
import os, sys

import matplotlib.pyplot as plt
import random
from modal_dielectric import ModalDielectric
from modal_faraday import ModalFaraday
from modal_magnetization import ModalMagnetization

matplotlib.use("Qt5Agg")  # 声明使用QT5
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from dielectric_simulator import DielectricSimulator, func_Havriliak_Negami, \
    func_Cole_Cole, func_Cole_Davidson, func_Debye


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

    def setupUi(self, MainWindow):
        # 继承调用父类的
        Ui_MainWindow.setupUi(self, MainWindow)
        # 菜单功能
        self.actionQuit.triggered.connect(on_Action_quit)
        self.actionSaveResultsAs.triggered.connect(self.on_Action_save_results_as)
        self.modal_dielectric.setupUi()

    def on_Action_save_results_as(self):
        self.modal_dielectric.on_Action_save_results_as()


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


class CharacterEncodingConversionDialog(Ui_CharacterEncodingConversionDialog):
    def __init__(self, parent=None, simulator=None):
        # TODO:
        super(CharacterEncodingConversionDialog, self).__init__()
        self.simulator = simulator


def setupUi(self, ParametersRangeSettingDialog, CharacterEncodingConversionDialog):
    Ui_ParametersRangeSettingDialog.setupUi(self, ParametersRangeSettingDialog)
    Ui_CharacterEncodingConversionDialog.setupUi(self, CharacterEncodingConversionDialog)
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


if __name__ == "__main__":
    # 必须添加应用程序，同时不要忘了sys.argv参数
    app = QtWidgets.QApplication(sys.argv)
    # 分别对窗体进行实例化
    mainWindow = QtWidgets.QMainWindow()
    parametersRangeSettingDialog = QtWidgets.QDialog()
    characterEncodingConversionDialog = QtWidgets.QDialog()
    # 固定主窗口尺寸
    # mainWindow.setFixedSize(mainWindow.width(), mainWindow.height())
    mainWindow.setFixedSize(1080, 720)
    # 包装
    simulator = Simulator()
    parametersRangeSettingDialogWindow = ParametersRangeSettingDialog(simulator=simulator)
    characterEncodingConversionWindow = CharacterEncodingConversionDialog(simulator=simulator)
    # 分别初始化UI
    simulator.setupUi(mainWindow)
    parametersRangeSettingDialogWindow.setupUi(parametersRangeSettingDialog)
    characterEncodingConversionWindow.setupUi(characterEncodingConversionDialog)
    # 连接窗体
    simulator.actionParameters_Range.triggered.connect(parametersRangeSettingDialog.show)
    simulator.actionUTF_8_Conversion.triggered.connect(characterEncodingConversionDialog.show)

    mainWindow.show()  # show（）显示主窗口

    # 软件正常退出
    sys.exit(app.exec_())
