from PyQt5 import QtCore, QtGui, QtWidgets
from simulator_ui import Ui_MainWindow
import numpy as np
import os, sys


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
        # 设定ListView内的Model
        self.ListView_file.setModel(self.list_view_file_model)
        # 禁止编辑
        self.ListView_file.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        # 设定Slider
        self.Slider_alpha.valueChanged.connect(self.on_Slider_alpha_valueChanged)
        self.Slider_beta.valueChanged.connect(self.on_Slider_beta_valueChanged)
        self.Slider_tau.valueChanged.connect(self.on_Slider_tau_valueChanged)
        self.Slider_epsiloninf.valueChanged.connect(self.on_Slider_epsiloninf_valueChanged)
        self.Slider_deltaepsilon.valueChanged.connect(self.on_Slider_deltaepsilon_valueChanged)

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
        file_name, file_type = QtWidgets.QFileDialog.getOpenFileNames()
        self.file_path = file_name
        self.file_name = []
        for i in range(file_name.__len__()):
            self.file_name.append(file_name[i].split('/')[-1])
        self.list_view_file_model.setStringList(self.file_name)
        print(self.file_path)
        print(self.file_name)

    def on_PushButton_dir_clicked(self):
        dir = QtWidgets.QFileDialog.getExistingDirectory()
        self.file_name = os.listdir(dir)
        self.file_path = []
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
        print('SIMULATE!!!')
        print(self.file_path)
        print(self.alpha, self.beta, self.tau, self.epsiloninf, self.deltaepsilon)
        # TODO:传输文件数据调取模拟模块


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
