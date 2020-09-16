import matplotlib
from PyQt5 import QtCore, QtGui, QtWidgets
from simulator_ui import Ui_MainWindow
import numpy as np
import os, sys

import matplotlib.pyplot as plt
import random
from modal_dielectric import ModalDielectric

matplotlib.use("Qt5Agg")  # 声明使用QT5
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from dielectricrelaxationsimulator import DielectricRelaxationSimulator, func_Havriliak_Negami, \
    func_Cole_Cole, func_Cole_Davidson, func_Debye


def on_Action_quit():
    sys.exit(0)
    pass


class Simulator(Ui_MainWindow):

    def __init__(self, parent=None):

        super(Simulator, self).__init__()
        # 注意：这里会把自己作为模块连接这个主类的引用传过去
        self.modal_dielectric = ModalDielectric(parent=self)

        # TODO:初始化磁化模块
        # TODO:初始化法拉第模块

    def setupUi(self, MainWindow):
        # 继承调用父类的
        Ui_MainWindow.setupUi(self, MainWindow)
        # 菜单功能
        self.actionQuit.triggered.connect(on_Action_quit)
        self.actionSaveResultsAs.triggered.connect(self.on_Action_save_results_as)
        self.actionParameters_Range.triggered.connect(self.on_Action_parameters_range)
        self.modal_dielectric.setupUi()

    def on_Action_save_results_as(self):
        self.modal_dielectric.on_Action_save_results_as()

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
