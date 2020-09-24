import matplotlib
from PyQt5 import QtCore, QtGui, QtWidgets
from simulator_ui import Ui_MainWindow
import numpy as np
import os, sys
import matplotlib.pyplot as plt
import random

matplotlib.use("Qt5Agg")  # 声明使用QT5
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas


class ModalMagnetization(object):

    # TODO:
    def __init__(self,parent):
        self.parent = parent
        # 模型 1:Jiles-Atherton,模型 2:Brillouin,模型 3:Langevin,模型 4:Takacs
        # 默认模型为1:Jiles-Atherton
        self.model = 1
        # 饱和磁化强度(A/m)
        self.sat_mz_min = 1.0
        self.sat_mz_max = 10000.0
        # 磁耦合系数(J/(Tm^3))
        self.a_min = 0.1
        self.a_max = 1.0
        # 不可逆损耗系数(??)
        self.k_min = 0.0
        self.k_max = 1.0
        # 磁畴耦合系数(??)
        self.alpha_min = 0.0
        self.alpha_max = 1.0
        # 可逆磁化系数(1)
        self.c_min = 0.0
        self.c_max = 1.0
        # 轨道量子数，是自然数(1)
        self.j_min = 1
        self.j_max = 10000
        # 轨道量子数的朗德因子(1)
        self.g_j_min = 1
        self.g_j_max = 3
        # 初始化参数
        self.sat_mz = self.sat_mz_max
        self.a = self.a_max
        self.k = self.k_max
        self.alpha = self.alpha_min
        self.c = self.c_min
        self.j = self.j_min
        self.g_j = self.g_j_min
        # 录入文件相关系数和自定义UI组件
        self.file_path = []
        self.file_name = []
        self.list_view_file_model = QtCore.QStringListModel()
        self.list_view_file_model.setStringList(self.file_name)
        # 结果的自定义UI组件
        self.list_view_results_model = QtGui.QStandardItemModel()
        # 结果字典，键是文件名
        self.result_dict = {}
        # plot相关的弹窗的UI及其初始化
        self.dialog_plot = QtWidgets.QDialog()



