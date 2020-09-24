import matplotlib
from PyQt5 import QtCore, QtGui, QtWidgets
from simulator_ui import Ui_MainWindow
import numpy as np
import os, sys
import matplotlib.pyplot as plt
import random

matplotlib.use("Qt5Agg")  # 声明使用QT5
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas


class ModalFaraday(object):

    # TODO:
    def __init__(self,parent):
        self.parent = parent
