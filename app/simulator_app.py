from PyQt5 import QtCore, QtWidgets
import os, sys
from ui.simulator import Simulator,ParametersRangeSettingDialog,CharacterEncodingConversionDialog,MagneticDataConversionDialog


if __name__ == "__main__":
    # 必须添加应用程序，同时不要忘了sys.argv参数
    app = QtWidgets.QApplication(sys.argv)
    # 分别对窗体进行实例化
    mainWindow = QtWidgets.QMainWindow()
    parametersRangeSettingDialog = QtWidgets.QDialog()
    characterEncodingConversionDialog = QtWidgets.QDialog()
    magneticDataConversionDialog = QtWidgets.QDialog()
    # 包装
    simulator = Simulator()
    parametersRangeSettingDialogWindow = ParametersRangeSettingDialog(simulator=simulator)
    characterEncodingConversionWindow = CharacterEncodingConversionDialog(simulator=simulator)
    magneticDataConversionWindow = MagneticDataConversionDialog(simulator=simulator)
    # 分别初始化UI
    simulator.setupUi(mainWindow)
    parametersRangeSettingDialogWindow.setupUi(parametersRangeSettingDialog)
    characterEncodingConversionWindow.setupUi(characterEncodingConversionDialog)
    magneticDataConversionWindow.setupUi(magneticDataConversionDialog)
    # 连接窗体
    simulator.actionParameters_Range.triggered.connect(parametersRangeSettingDialog.show)
    simulator.actionUTF_8_Conversion.triggered.connect(characterEncodingConversionDialog.show)
    simulator.actionMagnetic_Data_Conversion_SI_CGS.triggered.connect(magneticDataConversionDialog.show)

    mainWindow.show()  # show（）显示主窗口

    # 软件正常退出
    sys.exit(app.exec_())