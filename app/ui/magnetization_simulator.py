import numpy as np
from scipy.optimize import curve_fit


class MagnetizationSimulator:
    def __init__(self, model, file_path, p0=None, bounds=()):
        self.model = model
        self.file_path = file_path
        # 一个初始尝试值的数组
        self.p0 = p0
        # 一个参数的边间turple，左边数组下界，右边数组上届
        self.bounds = bounds

    def get_data(self):

        # TODO:

        pass
