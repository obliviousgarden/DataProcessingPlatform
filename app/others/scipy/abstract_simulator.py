from abc import abstractmethod, ABCMeta


class AbstractSimulator(metaclass=ABCMeta):
    def __init__(self, model, file_path, p0=None, bounds=()):
        self.model = model
        self.file_path = file_path
        # 一个初始尝试值的数组
        self.p0 = p0
        # 一个参数的边间turple，左边数组下界，右边数组上届
        self.bounds = bounds

    @abstractmethod
    def get_data(self):
        pass

    @abstractmethod
    def simulate(self):
        pass
