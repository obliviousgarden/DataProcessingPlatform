import numpy as np
from scipy.optimize import curve_fit
from app.utils import sci_const
# 折射率的模拟已经被解耦出去，这里只有等效介质的模拟不过其中调用了折射率模拟的模块

class FaradaySimulator:
    def __init__(self, refractive_index_model:int, effective_medium_model:int, file_path_m:str, file_path_d:str,
                 first_pos_info_tuple=(), wl_bound=(), p0=None, bounds=()):
        # 折射率的拟合模型
        self.refractive_index_model = refractive_index_model
        # 等效介质模型
        self.effective_medium_model = effective_medium_model
        # 金属材料的折射率n和消光系数k
        self.file_path_m = file_path_m
        # 介电材料的折射率n和消光系数k
        self.file_path_d = file_path_d
        # 接受并解析第1个数据点的位置信息及其单位信息first_pos_info_tuple
        ((self.wl_unit_m, self.start_row_m, self.wl_start_rol_m, self.n_start_rol_m, self.k_start_rol_m),
         (self.wl_unit_d, self.start_row_d, self.wl_start_rol_d, self.n_start_rol_d,
          self.k_start_rol_d)) = first_pos_info_tuple
        # 频率/波长范围
        (self.wl_unit_bound, self.wl_start, self.wl_end) = wl_bound
        # 一个初始尝试值的数组
        self.p0 = p0
        # 一个参数的边间turple，左边数组下界，右边数组上届
        self.bounds = bounds

    def open_file(self, path):
        try:
            with open(path, 'r', encoding="utf-8") as file:
                lines = file.readlines()
        except UnicodeDecodeError:
            with open(path, 'r', encoding="shift_jis") as file:
                lines = file.readlines()
        return lines

    def get_data(self, file_path:str):
        pass

    def simulate(self, file_path:str):
        # ROUTE A:
        # 1 实验复折射率M,D-->计算复折射率M,D
        # 2 计算复折射率M,D-->计算复介电常数M,D
        # 3 计算复介电常数M,D-->有效介质近似-->计算复介电常数EFF
        # 4 计算复介电常数EFF-->计算FARADAY
        # ROUTE B:
        # 1 实验复折射率M,D-->计算复折射率M,D
        # 2,3 计算复折射率M,D-->有效介质近似(需要自己设计)-->计算复折射率EFF
        # 4 计算复折射率EFF-->计算FARADAY
        print('模拟折射率色散方程')
        print('计算方程')
        print('开始模拟')
        wl_raw, theta_raw = self.get_data(file_path)
        print('获取数据完毕', wl_raw, theta_raw)



class RefractiveIndexModel:
    # 波长是um
    def __init__(self):
        print('RefractiveIndexModel.__init__')

    def model_cauthy(self, wl, a, b):
        n = np.add(a, np.divide(b, np.square(wl)))
        return n

    def model_sellmeier_3(self, wl, b1, b2, b3, c1, c2, c3):
        wl2 = np.square(wl)
        term_1 = np.divide(np.divide(
            np.multiply(b1, wl2),
            np.subtract(wl2, np.square(c1))))
        term_2 = np.divide(np.divide(
            np.multiply(b2, wl2),
            np.subtract(wl2, np.square(c2))))
        term_3 = np.divide(np.divide(
            np.multiply(b3, wl2),
            np.subtract(wl2, np.square(c3))))
        complex_n = np.sqrt(np.add(1, np.add(term_1, np.add(term_2, term_3))))
        n = np.real(complex_n)
        k = np.imag(complex_n)
        result = np.hstack([n, k])
        return result

    def model_sellmeier_2(self, wl, b1, b2, c1, c2):
        wl2 = np.square(wl)
        term_1 = np.divide(np.divide(
            np.multiply(b1, wl2),
            np.subtract(wl2, np.square(c1))))
        term_2 = np.divide(np.divide(
            np.multiply(b2, wl2),
            np.subtract(wl2, np.square(c2))))
        complex_n = np.sqrt(np.add(1, np.add(term_1, term_2)))
        n = np.real(complex_n)
        k = np.imag(complex_n)
        result = np.hstack([n, k])
        return result


class EffectiveMediumModel:
    def __init__(self):
        print('EffectiveMediumModel.__init__')

    def model_maxwell_garnett(self, epsilon_m, epsilon_d, c_m):
        term_1 = np.subtract(np.add(epsilon_m, np.multiply(2, epsilon_d)),
                             np.multiply(c_m, np.subtract(epsilon_m, epsilon_d)))
        term_2 = np.divide(np.subtract(epsilon_m, epsilon_d),
                           term_1)
        term_3 = np.multiply(3. * c_m, term_2)
        epsilon_eff = np.multiply(epsilon_d, np.add(1, term_3))
        return epsilon_eff

    def model_bruggeman(self, epsilon_m, epsilon_d, c_m):
        h_b = np.subtract(np.multiply(2 - 3 * c_m, epsilon_d),
                          np.multiply(1 - 3 * c_m, epsilon_m))
        term_1 = np.sqrt(np.add(np.square(h_b),
                                np.multiply(8., np.multiply(epsilon_m, epsilon_d))))
        epsilon_eff = np.divide(np.add(h_b,
                                       term_1),
                                4.)
        return epsilon_eff

    def model_belyaev(self, epsilon_m, epsilon_d, c_m, wl, a, miu_m):
        omega = np.divide(2 * np.pi, wl)
        k_m = np.multiply(np.sqrt(epsilon_m, miu_m),
                          np.divide(omega, sci_const.c))
        x = np.multiply(k_m, a)
        term_x = np.subtract(1, np.multiply(x, np.reciprocal(np.tan(x))))
        j_x = np.multiply(2,
                          np.divide(term_x,
                                    np.subtract(np.square(x), term_x)))
        epsilon_m_times_j = np.multiply(epsilon_m, j_x)
        h_b = np.subtract(np.multiply(2 - 3 * c_m, epsilon_d),
                          np.multiply(1 - 3 * c_m, epsilon_m_times_j))
        term_1 = np.sqrt(np.add(np.square(h_b),
                                np.multiply(8., np.multiply(epsilon_m_times_j, epsilon_d))))
        epsilon_eff = np.divide(np.add(h_b,
                                       term_1),
                                4.)
        return epsilon_eff

class FaradayModel:
    def __init__(self):
        print('FaradayModel.__init__')

    def model_on_refractive_index(self):
        pass

    def model_on_epsilon(self, wl, c_m):
        pass

if __name__ == '__main__':
    file_path_m = "D:\\PycharmProjects\\AIProcessingPlatform\\app\\ui\\Co.csv"
    file_path_d = "D:\\PycharmProjects\\AIProcessingPlatform\\app\\ui\\SrF2.csv"
    first_pos_info_tuple = (('um', 3, 1, 4, 5),
                            ('um', 3, 1, 4, 5))
    wl_bound = ('nm', 405., 1550.)

    # m = - 1. + 0.j
    # n = np.sqrt(m)
    print(sci_const.c)

    # Refractive Index Models:
    # 1. Cauthy: n = A + B/λ^2
    # 2. Sellmeier-3: n^2 = 1 + B_1*λ^2/(λ^2-C_1^2) + B_2*λ^2/(λ^2-C_2^2) + B_3*λ^2/(λ^2-C_3^2)
    # 3. Sellmeier-2: n^2 = 1 + B_1*λ^2/(λ^2-C_1^2) + B_2*λ^2/(λ^2-C_2^2)
    # Effective Medium Models:
    # 1. Maxwell Garnett Equation
    # 2. Bruggeman's Model
    # 3. Belyaev's Model

    file_path = "D:\\PycharmProjects\\AIProcessingPlatform\\app\\ui\\theta-wavelength.csv"
    d = 0.1  # 膜厚 um
    # B_1,B_2,B_3,C_1,C_2,C_3,c_m
    p0 = [0.67805894, 0.37140533, 3.3485284, 0.05628989, 0.10801027, 39.906666, 0.26]
    bounds = ([0.1, 0.1, 0.1, 0.01, 0.01, 0.01, 0.],
              [5, 5, 5, 50, 50, 50, 1.])
    faraday_simulator = FaradaySimulator(
        refractive_index_model=2,
        effective_medium_model=1,
        file_path_m=file_path_m,
        file_path_d=file_path_d,
        first_pos_info_tuple=first_pos_info_tuple,
        wl_bound=wl_bound,
        p0=p0,
        bounds=bounds
    )
    faraday_simulator.simulate()
