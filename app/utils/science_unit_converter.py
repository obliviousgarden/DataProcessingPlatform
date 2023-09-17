from app.utils.science_unit import Unit, ScienceUnit
import numpy as np
from app.utils import sci_const

class ScienceUnitConverter:
    def __init__(self):
        # from A/m to [A/m,T,G,kG,Oe,kOe,emu/cm^3]
        magnetic_convert_factor_list = [1., (4 * np.pi) / 1.e7, (4 * np.pi) / 1.e3, 4 * np.pi / 1.e6,
                                        (4 * np.pi) / 1.e3, 4 * np.pi / 1.e6, 1.e3]
        self.magnetic_convert_factor_matrix = []
        for factor in magnetic_convert_factor_list:
            self.magnetic_convert_factor_matrix.append(np.divide(magnetic_convert_factor_list, factor))
        # from um to [um,nm,A,cm]
        wavelength_convert_factor_list = [1., 1.e3, 1.e4, 1 / 1.e4]
        self.wavelength_convert_factor_matrix = []
        for factor in wavelength_convert_factor_list:
            self.wavelength_convert_factor_matrix.append(np.divide(wavelength_convert_factor_list, factor))
        # from um_1 to [um_1,nm_1,A_1,cm_1,eV]
        wavenumber_convert_factor_list = [1., 1 / 1.e3, 1 / 1.e4, 1.e4,
                                          sci_const.e / (sci_const.h * sci_const.c * 1.e6)]
        self.wavenumber_convert_factor_matrix = []
        for factor in wavenumber_convert_factor_list:
            self.wavenumber_convert_factor_matrix.append(np.divide(wavenumber_convert_factor_list, factor))

    @staticmethod
    def convert(from_unit_class, to_unit_class: None, from_unit: Unit, to_unit: Unit, value):
        instance = ScienceUnitConverter()
        if to_unit_class is None:
            to_unit_class = from_unit_class
        result = np.nan
        if from_unit_class == ScienceUnit.Magnetization and from_unit_class == to_unit_class:
            result = instance.__magnetization_unit_converter(from_unit=from_unit, to_unit=to_unit, value=value)
        elif from_unit_class == ScienceUnit.Length:
            if to_unit_class == ScienceUnit.Length:
                result = instance.__wavelength_unit_converter(from_unit=from_unit, to_unit=to_unit, value=value)
            elif to_unit_class == ScienceUnit.Wavenumber:
                result_0 = instance.__wavelength_unit_converter(from_unit=from_unit, to_unit=ScienceUnit.Length.um.value, value=value)
                result = instance.__wavenumber_unit_converter(from_unit=ScienceUnit.Wavenumber.um_1.value, to_unit=to_unit, value=np.reciprocal(result_0))
            else:
                print("WRONG. ScienceUnitConverter to_unit_type.")
        elif from_unit_class == ScienceUnit.Wavenumber:
            if to_unit_class == ScienceUnit.Wavenumber:
                result = instance.__wavenumber_unit_converter(from_unit=from_unit, to_unit=to_unit, value=value)
            elif to_unit_class == ScienceUnit.Length:
                result_0 = instance.__wavenumber_unit_converter(from_unit=from_unit, to_unit=ScienceUnit.Wavenumber.um_1.value, value=value)
                result = instance.__wavenumber_unit_converter(from_unit=ScienceUnit.Length.um.value, to_unit=to_unit, value=np.reciprocal(result_0))
            else:
                print("WRONG. ScienceUnitConverter to_unit_type.")
        else:
            print("ScienceUnitConverter: Unknown ScienceUnitConverterType")
        return result

    def __magnetization_unit_converter(self, from_unit: Unit, to_unit: Unit, value):
        result = value * self.magnetic_convert_factor_matrix[from_unit.index][to_unit.index]
        return result

    def __wavelength_unit_converter(self, from_unit: Unit, to_unit: Unit, value):
        result = value * self.wavelength_convert_factor_matrix[from_unit.index][to_unit.index]
        return result

    def __wavenumber_unit_converter(self, from_unit: Unit, to_unit: Unit, value):
        result = value * self.wavenumber_convert_factor_matrix[from_unit.index][to_unit.index]
        return result


if __name__ == '__main__':
    from_unit = ScienceUnit.Wavenumber.eV
    to_unit = ScienceUnit.Wavenumber.um_1
    print(ScienceUnitConverter.convert(from_unit_class= from_unit.__class__,
                                       to_unit_class= to_unit.__class__,
                                       from_unit=from_unit.value,
                                       to_unit=to_unit.value,
                                       value=1.))
