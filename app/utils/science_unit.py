from aenum import Enum, unique, skip,EnumType
import numpy as np
from app.utils import sci_const
from app.utils.science_base import Unit



@unique
class ScienceUnit(Enum):
    @unique
    class Time(Enum):
        s = Unit(index=0, symbol="s", ratio=1., description="Second")

    @unique
    class Length(Enum):
        m = Unit(index=0, symbol="m", ratio=1., description="Meter")
        cm = Unit(index=1, symbol="cm", ratio=1./1.e2,description="Centimeter")
        mm = Unit(index=2, symbol="mm", ratio=1./1.e3, description="Millimeter")
        um = Unit(index=3, symbol="um", ratio=1./1.e6, description="Micrometer")
        nm = Unit(index=4, symbol="nm", ratio=1./1.e9, description="Nanometer")
        A = Unit(index=5, symbol="A", ratio=1./1.e10, description="Angstrom")

    @unique
    class Area(Enum):
        m2 = Unit(index=0, symbol="m^2", ratio=1., description="Square Meter")
        cm2 = Unit(index=1, symbol="cm^2", ratio=1./1.e4, description="Square Centimeter")
        mm2 = Unit(index=2, symbol="mm^2", ratio=1./1.e6, description="Square Millimeter")
        um2 = Unit(index=3, symbol="um^2", ratio=1./1.e12, description="Square Micrometer")
        nm2 = Unit(index=4, symbol="nm^2", ratio=1./1.e18, description="Square Nanometer")

    @unique
    class Volume(Enum):
        m3 = Unit(index=0, symbol="m^3", ratio=1., description="Cubic Meter")
        cm3 = Unit(index=1, symbol="cm^3", ratio=1./1.e6, description="Cubic Centimeter")
        mm3 = Unit(index=2, symbol="mm^3", ratio=1./1.e9, description="Cubic Millimeter")
        um3 = Unit(index=3, symbol="um^3", ratio=1./1.e18, description="Cubic Micrometer")
        nm3 = Unit(index=4, symbol="nm^3", ratio=1./1.e27, description="Cubic Nanometer")

    @unique
    class Mass(Enum):
        kg = Unit(index=0, symbol="kg", ratio=1., description="Kilogram")
        g = Unit(index=1, symbol="g", ratio=1./1.e3, description="Gram")
        mg = Unit(index=2, symbol="mg", ratio=1./1.e6, description="Milligram")
        ug = Unit(index=3, symbol="ug", ratio=1./1.e9, description="Microgram")

    @unique
    class Voltage(Enum):
        V = Unit(index=0, symbol="V", ratio=1., description="Volt")
        mV = Unit(index=1, symbol="mV", ratio=1./1.e3, description="Millivolt")

    @unique
    class Frequency(Enum):
        Hz = Unit(index=0, symbol="Hz", ratio=1., description="Hertz")
        kHz = Unit(index=1, symbol="kHz", ratio=1.e3, description="Kilohertz")
        MHz = Unit(index=2, symbol="MHz", ratio=1.e6, description="Megahertz")
        GHz = Unit(index=3, symbol="GHz", ratio=1.e9, description="Gigahertz")
        THz = Unit(index=4, symbol="THz", ratio=1.e12, description="Terahertz")

    @unique
    class Capacitance(Enum):
        F = Unit(index=0, symbol="F", ratio=1., description="Farad")
        mF = Unit(index=1, symbol="mF", ratio=1./1.e3, description="Millifarad")
        uF = Unit(index=2, symbol="uF", ratio=1./1.e6, description="Microfarad")
        nF = Unit(index=3, symbol="nF", ratio=1./1.e9, description="Nanofarad")
        pF = Unit(index=4, symbol="pF", ratio=1./1.e12, description="Picofarad")
        fF = Unit(index=5, symbol="fF", ratio=1./1.e15, description="Femtofarad")

    @unique
    class Magnetization(Enum):
        A_m_1 = Unit(index=0, symbol="A/m", ratio=1., description="Ampere per Meter")
        T = Unit(index=1, symbol="T", ratio=1.e7/(4.*np.pi), description="Tesla")
        G = Unit(index=2, symbol="G", ratio=1.e3/(4.*np.pi), description="Gauss")
        kG = Unit(index=3, symbol="kG", ratio=1.e6/(4.*np.pi), description="kiloGauss")
        Oe = Unit(index=4, symbol="Oe", ratio=1.e3/(4.*np.pi), description="Oersted")
        kOe = Unit(index=5, symbol="kOe", ratio=1.e6/(4.*np.pi), description="kiloOersted")
        emu_cm_3 = Unit(index=6, symbol="emu/cm^3", ratio=1.e3, description="emu per Cubic Centimeter")

    @unique
    class Permeability(Enum):
        H_m_1 = Unit(index=0, symbol="H/m", ratio=1., description="Henry per Meter")

    @unique
    class Wavenumber(Enum):
        m_1 = Unit(index=0, symbol="m-1", ratio=1., description="Wavenumber, micrometer-1")
        cm_1 = Unit(index=1, symbol="cm-1", ratio=1.e2, description="Wavenumber, centimeter-1")
        mm_1 = Unit(index=2, symbol="mm-1", ratio=1.e3, description="Wavenumber, centimeter-1")
        um_1 = Unit(index=3, symbol="um-1", ratio=1.e6, description="Wavenumber, micrometer-1")
        nm_1 = Unit(index=4, symbol="nm-1", ratio=1.e9, description="Wavenumber, nanometer-1")
        A_1 = Unit(index=5, symbol="A-1", ratio=1.e10, description="Wavenumber, Angstrom-1")
        eV = Unit(index=6, symbol="eV", ratio=sci_const.h*sci_const.c/sci_const.e, description="Wavenumber, Energy, electronVolt")

    @unique
    class Dimensionless(Enum):
        DN = Unit(index=0, symbol="1", ratio=1., description="Dimensionless Number, no unit")

    @unique
    class AtomicContent(Enum):
        at = Unit(index=0, symbol="at.%", ratio=1., description="Atomic Percent, at.%")

    @unique
    class Unknown(Enum):
        unkn = Unit(index=0, symbol="unknown", ratio=1., description="Unknown unit")

    @staticmethod
    def get_from_symbol(symbol: str)->Unit:
        for key,value in ScienceUnit.__dict__.items():
            # print(key,value,type(value).__name__)
            if type(value).__name__ == 'EnumType':
                for name,unit in value.__dict__['_member_map_'].items():
                    if unit.value.get_symbol() == symbol:
                        return unit.value
        return ScienceUnit.Unknown.unkn.value

    @staticmethod
    def get_from_description_with_symbol_bracket(description_with_symbol_bracket: str)->Unit:
        for key,value in ScienceUnit.__dict__.items():
            # print(key,value,type(value).__name__)
            if type(value).__name__ == 'EnumType':
                for name,unit in value.__dict__['_member_map_'].items():
                    if unit.value.get_description_with_symbol_bracket() == description_with_symbol_bracket:
                        return unit.value
        return ScienceUnit.Unknown.unkn

    @staticmethod
    def is_symbol_repeat():
        symbol_list = []
        for key,value in ScienceUnit.__dict__.items():
            print(key,value,type(value).__name__)
            if type(value).__name__ == 'EnumType':
                # print(value.__dict__['_member_map_'])
                for name,unit in value.__dict__['_member_map_'].items():
                    symbol_list.append(unit.value.get_symbol())
                    print(value)
        print("symbol_list:",symbol_list)
        return len(symbol_list) == len(set(symbol_list))

    @staticmethod
    def classify_unit(classified_unit:Unit)->str:
        classification_str = 'Unknown'
        for key,value in ScienceUnit.__dict__.items():
            if type(value).__name__ == 'EnumType':
                # print(key,value.__dict__['_member_map_'])
                for name,unit in value.__dict__['_member_map_'].items():
                    if classified_unit == unit.value:
                        classification_str = key
        return classification_str

    @staticmethod
    def get_unit_list_by_classification(classification):
        unit_list = []
        if type(classification) is str:
            # print("get_unit_list_by_classification FROM str")
            classification_name = classification
        elif type(classification) is EnumType:
            # print("get_unit_list_by_classification FROM EnumType")
            classification_name = classification.__name__
        else:
            print("WRONG INPUT PARAMETER")
            return []

        for key,value in ScienceUnit.__dict__.items():
            if type(value).__name__ == 'EnumType' and key == classification_name:
                for name,unit in value.__dict__['_member_map_'].items():
                    unit_list.append(unit.value)
        return unit_list

def science_unit_convert(from_list:list,from_unit:Unit,to_unit:Unit)->list:
    # FIXME:ScienceUnit 转为 Unit
    # print(from_list,from_unit,to_unit)
    # print(ScienceUnit.classify_unit(from_unit),ScienceUnit.classify_unit(to_unit),ScienceUnit.classify_unit(from_unit)==ScienceUnit.classify_unit(to_unit))
    for index in range(from_list.__len__()):
        if from_list[index] is None:
            from_list[index] = 0.
    if ScienceUnit.classify_unit(from_unit) == ScienceUnit.classify_unit(to_unit):
        return np.multiply(from_list,from_unit.get_ratio()/to_unit.get_ratio())
    elif from_unit == ScienceUnit.Permeability.H_m_1.value and to_unit == ScienceUnit.Dimensionless.DN.value:
        return np.multiply(from_list,1.e7/(4.*np.pi))
    elif from_unit == ScienceUnit.Dimensionless.DN.value and to_unit == ScienceUnit.Permeability.H_m_1.value:
        return np.multiply(from_list,4.*np.pi/1.e7)
    else:
        print('ERROR,No conversion between different types of units.')
        return np.nan

SI_UNIT_LIST = [ScienceUnit.Magnetization.A_m_1.value,ScienceUnit.Magnetization.T.value,ScienceUnit.Length.m.value,ScienceUnit.Length.mm.value,ScienceUnit.Length.um.value,ScienceUnit.Length.nm.value,ScienceUnit.Length.A.value]
CGS_UNIT_LIST = [ScienceUnit.Magnetization.G.value,ScienceUnit.Magnetization.kG.value,ScienceUnit.Magnetization.Oe.value,ScienceUnit.Magnetization.kOe.value,ScienceUnit.Magnetization.emu_cm_3,ScienceUnit.Length.cm.value]




if __name__ == '__main__':
    # print(ScienceUnit.is_symbol_repeat())
    # print(ScienceUnit.get_from_symbol('kOe').get_description())
    # a_unit = ScienceUnit.Length.m.value
    # print(ScienceUnit.classify_unit(ScienceUnit.Length.m.value))
    # print(ScienceUnit.classify_unit(ScienceUnit.Length.cm.value))
    print(science_unit_convert([1],ScienceUnit.Capacitance.F.value,ScienceUnit.Capacitance.fF.value))
    # print(ScienceUnit.get_from_symbol('1'))
    # print(ScienceUnit.get_unit_list_by_classification("Magnetization"))
    # print(ScienceUnit.get_unit_list_by_classification(ScienceUnit.Magnetization))
    # print(ScienceUnit.Magnetization.__name__)
