from aenum import Enum, unique, skip


class Unit:
    def __init__(self, index: int, symbol: str, description: str):
        self.index = index
        self.symbol = symbol
        self.description = description


@unique
class ScienceUnit(Enum):
    @unique
    class Magnetization(Enum):
        A_m_1 = Unit(index=0, symbol="A/m", description="Ampere per Meter")
        T = Unit(index=1, symbol="T", description="Tesla")
        G = Unit(index=2, symbol="G", description="Gauss")
        kG = Unit(index=3, symbol="kG", description="kiloGauss")
        Oe = Unit(index=4, symbol="Oe", description="Oersted")
        kOe = Unit(index=5, symbol="kOe", description="kiloOersted")
        emu_cm_3 = Unit(index=6, symbol="emu/cm^3", description="emu per Cubic Centimeter")

    @unique
    class Wavelength(Enum):
        um = Unit(index=0, symbol="um", description="Wavelength, micrometer")
        nm = Unit(index=1, symbol="nm", description="Wavelength, nanometer")
        A = Unit(index=2, symbol="A", description="Wavelength, Angstrom")
        cm = Unit(index=3, symbol="cm", description="Wavelength, centimeter")

    @unique
    class Wavenumber(Enum):
        um_1 = Unit(index=0, symbol="um-1", description="Wavenumber, micrometer-1")
        nm_1 = Unit(index=1, symbol="nm-1", description="Wavenumber, nanometer-1")
        A_1 = Unit(index=2, symbol="A-1", description="Wavenumber, Angstrom-1")
        cm_1 = Unit(index=3, symbol="cm-1", description="Wavenumber, centimeter-1")
        eV = Unit(index=4, symbol="eV", description="Wavenumber, Energy, electronVolt")


if __name__ == '__main__':
    print(ScienceUnit.Magnetic.IS.value.index)
    print(ScienceUnit.Magnetic.IS.value.symbol)
    print(ScienceUnit.Magnetic.IS.value.description)
