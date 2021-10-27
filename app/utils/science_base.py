

class Unit:
    def __init__(self, index: int, symbol: str, ratio:float, description: str):
        self.index = index
        self.symbol = symbol
        self.ratio = ratio
        self.description = description

    def get_index(self):
        return self.index

    def get_symbol(self):
        return self.symbol

    def get_ratio(self):
        return self.ratio

    def get_description_with_symbol_bracket(self):
        return self.get_description()+self.get_symbol_bracket()

    def get_symbol_bracket(self):
        return '(' + str(self.symbol) + ')'

    def get_description(self):
        return self.description


class PhysicalQuantity:
    def __init__(self, name: str, unit: Unit, data:list):
        self.name = name
        self.unit = unit
        self.data = data

    def get_name(self):
        return self.name

    def get_unit(self):
        return self.unit

    def get_data(self):
        return self.data

    def get_name_with_unit(self):
        return self.name + self.unit.get_symbol_bracket()