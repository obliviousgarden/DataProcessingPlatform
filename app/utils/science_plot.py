import numpy as np
import matplotlib.pyplot as plt


class SciencePlotData:
    def __init__(self):
        self.plot_data = {}

    def add_figure_info(self, figure_title, x_label, y_label):
        self.plot_data[figure_title] = {'x_label': x_label, 'y_label': y_label, 'y_data_dict': {}}

    def add_plot_data(self, figure_title, x_data, y_data, y_legend):
        if figure_title not in self.plot_data.keys():
            print("ERROR: unknown figure_title")
        if y_legend in self.plot_data[figure_title]['y_data_dict']:
            print("ERROR: y_legend Exist")
        else:
            self.plot_data[figure_title]['y_data_dict'][y_legend] = [x_data,y_data]

    def count(self) -> int:
        return self.plot_data.__len__()

    def figure_titles(self) -> list:
        return list(self.plot_data.keys())

    def y_legends(self, figure_title) -> list:
        return list(self.plot_data[figure_title]['y_data_dict'].keys())

    def x_data(self, figure_title, y_legend) -> list:
        return list(self.plot_data[figure_title]['y_data_dict'][y_legend][0])

    def y_data(self, figure_title, y_legend):
        return list(self.plot_data[figure_title]['y_data_dict'][y_legend][1])

    def x_label(self,figure_title) -> str:
        return str(self.plot_data[figure_title]['x_label'])

    def y_label(self,figure_title) -> str:
        return str(self.plot_data[figure_title]['y_label'])


class SciencePlot:
    @staticmethod
    def sci_plot(data: SciencePlotData):
        count = data.count()
        row_count = 0
        col_count = 0
        if count in [1]:
            row_count = 1
            col_count = 1
        elif count in [2]:
            row_count = 1
            col_count = 2
        elif count in [3, 4]:
            row_count = 2
            col_count = 2
        elif count in [5, 6]:
            row_count = 2
            col_count = 3
        elif count in [7, 8, 9]:
            row_count = 3
            col_count = 3
        else:
            print("SciencePlot.sci_plot: Too many sub-figures.")
        i = 0
        for figure_title in data.figure_titles():
            i = i + 1
            ax = plt.subplot(row_count, col_count, i)
            plt.sca(ax)
            plt.title(figure_title)
            plt.xlabel(data.x_label(figure_title))
            plt.ylabel(data.y_label(figure_title))
            for y_label in data.y_legends(figure_title=figure_title):
                plt.plot(data.x_data(figure_title=figure_title, y_legend=y_label),
                         data.y_data(figure_title=figure_title, y_legend=y_label))
            plt.legend(data.y_legends(figure_title=figure_title))
        plt.show()


if __name__ == "__main__":
    x_list = [1, 2, 3]
    x1_list = [-1, 0, 1]
    y1_list = [3, 4, 5]
    y2_list = [2, 4, 6]
    y3_list = [3, 6, 9]
    plot_data = SciencePlotData()
    plot_data.add_figure_info(figure_title='title-1', x_label='x_label_1', y_label='y_label_1')
    plot_data.add_plot_data(figure_title='title-1', x_data=x_list, y_data=y1_list, y_legend='y_legend_1')
    plot_data.add_plot_data(figure_title='title-1', x_data=x_list, y_data=y2_list, y_legend='y_legend_2')
    plot_data.add_figure_info(figure_title='title-2', x_label='x_label_2', y_label='y_label_2')
    plot_data.add_plot_data(figure_title='title-2', x_data=x1_list, y_data=y2_list, y_legend='y_legend_2')
    plot_data.add_plot_data(figure_title='title-2', x_data=x1_list, y_data=y3_list, y_legend='y_legend_3')
    SciencePlot.sci_plot(plot_data)
