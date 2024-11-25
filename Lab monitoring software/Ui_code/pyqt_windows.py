import sys
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QPushButton, QWidget
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)


class MatplotlibWidget(FigureCanvas):
    def __init__(self, parent=None):
        # Create a Matplotlib figure
        self.fig, self.ax = plt.subplots()
        super().__init__(self.fig)

        # Optional: Set parent widget
        if parent is not None:
            self.setParent(parent)

    def plot_graph(self, x, y, title, xlabel, ylabel, data_label):
        # Plot a simple graph
        self.ax.clear()
        self.ax.scatter(x, y, label=data_label)
        self.ax.legend()
        self.ax.set_title(title)
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        self.draw()