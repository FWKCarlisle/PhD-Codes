import numpy as np
import matplotlib.pyplot as plt
import time

import sys
from PyQt5.QtWidgets import (
    QApplication,QComboBox, QMainWindow,
    QLabel, QVBoxLayout,QHBoxLayout, 
    QPushButton, QWidget, QLineEdit,
)
from PyQt5.QtCore import QTimer
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
)

class MatplotlibWidget(FigureCanvas):
    def __init__(self, parent=None):
        # Create a Matplotlib figure
        self.fig, self.ax = plt.subplots()
        super().__init__(self.fig)

        # Optional: Set parent widget
        if parent is not None:
            self.setParent(parent)

    def plot_graph(self, x, y, title, xlabel, ylabel, data_label, type = 'Scatter'):
        # Plot a simple graph
        self.ax.clear()
        if type == 'Line':
            self.ax.plot(x, y, label=data_label)
        else:
            self.ax.scatter(x, y, label=data_label)
        
        self.ax.legend()
        self.ax.set_title(title)
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        self.draw()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PyQt5 with Matplotlib")
        self.setGeometry(100, 100, 800, 600)

        self.data = []
        self.time = []

        self.data_list_len = 300
        self.refresh_rate = 300
        self.graph_type = 'scatter'


        self.setWindowTitle("PyQt5: Graph with Input Boxes")
        self.setGeometry(100, 100, 800, 600)

        # Central widget
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        # Main layout
        main_layout = QHBoxLayout()

        # Add Matplotlib canvas (graph)
        self.canvas = MatplotlibWidget(central_widget)
        main_layout.addWidget(self.canvas, stretch=3)  # Stretch to allocate space

        # Add a label



        label_layout = QVBoxLayout()

        self.temp_label = QLabel("Temperature: K")
        label_layout.addWidget(self.temp_label)

        # Label and input box 1
        label1 = QLabel("Refresh rate (ms):")
        self.rate_label = QLineEdit(f"{self.refresh_rate}")
        label_layout.addWidget(label1)
        label_layout.addWidget(self.rate_label)

        label1 = QLabel("Amount of points:")
        self.num_points = QLineEdit(f"{self.data_list_len}")
        label_layout.addWidget(label1)
        label_layout.addWidget(self.num_points)

        label1 = QLabel("Graph type:")
        self.type_box = QComboBox()
        self.type_box.addItem("Scatter")
        self.type_box.addItem("Line")
        label_layout.addWidget(label1)
        label_layout.addWidget(self.type_box)



        # Add a button
        button = QPushButton("Update graph settings", self)
        button.clicked.connect(self.update_graph_settings)
        label_layout.addWidget(button)

        # Stretch to push content to the top
        label_layout.addStretch()

        main_layout.addLayout(label_layout, stretch=1)

        self.init_time = time.time()
        self.timer = QTimer(self)  # Connect to the method
        self.timer.start()  # Update every 1000ms (1s)
        self.timer.timeout.connect(self.update_graph)
        self.timer.setInterval(self.refresh_rate)
        # Set layout to central widget
        central_widget.setLayout(main_layout)

    
    def update_graph_settings(self):
        print("Updating graph settings")
        self.data_list_len = int(self.num_points.text())
        self.timer.setInterval(int(self.rate_label.text()))
        self.graph_type = self.type_box.currentText()
        self.data = self.data[-self.data_list_len:] #set the data to the list length
        self.time = self.time[-self.data_list_len:]

    def update_data(self):
        self.data.append(self.read_data())
        time_diff = time.time() - self.init_time
        self.time.append(time_diff)

        if len(self.data) > self.data_list_len:
            self.data.pop(0)
            self.time.pop(0)

    def update_graph(self):
        self.update_data()
        self.canvas.plot_graph(self.time, self.data, "Rand data over time", "Time", "Data", "Random Data", self.graph_type)
        self.temp_label.setText(f"Temperature: {self.data[-1]} K")
    def on_button_click(self):
        self.label.setText("Button Clicked!")

    def read_data(self):
        live_data = np.random.randint(0, 10)
        return live_data




# Main code to run the application
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
