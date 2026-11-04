import datetime
import serial
import numpy as np
import matplotlib.pyplot as plt
import time
# import serial

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
    def __init__(self, parent=None, twin=False, ylabel = "variable"):
        self.fig, self.ax = plt.subplots()
        super().__init__(self.fig)

        if parent is not None:
            self.setParent(parent)

        self.scatter = self.ax.scatter([], [], label="Data")
        self.scatter2 = None
        self.ax2 = None

        if twin:
            self.ax2 = self.ax.twinx()
            self.scatter2 = self.ax2.scatter([], [], color='orange', label="Data 2")
            self.ax2.set_ylabel("Pressure 2")
            self.ax2.grid(False)

        self.ax.set_xlabel("Time")
        self.ax.set_ylabel(ylabel)

    def _set_y_limits(self, ax, y_values):
        y_min, y_max = min(y_values), max(y_values)
        if y_min == y_max:
            y_min -= 1
            y_max += 1
        pad = 0.05 * (y_max - y_min)
        ax.set_ylim(y_min - pad, y_max + pad)

    def update_scatter(self, x, y1, title, y2=None, label1="Data", label2="Data 2"):
        if len(x) == 0:
            return

        self.scatter.set_offsets(np.column_stack((x, y1)))
        self.scatter.set_label(label1)

        if y2 is not None and self.scatter2 is not None:
            self.scatter2.set_offsets(np.column_stack((x, y2)))
            self.scatter2.set_label(label2)

            self._set_y_limits(self.ax, y1)
            self._set_y_limits(self.ax2, y2)
        else:
            self._set_y_limits(self.ax, y1)

        x_min, x_max = min(x), max(x)
        if x_min == x_max:
            x_min -= 1
            x_max += 1
        x_pad = 0.05 * (x_max - x_min)
        self.ax.set_xlim(x_min - x_pad, x_max + x_pad)
        self.ax.set_title(title)
        self.draw_idle()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
    #     self.set_up_server))
    # #    self.ser = serial.Serial("COM3", 9600, serial.SEVENBITS,
    # #     serial.PARITY_ODD, serial.STOPBITS_ONE )
    # # #ser.baudrate = int(9600)
    # # #ser.port = "COM4"
    # #     ser.timeout = int(5)

        self.T1 = serial.Serial(
                    port='/dev/ttyUSB0',
                    # port='/dev/ttyUSB1',
                    baudrate=9600,
                    parity=serial.PARITY_NONE,
                    stopbits=serial.STOPBITS_ONE,
                    bytesize=serial.EIGHTBITS,
                    xonxoff=False,
                    timeout = 0.5,
                )

        self.P1 = serial.Serial(
                    # port='COM3',
                    port='/dev/ttyUSB1',
                    baudrate=9600,
                    parity=serial.PARITY_NONE,
                    stopbits=serial.STOPBITS_ONE,
                    bytesize=serial.EIGHTBITS,
                    xonxoff=False,
                    timeout = 0.5,
                )

        self.P2 = serial.Serial(
                            # port='COM4',
                            port='/dev/ttyUSB2',
                            baudrate=9600,
                            parity=serial.PARITY_NONE,
                            stopbits=serial.STOPBITS_ONE,
                            bytesize=serial.EIGHTBITS,
                            xonxoff=False,
                            timeout = 0.5,)
        

        self.T_data = []
        self.P_data_1 = []
        self.P_data_2 = []
        self.LHe_data = []
        self.T_time = []
        self.P_time = []
        self.LHe_time = []

        self.data_list_len = 1000
        self.T_refresh_rate = 1000
        self.P_refresh_rate = 3000
        self.LHe_refresh_rate = 10000
        self.graph_type = 'scatter'


        self.setWindowTitle("A104 UI")
        self.setGeometry(100, 100, 800, 600)

        # Central widget
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        # Main layout
        top_layout = QHBoxLayout()
        main_layout = QVBoxLayout()
        bottom_layout = QHBoxLayout()

        # Add Matplotlib canvas (graph)
        self.Temp = MatplotlibWidget(central_widget, ylabel = "Temperature (K)")
        self.Pressure_1 = MatplotlibWidget(central_widget, ylabel = "Pressure (mbar)")
        self.Pressure_2 = MatplotlibWidget(central_widget, ylabel = "Pressure (mbar)")
        self.LHe = MatplotlibWidget(central_widget, ylabel = "LHe (mm)")
        top_layout.addWidget(self.LHe, stretch=3)  # Stretch to allocate space
        pressure_layout = QVBoxLayout()
        pressure_layout.addWidget(self.Pressure_1, stretch=1)
        pressure_layout.addWidget(self.Pressure_2, stretch=1)
        top_layout.addLayout(pressure_layout, stretch=3)  # Stretch to allocate space
        bottom_layout.addWidget(self.Temp, stretch=3)  # Stretch to allocate space


        main_layout.addLayout(top_layout, stretch=1)  # Add bottom layout with less stretch#
        main_layout.addLayout(bottom_layout, stretch=1)  # Add bottom layout with less stretch


        label_layout = QVBoxLayout()

        label1 = QLabel("Refresh rate (ms):")
        label_layout.addWidget(label1)

        refresh_rate_labels_layout = QHBoxLayout()
        refresh_rate_layout = QHBoxLayout()
        label2 = QLabel("Temperature refresh rate (ms):")
        self.temp_rate_label = QLineEdit(f"{self.T_refresh_rate}")
        label3 = QLabel("Pressure refresh rate (ms):")
        self.pressure_rate_label = QLineEdit(f"{self.P_refresh_rate}")
        label4 = QLabel("LHe refresh rate (ms):")
        self.lhe_rate_label = QLineEdit(f"{self.LHe_refresh_rate}")

        refresh_rate_labels_layout.addWidget(label2)
        refresh_rate_labels_layout.addWidget(label3)
        refresh_rate_labels_layout.addWidget(label4)

        refresh_rate_layout.addWidget(self.temp_rate_label)
        refresh_rate_layout.addWidget(self.pressure_rate_label)
        refresh_rate_layout.addWidget(self.lhe_rate_label)

        label_layout.addLayout(refresh_rate_labels_layout)
        label_layout.addLayout(refresh_rate_layout)

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

        bottom_layout.addLayout(label_layout, stretch=1)
        main_layout.addLayout(bottom_layout, stretch=1)  # Add bottom layout with less stretch
        self.init_time = time.time()
        self.T_timer = QTimer(self)
        self.pressure_timer = QTimer(self)
        self.LHe_timer = QTimer(self)
        
        self.T_timer.start()
        self.pressure_timer.start()
        self.LHe_timer.start()
        self.T_timer.timeout.connect(lambda: self.update_T_graph(self.Temp))
        self.pressure_timer.timeout.connect(self.update_pressure_graphs)
        self.LHe_timer.timeout.connect(lambda: self.update_LHe_graph(self.LHe))
        self.T_timer.setInterval(self.T_refresh_rate)
        self.pressure_timer.setInterval(self.P_refresh_rate)
        self.LHe_timer.setInterval(self.LHe_refresh_rate)

        

        # Set layout to central widget
        central_widget.setLayout(main_layout)

    
    def update_graph_settings(self):
        print("Updating graph settings")
        self.data_list_len = int(self.num_points.text())


        self.T_timer.setInterval(int(self.temp_rate_label.text()))
        self.pressure_timer.setInterval(int(self.pressure_rate_label.text()))
        self.LHe_timer.setInterval(int(self.lhe_rate_label.text()))


        self.graph_type = self.type_box.currentText()
        self.T_data = self.T_data[-self.data_list_len:] #set the data to the list length
        self.T_time = self.T_time[-self.data_list_len:]
        self.P_data_1 = self.P_data_1[-self.data_list_len:]
        self.P_data_2 = self.P_data_2[-self.data_list_len:]
        self.P_time = self.P_time[-self.data_list_len:]
        self.LHe_data = self.LHe_data[-self.data_list_len:]
        self.LHe_time = self.LHe_time[-self.data_list_len:]

    def update_T_data(self, type):
        self.T_data.append(self.read_data(type))
        # self.data.append(self.read_data(type))
        time_diff = time.time() - self.init_time
        self.T_time.append(time_diff)

        if len(self.T_data) > self.data_list_len:
            self.T_data.pop(0)
            self.T_time.pop(0)

    def update_P_data(self):
        p1 = self.read_data("Pressure 1")
        p2 = self.read_data("Pressure 2")

        if p1 is None:
            p1 = self.P_data_1[-1] if self.P_data_1 else 0.0
        if p2 is None:
            p2 = self.P_data_2[-1] if self.P_data_2 else 0.0
    
        self.P_data_1.append(p1)
        self.P_data_2.append(p2)

        time_diff = time.time() - self.init_time
        self.P_time.append(time_diff)

        if len(self.P_data_1) > self.data_list_len:
            self.P_data_1.pop(0)
            self.P_data_2.pop(0)
            self.P_time.pop(0)

    def update_LHe_data(self, type):
        self.LHe_data.append(self.read_data(type))
        time_diff = time.time() - self.init_time
        self.LHe_time.append(time_diff)

        if len(self.LHe_data) > self.data_list_len:
            self.LHe_data.pop(0)
            self.LHe_time.pop(0)

    def update_T_graph(self, graph):
        self.update_T_data("Temperature")
        graph.update_scatter(self.T_time, self.T_data, f"Temperature - {self.T_data[-1]}K ")
    def update_pressure_graphs(self):
        self.update_P_data()

        self.Pressure_1.update_scatter(
            self.P_time,
            self.P_data_1,
            f"Pressure - P1: {self.P_data_1[-1]} mbar"
        )

        self.Pressure_2.update_scatter(
            self.P_time,
            self.P_data_2,
            f"Pressure - P2: {self.P_data_2[-1]} mbar"
        )

    def update_LHe_graph(self, graph):
        self.update_LHe_data("LHe")
        graph.update_scatter(self.LHe_time, self.LHe_data, f"LHe - {self.LHe_data[-1]}K (Last updated: {datetime.datetime.now().strftime('%H:%M:%S')})")

    def on_button_click(self):
        self.label.setText("Button Clicked!")

    def read_data(self, kind):
        if kind == "Temperature":
            self.T1.reset_input_buffer()
            self.T1.write(b'KRDG? A\r\n')
            read_temp_data = self.T1.readline()
            live_temp = read_temp_data.decode()
            live_data = float(live_temp[:-2])
            return live_data

        if kind == "Pressure 1":
            try:
                self.P1.reset_input_buffer()
                self.P1.write(b'*j01\r\n')
                time.sleep(0.05)
                self.P1.write(b'*S0\r\n')
                time.sleep(0.1)

                line = self.P1.readline()
                # print(f"Pressure 1 read line: {line}")
                if not line:
                    return None

                text = line.decode("ascii", errors="ignore").strip()
                parts = text.split(",")
                if not parts:
                    return None

                token = parts[0].split("@")
                if len(token) < 4:
                    return None

                return float(token[3])
            except Exception as exc:
                print(f"Pressure 1 read failed: {exc}")
                return None

        if kind == "Pressure 2":
            try:
                self.P2.reset_input_buffer()
                self.P2.write(b'*J01\r\n')
                time.sleep(0.05)
                self.P2.write(b'*S0\r\n')
                time.sleep(0.1)

                line = self.P2.readline()
                # print(f"Pressure 2 read line: {line}")
                if not line:
                    return None

                text = line.decode("ascii", errors="ignore").strip()
                parts = text.split(",")
                if not parts:
                    return None

                token = parts[0].split("@")
                if len(token) < 4:
                    return None

                return float(token[3])
            except Exception as exc:
                print(f"Pressure 2 read failed: {exc}")
                return None

        if kind == "LHe":
            return np.random.randint(0, 5)

        return None




# Main code to run the application
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
