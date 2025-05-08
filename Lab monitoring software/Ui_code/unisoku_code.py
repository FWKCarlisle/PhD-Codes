"""
Created on Wed Apr  2 09:00:33 2025

@author: CRYOGENICSYSTEM
"""
"""
Created on Fri Mar 28 14:24:46 2025

@author: Unisoku
"""
import numpy as np
import matplotlib.pyplot as plt
import time
import serial

import sys
from PyQt5.QtWidgets import (
    QApplication,QComboBox, QMainWindow,
    QLabel, QVBoxLayout,QHBoxLayout,
    QPushButton, QWidget, QLineEdit,
    QCheckBox,
)
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QFont
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

    def plot_graph(self, x, y,   data_label,marker_size, type = 'Scatter',colour="b"):
        # Plot a simple graph
       
        if type == 'Line':
            self.ax.plot(x, y, c=colour,label=data_label)
        else:
            self.ax.scatter(x, y, c=colour,label=data_label,s=marker_size)
       
       
       
    def draw_graph(self,title,xlabel, ylabel):
        self.ax.legend()
        self.ax.set_title(title)
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        self.draw()

class MainWindow(QMainWindow):
    def __init__(self):
        data = []
        timestamps = []
       
        self.ser = serial.Serial("COM6", 9600, serial.EIGHTBITS,
            serial.PARITY_NONE, serial.STOPBITS_ONE )
        #ser.baudrate = int(9600)
        #ser.port = "COM4"
        self.ser.timeout = int(5)
       
        super().__init__()
        self.setWindowTitle("PyQt5 with Matplotlib")
        self.setGeometry(100, 100, 800, 600)
       
        self.marker_size = 50
        self.STM = []
        self.Sorb = []
        self.ThreeHe = []
        self.OneK =[]
        self.time = []
        self.temp_location = "STM"
        self.temp_locations = {"STM": 1, "Sorb": 0, "3He": 0, "1K pot": 0}
        self.temp_locations_list = ",".join([key for key in self.temp_locations.keys() if self.temp_locations[key]==1])
       
        self.data_list_len = 5000
        self.refresh_rate = 300
        self.graph_type = 'scatter'


        self.setWindowTitle("Plotting system temps over time")
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
        font = QFont()
        font.setPointSize(14)
        font.setBold(True)

        self.temp_label_STM = QLabel("Temp (STM): K")
        self.temp_label_STM.setFont(font)
        self.temp_label_Sorb = QLabel("Temp (Sorb): K")
        self.temp_label_Sorb.setFont(font)
        self.temp_label_ThreeHe = QLabel("Temp (3He): K")
        self.temp_label_ThreeHe.setFont(font)
        self.temp_label_OneK = QLabel("Temp (1K): K")
        self.temp_label_OneK.setFont(font)
        label_layout.addWidget(self.temp_label_STM)
        label_layout.addWidget(self.temp_label_Sorb)
        label_layout.addWidget(self.temp_label_ThreeHe)
        label_layout.addWidget(self.temp_label_OneK)
        # Label and input box 1
        label1 = QLabel("Refresh rate (ms):")
        self.rate_label = QLineEdit(f"{self.refresh_rate}")
        self.rate_label.setFont(font)
        label_layout.addWidget(label1)
        label_layout.addWidget(self.rate_label)

        label1 = QLabel("Amount of points:")
        self.num_points = QLineEdit(f"{self.data_list_len}")
        self.num_points.setFont(font)
        label_layout.addWidget(label1)
        label_layout.addWidget(self.num_points)

        label1 = QLabel("Graph type:")
        self.type_box = QComboBox()
        self.type_box.addItem("Scatter")
        self.type_box.addItem("Line")
        self.type_box.setFont(font)
        label_layout.addWidget(label1)
        label_layout.addWidget(self.type_box)
       
       
   
   
        self.checkBox_STM = QCheckBox("STM" )
        self.checkBox_STM.stateChanged.connect(self.checkedSTM)
        self.checkBox_STM.setChecked(True)
        label_layout.addWidget(self.checkBox_STM)
       
        self.checkBox_Sorb = QCheckBox("Sorb")
        self.checkBox_Sorb.stateChanged.connect(self.checkedSorb)
        label_layout.addWidget(self.checkBox_Sorb)
       
        self.checkBox_ThreeHe = QCheckBox("3He")
        self.checkBox_ThreeHe.stateChanged.connect(self.checkedThreeHe)
        label_layout.addWidget(self.checkBox_ThreeHe)
       
        self.checkBox_OneK = QCheckBox("1k pot")
        self.checkBox_OneK.stateChanged.connect(self.checkedOneK)
        label_layout.addWidget(self.checkBox_OneK)
   
        # Add a button
        button = QPushButton("Update graph settings", self)
        button.setFont(font)
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

    def checkedSTM(self, checked):
        if checked:
            self.temp_locations['STM']= 1
        else:
            self.temp_locations['STM']= 0
        self.show()
       
    def checkedSorb(self, checked):
        if checked:
            self.temp_locations['Sorb']= 1
        else:
            self.temp_locations['Sorb']= 0
        self.show()
       
    def checkedThreeHe(self, checked):
        if checked:
            self.temp_locations['3He']= 1
        else:
            self.temp_locations['3He']= 0
        self.show()
       
    def checkedOneK(self, checked):
        if checked:
            self.temp_locations['1K pot']= 1
        else:
            self.temp_locations['1K pot']= 0
        self.show()
       
    def update_graph_settings(self):
        print("Updating graph settings")
        self.data_list_len = int(self.num_points.text())
        self.timer.setInterval(int(self.rate_label.text()))
        self.graph_type = self.type_box.currentText()
       
       
        temp_locations_list = ",".join([key for key in self.temp_locations.keys() if self.temp_locations[key]==1])
       
        if temp_locations_list != self.temp_locations_list:
            self.STM = []
            self.Sorb = []
            self.ThreeHe = []
            self.OneK =[]
            self.time = []
            self.init_time = time.time()
           
        self.temp_locations_list = temp_locations_list
           
       
       
    def resizeEvent(self, event):
        width = self.width()
        height = self.height()
        font_size = int(width / 40)
        font = QFont()
        font.setPointSize(font_size)
        self.temp_label_STM.setFont(font)
        self.temp_label_Sorb.setFont(font)
        self.temp_label_OneK.setFont(font)
        self.temp_label_ThreeHe.setFont(font)
        self.rate_label.setFont(font)
        self.num_points.setFont(font)
        self.type_box.setFont(font)
        # Dynamically adjust the marker size based on window size
        # Marker size will range from 10 to 200, depending on the window's width
        marker_size = max(10, min(int(width / 20), 200))  # Control the size range as needed
        self.marker_size = marker_size
        super().resizeEvent(event)
       
       
    def update_data(self):
        STM, Sorb, one_k, three_He = self.read_data()
       
        self.STM,  pop1 = self.update_arrays(STM, self.STM, self.data_list_len)
        self.Sorb, pop2 = self.update_arrays(Sorb, self.Sorb, self.data_list_len)
        self.OneK, pop3 = self.update_arrays(one_k, self.OneK, self.data_list_len)
        self.ThreeHe, pop4 = self.update_arrays(three_He, self.ThreeHe, self.data_list_len)
       
       
       
       
        time_diff = time.time() - self.init_time
        self.time.append(time_diff)

        if pop1 == True or pop2 == True or pop3 == True or pop4 == True:
            self.time.pop(0)
           
    def update_arrays (self,value, array, list_len):
       
        POP = False
       
        if value is not None:
            array.append(value)
           
        if len(array) > list_len:
            POP = True
            array.pop(0)
           
        return array, POP
           
           
       
    def update_graph(self):
        self.update_data()
       
        self.canvas.ax.clear()
        #print(self.temp_locations_list)
        if "Sorb" in self.temp_locations_list:
            #print("Yes", len(self.time), len(self.Sorb))
            self.canvas.plot_graph(self.time, self.Sorb, "Sorb",self.marker_size ,self.graph_type,colour="r")
            self.temp_label_Sorb.setText(f"Temp (Sorb): {self.Sorb[-1]} K")
        if "STM" in self.temp_locations_list:
            #print("No", len(self.time), len(self.STM))
            self.canvas.plot_graph(self.time, self.STM, "STM",self.marker_size ,self.graph_type,colour="b")
            self.temp_label_STM.setText(f"Temp (STM): {self.STM[-1]} K")
        if "1K pot" in self.temp_locations_list:
            #print("Maybe", len(self.time), len(self.OneK))
            self.canvas.plot_graph(self.time, self.OneK, "1K",self.marker_size ,self.graph_type,colour="g")
            self.temp_label_OneK.setText(f"Temp (1K): {self.OneK[-1]} K")
        if "3He" in self.temp_locations_list:
            #print("Maybe", len(self.time), len(self.ThreeHe))
            self.canvas.plot_graph(self.time, self.ThreeHe, "3He",self.marker_size ,self.graph_type,colour="orange")
            self.temp_label_ThreeHe.setText(f"Temperature (3He): {self.ThreeHe[-1]} K")
        self.canvas.draw_graph("Temp data against time", "Time (s)", "Temp (K)")
       
       
       
    def on_button_click(self):
        self.label.setText("Button Clicked!")

    def read_data(self):
       
        Sorb = None
        one_k = None
        three_He = None
        STM = None
       
        if "STM" in self.temp_locations_list:
            self.ser.write(b'input? d\r\n')
            STM = round(float(self.ser.readline().decode()),4)
        if "Sorb" in self.temp_locations_list:
            self.ser.write(b'input? a\r\n')
            Sorb = round(float(self.ser.readline().decode()),4)
        if "1K pot" in self.temp_locations_list:  
            self.ser.write(b'input? b\r\n')
            one_k = round(float(self.ser.readline().decode()),4)
        if "3He" in self.temp_locations_list:
            self.ser.write(b'input? c\r\n')
            three_He = round(float(self.ser.readline().decode()),4)
       
   
       
        return STM, Sorb, one_k, three_He




# Main code to run the application
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
________________________________________
From: Frederick Carlisle <ppxfc1@exmail.nottingham.ac.uk>
Sent: 08 May 2025 11:53
To: Frederick Carlisle <ppxfc1@exmail.nottingham.ac.uk>
Subject: 
 
import tkinter as tk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.ticker as ticker
import serial
import time
import datetime

# Initialize the data list
data = []
timestamps = []

Temp_or_pressure = False #True = temp, false = pressure
if Temp_or_pressure:
    print("Graphing temp")
    bits = serial.SEVENBITS
    parity = serial.PARITY_ODD
else:
    print("Graphing pressure")
    bits = serial.EIGHTBITS
    parity = serial.PARITY_NONE

ser = serial.Serial("COM3", 9600, bits,
    parity, serial.STOPBITS_ONE )
# ser.baudrate = int(9600)
#ser.port = "COM4"
ser.timeout = int(5)
#ser
# ser.open()

# Function to update the window with new data
def update_graph():
    #Update value for temp from Lakeshore
    if Temp_or_pressure:
        ser.write('KRDG? A\r\n'.encode())
        read_temp_data = ser.readline()
        output = read_temp_data.decode()
        print(output)
        new_value=float(output[:-2])
        reading = "T_Cryo = "+output[:-2]+" K"
    else:
        # time.sleep(1)
        for i in range(5):
            ser.write('*S'.encode())
            read_pressure_data = ser.readline()
            output = read_pressure_data.decode()
            # print(output)
            if output[0] == '"':
                print(output.strip().split('@'))
                readout = output.strip().split('@')[-1]
                new_value = readout[:-3]
                if readout[-2] == 'M':
                    units = "mBar"
                elif readout[-2] == 'T':
                    units = "Torr"
                elif readout[-2] == 'P':
                    units = "Pa"
                else:
                    print(output[-3])
                    units = "Unknown"
                reading = f"{new_value} {units}"
    print(new_value, datetime.datetime.now())
    data.append(float(new_value)) #add new value to the graphing list
   
    timestamps.append(time.time())
       
    # print(timestamps[-1])
    # Limit the list to the last 25 items to keep the graph to 25 points
    if len(data) > 5000:
        data.pop(0)
        timestamps.pop(0)
   
    # Clear the previous plot
    ax.clear()
   
 
    # Plot the updated data
    ax.scatter(timestamps, data, marker='o')
    ax.set_title("Real-Time Line Graph")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Value")
    # ax.set_ylim(0, 100)  # Set y-axis limits for consistent scaling
   
    # ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
   
    # Refresh the canvas to display the new plot and update the label
    label.config(text=reading)
    canvas.draw()
   
    # Schedule the next update in 1 second
    root.after(5000, update_graph)

# Create the main window
root = tk.Tk()
root.title("Real-Time Graph Display")

initial_number = 0
label = tk.Label(root,text=initial_number,font=("Helvetica",79))
label.pack(expand=True)

# Create a Matplotlib figure
fig = Figure(figsize=(5, 4), dpi=100)
ax = fig.add_subplot(111)

# Embed the figure in the Tkinter window
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.draw()
canvas.get_tk_widget().pack(expand=True)

# Schedule the first update
update_graph()

# Run the Tkinter event loop
root.mainloop()

