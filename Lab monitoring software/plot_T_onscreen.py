import tkinter as tk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.ticker as ticker
import random
import serial
import time
import tkinter as tk
from tkinter import *
import random

# Initialize the data list
data = []


ser = serial.Serial("COM3", 9600, serial.SEVENBITS,
    serial.PARITY_ODD, serial.STOPBITS_ONE )
#ser.baudrate = int(9600)
#ser.port = "COM4"
ser.timeout = int(5)
#ser
#ser.open()

# Function to update the graph with new data
def update_graph():
     #Generate a new random number and append it to the data list
    #new_value = random.randint(0, 100)
    ser.write('KRDG? A\r\n'.encode())
    read_pressure_data = ser.readline()
    test = read_pressure_data.decode()
    new_value=float(test[:-2])
    tempreading = "Cryostat T = "+test[:-2]+" K"
    
    data.append(new_value)
    
    # Limit the list to the last 25 items to keep the graph to 25 points
    if len(data) > 25:
        data.pop(0)
    
    # Clear the previous plot
    ax.clear()
    
    # Plot the updated data
    ax.plot(data, marker='o')
    ax.set_title("Real-Time Line Graph")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Value")
    # ax.set_ylim(0, 100)  # Set y-axis limits for consistent scaling
    
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
    
    # Refresh the canvas to display the new plot
    canvas.draw()
    
    # Schedule the next update in 1 second
    root.after(500, update_graph)

# Create the main window
root = tk.Tk()
root.title("Real-Time Graph Display")

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
