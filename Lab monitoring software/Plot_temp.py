import tkinter as tk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.ticker as ticker
import serial
import pyautogui
from tkinter import ttk
import time


# Initialize the data list
data = []
timestamps = []

ser = serial.Serial("COM3", 9600, serial.SEVENBITS,
    serial.PARITY_ODD, serial.STOPBITS_ONE )
#ser.baudrate = int(9600)
#ser.port = "COM4"
ser.timeout = int(5)
#ser
#ser.open()

# Function to update the window with new data
def update_graph(update_rate=500, data_list_len=500, graph_type="Scatter", count = 0):
    count += 1
    #Update value for temp from Lakeshore
    ser.write('KRDG? A\r\n'.encode())
    read_pressure_data = ser.readline()
    test = read_pressure_data.decode()
    print(test)
    # new_value=float(test[:-2])
    tempreading = "T_Cryo = "+test[:-2]+" K"
    
    data.append(new_value) #add new value to the graphing list
    
    timestamps.append(time.time())
        
    # Limit the list to the last 25 items to keep the graph to 25 points
    if len(data) > data_list_len:
        data.pop(0)
        timestamps.pop(0)
    
    # Clear the previous plot
    ax.clear()
    if count == 10:
        pyautogui.press('scrolllock')
        print("Scrolllock")
        pyautogui.press('scrolllock')
        count = 0 
 
    # Plot the updated data
    if graph_type == "Scatter": 
        ax.scatter(timestamps, data, marker='o',s=4)
    else:
        ax.plot(timestamps, data, marker='o')
    ax.set_title("Real-Time Line Graph")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Value")
    # ax.set_ylim(0, 100)  # Set y-axis limits for consistent scaling
    
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
    
    # Refresh the canvas to display the new plot and update the label
    label.config(text=tempreading)
    canvas.draw()
    
    # Schedule the next update in 1 second
    root.after(update_rate, update_graph)
    
def button_press(Graph_type):
    update_rate = int(speed.get())
    data_list_len = int(points.get())
    
    update_graph(update_rate,data_list_len,Graph_type)

# Create the main window
root = tk.Tk()
root.title("Real-Time Graph Display")

initial_number = 0
label = tk.Label(root,text=initial_number,font=("Helvetica",79))
label.pack(expand=True)

# Create a Matplotlib figure
fig = Figure(figsize=(5, 4), dpi=100)
ax = fig.add_subplot(111)

# label = ttk.Label(root, text="Enter a refresh rate (ms):")
# label.pack(pady=5)

# speed = ttk.Entry(root)
# speed.pack(pady=10)
# speed.insert(0, "500")

# label = ttk.Label(root, text="Enter an amount of data points to show (Default=500):")
# label.pack(pady=5)

# points = ttk.Entry(root)
# points.pack(pady=10)
# points.insert(0, "1000")

# Gradient_100 = ttk.Label(root, text="Gradient over all data: ")
# Gradient_100.pack(pady=10)

# Gradient_50 = ttk.Label(root, text="Gradient over all data: ")
# Gradient_50.pack(pady=10)

# Gradient_10 = ttk.Label(root, text="Gradient over all data: ")
# Gradient_10.pack(pady=10)



# Embed the figure in the Tkinter window
canvas = FigureCanvasTkAgg(fig, master=root)

# button = tk.Button(root, 
#                    text="Scatter", 
#                    fg="red",
#                    command=button_press("Scatter"))
# button.pack(side=tk.LEFT)
# slogan = tk.Button(root,
#                    text="Line",
#                    command=button_press("Line"))
# slogan.pack(side=tk.LEFT)
# Schedule the first update
canvas.draw()
canvas.get_tk_widget().pack(expand=True)

update_graph()

# Run the Tkinter event loop
root.mainloop()