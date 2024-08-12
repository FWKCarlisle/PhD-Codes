import serial
import tkinter as tk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.ticker as ticker
import time

# Initialize the arrays
data = []
timestamps = []


# Initialize the serial connection
hemeter = serial.Serial('COM3', 9600, serial.EIGHTBITS, serial.PARITY_NONE, serial.STOPBITS_ONE, xonxoff=True)
hemeter.timeout = int(5)

# define functions for the serial connection
def read_level():
    hemeter.write( "T \r\n".encode())
    hemeter.write( "G \r\n".encode())
    reply = hemeter.readline()
    return reply

def read_status():
    hemeter.write( "S \r\n".encode())
    reply = hemeter.readline()
    return reply

def change_mode(mode): #mode = 0 for STBY, 1 for Slow, 2 for Fast, 3 for continuous
    hemeter.write(f"M{mode} \r\n".encode())
    reply = hemeter.readline()
    return reply

# Function to update the window with new data
def update_graph():
    #Update value for temp from Lakeshore
    level = read_level()[2:6].decode()
    
    data.append(int(level)) #add new value to the graphing list
    
    timestamps.append(time.time())
        
  
    # Limit the list to the last 25 items to keep the graph to 25 points
    if len(data) > 10:
        data.pop(0)
        timestamps.pop(0)
    
    # Clear the previous plot
    ax.clear()
    
 
    # Plot the updated data
    ax.plot(timestamps, data, marker='o')
    ax.set_title("Real-Time Line Graph")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Value")
    # ax.set_ylim(0, 100)  # Set y-axis limits for consistent scaling
    
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
    
    # Refresh the canvas to display the new plot and update the label
    label.config(text= level + " mm")
    canvas.draw()
    
    # Schedule the next update in 1 second
    root.after(500, update_graph)

# Create the main window
root = tk.Tk()
root.title("Real-Time Graph Display")

initial_number = 0
label = tk.Label(root,text=initial_number,font=("Helvetica",50))
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

# Add buttons to change the mode
def change_mode_button(mode):
    change_mode(mode)
    print("Mode changed to " + str(mode))

button1 = tk.Button(root, text="STBY", command=lambda: change_mode_button(0))
button1.pack(side=tk.LEFT)

button2 = tk.Button(root, text="Slow", command=lambda: change_mode_button(1))
button2.pack(side=tk.LEFT)

button3 = tk.Button(root, text="Fast", command=lambda: change_mode_button(2))
button3.pack(side=tk.LEFT)

button4 = tk.Button(root, text="Cont", command=lambda: change_mode_button(3))
button4.pack(side=tk.LEFT)

# Run the Tkinter event loop
root.mainloop()