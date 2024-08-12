"""
Created on Wed Aug  7 13:52:51 2024

@author: ppztsj
"""

"Test script for opening communications with Lakeshore 331 Temperature controller"

import serial
import time
import tkinter as tk
from tkinter import *
import random

# LSserial = serial.Serial( 'COM4', 9600, serial.SEVENBITS, serial.PARITY_ODD, 
#                        serial.STOPBITS_ONE )

# #LSserial.open()
# print("Connecting")
# # Query message :
# #     CRDG = Celsius Reading Query
# #     A is the input can be A or B
# #     Terminators are <CR><LF>
# LSserial.write( 'PID? 1 /r/n'.encode() )
# reply = LSserial.readline()
# print(reply)

# LSserial.close()



#%% Setup the connection here...

#Setup the connection to the controller
ser = serial.Serial("COM3", 9600, serial.SEVENBITS,
    serial.PARITY_ODD, serial.STOPBITS_ONE )
#ser.baudrate = int(9600)
#ser.port = "COM4"
ser.timeout = int(5)
#ser
#ser.open()

#%% Make basic Tkinter window to display temperature
def Draw():
    global text


    frame=tk.Frame(root,width=100,height=100,relief='solid',bd=1)
    frame.place(x=10,y=10)
    text=tk.Label(frame,text='0')
    text.pack(expand=True)

def Refresher():
    global label

    ser.write('KRDG? A\r\n'.encode())
    read_pressure_data = ser.readline()
    test = read_pressure_data.decode()
    tempreading = "Cryostat T = "+test[:-2]+" K"

    label.config(text=tempreading)
    root.after(1000, Refresher) # every second...

root=tk.Tk()

root.title("Temp readings")
root.geometry("400x300")
initial_number = 0
label = tk.Label(root,text=initial_number,font=("Helvetica",79))
label.pack(expand=True)
# Draw()
#Refresher()
# root.mainloop()


#%% Loop to ask for temperature in K every second

count = 1

ser.write('KRDG? A\r\n'.encode())
read_pressure_data = ser.readline()
#ser.flush()
#a = str(read_pressure_data).split(",")
test = read_pressure_data.decode()
# tempreading = "Cryostat T = "+test[:-2]+" K"
tempreading = str(random.randint(100,999))
#print(type(test))
#test1 = bytes(a[0], 'utf-8')
#Refresher()
root.after(1000,Refresher)
root.mainloop()
#print(tempreading)
