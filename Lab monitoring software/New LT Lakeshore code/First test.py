# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 13:52:51 2024

@author: ppztsj
"""

"Test script for opening communications with Lakeshore 331 Temperature controller"

import serial

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
ser = serial.Serial("COM4", 9600, serial.SEVENBITS,
    serial.PARITY_ODD, serial.STOPBITS_ONE )
#ser.baudrate = int(9600)
#ser.port = "COM4"
ser.timeout = int(5)
#ser
ser.open()

#%%

ser.write('PID? 1\r\n'.encode())
read_pressure_data = ser.readline()
#ser.flush()
a = str(read_pressure_data).split(",")
test = read_pressure_data.decode()

test1 = bytes(a[0], 'utf-8')

