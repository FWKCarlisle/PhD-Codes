# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 11:52:46 2021

@author: ppxme
Edited for use with Lakeshore 331 by ppztj1

V1: Modifying the code for the Lakeshore 335 to work with the Lakeshore 331 
which is an older model and does not have support from the Lakeshore Python 
module. Commenting out the sections which handle allowing the user to change 
the heater setpoint for now, starting by making sure the code can first read 
the temperature of the STM and plot it.


"""


#from lakeshore.model_335 import *
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
# import numpy as np
import matplotlib.widgets as widgets
import time
import serial

plt.close('all')

#%% Make figure window

def switch_off_callback(event):
    # Stops aquiring data
    global switchon
    switchon = False
    print('Stopped recording data')

fig = plt.figure(figsize=(15,5))

#Button to press
offax = plt.axes([0.8,0.75,0.1,0.1])
offHandle = widgets.Button(offax,'Stop recording')
offHandle.on_clicked(switch_off_callback)
switchon = True

#Value to enter
def submit(text):
    text_org = text
    text = text.replace('.','',1)
    if text.isnumeric():
        text_number = float(text_org)
        if text_number>1 and text_number<310:
            #my_model_335.set_control_setpoint(2,text_number)
            print('SET NUMBER '+str(text_number))
        else:
            print('Enter a number between 1K and 310K')
            
    else:
        print('Enter numeric value')
            
axbox = plt.axes([0.1, 0.05, 0.8, 0.075])
text_box = widgets.TextBox(axbox, 'Temp Setpoint (K)', initial='input value')
text_box.on_submit(submit)

#For heater range setting
rax = plt.axes([0.8, 0.5, 0.10, 0.15])
radio = widgets.RadioButtons(rax, ('Off', 'Low', 'Med', 'High'))

# def hzfunc(label):
#     print('Heater button press')
    
#     if label == 'Off':
#         my_model_335.set_heater_range(2, Model335HeaterRange.OFF)
#         print('Off')
#     if label == 'Low':
#         my_model_335.set_heater_range(2, Model335HeaterRange.LOW)
#         print('Low')
#     if label == 'Med':
#         my_model_335.set_heater_range(2, Model335HeaterRange.MEDIUM)
#         print('Med')
#     if label == 'High':
#         my_model_335.set_heater_range(2, Model335HeaterRange.HIGH)
#         print('High')
    
# radio.on_clicked(hzfunc)

def nothing_fun(label):
    print('Resetting label to Off')

#For stopping all 
def switch_off_callback1(event):
    # Stops aquiring data
    print('Stop heater!')
    #my_model_335.all_heaters_off()
    radio.set_active(0)
    


offax1 = plt.axes([0.8,0.25,0.1,0.1])
offHandle1 = widgets.Button(offax1,'Turn all heaters OFF')
offHandle1.on_clicked(switch_off_callback1)


ax = plt.axes([0.15,0.25,0.6,0.7])
plt.xlabel('Time(Seconds)')
plt.ylabel('Temperature (K)')



#%% Start time
start_time = datetime.now()


#%% 

# # Connect to the first available Model 335 temperature controller over USB using a baud rate of 57600
# my_model_335 = Model335(57600)

# # Seems to fail on the first attempt... but then is OK
# try:
#     value = my_model_335.get_all_kelvin_reading()
# except:
#     print('OK')

# #values = [1,0]


# Connect to the Lakeshore 331, need to connect via a COM port, if this doesn't
# work, consult Device Manager and check which COM port is being used for the 
# RS232-to-USB converter for the Lakeshore

# A Baud rate of 9600 is used because this is the highest rate this model of 
# Lakeshore is capable of, and the connection specifics are based on 
# information in the Lakeshore 331 manual

#Setup the connection to the controller
ser = serial.Serial("COM3", 9600, serial.SEVENBITS,
    serial.PARITY_ODD, serial.STOPBITS_ONE )

ser.timeout = int(5)


all_data_lakeshore = pd.DataFrame({'Time': [], 'Cryostat Temp (K)': []})#, 'Sample Temp(K)':[]})



count = 0
part = 1
    
while switchon == True:
    #values = my_model_335.get_all_kelvin_reading()
    ser.write('KRDG? A\r\n'.encode())
    read_kelvin_data = ser.readline()
    cryostat_temp = read_kelvin_data.decode()

    #test1 = bytes(a[0], 'utf-8')
    
    
    #Old way... use time.time() instead!
    #all_data_lakeshore = all_data_lakeshore.append({'Time': [(datetime.now()).strftime("%H:%M:%S")], 'Cryostat Temp (K)': [values[0]], 'Sample Temp(K)':[values[1]]}, ignore_index=True)
    all_data_lakeshore = all_data_lakeshore._append({'Time': [time.time()], 'Cryostat Temp (K)': [cryostat_temp]}, ignore_index=True)#, 'Sample Temp(K)':[values[1]]}, ignore_index=True)
    print(all_data_lakeshore)
    print(type(all_data_lakeshore['Time']))
    print(type(all_data_lakeshore['Cryostat Temp (K)']))
    ax.clear()
    plt.plot((all_data_lakeshore['Cryostat Temp (K)']).tolist(),'k')
    # plt.plot((all_data_lakeshore['Sample Temp(K)']).tolist(),'b')
    plt.xlabel('Time(Seconds)')
    plt.ylabel('Temperature (K)')
    
    count = count + 1
    if count % 1000 == 0:
        name = 'Time_'+start_time.strftime("%H_%M_%S")+'_Date_'+start_time.strftime("%d_%m_%y")+'_H_M_S___TVS_part_'+str(part)+'.csv'
        all_data_lakeshore.to_csv(name, encoding='utf-8', index=False)
        part = part + 1
    
    plt.pause(1)
    
    
    
    
#%% Save end data

name = 'Time_'+start_time.strftime("%H_%M_%S")+'_Date_'+start_time.strftime("%d_%m_%y")+'_H_M_S___TVS.csv'
all_data_lakeshore.to_csv(name, encoding='utf-8', index=False)
