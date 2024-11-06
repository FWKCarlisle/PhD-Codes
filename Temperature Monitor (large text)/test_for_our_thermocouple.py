# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 10:27:19 2023

@author: ppzme
"""

import minimalmodbus

PORT='COM4'
TEMP_REGISTER = 4096



#Set up instrument
instrument = minimalmodbus.Instrument(PORT,1,mode=minimalmodbus.MODE_ASCII)

#Make the settings explicit
instrument.serial.baudrate = 9600      # Baud
instrument.serial.bytesize = 8
instrument.serial.parity   = minimalmodbus.serial.PARITY_EVEN
instrument.serial.stopbits = 1
instrument.serial.timeout  = 1          # seconds

# Good practice
instrument.close_port_after_each_call = True
instrument.clear_buffers_before_each_transaction = True

# Read temperatureas a float
# if you need to read a 16 bit register use instrument.read_register()
i  =1
while i==1: 
    temperature = instrument.read_register(TEMP_REGISTER, functioncode=3)

    print('T = '+str(temperature/10)+'C')   

