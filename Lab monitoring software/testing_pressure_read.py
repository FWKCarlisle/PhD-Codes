import serial
import time


ser = serial.Serial("COM5", 9600, serial.EIGHTBITS,
    serial.PARITY_NONE, serial.STOPBITS_ONE )
#ser.baudrate = int(9600)
#ser.port = "COM4"
ser.timeout = int(5)


ser.write('*S'.encode())
read_pressure_data = ser.readline()
test = read_pressure_data.decode()
readout = test.strip().split('@')[3]
pressure = readout[:-2]
if readout[-2] == 'M':
    units = "mBar"
elif readout[-2] == 'T':
    units = "Torr"
elif readout[-2] == 'P':
    units = "Pa"
else:
    print(test[-2])
    units = "Unknown"
output = "Pressure = "+ pressure +" "+units
print(output)