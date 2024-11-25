import serial
import time
import minimalmodbus

# ser = serial.Serial("COM3", 19200, serial.EIGHTBITS,
#     serial.PARITY_NONE, serial.STOPBITS_ONE )
# #ser.baudrate = int(9600)
# #ser.port = "COM4"
# ser.timeout = int(5)
# if ser.is_open:
#     print("Serial port is open")
# ser.write(":>01?Iv!".encode())
# read_pressure_data = ser.readline()
# test = read_pressure_data.decode()
# print(read_pressure_data)

instrument = minimalmodbus.Instrument('COM3', 1)  # port name, slave address (in decimal)
instrument.serial.baudrate = 19200
instrument.serial.bytesize = 8
instrument.serial.parity   = serial.PARITY_NONE
instrument.serial.stopbits = 1
instrument.serial.timeout  = 0.05
instrument.mode = minimalmodbus.MODE_RTU

# ser = serial.Serial("COM3", 19200, serial.EIGHTBITS,
#     serial.PARITY_NONE, serial.STOPBITS_ONE )

# if ser.is_open:
#     print("Serial port is open")
# ser.write(">01?Iv!".encode())

# # read_pressure_data = ser.readline().decode()
# readout = ser.readline().decode()
# print(readout)
# Read temperature (PV = ProcessValue) from position 0
pressure = instrument.read_register(154, 4, functioncode=3)  # Registernumber, number of decimals
print(pressure)