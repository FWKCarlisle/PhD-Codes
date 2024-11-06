import minimalmodbus

#COM port settings
COM_port = 'COM3'
COM_baudrate = 19200
COM_timeout = 1

PORT= COM_port
# read_pressure = 


#Set up instrument
instrument = minimalmodbus.Instrument(PORT,1,mode=minimalmodbus.MODE_ASCII)

#Make the settings explicit
instrument.serial.baudrate = int(COM_baudrate)     # Baud
instrument.serial.bytesize = 8
instrument.serial.parity   = minimalmodbus.serial.PARITY_NONE
instrument.serial.stopbits = 1
instrument.serial.timeout  = int(COM_timeout)         # seconds

# Good practice
instrument.close_port_after_each_call = True
instrument.clear_buffers_before_each_transaction = True

# raw_data = instrument.read_registers(0, 2, functioncode=4)  # Read as input registers

# # Convert data to hexadecimal and interpret as little-endian ASCII
# byte_data = (raw_data[0] << 16) + raw_data[1]  # Combine two registers
# ascii_string = byte_data.to_bytes(4, byteorder='little').decode('ascii')

# print("Device ID:", ascii_string)

# import serial

# try:
#     ser = serial.Serial('COM3', baudrate=19200, timeout=1)
#     if ser.is_open:
#         print(f"Successfully opened {ser.port}")
#     ser.close()
# except serial.SerialException as e:
#     print(f"Failed to open the port: {e}")