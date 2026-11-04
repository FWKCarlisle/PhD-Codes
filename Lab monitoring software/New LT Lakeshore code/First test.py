
import serial
import time
bits = serial.EIGHTBITS
parity = serial.PARITY_NONE
stopbits = serial.STOPBITS_ONE


ser = serial.Serial("COM3", 9600, bits, parity, stopbits)
#ser.baudrate = int(9600)
#ser.port = "COM4"
ser.timeout = int(5)
#ser
# ser.open()

for i in range(5):
    ser.reset_input_buffer()
    # time.sleep()
    # ser.write(b'*j01\r\n')
    # time.sleep(0.1)

    ser.write(b'*S0\r\n')
    time.sleep(0.2)

    while ser.in_waiting:
        print(ser.readline())


