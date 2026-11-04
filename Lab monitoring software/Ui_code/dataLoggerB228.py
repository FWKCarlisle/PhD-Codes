import time
import serial
import csv
from datetime import datetime
import os

# Serial setup
temp_ser = serial.Serial("COM6", 9600, serial.EIGHTBITS, serial.PARITY_NONE, serial.STOPBITS_ONE)
pres_ser = serial.Serial("COM7", 9600, serial.EIGHTBITS, serial.PARITY_NONE, serial.STOPBITS_ONE)
temp_ser.timeout = 5
pres_ser.timeout = 5

fields = ["timestamp", "STM", "Sorb", "1K", "3He", "Prep", "Exchange"]

def get_filename():
    today = datetime.now().strftime("%Y-%m-%d")
    return f"data_{today}.txt"

def ensure_header(filename):
    if not os.path.exists(filename):
        with open(filename, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(fields)

print("Logging started. Press Ctrl+C to stop.")
try:
    while True:
        filename = get_filename()
        ensure_header(filename)

        timestamp = datetime.now().isoformat()

        temp_ser.write(b'input? d\r\n')
        STM = round(float(temp_ser.readline().decode().strip()), 4)

        temp_ser.write(b'input? a\r\n')
        Sorb = round(float(temp_ser.readline().decode().strip()), 4)

        temp_ser.write(b'input? b\r\n')
        one_k = round(float(temp_ser.readline().decode().strip()), 4)

        temp_ser.write(b'input? c\r\n')
        three_He = round(float(temp_ser.readline().decode().strip()), 4)

        pres_ser.write(b'#000F')
        pres_data = pres_ser.readline().decode().strip()
        prep, exchange, _ = pres_data.split(",")
        prep = float(prep[1:])
        exchange = float(exchange)

        with open(filename, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([timestamp, STM, Sorb, one_k, three_He, prep, exchange])

        time.sleep(5)

except KeyboardInterrupt:
    print("Logging stopped.")