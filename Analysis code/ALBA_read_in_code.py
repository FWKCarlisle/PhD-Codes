from PyMca5.PyMcaIO import specfilewrapper
import matplotlib.pyplot as plt
import numpy as np
# Sample data
filename = "data01.dat"
# Open the SPEC file
sf = specfilewrapper.Specfile("data01.dat")
# Select a scan
scan = sf.select("1")
# Pull out Columns
labels = scan.alllabels()  
# Check all the recorded data 
Pt_No = scan.datacol(1) # Pt_No
energy_mono_corrected = scan.datacol(2) #energy_mono_corrected
ideu71_motor_gap = scan.datacol(3) # ideu71_motor_gap
adc2_timer = scan.datacol(4) #adc2_timer
adc2_i1 = scan.datacol(5) #adc2_i1
adc2_i2 = scan.datacol(6) #adc2_i2
adc2_i3 = scan.datacol(7) #adc2_i3
adc2_i4 = scan.datacol(8) #adc2_i4
timer_mono_ct = scan.datacol(9) #timer_mono_ct
energy_mono_ct = scan.datacol(10) #energy_mono_ct
grpitch_mono_ct = scan.datacol(11) #grpitch_mono_ct
grenc_mono_ct = scan.datacol(12) # grenc_mono_ct
dt = scan.datacol(13) #dt
#plt.plot(energy_mono_ct,adc2_i3)
#Agrees with using PyMca GUI
#%% Metadata
Scan_command = scan.command(0) # Command used to run spectra
Scan_date = scan.date(0) # Date scan was taken on
# Header of individual scans
Scan_header_1 = scan.header("S")
Scan_header_2 = scan.header("U")
Scan_header_3 = scan.header("D")
Scan_header_4 = scan.header("C")
Metadata_labels = scan.header("O")
Metadata_data = scan.header("P")
labels = np.array([])
data = np.array([])
for i in range(0,19):
    arr = Metadata_labels[i].split() 
    arr1 = Metadata_data[i].split()
    arr = [item.strip() for item in arr]  
    arr1 = [item.strip() for item in arr1]  
    labels = np.append(labels,arr)
    data = np.append(data,arr1)

dictionary = dict(zip(labels, data))













