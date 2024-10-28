from Force_curve_testing import calc_force_trapz
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from KPFMFitLatestFile import Spectrum
import os

# Load the data
def convert_data(folder_path,specific_file = None, variable = 'dF'): 
    '''
    folder path: path to the folder containing the data
    variables: list of variables to be extracted from the data
    Allowed variables: "dF", "Amplitude", "Excitation"
    These are from the .dat file structure outputed from nanonis. 
    '''

    file_list = [f for f in os.listdir(folder_path) if f.endswith('.dat')]
    file_list.sort()
    # print(folder_path)
    # print(file_list)

    if specific_file != None:
        file_path = os.path.join(folder_path, specific_file)
        
        if variable == 'dF':
            channel = 'OC M1 Freq. Shift (Hz)'
        elif variable == 'Amplitude':
            channel = 'OC M1 Amplitude (V)'
        elif variable == 'Excitation':
            channel = 'OC M1 Excitation (V)'

        # file_spectrum = Spectrum(path=folder_path, fileName=specific_file, channel=channel)
        
        #open up the file and extract the data
        file_spectrum = Spectrum(path=folder_path, fileName=specific_file, channel='OC M1 Freq. Shift (Hz)')
        # file_spectrum.load_file()
        # x = file_spectrum.x

        return file_spectrum
    

        
            
def main():
    folder_path = r'C:\Users\ppxfc1\OneDrive - The University of Nottingham\Desktop\PhD\Data\Force_measurement_Sofia_28.10\force_measurement_raw'
    file_spectrum = convert_data(folder_path, specific_file='force_in_manipulation_run2_0_1_2_3_4_5_6_7_8_9_10_11_12_001.dat', variable='dF')
    # file_spectrum
    df = file_spectrum.x
    z = file_spectrum.y
    amplitude = 1e-10 #in pm
    k_spring = 2000 #in N/m
    frequency_res = 20000 #in Hz
    df = np.array(df)
    z = np.array(z)
    print("dF")
    print(df)
    print("z")
    print(z)
    forces_trapz = calc_force_trapz(z, df, amplitude, k_spring, frequency_res)
    print("forces_trapz")
    print(forces_trapz)


    print(len(forces_trapz), len(z))
    z = z[0:len(forces_trapz)]
    plt.plot(z, forces_trapz)
    plt.show()


if __name__ == "__main__":
    main()
    

