from Force_curve_testing import calc_force_trapz
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from spectra_set_analysis import SpectraSet
# Load the data

    

        
            
def main():
    folder_path = r'C:\Users\ppxfc1\OneDrive - The University of Nottingham\Desktop\PhD\Data\Force_measurement_Sofia_28.10\force_measurement_raw'
    file_list = [f for f in os.listdir(folder_path) if f.endswith('.dat')]
    file_spectrums = SpectraSet(folder_path, file_list)

    file_name = 'force_in_manipulation_run2_0_1_2_3_4_5_017.dat'
    print(len(file_name))
    spectrum_idx = file_spectrums.fileNames.index(file_name)
    spectrum_file = file_spectrums.spectraData[spectrum_idx]

    
    file_dict = {}
    for file_name in file_spectrums.fileNames:
        length = len(file_name)
        if length not in file_dict:
            file_dict[length] = []
        file_dict[length].append(file_name)
    
    df_set = file_dict.get(46) ## Set this to length of the file name
    test = []
    for file_name in df_set:
        split = file_name.split('_')
        if split[3] == 'run2' and split[10][0] == '0':
            file_index = file_name.split('_')[-1].split('.')[0]
            test.append(file_name)
    print(test)

    spectrum_files_for_z = []
    for i,name in enumerate(test):
        spectrum_files_for_z.append(file_spectrums.spectraData[file_spectrums.fileNames.index(test[i])])
    

    for i, spectrum in enumerate(spectrum_files_for_z):
        z = spectrum.ReadChannel('Z (m)')
        index = spectrum.ReadChannel('Index')
        df = spectrum.ReadChannel('OC M1 Freq. Shift (Hz)')

        plt.scatter(index, df)
        plt.title(test[i])
        plt.show()



    # index = spectrum_file.ReadChannel('Index')
    # df = spectrum_file.ReadChannel('OC M1 Freq. Shift (Hz)')

    # plt.scatter(index, df)
    # plt.show()


    amplitude = 1e-10 #in pm
    k_spring = 2000 #in N/m
    frequency_res = 20000 #in Hz
   
    # forces_trapz = calc_force_trapz(z, df, amplitude, k_spring, frequency_res)
    


    # print(len(forces_trapz), len(z))
    # z = z[0:len(forces_trapz)]
    # plt.plot(z, forces_trapz)
    # plt.show()


if __name__ == "__main__":
    main()
    

