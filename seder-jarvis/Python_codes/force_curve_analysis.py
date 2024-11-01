from Force_curve_testing import calc_force_trapz
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from spectra_set_analysis import SpectraSet
# import oct2py

# Load the data

    

        
            
def main():
    folder_path = r'C:\Users\ppxfc1\OneDrive - The University of Nottingham\Desktop\PhD\Orbital Memory\Nikhil_visit\data\24.08.2024'
    file_list = [f for f in os.listdir(folder_path) if f.endswith('.dat')]
    file_spectrums = SpectraSet(folder_path, file_list)

    amplitude = 1e-10 #in pm
    k_spring = 2000 #in N/m
    frequency_res = 20000 #in Hz
    
    
    file_name = 'Z-Spectroscopy_BP_00184.dat'
    print(len(file_name))
    spectrum_idx = file_spectrums.fileNames.index(file_name)
    spectrum_file = file_spectrums.spectraData[spectrum_idx]

    spectrum_files_for_z = []

    ### FOR SOFIA DATA
    # file_dict = {}
    # for file_name in file_spectrums.fileNames:
    #     length = len(file_name)
    #     if length not in file_dict:
    #         file_dict[length] = []
    #     file_dict[length].append(file_name)
    
    # df_set = file_dict.get(46) ## Set this to length of the file name
    # test = []
    # for file_name in df_set:
    #     split = file_name.split('_')
    #     if split[3] == 'run2' and split[10][0] == '0':
    #         file_index = file_name.split('_')[-1].split('.')[0]
    #         test.append(file_name)
    # print(test)

    # spectrum_files_for_z = []
    # for i,name in enumerate(test):
    #     spectrum_files_for_z.append(file_spectrums.spectraData[file_spectrums.fileNames.index(test[i])])
    
    ### FOR NIKHIL DATA
    spectrum_names = []
    for file_name in file_spectrums.fileNames:
        if 'Z-Spectroscopy_BP' in file_name:
            spectrum_names.append(file_name)
            spectrum_files_for_z.append(file_spectrums.spectraData[file_spectrums.fileNames.index(file_name)])
    # print(spectrum_names)
    for i, spectrum in enumerate(spectrum_files_for_z):
        fig, [dataAx, forceAx] = plt.subplots(nrows=2, ncols=1, sharex=False)
        z = np.array(spectrum.ReadChannel('Z rel (m)'))
        index = spectrum.ReadChannel('Index')
        df = np.array(spectrum.ReadChannel('OC M1 Freq. Shift (Hz)'))

        dataAx.scatter(z, df, s=0.3, label="Raw data")
        # print(spectrum_names[i])
        dataAx.set_title(spectrum_names[i])
        dataAx.set_xlabel('Z (m)')
        dataAx.set_ylabel('Frequency Shift (Hz)')
        dataAx.legend()
        
        # print("Z: ",z)
        # print("dF: ",df)
        #reverse the two arrays
        z = z[::-1]
        df = df[::-1]


        output_file_path = r"C:\Users\ppxfc1\OneDrive - The University of Nottingham\Desktop\PhD\Code\PhD-Codes\seder-jarvis\outputed_mathematica_data" + "\\" + spectrum_names[i][:-4] + '_z.txt'
        with open(output_file_path, 'w') as f:
            # f.write('Z (m) Frequency Shift (Hz) ' + spectrum_names[i] + '\n' )
            for j in range(len(z)):
                f.write(str(z[j]) + " "+ str(df[j]) +'\n')
        print(spectrum_names[i])
        forces_trapz = calc_force_trapz(z, df, amplitude, k_spring, frequency_res, abs_YN=False)


        mathematica_x = []
        mathematica_force = []

        with open(r"C:\Users\ppxfc1\OneDrive - The University of Nottingham\Desktop\PhD\Code\PhD-Codes\seder-jarvis\Force_SJ.csv") as file:
            for i in range(10):
                file.readline()
            for line in file:
                # print(line)
                line = line.split(",")
                mathematica_x.append(float(line[0]))
                mathematica_force.append(float(line[1]))

        # oc = oct2py.Oct2Py()


        # oc.addpath(r"C:\Users\ppxfc1\OneDrive - The University of Nottingham\Desktop\PhD\Code\PhD-Codes\seder-jarvis")  
        # oc.eval('sader_run.m')

        matlab_z = []
        matlab_force = []
        with open(r"C:\Users\ppxfc1\OneDrive - The University of Nottingham\Desktop\PhD\Code\PhD-Codes\seder-jarvis\matlab_z_f_output.txt") as file:
            file.readline()
            for line in file:
                line = line.split()
                matlab_z.append(float(line[0]))
                matlab_force.append(float(line[1]))
        
        matlab_z = np.array(matlab_z)
        matlab_force = np.array(matlab_force)
        mathematica_force = np.array(mathematica_force)
        z = z[0:len(forces_trapz)]


        #Plotting the forces
        forceAx.plot(z, forces_trapz*1e9, label='Python')
        forceAx.plot(mathematica_x, mathematica_force*1e9, label='Mathematica')
        forceAx.plot(matlab_z*1e-9, matlab_force, label='Matlab')

        # forceAx.set_title(spectrum_names[i])
        forceAx.set_xlabel('Z (m)')
        forceAx.set_ylabel('Force (nN)')
        plt.legend()
        plt.show()

if __name__ == "__main__":
    main()
    

