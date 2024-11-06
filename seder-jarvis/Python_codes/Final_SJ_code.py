import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from spectrum_analysis import Spectrum
import oct2py

def calc_force_trapz (z, df, A, k, f_0, abs_YN = True):
    Omega = df/f_0
    dOmega_dz = np.diff(Omega)/np.diff(z)

    z = z[:-1]
    # Delta_f = Delta_f[:-1]
    Omega = Omega[:-1]
    force = np.zeros(len(z) - 2)

    for j in range(len(z) - 2):
        # start at j+1 due to pole at t=z
        t = z[j+1:]
        
        # adjust length of Omega and dOmega_dz to length of t
        Omega_tmp = Omega[j+1:]
        dOmega_dz_tmp = dOmega_dz[j+1:]

        ### Abs to stop negative values, added 11:05, 29/10/24
        # abs_YN = True
        if abs_YN:

            integral = np.trapezoid((1 + np.sqrt(A) / (8 * np.sqrt(np.pi * abs(t - z[j])))) * Omega_tmp - 
                            A**(3/2) / np.sqrt(2 * abs(t - z[j])) * dOmega_dz_tmp, t)
            
            # correction terms for t=z from [2]
            corr1 = Omega[j] * (z[j+1] - z[j])
            corr2 = 2 * (np.sqrt(A) / (8 * np.sqrt(np.pi))) * Omega[j] * np.sqrt(abs(z[j+1] - z[j]))
            corr3 = (-2) * (A**(3/2) / np.sqrt(2)) * dOmega_dz[j] * np.sqrt(abs(z[j+1] - z[j]))

        else:
            
            inner = (1 + np.sqrt(A) / (8 * np.sqrt(np.pi * (t - z[j])))) * Omega_tmp - A**(3/2) / np.sqrt(2 * (t - z[j])) * dOmega_dz_tmp
        
            integral = np.trapz(inner, t)
            
            # correction terms for t=z from [2]
            corr1 = Omega[j] * (z[j+1] - z[j])
            corr2 = 2 * (np.sqrt(A) / (8 * np.sqrt(np.pi))) * Omega[j] * np.sqrt((z[j+1] - z[j]))
            corr3 = (-2) * (A**(3/2) / np.sqrt(2)) * dOmega_dz[j] * np.sqrt((z[j+1] - z[j]))
        force[j] = 2 * k * (corr1 + corr2 + corr3 + integral)
    return force


def file_to_force(file_path, output_path = None):

    if not os.path.exists(file_path):
        print("File path does not exist")
        raise FileNotFoundError  
    path = os.path.dirname(file_path)
    file_name = os.path.basename(file_path)
    # print(path, file_name)

    spectrum = Spectrum(path, file_name)

    z = np.array(spectrum.ReadChannel('Z rel (m)'))
    index = spectrum.ReadChannel('Index')
    df = np.array(spectrum.ReadChannel('OC M1 Freq. Shift (Hz)'))
    
    amplitude = 100e-12 #1 angstrom Amplitude of the oscillation (m)
    k_spring = 1900 #Spring constant of cantilever (N/m)
    frequency_res = 20000 #Resonant frequency far from surface (Hz)

    force = calc_force_trapz(z, df, amplitude, k_spring, frequency_res)

    with oct2py.Oct2Py() as oc:
        oc.addpath(r"C:\Users\ppxfc1\OneDrive - The University of Nottingham\Desktop\PhD\Code\PhD-Codes\seder-jarvis\Matlab_codes")
        M_f= oc.saderF(z, df, amplitude, k_spring, frequency_res)

        # print("Matlab force: ", M_f)

    
    

    
    z = z[:len(force)] #The method to calculate force returns a force array that has several elements removed due to filtering and method. So set Z length to be the same for ease

    if output_path == None:
        output_path = os.path.join(path, file_name[:-4] + '_force.txt')
    if os.path.exists(output_path):
        with open(output_path, 'w') as f:
            f.write('Z (m) , Force (N) , ' + file_name + '\n' )
            for j in range(len(z)):
                f.write(str(z[j]) + " , "+ str(force[j]) +'\n')
    else:
        with open(output_path, 'x') as f:
            f.write('Z (m) , Force (N) , ' + file_name + '\n' )
            for j in range(len(z)):
                f.write(str(z[j]) + " , "+ str(force[j]) +'\n')
    print("Force calculated and saved to ", output_path)
    return 1



if __name__ == "__main__":
    file_path = r"C:\Users\ppxfc1\OneDrive - The University of Nottingham\Desktop\PhD\Orbital Memory\Nikhil_visit\data\24.08.2024\Z-Spectroscopy_BP_00188.dat"
    output = r"C:\Users\ppxfc1\OneDrive - The University of Nottingham\Desktop\PhD\Orbital Memory\Nikhil_visit\data\24.08.2024\Z-Spectroscopy_BP_00188_z.txt"
    
    
    file_to_force(file_path, output)

    
   