from Force_curve_testing import calc_force_trapz
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from spectra_set_analysis import SpectraSet


def main ():
    
    python_df = []
    python_z = []
    with open(r"C:\Users\ppxfc1\OneDrive - The University of Nottingham\Desktop\PhD\Code\PhD-Codes\seder-jarvis\outputed_mathematica_data\Z-Spectroscopy_BP_00179_z.txt") as f:
        for line in f:
            line = line.strip().split(" ")
            print(line)
            python_z.append(float(line[0]))
            python_df.append(float(line[1]))

    python_df = np.array(python_df)
    python_z = np.array(python_z)
    python_force = calc_force_trapz(python_z, python_df, 1e-10, 2000, 10000)

    python_z = python_z[:len(python_force)]



    matlab_df = []
    matlab_z = []
    with open(r"C:\Users\ppxfc1\OneDrive - The University of Nottingham\Desktop\PhD\Code\PhD-Codes\seder-jarvis\matlab_z_f_output.txt") as f:
        f.readline()
        for line in f:
            line = line.strip().split(" ")
            print(line)
            matlab_z.append(float(line[0]))
            matlab_df.append(float(line[1]))

    matlab_df = np.array(matlab_df)
    matlab_z = np.array(matlab_z)
    python_force = np.array(python_force)

    plt.plot(matlab_z, matlab_df, label="Matlab")   
    plt.plot(python_z*1e9, python_force*1e9, label="Python")
    plt.xlabel("Z (m)")
    plt.ylabel("Force (nN)")
    plt.legend()
    plt.show()
    return 1


main()