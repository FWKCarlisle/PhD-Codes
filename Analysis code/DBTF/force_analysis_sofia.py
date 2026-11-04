# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 14:26:52 2023

@author: Freddie Carlisle
"""

#%% Import the modules
import numpy as np
from cycler import cycler
from numpy.polynomial import polynomial as P
from scipy.signal import savgol_filter
from scipy.integrate import trapezoid
import multiprocessing as mp
import re
import io
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import ruptures as rpt
import struct
import datetime
import os
import glob
from pathlib import Path
import pptx
import pptx.util
import oct2py
import time
import math
import matplotlib.cbook as cbook
import matplotlib.colors as colors


class output_data_spectra_dat():
    
    def get_file(self, pathname, filename, show_methods=0):
        #Find get the file name
        self.filename_path = pathname + "\\" + filename
        
        #Now we want to get the data and read it. 
        self.meta = self.load_file(show_methods)
        
        if show_methods == 1:
            self.show_method_fun()
    
    def convert_time_to_epoch(self, time):
        dt = datetime.datetime.strptime(str(time), '%d.%m.%Y %H:%M:%S')
        return dt.timestamp()

    def load_file(self, show_methods):
        #Open the file. The data and headings start after the '[DATA]' section.
        f = open(self.filename_path, "r")
        f.seek(0)
        file_all_data = f.read()
        f.close()
        
        #Split at the '[DATA]' section. The second in the list is our data with the headers.
        file_data = re.split(r"\[DATA\]\n",file_all_data)[1]
        meta_data = re.split(r"\[DATA\]\n",file_all_data)[0]
        # print(meta_data)
        #Extract the time infomation:
        # self.start_time = self.convert_time_to_epoch(meta_data[meta_data.index('Start time')+11:meta_data.index('Start time')+11+19])
        self.start_time = self.convert_time_to_epoch(meta_data[meta_data.index('Saved Date')+11:meta_data.index('Saved Date')+11+19])
        

        meta_split = meta_data.split('\t')
        zrel_line = meta_data.split("\n")[6]
        self.zrel = float(zrel_line.split('\t')[2])  # Extracting the Z relative value 
        #Extract position infomation: 
        # self.x_pos = float(file_data.split('X (m)\t')[1].split('\n')[0][0:-1])
        # self.y_pos = float(file_data.split('Y (m)\t')[1].split('\n')[0][0:-1])
        
        
        #Load as df to make life easy
        self.df = pd.read_csv(io.StringIO(file_data),delimiter = '\t')
        
        self.list_of_methods = list(self.df)
        return meta_data
     
    def show_method_fun(self):
        
        print('Possible methods to use are:')
        
        for count, i in enumerate(self.list_of_methods):
            print(str(count) +') '+ i)
        
    def give_data(self, method_number):
        
        #if type(method_number) is int:
            #Do nothing, as this is the expected format
        if type(method_number) is str:
            #Convert to number but use the lookup table
            for count, i in enumerate(self.list_of_methods):
                if method_number in i:
                    method_number = count
                    break
            if method_number is not count:
                #We did not convert to an int value
                raise Exception("This input is not found in the list, check spelling is correct!")
                
        elif type(method_number) is not int: 
            raise Exception("Must be either and int or str variable")
        
        name_to_export = self.list_of_methods[method_number]
        
        exported_data = (self.df[name_to_export],self.list_of_methods[method_number])
        
        return exported_data
    

class Spectrum(output_data_spectra_dat):
    
    def __init__(self, path, fileName, channel):
        """
        Parameters
        ----------
        path : str
            path to spectra file.
        fileName : str
            spectra file name (dat file).
        channel : str
            Channel to read.
            For KPFM it'll be 'OC M1 Freq. Shift [AVG] (Hz)'.
            If the given channel is not found, the available channels will be 
            returned. So, if you're not sure of the exact channel name, just 
            type in non sense.

        Returns
        -------
        Class instance for the specified spectra file, with 2 attributes: the x
        and y data for the specified channel. 
        
        """
        super().__init__()

        self.path = path
        self.fileName = fileName
        self.channel = channel
        
        self.x, self.y = self._OpenSpectra(path=self.path, fileName=self.fileName, channel = self.channel)
        
    
    
    def _OpenSpectra(self, path, fileName, channel):
        """
        Parameters
        ----------
        path : str
            path to spectra file.
        filename : str
            spectra file (dat file).
        channel : str
            Channel to read.
            

        Returns
        -------
        x : 1D array
            spectra x data. For KPFM it'll be bias in Volts.
        y : 1D array
            spectra y data. For KPFM it'll be freq shift in Hz.

        """
        self.get_file(path, fileName)  
        
        # if channel not in file, print list of possible channels
        if channel not in list(self.df): 
            print('Choice of channel not found in ' + fileName)
            self.show_method_fun()
         
        
        x = self.give_data(0)[0] 
        y = self.give_data(channel)[0]

        return x, y 
def setup_presentation():
    prs = pptx.Presentation()
    return prs

def save_to_presentation(prs, save_path,image_path,  filename = "presentation", save= "900mV"):
    slide_layout = prs.slide_layouts[5]
    slide = prs.slides.add_slide(slide_layout)
    
    # Step 4: Add image to the slide
    left = pptx.util.Inches(0.25)   # Adjust positioning
    top = pptx.util.Inches(1.5)
    width = pptx.util.Inches(8)  # Set width (optional)
    height = pptx.util.Inches(6) # Set height (optional)
    # print("Image path: ", image_path)
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    
    slide.shapes.add_picture(image_path, left, top, width, height)

    # Step 5: Save the PowerPoint file
    pptx_filename = save_path + fr"\{filename}.pptx"
    prs.save(pptx_filename)


def magnitude (value):
    if (value == 0): return 0
    return int(math.floor(math.log10(abs(value))))

def clean_matrix_for_plot(matrix,order_of_mag, is_neg=False):

    max_row_length = max(len(row) for row in matrix)
    matrix_for_plot = np.array([np.pad(row, (0, max_row_length - len(row)), constant_values=np.nan) for row in matrix])

    #find the max and min of the matrix
    matrix_for_plot = np.nan_to_num(matrix_for_plot, nan=0.0001e-12)
    matrix_for_plot = matrix_for_plot * (10**(order_of_mag-1))
    if is_neg:
        matrix_for_plot = -matrix_for_plot
        current_matrix_offset = matrix_for_plot.min() 
        matrix_for_plot += abs(current_matrix_offset) + 0.01e-12
    # Plot the heatmap

    return max_row_length, matrix_for_plot

def calc_force_trapz (z, df, A, k, f_0, option = "normal"):
    # print(z[i], df[i], A, k, f_0)
    Omega = df
    
    # Delta_f = df
    # print("df: ",df, " Omega: ", Omega)
    # dOmega_dz = np.diff(Omega)/np.diff(z)
    dOmega_dz = np.gradient(Omega, z)
    df = df[:-1]  # Remove the last element to match the length of z
    z = -z[:-1]
    
    # dOmega_dz = np.flip(dOmega_dz)
    # Delta_f = Delta_f[:-1]
    Omega = Omega[:-1]
    dOmega_dz = dOmega_dz[:-1]
    force = np.zeros(len(z) - 2)

    # print(f"len z: {len(z)}, len df: {len(df)}, len Omega: {len(Omega)}, len dOmega_dz: {len(dOmega_dz)}")
    

    for j in range(len(z) - 2):
        # start at j+1 due to pole at t=z
        t = z[j+1:]
        # adjust length of Omega and dOmega_dz to length of t
        Omega_tmp = Omega[j+1:]
        dOmega_dz_tmp = dOmega_dz[j+1:]
        # print(f"t: {t}, Omega_tmp: {Omega_tmp}, dOmega_dz_tmp: {dOmega_dz_tmp}")
        # print(f"len t: {len(t)}, len Omega_tmp: {len(Omega_tmp)}, len dOmega_dz_tmp: {len(dOmega_dz_tmp)}")
        # print(f"t: {t[j]}, Omega_tmp: {Omega_tmp[j]}, dOmega_dz_tmp: {dOmega_dz_tmp[j]}")
        ### Abs to stop negative values, added 11:05, 29/10/24
        # abs_YN = True
        if option == "normal":
            inter = (1 + np.sqrt(A) / (8 * np.sqrt(np.pi * abs(t - z[j])))) * Omega_tmp - A**(3/2) / np.sqrt(2 * abs(t - z[j])) * dOmega_dz_tmp
            integral = np.trapezoid(inter, t)
            
            # correction terms for t=z from [2]
            corr1 = Omega[j] * (z[j+1] - z[j])
            corr2 = 2 * (np.sqrt(A) / (8 * np.sqrt(np.pi))) * Omega[j] * np.sqrt(abs(z[j+1] - z[j]))
            corr3 = (-2) * (A**(3/2) / np.sqrt(2)) * dOmega_dz[j] * np.sqrt(abs(z[j+1] - z[j]))

            # force[j] = 2 * k * (corr1 + corr2 + corr3 + integral)
            force[j] = 2 * k/f_0 *  (integral)

        elif option == "abs":
            
            inner = (1 + np.sqrt(A) / (8 * np.sqrt(np.pi * (t - z[j])))) * Omega_tmp - A**(3/2) / np.sqrt(2 * (t - z[j])) * dOmega_dz_tmp
            integral = np.trapezoid(inner, t)
            
            # correction terms for t=z from [2]
            corr1 = Omega[j] * (z[j+1] - z[j])
            corr2 = 2 * (np.sqrt(A) / (8 * np.sqrt(np.pi))) * Omega[j] * np.sqrt((z[j+1] - z[j]))
            corr3 = (-2) * (A**(3/2) / np.sqrt(2)) * dOmega_dz[j] * np.sqrt((z[j+1] - z[j]))

            force[j] = 2 * k * (corr1 + corr2 + corr3 + integral)
        elif option == "simple":

            df_tmp = df[j+1:]
            z_tmp = z[j+1:]
            # print(len(df_tmp), len(z_tmp))
            integrand = (2 * k / f_0) * df_tmp
            force[j] = np.trapezoid(integrand, z_tmp)
        
        
    return force

def ktscap_from_df(df, k, f0):
    """Rewrite Δf(z) as force gradient."""
    return -2*k*df/f0  # N/m

def feven_deconv(zk, ktscap, A0, sgwin=9, sgdegree=3, tozero=50):
    """
    Sader–Jarvis inversion (converted from Philipp Rahe's Matlab Sader-Jarvis code, Ver 8, 2020) to recover conservative
    tip–sample force from frequency-gradient data.

    Parameters
    ----------
    zk : array
        z positions (m)
    ktscap : array
        Force gradient k_ts^cap(z) [N/m], from Δf via FM-AFM.
    A0 : float
        Oscillation amplitude (m).
    sgwin, sgdegree : int
        Savitzky–Golay smoothing window length and polynomial degree.
        sgwin must be odd and > sgdegree for smoothing to be applied.
    tozero : int
        Number of far-z points over which to subtract mean from ktscap.
    """
    zk = np.asarray(zk, dtype=float)
    ktscap = np.asarray(ktscap, dtype=float)
    N = len(zk)

    # --- Shift so far tail averages to zero ---
    if tozero > 0 and tozero < N:
        ktscap = ktscap - np.mean(ktscap[-tozero:])

    # --- Optionally smooth & compute derivative ---
    if (sgwin > 1) and (sgdegree >= 1) and (sgwin > sgdegree) and (sgwin % 2 == 1):
        # Smoothed derivative via Savitzky–Golay filter
        ddfdz = savgol_filter(ktscap, sgwin, sgdegree, deriv=1, delta=np.mean(np.diff(zk)))
    else:
        # Central differences (fallback)
        if sgwin != 0:
            print("Warning: invalid Savitzky–Golay parameters, using central differences.")
        ddfdz = np.zeros_like(ktscap)
        ddfdz[0] = (ktscap[1] - ktscap[0]) / (zk[1] - zk[0])
        ddfdz[1:-1] = (ktscap[2:] - ktscap[:-2]) / (zk[2:] - zk[:-2])
        ddfdz[-1] = (ktscap[-1] - ktscap[-2]) / (zk[-1] - zk[-2])

    # --- Local spacing array (central differences) ---
    ddz = np.zeros_like(zk)
    ddz[0] = np.abs(zk[1] - zk[0])
    ddz[1:-1] = np.abs((zk[2:] - zk[:-2]) / 2.0)
    ddz[-1] = np.abs(zk[-1] - zk[-2])

    Feven = np.zeros(N-2)

    # --- Main loop over z_j ---
    for j in range(N-2):
        # Forward distances from current basepoint
        dz_seg = zk[(j+1):] - zk[j]
        mask = dz_seg < 0  # only forward, positive separation #Changed to backward for testing
        dz_seg = np.abs(dz_seg[mask])
        kts_seg = np.abs(ktscap[(j+1):][mask])
        ddf_seg = np.abs(ddfdz[(j+1):][mask])
        if len(dz_seg) == 0:
            Feven[j] = np.nan
            continue

        # --- MAIN SADER–JARVIS INTEGRAL ---
        # This is the nasty integral in the SJ formula:
        #  F_even(z_j) = -∫_{z_j}^∞ [ (1 + sqrt(A0/(64π(u-z_j)))) ktscap(u)
        #                             - sqrt(A0^3/(2(u-z_j))) d(ktscap)/du ]
        #                 du
        # The kernel has a 1/sqrt(u - z_j) singularity at the lower limit,
        # so u=z_j is excluded here and handled by corr0..corr2 below.
        
        int_term = -trapezoid(
            (1 + np.sqrt(A0/(64*np.pi*dz_seg))) * kts_seg
            - np.sqrt(A0**3/(2*dz_seg)) * ddf_seg,
            dz_seg + zk[j]
        )

        # Corrections for singular part at u = z_j
        corr0 = -ktscap[j] * ddz[j]
        corr1 = -np.sqrt(A0/(16*np.pi)) * ktscap[j] * np.sqrt(ddz[j])
        corr2 =  np.sqrt(4*A0**3/2) * ddfdz[j] * np.sqrt(ddz[j])

        # Combine
        Feven[j] = int_term + corr0 + corr1 + corr2

    zFeven = zk[:N-2]
    return zFeven, Feven

def calc_force_matrix(z, df_matrix, A, k, f_0, ppt_params, prs, transpose = False):
    # print(f"len extra {len(ppt_params)}")
    powerpoint_path, number, z_rel = ppt_params
    k = 1800  # N/m
    f_0 = 23000  # Hz
    Fzarr=[]
    z = np.array(z) * -1
    kzarr = np.array(df_matrix) * (2 * k / f_0)

    if transpose:
        for i in range(0,len(z)):
            Fzx=[]
            for j in range(5,len(kzarr[0])):
                # Fzx.append(np.trapezoid(kzarr[0:j][:,i],z[0:j]))
                Fzx.append(np.trapezoid(kzarr[i, 0:j], z[0:j]))

                
            Fzarr.append(np.array(Fzx))
            plt.plot(z[5:], Fzx, label=f'Index {i} Simple Force')
    else:
        for i in range(0,len(kzarr[0])):
            Fzx=[]
            for j in range(5,len(z)):
                Fzx.append(np.trapezoid(kzarr[0:j][:,i],z[0:j]))
                
            Fzarr.append(np.array(Fzx))
            plt.plot(z[5:], Fzx, label=f'Index {i} Simple Force')
    
    plt.xlabel('Z relative (m)')
    plt.ylabel('Force (N)')
    plt.title(f"Force vs Z relative for {number} at index 10")
    plt.savefig(f"{powerpoint_path}\\temp_img.png")
    plt.close()
    print(f"Saving to PowerPoint heatmap... {number} current")
    save_to_presentation(prs, powerpoint_path, f"{powerpoint_path}\\temp_img.png", filename=number, save=z_rel)

def df_analysis(df, z_rel_arr, sensor_params, ppt_params): #Add prs back in for individual force curves
    amplitude, k_spring, frequency_res = sensor_params
    powerpoint_path, number,  z_rel = ppt_params
    # print("df", df, " z ", z_rel_arr)

    # df = np.flip(df)
    # z_rel_arr = np.flip(z_rel_arr)


    Py_force = calc_force_trapz(np.array(z_rel_arr), np.array(df), amplitude, k_spring, frequency_res, option="normal")
    simple_force = calc_force_trapz(np.array(z_rel_arr), np.array(df), amplitude, k_spring, frequency_res, option="simple")
    no_abs_force = calc_force_trapz(np.array(z_rel_arr), np.array(df), amplitude, k_spring, frequency_res, option="abs")
    ktscap = ktscap_from_df(df, k_spring, frequency_res)
    z_phil, f_phil = feven_deconv(np.array(z_rel_arr), ktscap, amplitude)
    with oct2py.Oct2Py() as oc:
        oc.addpath(r"C:\Users\ppxfc1\OneDrive - The University of Nottingham\Desktop\PhD\Code\PhD-Codes\seder-jarvis\Matlab_codes")
        M_f= oc.saderF(z_rel_arr, df,  frequency_res, k_spring, amplitude)

        # print("Matlab force: ", M_f)

        oc.exit()
        

    z_force = np.array(z_rel_arr[:len(Py_force)]) 
    # print(len(z), len(Py_force), len(M_f))
    # print("Python force, matlab force, difference, Z")
    # for i in range(len(Py_force)):
    #     print(Py_force[i], np.real(M_f[i]), (Py_force[i] - M_f[i]),z_force[i])
    print("Python force, simple force, no_abs_force, phil force, matlab force")
    print(Py_force[-1],simple_force[-1], no_abs_force[-1], f_phil[-1],M_f[-1])
    M_f = M_f.ravel()

    eps   = 1.602e-19   # J  (~1 eV depth). Stick to SI units -- fewer conversion factors to worry about
    sigma = 1e-9        # m  -> well min ~ 1.0 nm

    U_true = 4*eps*((sigma/z_force)**12 - (sigma/z_force)**6)
    F_true = -np.gradient(U_true, z_force)  

    mask = abs(F_true) < 100 
    F_true = F_true[mask]
    

    difference = Py_force - np.real(M_f)
    # print(difference)
    # print("Z, Py_force, M_f, difference")
    # for i, value in enumerate(difference):
    #     if i % 10 == 0:
    #         print(z_force[i], Py_force[i], M_f[i], value)
    fig, [dfAx, forceAx, differenceAx] = plt.subplots(3,1, sharex = True)
    
    dfAx.plot(z_rel_arr, df, label = 'Frequency shift')
    dfAx.set_ylabel('Frequency shift (Hz)')
    dfAx.legend()

    forceAx.plot(z_force, np.real(Py_force), label = 'Python force')
    forceAx.plot(z_force, np.real(M_f), label = 'Matlab force')
    forceAx.plot(z_force, simple_force, label = 'Simple force')
    forceAx.plot(z_phil, f_phil, label = 'Phil force', linestyle='-.')
    forceAx.set_ylabel('Force (N)')
    forceAx.legend()

    differenceAx.plot(z_force[mask], F_true, label = 'True force LJ', linestyle='--')
    differenceAx.set_ylabel('force (nN)')
    differenceAx.legend()

    plt.xlabel('Z (m)')
    # plt.show()
    # plt.close()   # plt.show()

    plt.savefig(f"{powerpoint_path}\\temp_img.png")
    plt.close()
    # save_to_presentation(prs, powerpoint_path, f"{powerpoint_path}\\temp_img.png", filename=number, save=z_rel) #Add prs back in for individual force curves. 
        
    return [Py_force, M_f, simple_force, no_abs_force, f_phil], [z_force, z_phil]

def worker(df, z_rel_arr, params, extras):
    """Wrapper function to call df_analysis for one df."""
    forces, zs = df_analysis(df, z_rel_arr, params, extras)
    return forces, zs#

def parallel_analysis(fs_t, z_rel_arr, params, extras):
    prs = extras[2]
    extras = extras[:2] + extras[3:] #Remove prs for parallel analysis
        
    calc_force_matrix(z_rel_arr, fs_t, params[0], params[1], params[2], extras, prs=prs) #Calculate the force matrix once
        
    tasks = [(df, z_rel_arr, params, extras) for df in fs_t]
    
    with mp.Pool() as pool:
        results = pool.starmap(worker, tasks)

    return results

def serial_analysis(fs_t, z_rel_arr, params, extras):
    results = []
    prs = extras[2] #Remove prs for serial analysis
    calc_force_matrix(z_rel_arr, fs_t, params[0], params[1], params[2], extras, prs=prs)

    for i, df in enumerate(fs_t):
        print("Calculating force for index:", i, " out of ", len(fs_t))
        #Forces are [Python Force, Matlab Force, Simple Force, No Abs Force]
        
        forces, zs = df_analysis(df, z_rel_arr, params,extras)
        results.append((forces, zs))
    return results

def sj_analysis(fs_t, z_rel_arr, ppt_params):
    powerpoint_path, prs, number, z_rel = ppt_params
    amplitude = 100e-12 #1 angstrom Amplitude of the oscillation (m)
    k_spring = 1800 #Spring constant of cantilever (N/m)
    frequency_res = 23000 #Resonant frequency far from surface (Hz)
    start_time = time.perf_counter()
    
    

    params = [amplitude, k_spring, frequency_res]
    extras = [powerpoint_path, number, prs, z_rel] #Add prs back in for individual force curves

    #decide between parallel and serial analysis

    # force = calc_force_matrix(z_rel_arr, fs_t, amplitude, k_spring, frequency_res)

    results = parallel_analysis(fs_t, z_rel_arr, params, extras)
    # results = serial_analysis(fs_t, z_rel_arr, params, extras)

    # Unpack results
    force_arrays, z_force_arr, heat_matrix, phil_z = [], [], [], []
    for i , (forces, zs) in enumerate(results):
        force_arrays.append(forces)
        z_force_arr.append(zs[0])
        heat_matrix.append(forces[2])
        phil_z.append(zs[1])
        plt.plot(zs[0], forces[2], label=f'Index {i} Simple Force')
        
    finish_time = time.perf_counter()

    print(f"time taken was {finish_time-start_time}")
    # force_arrays = []
    # z_force_arr = []
    # heat_matrix = []
    # phil_z = []
    # for i, df in enumerate(fs_t):
    #     print("Calculating force for index:", i, " out of ", len(fs_t))
    #     #Forces are [Python Force, Matlab Force, Simple Force, No Abs Force]
        
    #     forces, zs = df_analysis(df, z_rel_arr, [amplitude, k_spring, frequency_res], [powerpoint_path, number, z_rel])

    #     force_arrays.append(forces)
    #     z_force_arr.append(zs[0])
    #     heat_matrix.append(forces[2])
    #     phil_z.append(zs[1])
    #     plt.plot(zs[0], forces[2], label=f'Index {i} Simple Force')

    # finish_time = time.perf_counter()

    # print(f"time taken was {finish_time-start_time}")

        
    plt.xlabel('Z relative (nm)')
    plt.ylabel('Force (N)')
    plt.title(f"Simple Force vs Z relative for {number}")
    plt.savefig(f"{powerpoint_path}\\temp_img.png")
    save_to_presentation(prs, powerpoint_path, f"{powerpoint_path}\\temp_img.png", filename=number, save=z_rel)
    plt.close()

    for i, force_array in enumerate(force_arrays):
        plt.plot(z_force_arr[i], force_array[0], label=f'Index {i} Python Force')
    plt.xlabel('Z relative (nm)')
    plt.ylabel('Force (N)')
    plt.title(f"Python Force vs Z relative for {number}")
    plt.savefig(f"{powerpoint_path}\\temp_img.png")
    save_to_presentation(prs, powerpoint_path, f"{powerpoint_path}\\temp_img.png", filename=number, save=z_rel)
    plt.close()

    for i, force_array in enumerate(force_arrays):
        plt.plot(z_force_arr[i], force_array[1], label=f'Index {i} Matlab Force')
    plt.xlabel('Z relative (nm)')
    plt.ylabel('Force (N)')
    plt.title(f"Matlab Force vs Z relative for {number}")
    plt.savefig(f"{powerpoint_path}\\temp_img.png")
    save_to_presentation(prs, powerpoint_path, f"{powerpoint_path}\\temp_img.png", filename=number, save=z_rel)
    plt.close()
    for i, force_array in enumerate(force_arrays):
        plt.plot(phil_z[i], force_array[4], label=f'Index {i} Phil Force')
    plt.xlabel('Z relative (nm)')
    plt.ylabel('Force (N)')
    plt.title(f"Phil Force vs Z relative for {number}")
    plt.savefig(f"{powerpoint_path}\\temp_img.png")
    save_to_presentation(prs, powerpoint_path, f"{powerpoint_path}\\temp_img.png", filename=number, save=z_rel)
    plt.close()
  
    
    
    for i, force_array in enumerate(force_arrays):
        plt.plot(z_force_arr[i], force_array[3], label=f'Index {i} No Abs Force')
        
    
    plt.xlabel('Z relative (nm)')
    plt.ylabel('Force (N)')
    plt.title(f"No Abs Force vs Z relative for {number}")
    plt.savefig(f"{powerpoint_path}\\temp_img.png")
    save_to_presentation(prs, powerpoint_path, f"{powerpoint_path}\\temp_img.png", filename=number, save=z_rel)
    plt.close()
    

    max_row_length_f, force_matrix = clean_matrix_for_plot(heat_matrix, 1, is_neg=False)
    # force_matrix = force_matrix[0, :, :]  # Select the first slice if applicable

    print(f"Max row length for force matrix: {max_row_length_f}")
    print(f"Force matrix shape: {force_matrix.shape}")
    print(f"Z force array length: {len(z_force_arr[i])}") 
    # plt.figure(figsize=(12, 8))
    # plt.pcolormesh(z_rel_arr[3:],np.arange(max_row_length_f),  force_matrix, cmap='viridis', shading='auto')
    # plt.colorbar(label='force (N?)')
    # plt.ylabel('Height ?')
    # plt.xlabel('Z relative (nm)')
    # # plt.show()
    # plt.savefig(f"{powerpoint_path}\\temp_img.png")
    # plt.close()
    # save_to_presentation(prs, powerpoint_path, f"{powerpoint_path}\\temp_img.png", filename=number, save=z_rel)


def dir_full_analysis(dir):

    number = dir
    dir_path = rf"D:\DBTF\manips\{number}"
    powerpoint_path = rf"C:\Users\ppxfc1\OneDrive - The University of Nottingham\Desktop\PhD\Code\PhD-Codes\Analysis code\DBTF\force_powerpoints"

    files = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]
    files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    
    data_arrays = []
    prs = setup_presentation()
    pptx_filename = powerpoint_path + fr"\{number}.pptx"
    if os.path.exists(pptx_filename):
        os.remove(pptx_filename)
    for i, file in enumerate(files):
        # Remove the existing file if it exists
        times = []
        file_spec = Spectrum(path=dir_path, fileName=file, channel='OC M1 Freq. Shift (Hz)')

        current = file_spec.x
        freq_shift = file_spec.y
        z = file_spec.give_data("Z (m)")[0][0]
        z_rel = file_spec.zrel

        indicies =  np.arange(0,len(current))

        data_arrays.append([current, freq_shift, z_rel, indicies])

        plt.plot(np.arange(0,len(current)), current,"o-", label=f'{z*1e9:.2f}m')
        plt.xlabel('Indicies')
        plt.ylabel('Current (A)')
        plt.title(f"Current vs Indicies for {z_rel*1e9:.2f}nm Z relative {file}")
        plt.savefig(f"{powerpoint_path}\\temp_img.png")
        plt.close()
        # print(f"Saving to PowerPoint... {file}_{z_rel*1e9:.2f}nm Z relative")
        save_to_presentation(prs, powerpoint_path, f"{powerpoint_path}\\temp_img.png", filename=number, save=z_rel)
        

    #Plot the current matrix
    z_rel_arr = []
    imdata_c = []
    imdata_f = []
    imdata_dc = []
    imdata_df = []
    is_neg = False
    # Iterate through data_arrays to extract current and Z_rel
    for i, data in enumerate(data_arrays):
        current, freq_shift, z_rel, indicies = data
        if min(current) < 0:
            is_neg = True
        imdata_c.append(np.array(current))  # Append current arrays
        imdata_f.append(np.array(freq_shift))  # Append freq_shift arrays
        imdata_dc.append(np.gradient(current))
        imdata_df.append(np.gradient(freq_shift))

        z_rel_arr.append(z_rel + 200e-12)  # Convert Z_rel to nanometers


    
    
   
    
    max_row_length_c, current_matrix = clean_matrix_for_plot(imdata_c,12, is_neg=is_neg)
    max_row_length_f, freq_shift_matrix = clean_matrix_for_plot(imdata_f,1, is_neg=is_neg)
    max_row_length_dc, current_matrix_dc = clean_matrix_for_plot(imdata_dc,1, is_neg=is_neg)
    max_row_length_df, freq_shift_matrix_df = clean_matrix_for_plot(imdata_df,1, is_neg=is_neg)
    
    calc_force_matrix(z_rel_arr, freq_shift_matrix, 100e-12, 1800, 23000, [powerpoint_path, number, z_rel], prs)
    
    

    fs_t_2 = np.zeros((max_row_length_f, len(freq_shift_matrix)))
    for i, row in enumerate(freq_shift_matrix):
        for j, value in enumerate(row):
            fs_t_2[j,i] = value


    cm_t = np.transpose(current_matrix)
    fs_t = np.transpose(freq_shift_matrix)
    cm_dc_t = np.transpose(current_matrix_dc)
    fs_df_t = np.transpose(freq_shift_matrix_df)

    calc_force_matrix(z_rel_arr, fs_t, 100e-12, 1800, 23000, [powerpoint_path, number, z_rel], prs)
    calc_force_matrix(z_rel_arr, fs_t_2, 100e-12, 1800, 23000, [powerpoint_path, number, z_rel], prs)
    
    


    

    

    plt.figure(figsize=(12, 8))
    plt.pcolormesh(np.arange(max_row_length_c), z_rel_arr, current_matrix,norm=colors.LogNorm(vmin=current_matrix.min(), vmax=current_matrix.max()), cmap='viridis', shading='auto')
    plt.colorbar(label='Current (pA)')
    plt.xlabel('Indices')
    plt.ylabel('Z relative (nm)')
    if is_neg:
        plt.title('2D Heatmap of Current for Each Z_rel (Negative Current, - 5e-13 offset)')
    plt.title(f'2D Heatmap of Current for {number}')
    plt.savefig(f"{powerpoint_path}\\temp_img.png")
    plt.close()
    print(f"Saving to PowerPoint heatmap... {number} current")
    save_to_presentation(prs, powerpoint_path, f"{powerpoint_path}\\temp_img.png", filename=number, save=z_rel)
    
    if plot_individual_z_curves:
        print("Plotting individual current vs Z relative for each index...")
    
        for i, dfZ in enumerate(cm_t):
            plt.plot(z_rel_arr, dfZ, "o-")
            plt.xlabel('Z relative (nm)')
            plt.ylabel('Current (pA)')
            plt.title(f"Current vs Z relative for {number} at index {i}")
            plt.savefig(f"{powerpoint_path}\\temp_img.png")
            plt.close()
            save_to_presentation(prs, powerpoint_path, f"{powerpoint_path}\\temp_img.png", filename=number, save=z_rel)
        

    plt.figure(figsize=(12, 8))
    plt.pcolormesh(np.arange(max_row_length_c), z_rel_arr, freq_shift_matrix, cmap='viridis', shading='auto')
    plt.colorbar(label='Freq Shift (Hz)')
    plt.xlabel('Indices')
    plt.ylabel('Z relative (nm)')
    if is_neg:
        plt.title('2D Heatmap of Freq Shift for Each Z_rel (Negative Freq Shift , - 5e-13 offset)')
    plt.title(f'2D Heatmap of Freq Shift for {number}')
    plt.savefig(f"{powerpoint_path}\\temp_img.png")
    plt.close()
    print(f"Saving to PowerPoint heatmap... {number} freq shift")
    save_to_presentation(prs, powerpoint_path, f"{powerpoint_path}\\temp_img.png", filename=number, save=z_rel)

    if plot_individual_z_curves:
        print("Plotting individual freq shift vs Z relative for each index...")
    
        for i, dfZ in enumerate(fs_t):
            plt.plot(z_rel_arr, dfZ, "o-")
            plt.xlabel('Z relative (nm)')
            plt.ylabel('Freq Shift (Hz)')
            plt.title(f"Freq Shift vs Z relative for {number} at index {i}")
            plt.savefig(f"{powerpoint_path}\\temp_img.png")
            plt.close()
            save_to_presentation(prs, powerpoint_path, f"{powerpoint_path}\\temp_img.png", filename=number, save=z_rel)
    

    plt.figure(figsize=(12, 8))
    plt.pcolormesh(np.arange(max_row_length_c), z_rel_arr, current_matrix_dc,cmap='viridis', shading='auto')
    plt.colorbar(label='Current Gradient (pA/Index)')
    plt.xlabel('Indices')
    plt.ylabel('Z relative (nm)')
    if is_neg:
        plt.title('2D Heatmap of Current Gradient for Each Z_rel (Negative Current Gradient, - 5e-13 offset)')
    plt.title(f'2D Heatmap of Current Gradient for {number}')
    plt.savefig(f"{powerpoint_path}\\temp_img.png")
    plt.close()
    print(f"Saving to PowerPoint heatmap... {number} gradient")
    save_to_presentation(prs, powerpoint_path, f"{powerpoint_path}\\temp_img.png", filename=number, save=z_rel)

    plt.figure(figsize=(12, 8))
    plt.pcolormesh(np.arange(max_row_length_c), z_rel_arr, freq_shift_matrix_df, cmap='viridis', shading='auto')
    plt.colorbar(label='Freq Shift Gradient (Hz/Index)')
    plt.xlabel('Indices')
    plt.ylabel('Z relative (nm)')
    if is_neg:
        plt.title('2D Heatmap of Freq Shift Gradient for Each Z_rel (Negative Freq Shift Gradient, - 5e-13 offset)')
    plt.title(f'2D Heatmap of Freq Shift Gradient for {number}')
    plt.savefig(f"{powerpoint_path}\\temp_img.png")
    plt.close()
    print(f"Saving to PowerPoint heatmap... {number} freq shift gradient")
    save_to_presentation(prs, powerpoint_path, f"{powerpoint_path}\\temp_img.png", filename=number, save=z_rel)

    # plt.figure(figsize=(12, 8))
    # plt.pcolormesh(z_rel_arr,np.arange(max_row_length_c),  current_matrix.transpose(), cmap='viridis', shading='auto')


    sj_analysis(fs_t_2, z_rel_arr, [powerpoint_path, prs, number, z_rel])
    
def dir_sj_analysis(dir):
    number = dir
    dir_path = rf"D:\DBTF\manips\{number}"
    powerpoint_path = rf"C:\Users\ppxfc1\OneDrive - The University of Nottingham\Desktop\PhD\Code\PhD-Codes\Analysis code\DBTF\force_powerpoints"

    files = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]
    files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    
    data_arrays = []
    prs = setup_presentation()
    pptx_filename = powerpoint_path + fr"\{number}.pptx"
    if os.path.exists(pptx_filename):
        os.remove(pptx_filename)
    for i, file in enumerate(files):
        # Remove the existing file if it exists
        times = []
        file_spec = Spectrum(path=dir_path, fileName=file, channel='OC M1 Freq. Shift (Hz)')

        current = file_spec.x
        freq_shift = file_spec.y
        z = file_spec.give_data("Z (m)")[0][0]
        z_rel = file_spec.zrel

        indicies =  np.arange(0,len(current))

        data_arrays.append([current, freq_shift, z_rel, indicies])

    z_rel_arr = []
    imdata_c = []
    imdata_f = []
    imdata_dc = []
    imdata_df = []
    is_neg = False
    # Iterate through data_arrays to extract current and Z_rel
    for i, data in enumerate(data_arrays):
        current, freq_shift, z_rel, indicies = data
        if min(current) < 0:
            is_neg = True
        
        imdata_c.append(np.array(current))  # Append current arrays
        imdata_f.append(freq_shift)  # Append freq_shift arrays
        imdata_dc.append(np.gradient(current))
        imdata_df.append(np.gradient(freq_shift))

        z_rel_arr.append(z_rel)  

    
    print(imdata_f)
    fs_t = np.zeros((max(len(row) for row in imdata_f),len(imdata_f)))
    print(fs_t.shape)
    for i, row in enumerate(imdata_f):
        for j, value in enumerate(row):
            print(i, j, value)
            fs_t[j,i] = value
    print(len(imdata_f), len(imdata_f[0]))
    print(fs_t.shape)

    sj_analysis(fs_t, z_rel_arr, [powerpoint_path, prs, number, z_rel])


    print(dir)

if __name__ == "__main__":
    manip_dir = r"D:\DBTF\manips"
    plot_individual_z_curves = False
    dirs = [f for f in os.listdir(manip_dir) if os.path.isdir(os.path.join(manip_dir, f))]
    
    for dir in dirs:
        dir_full_analysis(dir)
        # dir_sj_analysis(dir)