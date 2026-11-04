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

def calc_force_matrix(z, stiffness, A, k, f_0, ppt_params, prs, transpose = False):
    # print(f"len extra {len(ppt_params)}")
    powerpoint_path, number, z_rel = ppt_params
   
    Fzarr=[]
    z = np.array(z) 

    if transpose:
        for i in range(0,len(z)):
            Fzx=[]
            for j in range(5,len(stiffness[0])):
                # Fzx.append(np.trapezoid(kzarr[0:j][:,i],z[0:j]))
                Fzx.append(np.trapezoid(stiffness[i, 0:j], z[0:j]))

                
            Fzarr.append(np.array(Fzx))
            plt.plot(z[5:], Fzx, label=f'Index {i} Simple Force')
    else:
        for i in range(0,len(stiffness[0])):
            Fzx=[]
            for j in range(5,len(z)):
                Fzx.append(np.trapezoid(stiffness[0:j][:,i],-z[0:j]))
                
            Fzarr.append(np.array(Fzx))
            plt.plot(z[5:], Fzx, label=f'Index {i} Simple Force')
    force_matrix = []
    for i, Fzx in enumerate(Fzarr):
        force_matrix.append(Fzx)

    
    plt.xlabel('Z relative (m)')
    plt.ylabel('Force (N)')
    plt.title(f"Force vs Z relative for {number} at index 10")
    plt.savefig(f"{powerpoint_path}\\temp_img.png")
    plt.close()
    save_to_presentation(prs, powerpoint_path, f"{powerpoint_path}\\temp_img.png", filename=number, save=z_rel)

    return force_matrix

def calc_force_SJ(z, z_rel_arr, stiffness, A, k, f_0, ppt_params = [], prs = None):
    # z = np.flip(z)
    dFdz = np.gradient(stiffness, z,)
    force = np.zeros_like(dFdz)

    for j in range(len(z) - 2):

        t = z[j+1:]
        stiff_tmp = stiffness[j+1:]
        dFdz_tmp = dFdz[j+1:]
        # inter = (1 + np.sqrt(A) / (8 * np.sqrt(np.pi * abs(t - z_rel_arr[j])))) * stiff_tmp - A**(3/2) / np.sqrt(2 * abs(t - z_rel_arr[j])) * dFdz_tmp
        inter = (1 + np.sqrt(A) / (8 * np.sqrt(np.pi * abs(t - z[j])))) * stiff_tmp - A**(3/2) / np.sqrt(2 * abs(t - z[j])) * dFdz_tmp
        
        
        integral = np.trapezoid(inter, t)

        
        # corr1 = stiffness[j] * (z[j+1] - z[j])
        # corr2 = 2 * (np.sqrt(A) / (8 * np.sqrt(np.pi))) * stiffness[j] * np.sqrt(abs(z[j+1] - z[j]))
        # corr3 = (-2) * (A**(3/2) / np.sqrt(2)) * dFdz[j] * np.sqrt(abs(z[j+1] - z[j]))

        # force[j] = integral + corr1 + corr2 + corr3
        force[j] = integral 

    return force


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

    move_file = 51 #43 for 16
    for i, file in enumerate(files):
        # Remove the existing file if it exists
        if i > move_file:
            break
        file_spec = Spectrum(path=dir_path, fileName=file, channel='OC M1 Freq. Shift (Hz)')

        current = file_spec.x
        freq_shift = file_spec.y
        z = file_spec.give_data("Z (m)")[0] ## Use this as its a standard height in the middle of measurement
        x = file_spec.give_data("X (m)")[0]
        y = file_spec.give_data("Y (m)")[0]
        z_rel = file_spec.zrel

        indicies =  np.arange(0,len(current))

        data_arrays.append([current, freq_shift, z, indicies, x, y, z_rel])

        plt.plot(np.arange(0,len(current)), current,"o-", label=f'{z[400]*1e9:.2f}m')
        plt.xlabel('Indicies')
        plt.ylabel('Current (A)')
        plt.title(f"Current vs Indicies for {z[400]*1e9:.2f}nm Z relative {file}")
        plt.savefig(f"{powerpoint_path}\\temp_img.png")
        plt.close()
        # print(f"Saving to PowerPoint... {file}_{z_rel*1e9:.2f}nm Z relative")
        # save_to_presentation(prs, powerpoint_path, f"{powerpoint_path}\\temp_img.png", filename=number, save=z_rel)
        
    z_arrs = []
    zs = []
    imdata_c = []
    stiffness_arr = []
    x_arrs = []
    y_arrs = []
    z_rel_arr = []
    is_neg = False
    A = 100e-12
    k = 1800
    f_0 = 23000
    # Iterate through data_arrays to extract current and Z_rel
    for i, data in enumerate(data_arrays):
        current, freq_shift, z, indicies, x, y, z_rel = data

        #cut the last set of points off to get rid of divergence


        background = (np.mean(freq_shift[0:50]) + np.mean(freq_shift[-50:-1]))*0.5
        freq_shift = freq_shift - background
        if min(current) < 0:
            is_neg = True
        
        stiffness = np.array(freq_shift)* 2 * k / f_0
        imdata_c.append(np.array(current))  # Append current arrays
        stiffness_arr.append(stiffness)  # Append freq_shift arrays
        z_arrs.append(z) # List of individual z arrays for each measurement
        zs.append(z[400]) # List of individual z heights for each measurement
        x_arrs.append(data[4])
        y_arrs.append(data[5])
        z_rel_arr.append(z_rel)
        plt.plot(x, stiffness, "-o", label=f'{z_rel*1e9:.2f}nm')
    plt.xlabel('Z (m)')
    plt.ylabel('Stiffness (N/m)')
    plt.title(f"Stiffness vs Z for {number}")
    plt.savefig(f"{powerpoint_path}\\temp_img.png")
    plt.close()
    save_to_presentation(prs, powerpoint_path, f"{powerpoint_path}\\temp_img.png", filename=number, save="All Z rel")


    #Cut the list down to the smallest array size
    min_length = min(len(arr) for arr in imdata_c)
    imdata_c = [arr[:min_length] for arr in imdata_c]
    stiffness_arr = [arr[:min_length] for arr in stiffness_arr]

    imdata_c = np.array(imdata_c)
    stiffness_arr = np.array(stiffness_arr)
    zs = np.array(zs)

    

    
    force_matrix_simple = calc_force_matrix(zs, stiffness_arr, A, k, f_0, (powerpoint_path, number, z_rel), prs)
    force_matrix_simple = np.array(force_matrix_simple)
    Fx_matrix = []
    print(f"Len z_rel_arr: {force_matrix_simple.shape[1]}, X {len(x_arrs[i][:min_length])}, FMS {force_matrix_simple.shape}  ")
    for i in range(force_matrix_simple.shape[1]):
        F_x = []
        for j in range(len(x_arrs[i][:min_length])):
            F_x.append(force_matrix_simple[j][i])
        Fx_matrix.append(F_x)

        F_x = np.array(F_x)
        plt.plot(x_arrs[i][:min_length], F_x , "-",alpha=(i/len(z_rel_arr)), label=f'{z_rel_arr[i]*1e9:.2f}nm')
    plt.title("Force against X (simple)")
    plt.xlabel("x (nm)")
    plt.ylabel("F (nN)")
    plt.savefig(f"{powerpoint_path}\\temp_img.png")
    plt.show()
    plt.close()
    save_to_presentation(prs, powerpoint_path, f"{powerpoint_path}\\temp_img.png", filename=number, save="All Z rel")


    potential = calc_force_matrix(zs, Fx_matrix, A, k, f_0, (powerpoint_path, number, z_rel), prs)
    

    stiffness_arr_t = stiffness_arr.T
    # for i in range(len(stiffness_arr[0])):
    #     for j in range(zs.shape[0]): 
    #         stiffness_arr_t.append(stiffness_arr[j][i])
    
    for stiff in stiffness_arr_t:
        force_sj = calc_force_SJ(zs, z_rel_arr, stiff, A, k, f_0, (powerpoint_path, number, z_rel), prs)
        plt.plot(np.array(zs), np.array(force_sj)*1e9)
    plt.title("Force against Z (SJ)")
    plt.xlabel("Z (nm)")
    plt.ylabel("F (nN)")
    plt.savefig(f"{powerpoint_path}\\temp_img.png")
    # plt.show()
    plt.close()
    save_to_presentation(prs, powerpoint_path, f"{powerpoint_path}\\temp_img.png", filename=number, save=z_rel)


    

if __name__ == "__main__":
    manip_dir = r"D:\DBTF\manips"
    plot_individual_z_curves = False
    dirs = [f for f in os.listdir(manip_dir) if os.path.isdir(os.path.join(manip_dir, f))]
    
    dir_full_analysis(dirs[13])