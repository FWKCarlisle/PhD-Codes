# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 14:26:52 2023

@author: ppzme
"""

#%% Import the modules
import numpy as np
from numpy.polynomial import polynomial as P
import re
import io
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import struct
import datetime
import os
import glob
from pathlib import Path


class output_data_spectra_dat():
    
    def get_file(self, pathname, filename, show_methods=0):
        
        
        #Find get the file name
        self.filename_path = pathname + "\\" + filename
        
        #Now we want to get the data and read it. 
        self.load_file(show_methods)
        
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
        #Extract position infomation: 
        # self.x_pos = float(file_data.split('X (m)\t')[1].split('\n')[0][0:-1])
        # self.y_pos = float(file_data.split('Y (m)\t')[1].split('\n')[0][0:-1])
        
        
        #Load as df to make life easy
        self.df = pd.read_csv(io.StringIO(file_data),delimiter = '\t')
        
        self.list_of_methods = list(self.df)
     
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
        
        self.x, self.y = self._OpenSpectra(path, fileName, channel)
        
    
    
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
    
def create_map(X,Y, height, label='Height map', scale=1,angle=55):

    map_fig, map_ax = plt.subplots()
    scatter = plt.scatter(X, Y, c=height, cmap='gist_rainbow', s=scale)
    save_path = data_dir + f"\\height_map_{file}_{bias_data[0]:.3f}mV.png"
    map_ax.plot(np.full(100, np.mean(X)), np.linspace(np.min(Y), np.max(Y), 100), 'k--', label='Median X value')
    map_fig.colorbar(scatter, label=label)
    map_ax.set_xlabel('X (m)')
    map_ax.set_ylabel('Y (m)')
    map_ax.set_xlim(left=0)
    map_ax.set_ylim(bottom=0)
    map_ax.set_title('Height map')
    map_fig.savefig(save_path)
    print(f"Saved to: {save_path}")
    # plt.show()
    plt.close(map_fig)
    
    
    # return heatmap

def rotate_map(X, Y, angle):

    angle_radians = np.radians(angle)  # Convert to radians

    # Clockwise rotation matrix
    rotation_matrix = np.array([
        [np.cos(angle_radians), np.sin(angle_radians)],
        [-np.sin(angle_radians), np.cos(angle_radians)]
    ])

    # Stack X and Y data as a 2xN array
    points = np.vstack((X, Y))

    # Apply the rotation matrix
    rotated_points = rotation_matrix @ points

    # Extract rotated X and Y
    x_rotated = rotated_points[0, :]
    y_rotated = rotated_points[1, :]
    return x_rotated, y_rotated

def remove_small_values(data, threshold):
    index_list = []
    for i, datum in enumerate(data):
        if datum < threshold:
            index_list.append(i)
            data[i] = 0
    return data, index_list

def line_profile(x, y, height, label, threshold_percentage = 1.5,):

    median_x = np.median(x) #median to avoid outliers

    lower_bound = median_x * (1-(threshold_percentage/100))  # 1% below the median
    upper_bound = median_x * (1+(threshold_percentage/100))  # 1% above the median


    x_indices_within_range = np.where((x >= lower_bound) & (x <= upper_bound))[0]
    y_line = y[x_indices_within_range]
    height_profile = height[x_indices_within_range]

    indicies = np.argsort(y_line)
    y_line = y_line[indicies]
    height_profile = height_profile[indicies]

    global line_ax

    line_ax.plot(y_line, height_profile, label = label, alpha=0.8)
    
    return y_line, height_profile


def analysis_func(x, y, height, reference_height, label, scale=1, threshold_percentage = 10):
        # % of max value, aritrary threshold, can be changed but this is just to prove the concept
        reference_height_data, index_list = remove_small_values(reference_height, 0)
        
        
        height = np.delete(height, index_list)
        x = np.delete(x, index_list) #Remove out of threshold values
        y = np.delete(y, index_list)

        

        create_map(x, y, height, label, scale)
        y_line, height_profile = line_profile(x, y,height, label=label)
        return y_line, height_profile

        
if __name__ == "__main__":
    data_dir =r"C:\Users\ppxfc1\OneDrive - The University of Nottingham\Desktop\PhD\Data\DBTF_Heat_maps"
    file_names = [f.name for f in Path(data_dir).iterdir() if f.is_file()]
    file_names.sort()

    print(file_names)
    backgrounds_z = []
    backgrounds_current = []
    min_z_list = []
    min_current_list = []
    y_list = []
    height_list = []

    fig, line_ax = plt.subplots()
    for i, file in enumerate(file_names):
        if file[-1] == "t":
            spectrum = Spectrum(data_dir, file, 'Bias (V)')

            current_data, current_channel = spectrum.give_data('Current (A)')
            bias_data, bias_channel = spectrum.give_data('Bias (V)')
            x_data, x_channel = spectrum.give_data('X (m)')
            y_data, y_channel = spectrum.give_data('Y (m)')
            z_data, z_channel = spectrum.give_data('Z (m)')

            min_z = np.min(z_data)
            min_current = np.min(current_data)
            backgrounds_z.append(min_z)
            backgrounds_current.append(min_current)


            z_data = z_data - min_z
            current_data = current_data - min_current
            min_z_list.append(np.min(z_data))
            min_current_list.append(np.min(current_data))
            
            rotated_x, rotated_y = rotate_map(x_data, y_data, 55)

            if i == 0:
                zero_points = np.min(rotated_x) , np.min(rotated_y)
            rotated_x -= zero_points[0]
            rotated_y -= zero_points[1]

            
            y_line, height_profile = analysis_func(rotated_x, rotated_y,z_data, current_data, label=file, scale=1)
            y_list.append(y_line)
            height_list.append(height_profile)


    

line_ax.set_title("Line profile")
line_ax.set_xlabel("Y axis")
line_ax.set_ylabel("Z")
line_ax.legend()  # Add a legend to distinguish plots
# plt.grid(True)
plt.show()