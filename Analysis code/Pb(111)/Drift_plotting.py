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
        

        meta_split = meta_data.split('\t')  # Extracting the Z relative value 
        
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


def extract_z_array(dir_path, file_path):
    file_spec = Spectrum(path=dir_path, fileName=file_path, channel='Z (m)')

    current = file_spec.x
    z = file_spec.y
    return z 

if __name__ == "__main__":
    dir = r"D:"
    files = ['Z_response_monitor_creep_360mK_00013.dat', 'Z_response_monitor_creep_360mK_00012.dat', 'Z_response_monitor_creep_360mK_00011.dat', 'Z_response_monitor_creep_360mK_00010.dat', 'Z_response_monitor_creep_360mK_00009.dat', 'Z_response_monitor_creep_360mK_00008.dat', 'Z_response_monitor_creep_360mK_00007.dat']
    files = np.flip(files)
    # 'Z_response_monitor_creep_360mK_00003.dat','Z_response_monitor_creep_360mK_no_field00001.dat','Z_response_monitor_creep_360mK_00004.dat',Z_response_monitor_creep_360mK_00005.dat', 'Z_response_monitor_creep_360mK_00006.dat' ' , , 'Z_response_monitor_creep_360mK_Bz_200mT00001.dat'
    zs = []
    z_arr = []
    for file in files:
        z = extract_z_array(dir, file)
        
        zs.append([file, z])
        z_arr = z_arr + list(z)

    cut_off = 175000 #43871 for 4,5,6, # 123640 for 7,8,9,10,11,12,13

    time_arr = np.arange(0,len(z_arr)-cut_off) * 1e-3
    plt.plot(time_arr, np.array(z_arr[cut_off:])*1e9)
    plt.xlabel("Time (s)")
    plt.ylabel("Z (nm)")
    plt.title(f" Z drift over time at 360mK")
    plt.show()

        
    
        
