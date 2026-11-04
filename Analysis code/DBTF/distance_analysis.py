# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 14:26:52 2023

@author: Freddie Carlisle
"""

#%% Import the modules
import numpy as np
from cycler import cycler
from numpy.polynomial import polynomial as P
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
from time import sleep


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

def get_file_names(folder_path):
    # List all files in the folder
    file_names = [os.path.basename(f) for f in glob.glob(os.path.join(folder_path, "*"))]
    return file_names

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
    pptx_filename = save_path + fr"\{save}_{filename}.pptx"
    prs.save(pptx_filename)


if __name__ == "__main__":
    nine_dirpath = r"C:\Users\ppxfc1\OneDrive - The University of Nottingham\Kiraly Group\Data\DBTF\Distance_jumps\Distance_experiment_10_6_25\900mV"
    volt_dirpath = r"C:\Users\ppxfc1\OneDrive - The University of Nottingham\Kiraly Group\Data\DBTF\Distance_jumps\Distance_experiment_10_6_25\1V"
    powerpoint_path = r"Analysis code\DBTF\powerpoints"

    nine_or_volt = "1V"  # Change this to "1V" if you want to process the 1V data
    

    if nine_or_volt == "1V":
        dirpath = volt_dirpath
        save = "1V" # Change this to "1V" if you want to save the 1V data
    elif nine_or_volt == "900mV":
        dirpath = nine_dirpath
        save = "900mV"
    else:
        raise ValueError("Invalid option for nine_or_volt. Choose '1V' or '900mV'.")
    distances = get_file_names(dirpath)
    # distances = np.flip(distances)  # Reverse the order of distances if needed
    distance_paths_nine = []
    print(distances)

    for distance in distances:

        filepath = os.path.join(dirpath, distance)
        distance_paths_nine.append(filepath)

    # print(distance_paths_nine)
    
    all_times = []
    means = []
    medians = []
    errors = []
    
    image_path = powerpoint_path + r"\histogram_current.png"
    print("Image path: ", image_path)

    for i, distance_path in enumerate(distance_paths_nine):
        filenames = get_file_names(distance_paths_nine[i])
        # print(filenames)
        
        prs = setup_presentation()
        pptx_filename = powerpoint_path + fr"\{save}_{distances[i]}.pptx"
        if os.path.exists(pptx_filename):
            os.remove(pptx_filename)  # Remove the existing file if it exists
        times = []
        for j, filename in enumerate(filenames):
            
            file_spec = Spectrum(path=distance_path, fileName=filename, channel='X (m)')
            
            current = np.array(file_spec.x)
            X = np.array(file_spec.y)
            Y = np.array(file_spec.give_data("Y (m)")[0])  # Assuming the second column is Y
            index = np.arange(0, len(current),1)

            first_time = np.where(X == X[-1])[0][0]  # Find the first index where X is equal to the last value of X

            time_period = float(file_spec.meta.split('\n')[3].split('\t')[1])

            
            gradient = np.gradient(current)

            threshold = np.mean(current)*0.60
            
            # Find the index where the current is less than the threshold
            threshold_indices = np.where(current < threshold)[0]
            print(len(threshold_indices), " indices below threshold: ", threshold)
            indices_at_position = threshold_indices[threshold_indices >= first_time]
            # time = (len(threshold_indices) * time_period)/1000 #time to find it in seconds - Old method with offset
          
            if len(threshold_indices) == 0:
                print("No indices found at position after first time.")
                continue
            time = ((threshold_indices[-1] - first_time) * time_period)/1000 #time to find it in seconds
            print("Time to find threshold: ", time, " seconds with time period: ", time_period, "ms ", " difference between indices: ", threshold_indices[-1] - first_time)
            times.append(time)
            all_times.append(times)
            
            
            fig, [axI, axHyst, axPos] = plt.subplots(3, 1, figsize=(10, 6), sharex=False)
            fig, axI = plt.subplots(figsize=(10, 6), sharex=False)
            axI.axhline(threshold, color='red', linestyle='--', label='Min 20%')
            axI.axhline(np.mean(current), color='green', linestyle='--', label='Mean Current value')
            axI.plot(threshold_indices, current[threshold_indices], 'ro', label='Threshold Indices')
            axI.axvline(first_time, color='black', linestyle='--', label='Tip back on rail')
            axI.plot(indices_at_position, current[indices_at_position], 'go', label='Indices to time')
            axI.plot(index, current, label='Current')
            axI.set_title("Current vs Index for distance: " + distances[i] + " count " + str(j) + " time: " + str(round(time,4)) + " seconds")
            axI.set_xlabel("Index (ms)")
            axI.set_ylabel("Current (A)")
            axI.legend()

            
            axPos2 = axPos.twinx()  # Create a twin Axes sharing the x-axis
            axPos.scatter(index, X, label='X (m)',c="r", alpha=1)
            axPos2.scatter(index, Y, label='Y (m)',alpha=1)
            axPos.legend()
            



            axHyst.hist(current, bins=20, color='blue', alpha=0.7, edgecolor='black')
            axHyst.set_title("Histogram of Current for distance: " + distances[i] + " count " + str(j))
            axHyst.set_xlabel("Current (A)")
            axHyst.set_ylabel("Frequency")
            # plt.show()
            plt.savefig(image_path)
            plt.close()
            print("Saving distance: ", distances[i], " file: ", j, " Bias: ", save)
            save_to_presentation(prs, powerpoint_path,image_path, filename=distances[i], save=save) 
            sleep(0.5)


        print(times, np.mean(times))
        mean = np.mean(times)
        median = np.median(times)
        error = np.std(times) / np.sqrt(len(times))  # Standard error of the mean
        means.append(mean)
        medians.append(median)
        errors.append(error)
        plt.figure(figsize=(10, 6))
        plt.hist(times, bins=50, color='blue', alpha=0.7, edgecolor='black')
        plt.ylabel("Counts")
        plt.xlabel("Time (s)")
        plt.title("Histogram of Time for distance: " + distances[i] + " avg times: " + str(round(mean,4)) + f" (mean) for counts {len(times)}")
        # plt.show()
        plt.savefig(image_path)
        plt.close()
        save_to_presentation(prs, powerpoint_path,image_path, filename=distances[i], save=save) 
            
        # all_times.append(times)
    plt.close('all')
    fig = plt.figure(figsize=(10, 6))
    # plt.rc('axes', prop_cycle=cycler(color=['blue', 'green', 'red', 'orange', 'purple']))

    plt.hist(all_times, bins=40, alpha=1, label=f'Dataset')
    plt.ylabel("Counts")
    plt.xlabel("Time (s)")
    plt.title("Histogram of Time for all distances: " + str(np.median(means)) + " seconds")
    plt.show()



    mean_fig = plt.figure(figsize=(10, 6))
    # print(means)
    distances = [float(d.replace('m', '')) for d in distances]  # Convert distances to float
    # plt.scatter(distances, means, marker='o',color="b", label='Mean Times')
    plt.errorbar(distances, means, yerr=errors, fmt='o', color='b', label='Mean Times with Error')
    # plt.scatter(distances, medians, marker='o',color="r", label='Median Times')
    plt.xlabel("Distance (nm)",fontsize=18)
    plt.ylabel("Mean Time (s)",fontsize=18)
    # plt.title("Mean Times for each distance")
    plt.savefig(image_path)
    plt.show()
    