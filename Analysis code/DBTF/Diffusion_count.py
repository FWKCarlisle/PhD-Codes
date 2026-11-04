
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as spy
from scipy.special import erf
from scipy.optimize import curve_fit
import numpy as np
from scipy.special import erfinv
import sys
import os
import time
import re
import io
import datetime
import pptx
import pptx.util

from pathlib import Path
from PyQt5 import QtWidgets, QtCore
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from PyQt5 import QtWidgets, QtCore

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
        

        meta_split = meta_data.split('\n') # Extracting the Z relative value 
        #Extract position infomation: 
        self.sample_time = float(meta_data.split('Sample Period (ms)\t')[1].split('\n')[0][0:-1])
        # self.y_pos = float(meta_data.split('Y (m)\t')[1].split('\n')[0][0:-1])
        # self.z_pos = float(meta_data.split('Z (m)\t')[1].split('\n')[0][0:-1])
        
        
        #Load as df to make life easy
        self.df = pd.read_csv(io.StringIO(file_data),delimiter = '\t')
        self.meta_split = meta_split
        
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

        return x, y  # fallback will raise if not present
    


def create_datasets(dir, file_numbers=1):
    files = [f.name for f in Path(dir).iterdir() if f.is_file() and f.name.endswith('.dat')]
    files = sorted(files, key=lambda x: int(x.split('_')[4]))
    datasets = []
    number_list = []

    if type(file_numbers) == int:
        files = files[file_numbers-1:file_numbers]
    if type(file_numbers) == list:
        files = [files[i-1] for i in file_numbers]
    for i,file in enumerate(files):
        number = int(file.split('_')[4])
        if number not in number_list:
            number_list.append(number)
            # if single:
                # print('New number found: ' + str(number), ' number list: ' + str(number_list))
            datasets.append([])
        print(len(datasets), len(number_list), number, file)
        datasets[number_list.index(number)].append(Spectrum(dir, file, 'Z (m)'))
    return datasets

def find_times(z, time, current, bias):

    grad_z = np.gradient(z)
    grad_current = np.gradient(current)
    grad_bias = np.gradient(bias)

    sorted_GZ = sorted(enumerate(grad_z), key=lambda x: x[1], reverse=True)
    sorted_GZ_bck = sorted(enumerate(grad_z), key=lambda x: x[1], reverse=False)
    sorted_GC = sorted(enumerate(grad_current), key=lambda x: x[1], reverse=True)
    sorted_GC_bck = sorted(enumerate(grad_current), key=lambda x: x[1], reverse=False)


    bias_ramp_point_D = np.where(bias < 0.055)[0][0]
    bias_ramp_time_D = time[bias_ramp_point_D]

    bias_ramp_point_U = np.where(bias < 0.055)[-1][-1]
    bias_ramp_time_U = time[bias_ramp_point_U]

    retract_point = sorted_GZ[0][0]
    retract_time = time[retract_point]

    return_point = sorted_GZ_bck[0][0]
    return_time = time[return_point]

    feedback_point = sorted_GC[3][0]
    feedback_time = time[feedback_point]

    collect_point = sorted_GC[0][0]
    collect_time = time[collect_point]



    # bias_ramp = 


    # print("Sorted gradients:")
    # print("index Z| value Z| time Z| index current| value current| time current")
    # print(sorted_GZ[0],time[sorted_GZ[0][0]], sorted_GC[0],time[sorted_GC[0][0]])
    # print(sorted_GZ[1], time[sorted_GZ[1][0]], sorted_GC[1], time[sorted_GC[1][0]])
    # print(sorted_GZ[2], time[sorted_GZ[2][0]], sorted_GC[2], time[sorted_GC[2][0]])

    return retract_point, retract_time,  return_point, return_time, feedback_point, feedback_time, collect_point, collect_time, bias_ramp_point_U, bias_ramp_time_U, bias_ramp_point_D, bias_ramp_time_D    



def decide_event(z, current, time, retract_point, return_point, feedback_point, collect_point, bias_ramp_point_U, bias_ramp_time_U, bias_ramp_point_D, bias_ramp_time_D):


    # check if lost DBTF before retracting

    # check if current level after return changes

    #check if Z after feedback is returned is different

    #check if spike after feedback returned

    result = 0

    mean_before_retract = np.mean(z[bias_ramp_point_D:retract_point])
    retract_confidence = 1.001

    mean_after_feedback = np.mean(z[feedback_point:bias_ramp_point_U])
    feedback_confidence = 1.001

    tally = 0

    
    if  z[retract_point-1] < mean_before_retract * retract_confidence:
        print("DBTF lost before retracting")
        result = "Lost"
        return result, [retract_confidence, feedback_confidence, mean_before_retract, mean_after_feedback]

    
    if 1 > 1000:
        print("Current change after return before feedback")

        tally += 1

        # return 0, [retract_check]


    if mean_after_feedback < mean_before_retract * feedback_confidence: 
        print("Z after feedback is returned is different")

        tally += 1


        # return 0, [retract_check]

    if 1> 1000:
        print("There is a spike after feedback returned more than 2s after feedback returned")

        tally += 1

    #    return 0, [retract_check]

    print("Tally: " + str(tally))
    if tally == 0:

        print("DBTF was never lost")

        result = 1
    elif tally == 1:
        print("Tally doesn't equal more than 2, double check conditions")
    else: 
        print("DBTF was lost at some point")

        result = 0

    return result, [retract_confidence, feedback_confidence, mean_before_retract, mean_after_feedback]



    
def plot_things(spectrum, j, shared = None):



    # Normalise Z (mean or whatever)

    # Do gradient of Z 

    # We wanna check if current value is same when we return when its returning... if it is, 1
    #if not check Z when feedback is returned if its 0
    # if Z is not the same, find point in grad(Z) where it spikes after returning if it is 0
    # Measure time
    # if no spike is found, 0 and then invalidate the next one 

    Z = spectrum.y * 1e12
    Z = Z - np.mean(Z[0:10])  # Convert to nanometers and normalise

    current = spectrum.give_data('Current (A)')[0] * 1e12 # Convert to picoamperes
    bias = spectrum.give_data('Bias (V)')[0]

    time = spectrum.sample_time * np.arange(len(Z)) / 1000

    retract_point, retract_time,  return_point, return_time, feedback_point, feedback_time, collect_point, collect_time, bias_ramp_point_U, bias_ramp_time_U, bias_ramp_point_D, bias_ramp_time_D = find_times(Z, time, current, bias)


    result, data = decide_event(Z, current, time, retract_point, return_point, feedback_point, collect_point,  bias_ramp_point_U, bias_ramp_time_U, bias_ramp_point_D, bias_ramp_time_D)

    retract_confidence = data[0]
    feedback_confidence = data[1]
    mean_before_retract = data[2]
    mean_after_feedback = data[3]

    # Z = Z - np.mean(Z[0:10])  # Normalise Z to the mean of the first 10 points
    # current = current - np.mean(current[0:10])  # Normalise current to the mean of the first 10 points
    if shared is None:
        fig, ax = plt.subplots(1, 1, sharex=True)
    else: 
        fig,ax = shared[0]
        if shared[1] == 0:
            previous_time = 0
        else:
            previous_time = shared[1]
            # print(previous_time, retract_time)
            time_diff = (max(retract_time, previous_time) - min(retract_time, previous_time))
            print(time_diff)
            time = time + abs(time_diff)
            left, bottom, width, height = [0.68, 0.15, 0.2, 0.2]
    #         ax2 = fig.add_axes([left, bottom, width, height])
    # # ax2 = ax.twinx()

    #         ax2.plot(time, current, label='Current ' + str(j), color='red')
    #         ax2.set_ylabel('Current (pA)')
    #         ax2.set_xlim(retract_time*0.9, feedback_time*1.1)

            ax.vlines(x=bias_ramp_time_D, ymin = 20, ymax = 35,  color = 'black')
            ax.vlines(x = 0.5, ymin = 20, ymax = 35,  color = 'black' )
            ax.hlines(y = 27.5, xmin = 0.5 , xmax = bias_ramp_time_D, color = 'black')


            ax.vlines(x=bias_ramp_time_U, ymin = 20, ymax = 35,  color = 'black')
            ax.vlines(x=5.625, ymin = 20, ymax = 35,  color = 'black')
            ax.hlines(y = 27.5, xmin = bias_ramp_time_U, xmax = 5.625, color = 'black')
            



    ax.plot(time, Z, label='Spectrum ' + str(j))
    # ax.plot(time[bias_ramp_point_D:retract_point], Z[bias_ramp_point_D:retract_point], label='Mean before retract', color='green')
    # ax.plot(time[feedback_point:bias_ramp_point_U], Z[feedback_point:bias_ramp_point_U], label='Mean after feedback', color='purple')
    ax.set_ylabel(' Z (pm)')

    # ax.set_xlim(-0.1, 6)
    ax.set_ylim(-420, 100)


    
    

    # ax2.plot(time,bias, label='Bias ' + str(j), color='blue')
    # ax2.set_ylabel('Bias (V)')
    if shared is None:
        ax.axvline(x=time[retract_point-1], color='green', linestyle='--', label='Retract time')
        ax.axvline(x=return_time, color='orange', linestyle='--', label='Return time')
        ax.axvline(x=feedback_time, color='purple', linestyle='--', label='Feedback time')
        ax.axvline(x=collect_time, color='blue', linestyle='--', label='Collect time')
        # ax.axvline(x=time[retract_point-150], color='grey', linestyle='--', label='Zero line')

        ax.hlines(y=mean_before_retract, xmin=time[feedback_point],xmax=time[bias_ramp_point_U], color='black', linestyle='--')
        ax.fill_between(x=time[feedback_point:bias_ramp_point_U], y1=mean_before_retract *retract_confidence, y2=mean_before_retract * (1 - (retract_confidence - 1)), color='purple', alpha=0.2, label='bias decrease -> retract interva')
        

        ax.hlines(y=mean_before_retract, xmin=time[bias_ramp_point_D], xmax=time[retract_point-1], color='black', linestyle='--')
        ax.fill_between(x=time[bias_ramp_point_D:retract_point-1], y1=mean_before_retract *retract_confidence, y2=mean_before_retract * (1 - (retract_confidence - 1)), color='green', alpha=0.2, label='feedback -> bias increase interval')


        ax.hlines(y=mean_after_feedback,xmin=time[feedback_point],xmax=time[bias_ramp_point_U], color='purple', linestyle='--', label='Mean after feedback')
        ax.hlines(y=mean_before_retract, xmin=time[bias_ramp_point_D], xmax=time[retract_point-1], color='green', linestyle='--', label='Mean before retract')

        # ax.set_ylim(-20, -19)
        # plt.xlim(2, 5)
        # ax.legend()
        ax.set_title(spectrum.fileName)

    return result, retract_time




def main():

    # dir = r"C:\Users\ppxfc1\OneDrive - The University of Nottingham\Desktop\PhD\DBTF\November data\Dataset18\1" #40 is gone 
    dir = r"C:\Users\ppxfc1\OneDrive - The University of Nottingham\Desktop\PhD\DBTF\November data\Dataset11\0.1\28th" #40 is gone
    dir = r"C:\Users\ppxfc1\OneDrive - The University of Nottingham\Desktop\PhD\DBTF\November data\Dataset11\0.1\Test Data"


    numbers = [3,4]
    fig, ax = plt.subplots(1, 1, sharex=True)
    ax.set_prop_cycle(color=['red', 'blue'])


    datasets = create_datasets(dir, file_numbers = numbers)
    print('Number of datasets: ' + str(len(datasets)))
    for i, dataset in enumerate(datasets):
        if i < 5 :
            for j, spectrum in enumerate(dataset):
                if j == 0:
                    result, retract_time = plot_things(spectrum, j, shared=[[fig,ax], 0])
                else:
                    result, retract_time = plot_things(spectrum, j, shared=[[fig,ax],retract_time])
                # plt.ylim(np.min(Z)*1.15, np.max(Z)*1.15)
        plt.show()
    return 1



main()