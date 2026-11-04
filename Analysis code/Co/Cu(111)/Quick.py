import datetime
import numpy as np
import pandas as pd
import io
import re
from scipy.signal import savgol_filter

import matplotlib.pyplot as plt

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
    
def smooth_spectrum(y, window_len=11):

    return savgol_filter(y, window_len, 3)  # window size 11, polynomial order 3


def main():

    file_name = 'Cu(111)_00005.dat'
    file_name = 'Cu(111)_00067.dat'
    # file_name = 'Cu(111)_00050.dat'

    dir_path = rf"Analysis code\Co\Cu(111)"

    file_spec = Spectrum(path=dir_path, fileName=file_name, channel='LI Demod 1 Y (A)')

    bias = np.array(file_spec.x) * 1000 # Convert to mV
    demod = file_spec.y

    fig, [DIDV_ax, IETS_ax, divide_ax] = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

    DIDV_ax.plot(bias, demod, label='Raw dI/dV')
    DIDV_ax.plot(bias, smooth_spectrum(demod), label='Smoothed dI/dV', color='orange')
    DIDV_ax.set_ylabel('dI/dV (arb. units)')
    DIDV_ax.legend()


    Smooth = smooth_spectrum(demod)
    d2IdV2 = np.gradient(Smooth, bias)
    
    IETS_ax.plot(bias, d2IdV2, label='d²I/dV²', color='orange')
    IETS_ax.plot(bias, np.gradient(demod, bias), label='d²I/dV² (non-smoothed)', color='blue', alpha=0.25)
    IETS_ax.set_ylabel('d²I/dV² (arb. units)')
    IETS_ax.axhline(0, color='black', lw=0.5, ls='--')
    IETS_ax.axvline(0, color='black', lw=0.5, ls='--')
    IETS_ax.legend()

    divide_ax.plot(bias, (np.gradient(demod, bias)/demod), label='dI/dV (non-smoothed) / dI/dV', color='blue', alpha=0.25)
    divide_ax.plot(bias, (d2IdV2/Smooth), label='d²I/dV² / dI/dV', color='orange')
    divide_ax.plot(bias, (d2IdV2/demod), label='d²I/dV² / dI/dV (raw)', color='purple')

    
    plt.xlabel('Bias (mV)')
    plt.ylabel('d2I/dV2 (arb. units)')
    plt.title('IETS Spectrum')
    plt.axhline(0, color='black', lw=0.5, ls='--')
    plt.axvline(0, color='black', lw=0.5, ls='--')
    plt.legend() 
    plt.show()
    





    return 1


if __name__ == "__main__":
    main()