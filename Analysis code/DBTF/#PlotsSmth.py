import sys
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
import io
import scipy.signal as spy
from scipy import constants
from scipy.special import erf
from scipy.optimize import curve_fit
import numpy as np
from scipy.special import erfinv
from matplotlib.colors import SymLogNorm

import datetime

from pathlib import Path
import pptx
import pptx.util

from PyQt5 import QtWidgets, QtCore
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

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
        try:
            self.sample_time = float(meta_data.split('Sample Period (ms)\t')[1].split('\n')[0][0:-1])
        except Exception as e:
            print(f"Error extracting sample time: {e}")
            self.sample_time = None  # or set a default value if appropriate
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


def get_metadata(pathname,filename):
    m.get_file(pathname,filename,show_methods=0)
    m.load_file(show_methods=0)
    meta_data = m.meta_data
    Datetime = meta_data[:,1][meta_data[:,0]=='Saved Date'][0]
    date = re.split(r' ',Datetime)[0]
    time = re.split(r' ',Datetime)[1]
    return date,time



def get_time_array(pathname,whichdata,howfar,timeaway):
    #list all files of a certain dataset and timeaway from molecule
    all_files = os.listdir(pathname)
    matching_files = [f for f in all_files if f.startswith('Dataset_'+str(whichdata)+'_'+str(howfar)+'_'+str(timeaway)) and f.endswith('.dat')]
    #print(matching_files)
    timedate_array=[]
    for i in range(0,100):
        try:
            numfiles = str('00047')
            
            filenum = str(int(i))
            filename0 = 'Dataset_'+str(whichdata)+'_'+str(howfar)+'_'+str(timeaway)+'_'+str(filenum)+'_00001.dat'
            filename1 = 'Dataset_'+str(whichdata)+'_'+str(howfar)+'_'+str(timeaway)+'_'+str(filenum)+'_'+str(numfiles)+'.dat'
            
            date0,time0 = get_metadata(pathname,filename0)
            date1,time1 = get_metadata(pathname,filename1)
            
            timedate_array.append([date0,date1,time0,time1])
            print(filename0)
        except Exception as e:
            print(f"Ended at iteration {i}: {e}")
            break
        
    return timedate_array



def timeshift(time):
    timeparts = time.split(':')
    timeparts[0] = str(int(timeparts[0])+24)
    newtime = ':'.join(timeparts)
    return newtime


def time_to_seconds(t):
    h,m,s = map(int,t.split(':'))
    return h*3600 + m*60 + s
    
    
    
def import_temps(pathname):
    all_files = os.listdir(pathname)
    matching_files = [f for f in all_files if f.startswith('temp_data_2025') and f.endswith('.txt')]
    #print(matching_files)
    temparray=[]
    for fname in matching_files:
        
    
        with open(str(pathname) + '/' + str(fname), "r") as f:
            content = f.read().splitlines()
        
        tempdata = np.array([re.split(r"[,T]", line) for line in content])
        
        
        for i in range(1, len(tempdata[:,0])):
            # time handling
            dataparts = tempdata[i][1].split(':')
            dataparts[-1] = round(float(dataparts[-1]))
            dataparts[-1] = "{:02}".format(dataparts[-1])
            tempdata[i][1] = ':'.join(dataparts)
        
            # date handling
            dt = datetime.strptime(tempdata[i][0], "%Y-%m-%d")
            tempdata[i][0] = dt.strftime("%d.%m.%Y")
            
        tempdata = np.delete(tempdata,0,axis=0)
        tempdata = np.delete(tempdata,[3,4,5],axis=1)
            
        temparray.append(tempdata)
        
    #now sort by date in ascending order using the first date entry
    temparray = sorted(temparray,key=lambda a: datetime.strptime(a[0, 0], "%d.%m.%Y"))    
    
    
    return temparray





def time_to_temp(timeval,temparr):
    temparrchop = list(temparr[:,1])
    temparrchop_seconds = np.array([time_to_seconds(t) for t in temparrchop])
    timeval_seconds = time_to_seconds(timeval)
    idx = np.abs(temparrchop_seconds-timeval_seconds).argmin()
    return temparr[idx][2]



def get_data_array(timedates,temparray):
    data_array = []
    
    startdate = timedates[0][0]
    
    #look through first entry in temparray[i] and find one that matches date to data (startdate)
    for i in range(len(temparray)):
        if temparray[i][0][0] == startdate:
            dayind = i #find dayindex to use to call the right day for temperature array. use i+1 later if we need
    temps0 = temparray[dayind]
    temps1 = temparray[dayind]
    for i in range(len(timedates)):
        date0 = timedates[i][0]
        date1 = timedates[i][1]
        time0 = timedates[i][2]
        time1 = timedates[i][3]
        
        # if the time spreads over to the next day, change temparray to next one
        if date0 != startdate:
            #time0 = timeshift(time0)
            temps0 = temparray[dayind+1]
            
        if date1 != startdate:
            #time1 = timeshift(time1)
            temps1 = temparray[dayind+1]
            
        #now find closest match from time array to temp array 
        T0 = time_to_temp(time0,temps0)
        T1 = time_to_temp(time1,temps1)
        
        data_array.append([float(T0),float(T1)])
            
    return data_array



def generate_raildata(tempspath, countspath, timepath, pos,datasetnum,time):
    
    timedates = get_time_array(timepath,datasetnum,'10nm',time)
    temparray = import_temps(str(tempspath))
    data_array = np.array(get_data_array(timedates,temparray))
    
    # counts = import_counts(str(countspath) , str(pos)+'_dataset'+str(datasetnum)+'_'+str(time)+'s.csv')

    # countsarr = np.full(len(timedates),np.nan)
    # countsarr[counts[:, 0].astype(int)] = counts[:, 1]

    xdata = np.mean(data_array,axis=1)
    #ydata = countsarr
    xerr = np.abs(np.mean(data_array,axis=1)- np.array([data_array[:,0],data_array[:,1]]))
    #print(xerr)

    return np.array([xdata,xerr[0]])

import numpy as np
import matplotlib.pyplot as plt

def plot_didv_map(bias_arrays, didv_arrays):
    """
    Plot multiple dI/dV spectra as a 2D map.

    Parameters
    ----------
    bias_arrays : list of array-like
        Bias values for each spectrum.
    didv_arrays : list of array-like
        dI/dV values corresponding to each bias array.

    Each element of bias_arrays/didv_arrays is one spectrum.
    """

    # Assume all spectra have the same bias axis
    bias = np.asarray(bias_arrays[0])*1000
    didv = np.asarray(didv_arrays)*1e12

    plt.figure(figsize=(8, 5))

    plt.imshow(
        didv,
        aspect="auto",
        origin="lower",
        extent=[bias.min(), bias.max(), 0, len(didv) - 1],
        cmap="inferno",
    )

    plt.xlabel("Bias (mV)")
    plt.ylabel("Array #")
    plt.colorbar(label="dI/dV (pA)")
    plt.tight_layout()
    plt.show()

def flatten_spectrum(x, y):
    """Remove the linear background between the first and last points."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    background = np.interp(x, [x[0], x[-1]], [y[0], y[-1]])
    return y - background

def plot_waterfalls():
    data_dir = Path(r"C:\Users\ppxfc1\OneDrive - The University of Nottingham\Desktop\PhD\MnTe\Line 4 -  80mT - 1648 - 1847")

    files = sorted(data_dir.glob("*.dat"))

    dIdVArr = []
    dIdVarr_raw = []
    biasArr = []

    for file_path in files:  # limit for testing; remove [:50] to use all
        spec = Spectrum(str(data_dir), fileName=file_path.name, channel="LI Demod 1 Y (A)")
        dIdV = np.asarray(spec.give_data("LI Demod 1 Y (A)")[0])
        bias = np.asarray(spec.give_data("Bias (V)")[0])
        dIdVarr_raw.append(dIdV- np.mean(dIdV))  # store raw data for comparison
        dIdVArr.append(flatten_spectrum(bias, dIdV))  # baseline correction using the mean and gradient of the first point
        biasArr.append(bias)

    # for i in range(len(dIdVArr)):
    #     plt.plot(biasArr[i]*1000, dIdVArr[i]*1e12, label=f"Spectrum {i+1}")
    #     plt.plot(biasArr[i]*1000, dIdVarr_raw[i]*1e12, color='k', linestyle='--', label='Raw Data')
    #     plt.hlines(y=np.mean(dIdVArr[i]*1e12), xmin=biasArr[i].min()*1000, xmax=biasArr[i].max()*1000, color='r', linestyle='--', label=f'Mean {i+1}')
    #     plt.show()
    plot_didv_map(biasArr, dIdVArr)

def plot_barriers():
    old = [3.90,3.96,3.36,5.09,4.44,5.58,8.34,None,9.24,5.51,7.62,9.45,11.65,8.61,4.91,8.24,16.34,4.43,7.94,13.61,8.19,6.91,3.71,5.21,13.32,12.17,1.29,5.08,5.03,8.47,4.21,15.50]
    new_35 = [6.20,6.09,6.24,6.22,6.07,6.16,7.5,8.8,7.99,8.06,7.44,7.58,6.62,6.62,6.56,6.46,4.92,5.30,7.70,7.97,7.38,7.46,6.81,6.63,9.21,9.18,6.87,7.16,8.27,8.11,5.85,4.86]
    new_25 = [4.10,4.27,4.10,4.20,4.03,4.19,4.89,6.02,5.24,5.51,4.89,5.09,4.25,4.46,4.34,4.38,3.20,3.68,5.04,5.39,4.86,5.09,4.50,4.49,6.03,6.22,4.42,4.91,5.38,5.50,3.85,3.25]
    new_45 = [8.30,7.90,8.37,8.25,8.11,8.14,10.10,11.59,10.74,10.61,9.99,10.08,8.98,8.78,8.78,8.54,6.63,6.92,10.37,10.56,9.90,9.84,9.11,8.76,12.39,12.14,9.32,9.42,11.16,10.72,7.85,6.48]
    new_15 = [2.01,2.45,1.97,2.18,1.99,2.22,2.29,3.24,2.50,2.95,2.35,2.59,1.88,2.29,2.12,2.31,1.49,2.06,2.37,2.81,2.33,2.71,2.19,2.35,2.85,3.27,1.96,2.65,2.49,2.89,1.86,1.60]
    new_55 = [10.39,9.72,10.51,10.27,10.15,10.11,12.70,14.37,13.48,13.16,12.53,12.57,11.35,10.95,11.00,10.61,8.35,8.54,13.03,13.14,12.43,12.22,11.42,10.90,15.57,15.10,11.78,11.67,14.04,13.33,9.85,8.11]
    plt.plot(np.arange(0,16,1),np.arange(0,16,1),label='Ideal', color = "grey", linestyle='-.')
    plt.scatter(old, new_35, label='New (35)')
    plt.scatter(old, new_25, label='New (25)')
    plt.scatter(old, new_45, label='New (45)')
    plt.scatter(old, new_15, label='New (15)')
    plt.scatter(old, new_55, label='New (55)')
    plt.xlabel('Old barrier calculation (meV)')
    plt.ylabel('New barrier calculation (meV)')
    plt.legend()
    plt.show()
def sim_plot(Del_E, ax=None, arrehnius=True):

    Ea_meV = Del_E         # Activation barrier (meV)
    T_min = 1.5           # Minimum temperature (K)
    T_max = 300         # Maximum temperature (K)
    num_points = 2000     # Number of temperature points

    # Physical constants
    kB = 8.617333262e-5   # Boltzmann constant (eV/K)
    h = 4.135667696e-15   # Planck constant (eV.s)

    # Convert activation energy from meV to eV
    Ea = Ea_meV / 1000

    # ==========================================
    # 2. TEMPERATURE RANGE
    # ==========================================

    T = np.linspace(T_min, T_max, num_points)

    # Reciprocal temperature
    inv_T = 1 / T

    # ==========================================
    # 3. EYRING EQUATION
    # ==========================================

    # Eyring hopping rate
    rate_eyring = (kB * T / h) * np.exp(-Ea / (kB * T))

    # Natural logarithm of hopping rate
    ln_rate_eyring = np.log(rate_eyring/T)

    # ==========================================
    # 4. ARRHENIUS EQUATION
    # ==========================================

    # Arrhenius prefactor
    rate_0 = kB * T_max / h

    rate_arrhenius = rate_0 * np.exp(-Ea / (kB * T))

    ln_rate_arrhenius = np.log(rate_arrhenius)

    # ==========================================
    # 5. PLOT RESULTS
    # ==========================================

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    # Main Eyring plot
    ax.plot(inv_T, ln_rate_eyring,
            label=f"Eyring equation (Ea = {Ea_meV} meV)", linewidth=2)
    if arrehnius:
        ax.plot(inv_T, ln_rate_arrhenius,
                '--', label=f"Arrhenius equation (Ea = {Ea_meV} meV)", linewidth=2)

    ax.set_xlabel("1/T (K$^{-1}$)", fontsize=13)
    ax.set_ylabel("ln(Hopping rate/Temp / s$^{-1}$K$^{-1}$)", fontsize=13)

    ax.set_title(
        f"Eyring plot: Activation barrier = {Ea_meV} meV",
        fontsize=14
    )

    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # ==========================================
    # SECONDARY TOP AXIS: TEMPERATURE (K)
    # ==========================================

    # Convert 1/T to T
    def inverse_temp(x):
        return 1 / x

    # Convert T back to 1/T
    def reciprocal_temp(T):
        return 1 / T

    ax_top = ax.secondary_xaxis(
        'top',
        functions=(inverse_temp, reciprocal_temp)
    )

    ax_top.set_xlabel("Temperature (K)", fontsize=13)

    # Choose realistic temperature tick marks
    temp_ticks = [1.5, 2, 3, 4, 5, 6, 8, 10, 50, 300]

    ax_top.set_xticks(temp_ticks)
    ax_top.set_xticklabels([str(t) for t in temp_ticks])

    ax_top.tick_params(axis='x', labelsize=10)

    plt.tight_layout()
    # plt.show()

    # ==========================================
    # 6. PRINT SOME RESULTS
    # ==========================================

    print("Activation barrier:", Ea_meV, "meV")
    print("\nTemperature (K) | Hopping rate (s^-1)")

    for temp in [1.5, 2, 3, 4, 5, 7, 10, 50, 300]:
        rate = (kB * temp / h) * np.exp(-Ea / (kB * temp))
        print(f"{temp:10.2f}     | {rate:.4e}")




def plot_arrays():
    dir = r"D:"
    file = r"telegraph00001.dat"

    spec = Spectrum(dir, fileName=file, channel="Bias (V)")
    # bias = spec.x
    z = spec.give_data('Z (m)')[0]
    time = np.arange(len(z)) * spec.sample_time * 1e-3  # Convert to seconds
    plt.plot(time, z)
    plt.ylabel('Lateral Force (pN)')
    plt.xlabel('Time (s)')

    plt.show()

    return 1


def plot_diffusion():
    dir = r"C:\Users\ppxfc1\OneDrive - The University of Nottingham\Desktop\PhD\DBTF\November data\Dataset11\0.1\Test Data"
    files = [f.name for f in Path(dir).iterdir() if f.is_file() and f.name.endswith('.dat')]
    files = sorted(files, key=lambda x: int(x.split('_')[4]))
    print(files)

    for i, file in enumerate(files):
        if i > 1:  # Skip the first two files
            spec = Spectrum(dir, fileName=file, channel="Z (m)")
            z = spec.give_data('Z (m)')[0] * 1e12
            z = z - np.mean(z[:50])  # baseline correction using the mean of the first 10 points
            time = np.arange(len(z)) * spec.sample_time * 1e-3  # Convert to seconds
            plt.plot(time, z, label=file)


    plt.xlabel('Time (s)')
    plt.ylabel('Z (pm)')
    plt.xlim([0, 6.5])
    plt.ylim([-400, 100])
    plt.show()


def plot_datafiles():
    dir = r"D:\Nanoscience data\DBTF\Diffusion\Data"
    files = ["Dataset_13_test_15nm_0.1_0_00001.dat","Dataset_13_test_15nm_0.1_0_00004.dat","Dataset_13_test_15nm_0.1_0_00007.dat"]
    labels = ["First Run", "Repeat 4 ", "Repeat 7"]
    list = None

    fig, [ax, diffAx] = plt.subplots(2, 1, figsize=(8, 10))


    Zs = []
    Times = []

    for i, file in enumerate(files):

        spec = Spectrum(dir, fileName=file, channel="Z (m)")
        current = spec.give_data('Current (A)')[0]
        bias = spec.give_data('Bias (V)')
        Z = spec.y
        sample_period = spec.sample_time
        print(current)
            # Convert to ms

        # x -= np.mean(x)  # baseline correction using the mean of the first 10 points
        try:
            label = labels[i]
        except:
            label = file

        grad = np.gradient(Z)
        Max = np.argmax(grad)
        Min = np.argmin(grad)

        if i == 0:
            offset = Max
        diff = Max - offset

        Z = Z[diff:] * 1e9
        current = current[diff:] * 1e12
        time = np.arange(len(Z)) * sample_period * 1e-3 

        ax.plot(time, Z, label=label)
        diffAx.plot(time, current, label=label )
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Z (nm)')
    ax.set_xlim([2.5, 5.5])
    ax.set_ylim([6.8, 7.4])
    ax.legend()
    diffAx.set_xlabel('Time (s)')
    diffAx.set_ylabel('Current (pA)')
    diffAx.set_xlim([2.5, 5.5])
    diffAx.set_ylim([-50, 160])
    diffAx.legend()
    plt.show()

    return 1

def plot_force():

    LForce_Ag     = [3.9 , 2.445, 0.845, 0.06 , 0.025,0.01,2.75, 2.14, 0.695, 1.495, 2.835, 0.88, 0.07, 0.09, 0.01]
    LForce_Ag_err = [0.08, 0.125, 0.125, 0.10 , 0.10 ,0.1 ,0.14, 0.26, 0.1  , 0.1  , 0.1  , 0.12, 0.1 , 0.1 , 0.10]
    Bias_Ag       = [0   , -0.05, -0.25, -0.50,-0.75 ,-1.0, 0.0,0.125, 0.25 , 0.0  , 0.002, 0.25, 0.50, 0.75, 1.00]

    LForce_Cu       = [0.3 , 1.1, 0.07 , 1.04,0.575,0.001, 0.01, 0.001,1.225 ]
    LForce_Cu_err   = [0.1 , 0.385, 0.1  ,  0.1,0.145,0.1  , 0.1 , 0.1  ,0.205  ]
    Bias_Cu         = [0.25, -0.05, 0.350, 0.15,0.05 ,-0.15,-0.25, -0.35,0.0    ]

    LForce_Pb     = [0.61 , 0.29, 0.83, 0.9   ,1.995, 0.715 ]
    LForce_Pb_err = [0.1  , 0.1 , 0.1 , 0.1   ,0.1  , 0.1   ]
    Bias_Pb       = [0.005, 0.05, 0.05, 0.0005,0.1  , 0.5   ]

    Potential_Ag  = [14.57, 10.26, 2.62, 0, 0,0, 11.13, 8.56, 2.85, 5.37, 13.29, 4.22, 0.2,0.31,0.01]
    Potential_Ag_err = [0.5,0.2,0.2,0.2,0.1,0.05,0.5,0.5,0.5,0.4,0.3,0.3,0.05,0.1,0.05]
    


    fig, [ax, PAx] = plt.subplots(2,1, figsize=(5, 10))
    # plt.errorbar(Bias_Ag, LForce_Ag, yerr=LForce_Ag_err, fmt='o', label='Ag(111)')
    # plt.errorbar(Bias_Cu, LForce_Cu, yerr=LForce_Cu_err, fmt='o', label='Cu(111)')
    # plt.errorbar(Bias_Pb, LForce_Pb, yerr=LForce_Pb_err, fmt='o', label='Pb(111)')
    # PAx = ax.twiny()

    ax.errorbar( LForce_Ag,Bias_Ag, xerr=LForce_Ag_err, fmt='o',ms=7,color="black", capsize=5, label='Lateral Force (pN)')

    PAx.errorbar( Potential_Ag,Bias_Ag, xerr=Potential_Ag_err, fmt='D',color = "#7C03D2", ms=7,capsize=5, label='Potential (meV)')

    ax.relim()
    ax.autoscale_view()

    PAx.relim()
    PAx.autoscale_view()

    # Current limits
    xmin1, xmax1 = ax.get_xlim()
    xmin2, xmax2 = PAx.get_xlim()

    # Fractional position of zero on the force axis
    f = (0 - xmin1) / (xmax1 - xmin1)

    # Adjust ONLY the left limit of the potential axis so that zero aligns
    xmin2 = -f * xmax2 / (1 - f)

    PAx.set_xlim(xmin2, xmax2)

   

    ax.vlines(x=0,ymin=ax.get_ylim()[0], ymax=ax.get_ylim()[1], colors="#7895a4", linestyles='dashed')
    PAx.vlines(x=0,ymin=PAx.get_ylim()[0], ymax=PAx.get_ylim()[1], colors="#7895a4", linestyles='dashed')
    # PAx.vlines(x=0,ymin=PAx.get_ylim()[0], ymax=PAx.get_ylim()[1], colors='red', linestyles='dashed')

    # .errorbar( LForce_Cu,Bias_Cu, xerr=LForce_Cu_err, fmt='o',ms=7, capsize=5, label='Cu(111)')
    # plt.errorbar( LForce_Pb,Bias_Pb, xerr=LForce_Pb_err, fmt='o',ms=7, capsize=5, label='Pb(111)')
    ax.set_xlabel('Lateral Force (pN)')
    ax.set_ylabel('Bias (V)')
    # ax.legend()
    ax.set_yticks([-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0])
    PAx.set_yticks([-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0])


    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    ax.invert_yaxis()

    PAx.set_xlabel('Potential (meV)')
    PAx.set_ylabel('Bias (V)')
    PAx.xaxis.tick_bottom()
    PAx.xaxis.set_label_position('bottom')
    PAx.invert_yaxis()
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = PAx.get_legend_handles_labels()

    # ax.legend(handles1 + handles2, labels1 + labels2, loc='best')
    # PAx.invert_yaxis()


    # PAx.legend(loc='lower right')

    plt.show()



if __name__ == "__main__":
    # main_plot_diffusion()
    # plot_arrays()
    # plot_force()
    # plot_diffusion()
    # plot_barriers()
    fig, ax = plt.subplots(figsize=(10, 6))
    barriers = [1,2,3,4,5,6,7,8]
    for barrier in barriers:
        sim_plot(barrier, ax=ax, arrehnius=False)
    plt.show()



