# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 13:37:04 2024

@author: physicsuser
"""

"""
Created on Tue Mar 19 09:12:20 2024

@author: ppysa5

The Spectrum class encapsulates 1 spectra file. 
It is a subclass of the spectra opening class (output_data_spectra_dat)
so it inherits attributes found on the file's metadata (eg. xy position). 

If you're looking to perform some analysis on a spectrum, I'd recommend you:
    1. write the analysis that you wish to perform to the spectrum as a 
    separate class (eg. see KPFMSpectrumAnalysis class)
    2. import your individual spectrum analysis here
    3. write a new method that runs your analysis, so that the output is stored
    as a Spectrum attribute (eg. see Spectrum.KPFMAnalysis method)
This way, we'll be able to see, at glance, the different analysis that the
group has written.

"""

from read_spectra import output_data_spectra_dat
from KPFM_spectrum_analysis import KPFMSpectrumAnalysis
import matplotlib.pyplot as plt
import os
import numpy as np


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
    
    
    
    # =========================================================================
    # KPFM analysis
    # =========================================================================
    
    def KPFMAnalysis(self, fit_range=25, xAdatomCentre=None, yAdatomCentre=None, 
                     plotCalculation=False, axFit=None, axResiduals=None, e_min=None, e_max=None, offset=None ):
        """
        From KPFM spectra, we want to calculate the Vcontact value. This 
        involves fitting the spectrum data, df(V), to a parabola y=ax**2+bx+c
        (using the lmfit library). Vcontact is the parabola's minima, -b/(2a).
        
        We can get a sense of the error on Vcontact by propagating the 
        error found for the fitting parameters b and a. Note that this might 
        be an undestimate of the error, as other experimetal variables will 
        play a role, eg. the quality of the qPlus resonance, drift... 
        Nonetheless, it is useful information to have, as it tells us how 
        confident we can be on the Vcontact value based on how well the data 
        fits a parabola. Another way of assessing the fit's quality is by 
        plotting it (setting plotCalculation=True), or by inspecting the fit's
        stats using lmfit.fit_report(self.fitInfo).


        Parameters
        ----------
        xAdatomCentre : float, optional
            DESCRIPTION. The default is None, and r wonr
        yAdatomCentre : float, optional
            DESCRIPTION. The default is None.
        plotCalculation : Bool, optional
            If True plot of the spectrum, its found fit and its corresponding 2
            sigma conf band; the fit's minimum and its correspoding error bar 
            derived by propagating the error on the 
            fitting parameters. The default is False.

        Returns
        -------
        if plotCalculation == True, the matplotlib fig and ax will be returned
        in case we want to add a title etc.

        """
        
        kpfmAnalysis = KPFMSpectrumAnalysis(bias=self.x, df=self.y, fit_range=fit_range)
        self.E_min = e_min
        self.E_max = e_max
        self.vContact = kpfmAnalysis.CalcVContact(E_min=e_min, E_max=e_max)
        
        self.fit = kpfmAnalysis.fit
        self.residuals = kpfmAnalysis.fitInfo.residual
        self.dfAtVContact = kpfmAnalysis.dfAtVContact
        self.vContactErr = kpfmAnalysis.vContactErr
        self.dfAtVContactErr = kpfmAnalysis.dfAtVContactErr
        self.fitInfo = kpfmAnalysis.fitInfo
        self.bias = kpfmAnalysis.bias
        
        if xAdatomCentre != None and yAdatomCentre != None:
            self.r = kpfmAnalysis.CalcR(self.x_pos, self.y_pos, xAdatomCentre, yAdatomCentre)
            
        if plotCalculation == True: 
            axFit, axResiduals, axDataMinusFit = kpfmAnalysis.PlotVContactCalculation(axFit, axResiduals, offset)
            return axFit, axResiduals, axDataMinusFit


#%%
def FindLatestFile(path):

    latestTime = 0
    latestFile = None
    # iterate over the files in the directory using os.scandir
    for f in os.scandir(path):
        if f.is_file() and f.name.endswith('.dat'):
            # get the modification time of the file using entry.stat().st_mtime_ns
            time = f.stat().st_mtime_ns
            if time > latestTime:
                # update the most recent file and its modification time
                latestFile = f.name
                latestTime = time
    return latestFile

#%%
path = r"C:\Users\Fwkca\OneDrive\Desktop\PhD Data\Nikhil visit BP\Spatial 8"

# Get a list of all .dat files in the specified folder
file_list = [f for f in os.listdir(path) if f.endswith('.dat')]

# Iterate over each file in the list

V_contacts = []
V_contact_errs = []
max_residuals = []
max_biases = []
fileNames = []
well_depths = []

biases = []
dfs = []


count = 0

for file_name in file_list:
    if count < 25:
        Dip_start = 0.2
        Dip_end = 1

        fit_range = 30
        Cutoff = True
        offset = 1

        # Create a Spectrum instance for each file
        # example_spectrum = Spectrum(path=path, fileName=file_name, channel='OC M1 Freq. Shift [AVG] (Hz)')
        example_spectrum = Spectrum(path=path, fileName=file_name, channel='OC M1 Freq. Shift (Hz)')
        bias = example_spectrum.x
        df = example_spectrum.y
        # Run the KPFM spectrum analysis
        if offset != None or offset != 0 or Cutoff == True:
            example_spectrum.KPFMAnalysis(fit_range=fit_range,offset = offset,e_min = Dip_start, e_max = Dip_end, plotCalculation=True)
        else:
            example_spectrum.KPFMAnalysis(fit_range=fit_range, plotCalculation=True)

        # Access the analysis results
        vContact = example_spectrum.vContact
        fit = example_spectrum.fit
        dfAtVContact = example_spectrum.dfAtVContact
        vContactErr = example_spectrum.vContactErr
        dfAtVContactErr = example_spectrum.dfAtVContactErr
        residuals = example_spectrum.fitInfo.residual
        bias = example_spectrum.bias

        dfs.append(df)
        biases.append(bias)

        # Find the maximum residual and its corresponding bias value past 0.1V


        mask = bias > 0.1
        # print(np.any(mask))
        # print(len(bias),len(bias[mask]), len(residuals),len(residuals[mask]))

        max_residual = max(residuals[mask])
        max_bias = bias[mask]
        max_bias = max_bias[residuals[mask] == max_residual].values[0]

        residual_mean = np.mean(residuals)

        well_depth = max_residual - residual_mean 
        # print("Max residual:", max_residual)


        # print("File - ",file_name, " Bias " , max_bias, " Residual ", max_residual, " Well depth ", well_depth, " residual mean ", residual_mean)


        #
        # Store the analysis results for each file



        fileNames.append(file_name)
        V_contacts.append(vContact)
        V_contact_errs.append(vContactErr)
        max_residuals.append(max(residuals))
        max_biases.append(max_bias)
        well_depths.append(well_depth)
        plt.title(file_name)
        plt.show()


        count += 1
    # Do something with the analysis results for each file
    # ...
print("File names", fileNames)
print("Well positions", max_biases)
# print("Well Depths",well_depths)


# plt.clf()

print(len(biases),len(dfs))

# for i in range(1,4):
    # plt.plot(biases[i],dfs[i], label = file_list[i]) #To offset, add + i/2
plt.xlabel('Bias (V)')
plt.ylabel('Frequency Shift (Hz)')
plt.legend()
plt.show()
# ALL OLD BITS

# exampleSpectrum = Spectrum(path=path, fileName=fileName,
#                           channel='OC M1 Freq. Shift [AVG] (Hz)')
#                         #   channel='OC M1 Freq. Shift (Hz)')



# # The file's matadata can be accessed eg. the spectra position
# # (See output_data_spectra_dat for more info)
# x_pos = exampleSpectrum.x_pos
# y_pos = exampleSpectrum.y_pos

# # run the KPFM spectrum analysis
# # plt.figure(1)
# # exampleSpectrum.KPFMAnalysis(e_min=Dip_start, e_max=Dip_end, plotCalculation=True)
# # plt.figure(2)
# exampleSpectrum.KPFMAnalysis(plotCalculation=True)
# # the output of the analysis is stored as attributes:
# vContact = exampleSpectrum.vContact
# fit = exampleSpectrum.fit
# dfAtVContact = exampleSpectrum.dfAtVContact
# vContactErr = exampleSpectrum.vContactErr
# dfAtVContactErr = exampleSpectrum.dfAtVContactErr
# residuals = exampleSpectrum.fitInfo.residual
# ... more info is stored within exampleSpectrum.fitInfo object (which is a 
# lmfit ModelResult instance, See lmfitâ€™s ModelResult documentation)
# import lmfit
# print(lmfit.fit_report(exampleSpectrum.fitInfo))
# plt.show()