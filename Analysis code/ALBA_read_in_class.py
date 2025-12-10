# -*- coding: utf-8 -*-
"""
Created on Tue Dec  2 12:07:51 2025

@author: Wstev
"""
from PyMca5.PyMcaIO import specfilewrapper
import matplotlib.pyplot as plt
import numpy as np


class Spectra_data:
    def __init__(self,filename,scannumber):
        self.filename = filename
        self.scannumber = scannumber
        self.data, self.metadata = self._Read_in(self.filename,self.scannumber)

        self._get_important_data()

    def _get_important_data(self):
            self.energy = self.data["energy_mono_ct"]
            self.TEY = self.data["adc2_i3"]
            self.I0 = self.data["adc2_i2"]
            self.Field = float(self.metadata["magnet_y"])
            self.Temperature = float(self.metadata["xmcd_temp_1K"])
            self.Angle = self.metadata["magnet_z"]
            self.Polarisation = float(self.metadata["ideu71_motor_phase"])
            if self.Polarisation > 0:
                self.Polarisation = "pc"
            elif self.Polarisation < 0:
                self.Polarisation = "nc"
            else:
                return Exception("Polarisation value not recognised -> check ideu71_motor_phase")
            return 1
    
    def _Read_in(self,filename,scannumber):   
        """
        General function that reads in all the data for one spectra
        Parameters
        ----------
        filename : Str
            Name of file spectra have been saved in
        scannumber : Str
            Number of scan usualy in form 6.2 ectt.

        Returns 2 Dictionaries containing all scan data and all metadata
        -------
        None.
        """
        # Reading in the file
        sf = specfilewrapper.Specfile(filename)
        scan = sf.select(scannumber)
        labels = scan.alllabels()  
        Metadata_labels = scan.header("O")
        Metadata_data = scan.header("P")
        
        data = []
        labels1 = np.array([])
        data1 = np.array([])
        # Recorded data
        for i in range(1,len(labels)+1):
            arr = scan.datacol(i)
            data.append(arr)
        
        # Metadata
        for i in range(0,19):
            arr = Metadata_labels[i].split() 
            arr1 = Metadata_data[i].split()
            arr = [item.strip() for item in arr]  
            arr1 = [item.strip() for item in arr1]  
            labels1 = np.append(labels1,arr)
            data1 = np.append(data1,arr1)
        
        # Saving it all together in 2 dictionarys
        Metadata_dictionary = dict(zip(labels1,data1))
        Data_dictionary = dict(zip(labels,data))
        return Data_dictionary,Metadata_dictionary
        
def plot_spectra(TEYs, Energies, XMCD):

    TEY_pc, TEY_nc = TEYs
    Energy_pc, Energy_nc = Energies
    fig, [TeyAx, XMCDAx, CombinedAx] = plt.subplots(1,3, figsize=(8,12))

    TeyAx.plot(Energy_pc, TEY_pc, label='TEY PC', color='blue')
    TeyAx.plot(Energy_nc, TEY_nc, label='TEY NC', color='red')
    TeyAx.set_title('TEY Spectra')
    TeyAx.set_xlabel('Energy (eV)')
    TeyAx.set_ylabel('TEY')
    TeyAx.legend()

    XMCDAx.plot(Energy_pc, XMCD, label='XMCD', color='green')
    XMCDAx.axhline(0, color='black', linestyle='--')
    XMCDAx.set_title('XMCD Spectrum')
    XMCDAx.set_xlabel('Energy (eV)')
    XMCDAx.set_ylabel('XMCD')
    XMCDAx.legend()

    CombinedAx.plot(Energy_pc, TEY_pc, label='TEY PC', color='blue', alpha=0.5)
    CombinedAx.plot(Energy_nc, TEY_nc, label='TEY NC', color='red', alpha=0.5)
    CombinedAx.plot(Energy_pc, XMCD, label='XMCD', color='green')
    CombinedAx.axhline(0, color='black', linestyle='--')
    CombinedAx.set_title('Combined TEY and XMCD')
    CombinedAx.set_xlabel('Energy (eV)')
    CombinedAx.set_ylabel('Intensity')
    CombinedAx.legend()
    # plt.tight_layout()
    plt.show()
    return fig, [TeyAx, XMCDAx, CombinedAx]

    
def calculate_XMCD(filename,ScanNumber,ScanNumber2, plt_spectra=True):
    print("Creating Spectras - {}, {}".format(ScanNumber, ScanNumber2))
    Scan1 = Spectra_data(filename,ScanNumber)
    Scan2 = Spectra_data(filename,ScanNumber2)
    print("Calculating XMCD")
    XMCD = Scan2.TEY - Scan1.TEY 
    Field_avg = (Scan2.Field + Scan1.Field)/2
    Temp_avg = ((Scan2.Temperature + Scan1.Temperature)/2)

    if plt_spectra:
        print("Plotting Spectras")
        TEYs = (Scan1.TEY/Scan1.I0, Scan2.TEY/Scan2.I0)
        Energies = (Scan1.energy, Scan2.energy)
        plot_spectra(TEYs, Energies, XMCD)

    return XMCD, Field_avg, Temp_avg

def main():

    # Sample data
    filename = r"Analysis code\CrCl3_Alba\data01.dat"
    # Open the SPEC file
    scannumber = "68.1"
    scannumber2 = "67.1"

    calculate_XMCD(filename, scannumber, scannumber2,plt_spectra=True)

    return 1
if __name__ == "__main__":
    main()
        
        
        
        
        
        
        
        
        










