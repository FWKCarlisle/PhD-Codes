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
            
            self.TEY = self.TEY/self.I0
            
            self.Field = float(self.metadata["magnet_y"])
            self.Temperature = float(self.metadata["xmcd_temp_1K"])
            self.Angle = float(self.metadata["xmcd_rotation"])
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
        try:
            scan = sf.select(scannumber)
        except:
            raise Exception(f"No Scannumber: {scannumber}")
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
    print("Creating Spectras - {}, {}".format(ScanNumber, ScanNumber2))#
    print(filename,ScanNumber)
    Scan1 = Spectra_data(filename,ScanNumber)
    Scan2 = Spectra_data(filename,ScanNumber2)

    if Scan1.Polarisation == "pc" and Scan2.Polarisation == "nc":
        nc = Scan2
        pc = Scan1
    elif Scan1.Polarisation == "nc" and Scan2.Polarisation == "pc":
        nc = Scan1
        pc = Scan2
    else:
        return Exception("Scans provided are not one pc and one nc polarisation")

    print(len(pc.TEY), len(nc.TEY))
    plt.plot(pc.energy, pc.TEY, label='PC')
    plt.plot(nc.energy, nc.TEY, label='NC')
    plt.legend()
    plt.show()

    XMCD = (pc.TEY - nc.TEY)/(0.5*(pc.TEY + nc.TEY))
    Field_avg = (pc.Field + nc.Field)/2
    Temp_avg = ((pc.Temperature + nc.Temperature)/2)

    if plt_spectra:
        print("Plotting Spectras")
        TEYs = (Scan1.TEY/Scan1.I0, Scan2.TEY/Scan2.I0)
        Energies = (Scan1.energy, Scan2.energy)
        plot_spectra(TEYs, Energies, XMCD)

    return XMCD,pc.energy, nc.energy, Field_avg, Temp_avg

def find_nearest(array, value):
    
    array = np.asarray(array)
    idx = (np.abs(array-value)).argmin()
    
    return array[idx], idx
def meth_grab_direct_points(XMCD, energy, peak_1 = 576.15, peak_2 = 577.2):
    
    
    XXXXX, max_idx = find_nearest(energy, peak_1)
    XXXXX, min_idx = find_nearest(energy, peak_2)
    
    Max_point = XMCD[max_idx]
    Min_point = XMCD[min_idx]
    
    Hyst_point = Max_point - Min_point
    # plt.figure(1)
    
    # plt.plot(energy, XMCD)
    # plt.scatter(energy[max_idx], XMCD[min_idx])
    # plt.axvline(576.1)
    # plt.axvline(577.2)
    # plt.show()
    
    
    return Hyst_point
def plot_manual_hysteresis(list_scan_numbers, polerisation, filename, colour = "b", T_or_F = True, labels = None):

    num_loops = len(list_scan_numbers)/4
    print(f"Number of loops: {num_loops}")
    count = 0

    chunked_data = []

    positive_data = []
    negative_data = []

    xmcd_list = []
    Fields = []
    temps = []
    rotations = []
    Chunksize = 4
    while count < num_loops:
        # print(list_scan_numbers[count*4:count*4 + 4])
        chunked_data.append(list_scan_numbers[count*Chunksize:count*Chunksize + Chunksize])
        count += 1
    for chunk in chunked_data:
        # print(chunk)
        if polerisation == "+--+":
            # try:
            if Chunksize == 4:
                positive_data.append([chunk[0],chunk[3]])
                negative_data.append([chunk[1],chunk[2]])
            elif Chunksize == 2:
                positive_data.append(chunk[0])
                negative_data.append(chunk[1])
            # except:
            #     raise Exception(f"Wait a second: {len(chunk)}")
    pair_num = 0
    print(positive_data)
    for i, pair in enumerate(positive_data):
        for j, scan_number in enumerate(pair):
            if pair_num == 0:
                XMCD_pair = 0
                Field_pair = 0
                temp_pair = 0
            XMCD,pc_energy, nc_energy, Field_avg, Temp_avg, = calculate_XMCD(filename, scan_number, negative_data[i][j], plt_spectra=False)
        
            XMCD_pair += meth_grab_direct_points(XMCD, pc_energy)
            Field_pair += Field_avg
            temp_pair += Temp_avg
            if pair_num == 1:
                xmcd_list.append(XMCD_pair/2)
                Fields.append(Field_pair/2)
                temps.append(temp_pair/2)
            
            pair_num +=1
            if pair_num == 2:
                pair_num = 0 

    xmcd_array = np.array(xmcd_list)
    Fields_array =  np.array(Fields)
    temps_array = np.array(temps)
    
    plt.plot(Fields_array, xmcd_array,'o-', color=colour)


def main():

    # Sample data
    filename = r"Analysis code\CrCl3_Alba\data01_10Dec2025.dat"
    # Open the SPEC file
    SNs = [120,220]
    end_numbers = [219,319]

    colors = ['r','c',"y"]
    for i, start_number in enumerate(SNs):
        list_scannumbers = np.arange(start_number, end_numbers[i]+1 , 1)
        
        list_scannumbers = np.array(list_scannumbers)
        
        list_scannumbers = [str(i) + ".1" for i in list_scannumbers]
        

        plot_manual_hysteresis(list_scannumbers, "+--+", filename, colour=colors[i])
    plt.show()
    return 1


if __name__ == "__main__":
    main()
        
        
        
        
        
        
        
        
        










