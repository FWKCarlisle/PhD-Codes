# -*- coding: utf-8 -*-
"""
Created on Wed Dec 10 21:18:14 2025

@author: opbl29
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Dec  2 12:07:51 2025

@author: Wstev
"""
from PyMca5.PyMcaIO import specfilewrapper
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import simpson
import lmfit as lm
from Hysteresis_analysis import XMCD_data_analysis, manual_hyst


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
        
def plot_spectra(TEYs, Energies, XMCD, Field, temp):

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
    XMCDAx.axvline(576.15)
    XMCDAx.axvline(577.2)
    XMCDAx.set_title('XMCD Spectrum')
    XMCDAx.set_xlabel('Energy (eV)')
    XMCDAx.set_ylabel('XMCD')
    XMCDAx.legend()
    
    CombinedAx.plot(Energy_pc, TEY_pc, label='TEY PC', color='blue', alpha=0.5)
    CombinedAx.plot(Energy_nc, TEY_nc, label='TEY NC', color='red', alpha=0.5)
    CombinedAx.plot(Energy_pc, XMCD, label='XMCD', color='green')
    CombinedAx.axhline(0, color='black', linestyle='--')
    CombinedAx.set_title(f'{temp}')
    CombinedAx.set_xlabel('Energy (eV)')
    CombinedAx.set_ylabel('Intensity')
    CombinedAx.legend()
    # plt.tight_layout()
    
    XMCDAx.set_title(f"{Field}")
    fig.show()
    return fig, [TeyAx, XMCDAx, CombinedAx]

    
def calculate_XMCD(filename,ScanNumber,ScanNumber2, plt_spectra=True):
    print("Creating Spectras - {}, {}".format(ScanNumber, ScanNumber2))
    Scan1 = Spectra_data(filename,ScanNumber)
    Scan2 = Spectra_data(filename,ScanNumber2)
    

    # print("Calculating XMCD")
    
    if Scan1.Polarisation == "pc":
        pc_scan = Scan1
        nc_scan = Scan2
    elif Scan2.Polarisation == "pc":
        pc_scan = Scan2
        nc_scan = Scan1
    else:
        raise Exception(f"Polarisation of Scan1 and Scan2 is not PC: {Scan1.Polarisation}, {Scan2.Polarisation}")
    
    # pc_scan.TEY = pc_scan.TEY - np.mean(pc_scan.TEY[:30])
    # nc_scan.TEY = nc_scan.TEY - np.mean(nc_scan.TEY[:30])
    # print(len(pc_scan.TEY), len(nc_scan.TEY))
    # plt.plot(pc_scan.energy, pc_scan.TEY, label='PC')
    # plt.plot(nc_scan.energy, nc_scan.TEY, label='NC')
    # plt.legend()
    # plt.show()

    XMCD = (pc_scan.TEY - nc_scan.TEY)/(0.5*(pc_scan.TEY + nc_scan.TEY))
    # XMCD = XMCD / (0.5 * ((max(pc_scan.TEY) - min(pc_scan.TEY))+(max(nc_scan.TEY) - min(nc_scan.TEY))))
    # print(Scan2.Field, Scan1.Field)
    
    Field_avg = (pc_scan.Field + nc_scan.Field)/2
    Temp_avg = ((pc_scan.Temperature + nc_scan.Temperature)/2)
    XMCD =  normalise_XMCD(XMCD)
    if plt_spectra:
        # print("Plotting Spectras")
    
        TEYs = (pc_scan.TEY, nc_scan.TEY)
        Energies = (pc_scan.energy, nc_scan.energy)
        plot_spectra(TEYs, Energies, XMCD, Field_avg, Temp_avg)
        
    if max(nc_scan.TEY) > max(pc_scan.TEY):
        p_or_n = -1
    else:
        p_or_n = 1
    rotation = 0.5*(pc_scan.Angle + nc_scan.Angle)
    return XMCD, Field_avg, Temp_avg, pc_scan.energy, rotation,  p_or_n


def plot_hysteresis(filename, scan_numbers):

    for i, pair in enumerate(scan_numbers):
        SN1, SN2 = pair
        XMCD, Field_avg, Temp_avg = calculate_XMCD(filename, SN1, SN2, plt_spectra=True)

        plt.scatter(Field_avg, XMCD)

def normalise_XMCD(XMCD):
    background  = np.mean(XMCD[:50])
    XMCD = XMCD - background
    return XMCD

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

def meth_integrate_over_bounds(XMCD, energy, p_or_n):
    
    XMCD_abs = abs(XMCD)
    
    min_bound, max_bound = 574, 580
    
    mask = (energy >= min_bound) & (energy <= max_bound)
    
    filtered_energy = energy[mask]
    filtered_XMCD = XMCD_abs[mask]
    
    integration_result = simpson(y = filtered_XMCD, x=filtered_energy)
    
    integration_result = integration_result * p_or_n
            
    return integration_result

def meth_integrate_over_peaks(XMCD, energy, w=10, peak_1 = 576.15, peak_2 = 577.2):
    
    def get_regions(XMCD, energy, peak, w):
        min_bound, max_bound = peak-w, peak+w
        mask = (energy >= min_bound) & (energy <= max_bound)
        
        
        XMCD_filt = XMCD[mask]
        Energy_filt = energy[mask]
        return XMCD_filt, Energy_filt
     
    XMCD_1, Energy_1 = get_regions(XMCD, energy, peak_1, w)
    XMCD_2, Energy_2 = get_regions(XMCD, energy, peak_2, w)
    
    Peak_1 = simpson(y=XMCD_1, x=Energy_1)
    Peak_2 = simpson(y=XMCD_2, x=Energy_2)

    hyst_point = peak_1 - peak_2
    
    return hyst_point

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
    for i, pair in enumerate(positive_data):
        for j, scan_number in enumerate(pair):
            if pair_num == 0:
                XMCD_pair = 0
                Field_pair = 0
                temp_pair = 0
            XMCD, Field_avg, Temp_avg, energy, rotation, p_or_n = calculate_XMCD(filename, scan_number, negative_data[i][j], plt_spectra=False)
            
            
            if T_or_F:
                # xmcd_list.append(method4(XMCD, energy, p_or_n))
                xmcd_list.append(meth_grab_direct_points(XMCD, energy))
                Fields.append(Field_avg)
                temps.append(Temp_avg)
                rotations.append(rotation)
            else: 
                XMCD_pair += meth_grab_direct_points(XMCD, energy)
                # XMCD_pair += meth_integrate_over_bounds(XMCD, energy, p_or_n)
                # XMCD_pair += meth_grab_direct_points(XMCD, energy)
            # XMCD_pair += max(XMCD, key=abs)
                Field_pair += Field_avg
                temp_pair += Temp_avg
                if pair_num == 1:
                    xmcd_list.append(XMCD_pair/2)
                    Fields.append(Field_pair/2)
                    temps.append(temp_pair/2)
                
                pair_num +=1
                if pair_num == 2:
                    pair_num = 0 
            
    # plt.figure()
    if labels == None:
        label = f"{temp_pair/2.:4f}K"
    else:
        label = labels
        
    plt.figure(100) 
    if T_or_F:
        plt.plot(temps, xmcd_list, "o-", color = colour)
        # plt.plot(rotations,xmcd_list)
        plt.axhline(0, c='k')
        plt.xlabel("Temp (K)")
        plt.ylim(-0.01,1)
        plt.xlim(0,30)
    else:
        plt.plot(Fields, xmcd_list, "o-", color=colour, label=label)
        plt.xlabel("Field (T)") 
        plt.legend()
        plt.axhline(0, ls="--")
        plt.axvline(0,ls="--")
    plt.ylabel("XMCD (A.U)")          
   
    # plt.title(f"{temp_pair/2.:4f}K")       


    return 1

def plot_angle_dependance(list_scan_numbers, polerisation, filename, colour = "b",T_or_F = True, labels = None):

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
    Chunksize = 2
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
    pair_num = 0
    for name in positive_data:
        file = Spectra_data(filename, name)
        TEY = file.TEY
        Energy = file.energy
        angle = file.Angle
        xmcd_list.append(TEY)
        rotations.append(angle)
        Fields.append(Energy)
    TEY = np.mean(xmcd_list, axis=0 )
    rotations = np.mean(rotations,axis=0)
    Fields = np.mean(Fields, axis=0)
    

    
    def line(x, m, c):
        return m*x + c
    
    def linear_fit(nc,energy):
        nc_fit = nc[0:10]
        energy_fit = energy[0:10]
        energy_fit = np.append(energy_fit, energy[len(nc)-10:len(nc)])
        nc_fit = np.append(nc_fit, nc[len(nc)-10:len(nc)])
        model = lm.Model(line)
        params = model.make_params(m=1, c=1)
        result = model.ﬁt(nc_fit, params, x=energy_fit)
        fit_values = result.values
        c = fit_values["c"]
        m = fit_values["m"]
        linear_fit = m*energy + c
        nc = nc - linear_fit
        return nc
    
    
    TEY = linear_fit(TEY,Energy)
    TEY = (TEY - np.mean(TEY[:50]))
    TEY = TEY/max(TEY)
    #plt.plot(Fields, TEY, label = f"Angle: {rotations} ")
    plt.plot(Fields, TEY, label = f"name: {name} ")
    plt.legend()
    plt.xlabel('Energy (eV)')
    plt.ylabel('TEY')
    
    
    return 1

def plot_cl_edge(list_scan_numbers, filename, polerisation="+--+"):
    num_loops = len(list_scan_numbers)/4
    print(f"Number of loops: {num_loops}")
    count = 0

    chunked_data = []

    positive_data = []
    negative_data = []

    xmcd_list = []
    energies = []
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
    for i, pair in enumerate(positive_data):
        for j, scan_number in enumerate(pair):
            if pair_num == 0:
                XMCD_pair = 0
                Field_pair = 0
                temp_pair = 0
            scan = Spectra_data(filename, scan_number)
            
            XMCD, Field_avg, Temp_avg, energy, rotation, p_or_n = calculate_XMCD(filename, scan_number, negative_data[i][j], plt_spectra=False)
            
            XMCD_pair += XMCD
            Field_pair += energy
            
            if pair_num == 1:
                
                plt.plot(Field_pair/2, XMCD_pair/2)
                # plt.plot(scan.energy, scan.TEY)
            pair_num +=1
            if pair_num == 2:
                pair_num = 0 
    plt.xlabel("Energy (eV)")
    plt.ylabel("XMCD")
    return 1
                
            
def Alba():

    # Sample data
    # filename = r"L:\projects\cycle2025-II\20250340331-bkiraly\DATA\data01_10Dec2025"
    filename = r"Analysis code\CrCl3_Alba\data01_10Dec2025.dat"
    
    SN = [120,220] 
    end_numbers = [219,319]
    
    
    labels = ['HOPG - 12K', 'HOPG - 12K', 'HOPG - 11K', 'HOPG - 12K']

    colours = ['r','c',"y"]
    polerisation = "+--+"
    for i, start_number in enumerate(SN):
        list_scannumbers = np.arange(start_number, end_numbers[i]+1 , 1)
        
        list_scannumbers = np.array(list_scannumbers)
        
        list_scannumbers = [str(i) + ".1" for i in list_scannumbers]
        
        # print(list_scannumbers)
        
        #calculate_XMCD(filename, scannumber, scannumber2,plt_spectra=True)
        plot_manual_hysteresis(list_scannumbers,polerisation, filename,colour=colours[i],T_or_F=False)
    return 1
def diamond():

    manual_hyst()



    return 1
if __name__ == "__main__":
    plt.close()
    # main_dataset_1()
   
    Alba()
    diamond()
    plt.show()    
        
        
        
        
        
        
        
        










