
from read_spectra import output_data_spectra_dat
from KPFM_spectrum_analysis import KPFMSpectrumAnalysis
# from KPFMFitLatestFile import Spectrum
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
                     plotCalculation=False, axFit=None, axResiduals=None, e_min=None, e_max=None, ):
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
            axFit, axResiduals, axDataMinusFit = kpfmAnalysis.PlotVContactCalculation(axFit, axResiduals)
            return axFit, axResiduals, axDataMinusFit


path = r"C:\Users\Fwkca\OneDrive\Desktop\PhD Data\Nikhil visit BP\Spatial 7 - 22.08"

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
numbers = []


file_beginning = "Z-Spectroscopy_BP_"

# file_list = file_list[0:3]
on_atom_file = "00010"
# files = [
#         ["00030","00031","00032"],
#         ["00033","00034","00035"], #Sets of files with reference in the middle
#         ["00036","00037","00038"],
#         ["00039","00040","00041"],
#         ["00042","00043","00044"]]
files = ["00053","00052","00050","00049","00048","00047","00055","00056","00057","00061","00058",]
Dip_start = 0.2
Dip_end = 1
fit_range = 33

# type = "aba" # reference after every scan
type = "ab" # reference at start
z_rels = []
dfs = []
all_dfs = []
all_zs = []


if type == "aba":
    count = 0
    file_packet = 0
    for file_packets in files:
        fig, [axData, axMinus, axMinusData] = plt.subplots(nrows=3, ncols=1, sharex=True)
        print("A " , file_packets)
        for number in file_packets:
            
            file_name = file_beginning+number + ".dat"    
            print(file_name)
            # Create a Spectrum instance for each file
            # example_spectrum = Spectrum(path=path, fileName=file_name, channel='OC M1 Freq. Shift [AVG] (Hz)')
            example_spectrum = Spectrum(path=path, fileName=file_name, channel='OC M1 Freq. Shift (Hz)')
            
            z_rel = example_spectrum.x
            df = example_spectrum.y
            if count == 1:
                axData.plot(z_rel, df, label=file_name)
                ref_f = df
            else:
                axData.plot(z_rel, df, label=file_name)
            z_rels.append(z_rel)
            dfs.append(df)
            numbers.append(number)
            
            count += 1
            
            
            if count == 3:
                all_dfs.append(dfs)
                all_zs.append(z_rels)
                print("Count -", count, " file_packet - ", file_packet, " Z_rels ", len(z_rels), " dfs ", len(dfs))
                axMinus.plot(z_rels[0], dfs[0]-dfs[1], label="0-1")
                axMinus.plot(z_rels[0], dfs[2]-dfs[1], label="2-1")
                #plot a 0 line for reference
                axMinus.plot(z_rels[0], np.zeros(len(z_rels[0])), 'k--')
                axMinus.legend()

                axMinusData.plot(z_rels[0], dfs[0]-dfs[2], label="0-1")
                axMinusData.legend()


                plt.title(f"Z-Spectroscopy BP {file_packet} {numbers[0]}-{numbers[1]}-{numbers[2]}") 
                plt.xlabel('Relative Z (m)')
                plt.ylabel('Frequency Shift (Hz)')
                axData.legend()
                plt.show()
                plt.clf()
                print("Count -", count, " ", len(z_rels), " ", len(dfs)) 
                file_packet += 1
                count = 0
                z_rels = []
                dfs = []
                numbers = []
                if len(z_rels) == 0 and len(dfs) == 0:
                    print("Empty arrays")
                    
    fig, axData = plt.subplots()
    for i in range(len(all_dfs)):
        for j in range(len(all_dfs[i])):
            axData.plot(all_zs[i][j], all_dfs[i][j] + i/4, label=files[i][j])
    count = 0    


if type == "ab":
    
    on_atom_file_name = file_beginning+on_atom_file + ".dat"
    atom_spectrum = Spectrum(path=path, fileName=on_atom_file_name, channel='OC M1 Freq. Shift (Hz)')
    atom_df = atom_spectrum.y
    atom_z_rel = atom_spectrum.x
    

    print(atom_df)

    for number in files:
        fig, [axData, axMinus] = plt.subplots(nrows=2, ncols=1, sharex=True)

        file_name = file_beginning+number + ".dat"    
        print(file_name)
        # Create a Spectrum instance for each file
        # example_spectrum = Spectrum(path=path, fileName=file_name, channel='OC M1 Freq. Shift [AVG] (Hz)')
        example_spectrum = Spectrum(path=path, fileName=file_name, channel='OC M1 Freq. Shift (Hz)')
        
        z_rel = example_spectrum.x
        df = example_spectrum.y
        print(df)
        z_rels.append(z_rel)
        dfs.append(df)
        numbers.append(number)

        axData.plot(z_rel, df, label=file_name)
        axData.plot(atom_z_rel, atom_df, label=on_atom_file_name)
        axMinus.plot(z_rel, df - atom_df, label=file_name)
        
        plt.xlabel('Relative Z (m)')
        plt.ylabel('Frequency Shift (Hz)')
        plt.title(f"Z-Spectroscopy BP {number}")
        axData.legend()
        axMinus.legend()
        plt.show()
        
    plt.clf()
    for i in range(len(dfs)):
        plt.plot(z_rels[i], dfs[i] + i/4, label=files[i])


#plot all data on one graph with offsets between file packets

plt.title(f"Z-Spectroscopy BP All")
plt.xlabel('Relative Z (m)')
plt.ylabel('Frequency Shift (Hz)')
plt.legend()
plt.show()