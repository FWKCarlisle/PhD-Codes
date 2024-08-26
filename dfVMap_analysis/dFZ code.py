
from read_spectra import output_data_spectra_dat
from KPFM_spectrum_analysis import KPFMSpectrumAnalysis
# from KPFMFitLatestFile import Spectrum
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

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


path = r"C:\Users\Fwkca\OneDrive\Desktop\PhD Data\Nikhil visit BP\Spatial 10 - dFZ" # Path to the folder containing the .dat files

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


file_beginning = "Z-Spectroscopy_BP_" # The beginning of the file name

# file_list = file_list[0:3]
on_atom_file = "00183"
# files = [
#         ["00030","00031","00032"],
#         ["00033","00034","00035"], #Sets of files with reference in the middle
#         ["00036","00037","00038"],
#         ["00039","00040","00041"],
#         ["00042","00043","00044"]]
# files = ["00228","00230","00232","00234","00236","00238","00241","00244","00246","00248",] #Sets of files numbers to be plotted
# files = ["00228","00229","00231","00233","00235","00237","00240","00243","00245","00247",]
# files = ["00186","00187","00189","00191","00193","00195","00197","00203","00201","00204",]
# files = ["00186","00188","00190","00192","00194","00196","00198","00200","00202","00205",]
# files = ["00208","00209","00211","00213","00215","00217","00219","00221","00223","00225",]
files = ["00208","00210","00212","00214","00216","00218","00220","00222","00224","00226",]


# type = "aba" # reference after every scan
type = "ab" # reference at start 
z_rels = []
dfs = []
all_dfs = []
all_zs = []

def gaussian(x, a, x0, sigma):
    return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))

def lorentzian(x, x0, a, gamma):
            return a * gamma**2 / ((x - x0)**2 + gamma**2)

def fit_gaussian(x, y):
    popt, _ = curve_fit(gaussian, x, y, p0=[1, np.mean(x), np.std(x)])
    return popt

def fit_lorentzian(x, y):
    popt, _ = curve_fit(lorentzian, x, y, p0=[np.mean(x), 1, np.std(x)])
    return popt


def func(x, a, b, c, offset):
    return a * np.exp(-0.5 * np.power((x-b) / c, 2.0)) + offset

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
        
            # Get the x and y data for the specified channel

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
    
    offset = 1

    # print(atom_df)

    for number in files:
        fig, [axData, axMinus,axSmoothMinus] = plt.subplots(nrows=3, ncols=1, sharex=True)

        file_name = file_beginning+number + ".dat"    
        print(file_name)
        # Create a Spectrum instance for each file
        # example_spectrum = Spectrum(path=path, fileName=file_name, channel='OC M1 Freq. Shift [AVG] (Hz)')
        example_spectrum = Spectrum(path=path, fileName=file_name, channel='OC M1 Freq. Shift (Hz)')
        
        z_rel = example_spectrum.x
        df = example_spectrum.y
        z_rels.append(z_rel)
        dfs.append(df)
        numbers.append(number)

        smoothed_df = np.convolve(df, np.ones(5)/5, mode='same')
        smoothed_atom_df = np.convolve(atom_df, np.ones(5)/5, mode='same')

        axData.plot(z_rel, df, label=file_name)
        axData.plot(atom_z_rel, atom_df, label=on_atom_file_name)
        
        # axSmooth.plot(z_rel, smoothed_df, label=file_name)
        # axSmooth.plot(atom_z_rel, smoothed_atom_df, label=on_atom_file_name)


        Minus_curve = -(df - atom_df)
        # smoothed_minus = np.convolve(Minus_curve, np.ones(5)/5, mode='same')

       
        axMinus.plot(z_rel, Minus_curve, label=number)
        # axMinus.plot(z_rel, smoothed_minus, label="Smoothed")

        smoothed_minus = -(smoothed_df - smoothed_atom_df)
        axSmoothMinus.plot(z_rel, smoothed_minus, label=number)
        # only find peaks in the first half of the data 
        
        peak_index = np.argmax(Minus_curve)
        peak_z = z_rel[peak_index]
       
        if Minus_curve[peak_index] - Minus_curve[peak_index+1] > 0.1 or number == "00233":
            exclude_points = 2
             # if the peak is an outlier point, find the next point
            print("Peak is an outlier point")
            print("Peak next point: ", Minus_curve[peak_index] - Minus_curve[peak_index+1])

            mask = np.ones(len(Minus_curve), dtype=bool)
            start_exclude = max(0, peak_index - exclude_points)
            end_exclude = min(len(Minus_curve), peak_index + exclude_points + 1)
            mask[start_exclude:end_exclude] = False

            # Use the mask to find the new peak index
            filtered_curve = Minus_curve[mask]
            new_peak_index_in_filtered = np.argmax(filtered_curve)

            # Map the index back to the original array
            new_peak_index = np.where(mask)[0][new_peak_index_in_filtered]

            print("New curve max index: ", new_peak_index)
            print("New curve max: ", Minus_curve[new_peak_index])

            # Update the peak_index to the newly found peak
            peak_index = new_peak_index

            print("Curve max: ", Minus_curve[peak_index])
            peak_index = np.argmax(Minus_curve[0:peak_index])
            print("New curve max: ", Minus_curve[peak_index])
        
        fit_range = 60  # Number of points to include around the peak
        start = max(0, peak_index - fit_range)
        end = min(len(z_rel), peak_index + fit_range)

        x_data = z_rel[start:end]
        y_data = smoothed_minus[start:end]

        # axMinus.plot(x_data, y_data, 'ro', label="Data")

        if offset is not None:
        # Add an offset to the data
            # offset = abs(min(y_data)) + offset # Ensure all y_data values are positive
            offset = 0.1
            print("Offset: ", offset)
            y_data = y_data + offset
            # print("Y data: ", y_data)

        initial_guess = [peak_z, max(y_data) - 0.1, 1]
        initial_guess_1 = [1, np.mean(x_data), np.std(x_data)]
        # print("Initial guess: ", initial_guess_1)
        try:
            popt, pcov = curve_fit(gaussian, x_data, y_data, p0=initial_guess_1, maxfev=10000)
            print("Popt: ", popt)
        except RuntimeError as e:
            print(f"Error in curve fitting: {e}")
            # return axFit, axResiduals, axDataMinusFit

        x0, a, gamma = popt
        a = a * 1E10
        gamma = gamma * 1E10 
        fwhm = 2.355 * gamma

        perr = np.sqrt(np.diag(pcov))
        error_x0, error_a, error_gamma = perr
        error_a = error_a * 1E10
        error_gamma = error_gamma * 1E10

        error_fwhm = 2.355 * error_gamma

        print(f"Fitted parameters:")
        print(f"Center (x0): {x0} ± {error_x0}")
        print(f"Height (a): {a} ± {error_a}")
        print(f"FWHM: {fwhm} ± {error_fwhm}")


        # Plot the fitted Gaussian with offset
        x_fit = np.linspace(min(x_data), max(x_data), 1000)
        y_fit = gaussian(x_fit, *popt)
        # print("Y fit: ", y_fit)
        axSmoothMinus.plot(x_fit, y_fit-offset, 'b-', label=f'Lorentzian fit\nHeight: {a:.3f}\nFWHM: {fwhm:.3f}')



        plt.xlabel('Relative Z (m)')
        plt.ylabel('Frequency Shift (Hz)')
        plt.title(f"Z-Spectroscopy BP {number}")
        axData.legend()
        axMinus.legend()
        axSmoothMinus.legend()
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
