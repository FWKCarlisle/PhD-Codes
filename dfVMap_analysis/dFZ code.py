
from read_spectra import output_data_spectra_dat
from KPFM_spectrum_analysis import KPFMSpectrumAnalysis
# from KPFMFitLatestFile import Spectrum
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy.integrate import quad
from matplotlib.ticker import ScalarFormatter, MultipleLocator


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
on_atom_file = "00184"
# files = [
#         ["00015","00016","00017"],
#         ["00018","00019","00020"], #Sets of files with reference in the middle Spatial 7 Up
#         ["00021","00022","00023"],
#         ["00024","00025","00026"],
#         ["00027","00028","00029"]]
# files = [
#         ["00030","00031","00032"],
#         ["00033","00034","00035"], #Sets of files with reference in the middle # Spatial 7 Down 
#         ["00036","00037","00038"],
#         ["00039","00040","00041"],
#         ["00042","00043","00044"]]

# files = [
#         ["00030","00009","00032"],
#         ["00033","00009","00035"], #Sets of files with reference in the middle # Spatial 7 Down With 0009 as reference
#         ["00036","00009","00038"],
#         ["00039","00009","00041"],
#         ["00042","00009","00044"]]

# files = [
#         ["00015","00009","00017"],
#         ["00018","00009","00020"], #Sets of files with reference in the middle Spatial 7 Up With 0009 as reference
#         ["00021","00009","00023"],
#         ["00024","00009","00026"],
#         ["00027","00009","00029"]]

# spatital 7 reference: 00009
# files = ["00047","00048","00049","00051","00052","00053"] #Sets of files numbers to be plotted ~Spatial 7 Left
# files = ["00055","00056","00057","00061","00058"] #Sets of files numbers to be plotted ~Spatial 7 Right




#spatial 9 data UD Reference: 00106
# files = ["00119","00111","00112","00113","00114","00115","00116","00117","00118"]
#LR
# files = ["00131","00120","00121","00122","00123","00124","00125","00126","00127","00128","00129","00130",]

#spatital 10 reference: 00184
#UD1
files = ["00186","00187","00189","00191","00193","00195","00197","00203","00201","00204","00188","00190","00192","00194","00196","00198","00200","00202","00205",] #up -> down
# files = ["00186","00187","00189","00191","00193","00195","00197","00203","00201","00204",] #up
# files = ["00188","00190","00192","00194","00196","00198","00200","00202","00205",] #down
#LR
# files = ["00208","00209","00211","00213","00215","00217","00219","00221","00223","00225","00210","00212","00214","00216","00218","00220","00222","00224","00226",] #left -> right
# files = ["00208","00209","00211","00213","00215","00217","00219","00221","00223","00225",] #left
# files = ["00210","00212","00214","00216","00218","00220","00222","00224","00226",] #right

# UD2 ref: 00184
# files = ["00230","00232","00234","00236","00238","00241","00244","00246","00248","00228","00229","00231","00233","00235","00237","00240","00243","00245","00247"] #up -> down
# files = ["00230","00232","00234","00236","00238","00241","00244","00246","00248",] #down
# files = ["00228","00229","00231","00233","00235","00237","00240","00243","00245","00247",] #up

#spatial 11 ref: 00275
# UD
# files =  ["00284","00286","00288","00290","00293","00295","00297","00299","00301","00303"] # 27.08 Down
# files =  ["00287","00289","00291","00294","00296","00298","00300","00302"] # 27.08 Up
#LR
# files = ["00304","00306","00308","00318","00321","00324","00326",] # 27.08 left
# files = ["00305","00307","00317","00319","00323","00325",] # 27.08 right


#spatial 12 Left reference: 00349
# LR
# files = ["00350","00352","00354","00356","00358","00360","00362","00364","00366","00368",] # 27.08 left
# files = ["00351","00353","00355","00357","00359","00361","00363","00365","00367",] # 27.08 right
# UD 
# files = ["00370","00372","00374","00376","00378","00380","00382","00384","00386"] #up
# files = ["00369","00371","00373","00375","00377","00379","00381","00383","00385","00387",] #down

#spatitial 13 UD (401-422) reference: 00400
# files = ["00401","00403","00405","00407","00409","00411","00413","00415","00417","00419","00421"] #Up
# files = ["00404","00406","00408","00410","00412","00414","00416","00418","00420","00422"] #Down

# spatital 13 LR (423-441) reference: 00400
# files = ["00423", "00425","00427","00429","00431","00433","00435","00437","00441"] #Left
# files = ["00424", "00426","00428","00430","00432","00434","00436","00438","00440"] #Right
# files = ["00426","00428","00430","00432","00434","00436","00438","00440"] #Right #REMOVED 00424


# type = "aba" # reference after every scan
type = "ab" # reference at start 
fit_number = 120
z_rels = []
dfs = []
all_dfs = []
all_zs = []
As = []
As_err = []
FWHMS = []
FWHMS_err = []
Centers = []
Centers_err = []
numbers_end = []
integrals = []
integral_errs = []

def gaussian(x, a, x0, sigma):
    return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))

# def func(x, a, x0, sigma): 
#     return a * np.exp(-(x - x0) ** 2/(2*sigma**2))

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


def fit_and_plot(x_curve, y_curve, ax,exclusion_list, number, label, color,fit_no = 60, offset=None):
    peak_index = np.argmax(y_curve)
    peak_z = x_curve[peak_index]
    if y_curve[peak_index] - y_curve[peak_index+1] > 0.1 or number in exclusion_list:
        exclude_points = 2
            # if the peak is an outlier point, find the next point
        print("Peak is an outlier point")
        print("Peak next point: ", y_curve[peak_index] - y_curve[peak_index+1])

        mask = np.ones(len(y_curve), dtype=bool)
        start_exclude = max(0, peak_index - exclude_points)
        end_exclude = min(len(y_curve), peak_index + exclude_points + 1)
        mask[start_exclude:end_exclude] = False

        # Use the mask to find the new peak index
        filtered_curve = y_curve[mask]
        new_peak_index_in_filtered = np.argmax(filtered_curve)

        # Map the index back to the original array
        new_peak_index = np.where(mask)[0][new_peak_index_in_filtered]

        print("New curve max index: ", new_peak_index)
        print("New curve max: ", y_curve[new_peak_index])

        # Update the peak_index to the newly found peak
        peak_index = new_peak_index

        print("Curve max: ", y_curve[peak_index])
        peak_index = np.argmax(y_curve[0:peak_index])
        print("New curve max: ", y_curve[peak_index])

    
        
    fit_range = fit_no  # Number of points to include around the peak
    start = max(0, peak_index - fit_range)
    end = min(len(x_curve), peak_index + fit_range)
    print("Start: ", start, " End: ", end)
    Smooth_y = np.convolve(y_curve, np.ones(5)/5, mode='same')
    # print("Peak index: ", Smooth_y)
    x_data = x_curve[start:end]
    y_data = Smooth_y[start:end]
    # print("X data: ", x_data)
        # axMinus.plot(x_data, y_data, 'ro', label="Data")

    print("offset: ", offset)
    if offset is not None:
    # Add an offset to the data
        # offset = abs(min(y_data)) + offset # Ensure all y_data values are positive
        offset = 0
        print("Offset: ", offset)
        y_data = y_data + offset
        # print("Y data: ", y_data)
    ax.plot(x_data, y_data, 'ro',alpha = 0.45, label="Data")


    initial_guess = [peak_z, max(y_data) - 0.1, 1]
    initial_guess_1 = [0.1, np.mean(x_data), np.std(x_data)]

    # ax.plot(x_data, gaussian(x_data, *initial_guess_1), 'k-', label="Initial guess")
    print("Initial guess: ", initial_guess_1)
    # print("X data: ", x_data)
    # print("Y data: ", y_data)
    try:
        popt, pcov = curve_fit(gaussian, x_data, y_data,p0=initial_guess_1, maxfev=10000)
        # print("Popt: ", popt)
        x0, a, gamma = popt
        height = a * 1E10
        gamma = gamma * 1E10
        fwhm = 2.355 * gamma

        perr = np.sqrt(np.diag(pcov))
        error_x0, error_a, error_gamma = perr
        error_a = error_a * 1E10
        error_gamma = error_gamma * 1E10

        error_fwhm = 2.355 * error_gamma

        print(f"Fitted parameters:")
        print(f"Center (x0): {x0} ± {error_x0}")
        print(f"Height (a): {height} ± {error_a}")
        print(f"FWHM: {fwhm} ± {error_fwhm}")

        # Plot the fitted Gaussian with offset
        x_fit = np.linspace(min(x_data), max(x_data), 1000)
        y_fit = gaussian(x_fit, *popt)
        # print("Y fit: ", y_fit)
        ax.plot(x_fit, y_fit-offset, 'b-', label=f'Gaussian fit\nHigh Point: {height:.3f}\nFWHM: {fwhm:.3f}\nCenter: {x0:.3f}')

        integral, error = integrate_gaussian(popt, min(x_data), max(x_data))

        return x0, height, fwhm, error_x0, error_a, error_fwhm, integral, error
    except RuntimeError as e:
        print(f"Error in curve fitting: {e}")
        # return axFit, axResiduals, axDataMinusFit

        # print("Popt: ", popt)
        return None, None, None, None, None, None, None, None

def integrate_gaussian(params, x_min, x_max):
        a, x0, sigma = params
        integral, error = quad(lambda x: gaussian(x, a, x0, sigma), x_min, x_max)
        return integral, error        

if type == "aba":
    count = 0
    file_packet = 0
    for file_packets in files:
        fig, [axData, axMinus, axMinusData] = plt.subplots(nrows=3, ncols=1, sharex=True) 
        print("A " , file_packets)
        for number in file_packets:
            print("B ", number)
            
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
                minus_1 = -(dfs[0] - dfs[1])
                minus_2 = -(dfs[2] - dfs[1])
                axMinus.plot(z_rels[0], minus_1, label="0-1")
                axMinus.plot(z_rels[0], minus_2, label="2-1")
                #plot a 0 line for reference
                axMinus.plot(z_rels[0], np.zeros(len(z_rels[0])), 'k--')
                axMinus.legend()

                axMinusData.plot(z_rels[0], np.convolve(minus_1, np.ones(5)/5, mode='same'), label="Smoothed 0-1")
                axMinusData.plot(z_rels[0], np.convolve(minus_2, np.ones(5)/5, mode='same'), label="Smoothed 2-1")
                print("len z_rel: ", len(z_rel), " len minus_1: ", len(minus_1))
                x0, height, fwhm, error_x0, error_a, error_fwhm, integral, error = fit_and_plot(z_rel, minus_1, axMinusData, ["00288","00294"], number, "0-1", "r",fit_no=fit_number, offset=None)

                axMinusData.legend()

                As.append(height)
                FWHMS.append(fwhm)
                Centers.append(x0)
                As_err.append(error_a)
                Centers_err.append(error_x0)
                FWHMS_err.append(error_fwhm)
                numbers_end.append(number)
                integrals.append(integral)
                integral_errs.append(error)


                plt.title(f"Z-Spectroscopy BP {file_packet} {numbers[0]}-{numbers[1]}-{numbers[2]}") 
                plt.xlabel('Relative Z (m)')
                plt.ylabel('Frequency Shift (Hz)')
                axData.legend()
                plt.show()
                # print("Count -", count, " ", len(z_rels), " ", len(dfs)) 
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
    
    offset = 0

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
        atom_poly_coeffs = np.polyfit(atom_z_rel, atom_df, 8)

                # Subtract the polynomial curve from df
        atom_poly = np.polyval(atom_poly_coeffs, z_rel)


        smoothed_df = np.convolve(df, np.ones(5)/5, mode='same')
        smoothed_atom_df = np.convolve(atom_df, np.ones(5)/5, mode='same')
        smoothed_atom_poly = np.convolve(atom_poly, np.ones(5)/5, mode='same')
#fit a polynomial to atom_df
        

        axData.plot(z_rel, df, label=file_name)
        axData.plot(atom_z_rel, atom_df, label=on_atom_file_name)
        # axData.plot(atom_poly,atom_df, label=on_atom_file_name+ " polynomial")
        
        # axSmooth.plot(z_rel, smoothed_df, label=file_name)
        # axSmooth.plot(atom_z_rel, smoothed_atom_df, label=on_atom_file_name)


        
        old_Minus_curve = -(df - atom_df)
        Minus_curve = -(df - atom_df)
        # Minus_curve = -(df - atom_poly)
        # smoothed_minus = np.convolve(Minus_curve, np.ones(5)/5, mode='same')
        print("Lengths: ", len(z_rel), len(Minus_curve), len(df), len(atom_df))
        # print(len(z_rel), len(Minus_curve))
        axMinus.plot(z_rel, Minus_curve, label="Minus with Poly")
        axMinus.plot(z_rel,old_Minus_curve,alpha=00.75, label = "Minus without Poly")
        # axMinus.plot(z_rel, smoothed_minus, label="Smoothed")

        smoothed_minus = -(smoothed_df - smoothed_atom_df)
        smoothed_minus_poly = -(smoothed_df - smoothed_atom_poly)
        axSmoothMinus.plot(z_rel, smoothed_minus, label="Smoothed raw")
        axSmoothMinus.plot(z_rel, smoothed_minus_poly, alpha=00.75, label = "Smoothed Poly")

        # only find peaks in the first half of the data 

        x0, height, fwhm, error_x0, error_a, error_fwhm, integral, error = fit_and_plot(z_rel, Minus_curve, axSmoothMinus, ["00119","00111","00112",], number, "0-1", "b", fit_no=fit_number, offset=offset)
        # x0, height, fwhm, error_x0, error_a, error_fwhm, integral, error = fit_and_plot(z_rel, smoothed_minus_poly, axSmoothMinus, ["00119","00111","00112",], number, "0-1", "b", fit_no=fit_number, offset=offset)

        As.append(height)
        FWHMS.append(fwhm)
        Centers.append(x0)
        As_err.append(error_a)
        Centers_err.append(error_x0)
        FWHMS_err.append(error_fwhm)
        numbers_end.append(number)
        integrals.append(integral)
        integral_errs.append(error)

        # # Plot the fitted Gaussian with offset
        # x_fit = np.linspace(min(x_data), max(x_data), 1000)
        # y_fit = gaussian(x_fit, *popt)
        # # print("Y fit: ", y_fit)
        # axSmoothMinus.plot(x_fit, y_fit-offset, 'b-', label=f'Gaussian fit\nHigh Point: {height:.3f}\nFWHM: {fwhm:.3f}\nCenter: {x0:.3f}')



        plt.xlabel('Relative Z (m)')
        plt.ylabel('Frequency Shift (Hz)')
        plt.title(f"Z-Spectroscopy BP {number}")
        axData.legend()
        axMinus.legend()
        axSmoothMinus.legend()

        formatter = ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((-10, -10))
        axSmoothMinus.xaxis.set_major_formatter(formatter)
        # axSmoothMinus.yaxis.set_major_formatter(formatter)

                
        axSmoothMinus.xaxis.set_major_locator(MultipleLocator(1E-10))
        axSmoothMinus.yaxis.set_major_locator(MultipleLocator(0.05))


        plt.show()
        
    plt.clf()
    for i in range(len(dfs)):
        plt.plot(z_rels[i], dfs[i] + i/4, label=files[i])


#plot all data on one graph with offsets between file packets

plt.title(f"Z-Spectroscopy BP All")
plt.xlabel('Relative Z (m)')
plt.ylabel('Frequency Shift (Hz)')
plt.legend()
# plt.show()
print("Numbers: ", numbers_end)
print("As: ", As)
print("Errors on As: ", As_err)
print("FWHMS: ", FWHMS)
print("Errors on FWHMS: ", FWHMS_err)
print("Centers: ", Centers)
print("Errors on Centers: ", Centers_err)
print("Integrals: ", integrals)
print("Integral Errors: ", integral_errs)
