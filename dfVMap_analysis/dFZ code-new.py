
from read_spectra import output_data_spectra_dat
from KPFM_spectrum_analysis import KPFMSpectrumAnalysis
# from KPFMFitLatestFile import Spectrum
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.optimize import curve_fit
from scipy.integrate import trapz
from scipy.signal import find_peaks
from scipy.integrate import quad
from matplotlib.ticker import ScalarFormatter, MultipleLocator
from datetime import datetime
import time


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


# path = r"C:\Users\Fwkca\OneDrive\Desktop\PhD Data\Nikhil visit BP\Spatial 10 - dFZ" # Path to the folder containing the .dat files
path = r"C:\Users\Fwkca\OneDrive\Desktop\PhD Data\Nikhil visit BP\BPA3" # Path to the folder containing the .dat files

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


# file_beginning_atom = "Z-Spectroscopy_BP_" # The beginning of the file name
# file_beginning = "Z-Spectroscopy_BP_"
file_beginning_atom = "dfzMap_BPA3_"
file_beginning = "dfzMap_BPA3_"

# file_list = file_list[0:3]
on_atom_file = "00002"

#spatital 10 reference: 00184
#UD1
# files = ["00186","00187","00189","00191","00193","00195","00197","00203","00201","00204","00188","00190","00192","00194","00196","00198","00200","00202","00205",] #up -> down
# files = ["00186","00187","00189","00191","00193","00195","00197","00203","00201","00204",] #up
# files = ["00188","00190","00192","00194","00196","00198","00200","00202","00205",] #down
#LR
# files = ["00208","00209","00211","00213","00215","00217","00219","00221","00223","00225","00210","00212","00214","00216","00218","00220","00222","00224","00226",] #left -> right
# files = ["00208","00209","00211","00213","00215","00217","00219","00221","00223","00225",] #left
# files = ["00210","00212","00214","00216","00218","00220","00222","00224","00226",] #right

# UD2 ref: 00184
# files = ["00230","00232","00234","00236","00238","00241","00248","00228","00229","00231","00233","00235","00237","00240","00247"] #up -> down
# files = ["00230","00232","00234","00236","00238","00241","00244","00246","00248",] #down
# files = ["00228","00229","00231","00233","00235","00237","00240","00243","00245","00247",] #up

# BPA1ref (00001-00042) ref: 00546 
# files = ["00001","00002","00003","00004","00005","00006","00007","00008","00009","00010","00011","00012","00013","00014","00015","00016","00017","00018","00019","00020","00021"] #LR
# files = ["00022", "00023", "00024", "00025", "00026", "00027", "00028", "00029", "00030", "00031", "00032", "00033", "00034", "00035", "00036", "00037", "00038", "00039", "00040", "00041" ] #UD

# BPA2 (00001-00041) ref: BP1_45 ( excluding "00008",)
# files = ["00021","00001","00002","00003","00004","00005","00006","00007","00009","00010","00011","00012","00013","00014","00015","00016","00017","00018","00019","00020",] #UD
# files = ["00022","00023","00024","00025","00026","00027","00028","00029","00030","00031","00032","00033","00034","00035","00036","00037","00038","00039","00040","00041",] #LR

#BPA3 (00005-00085)ref: 00002 Excliding - 00059, 00071, 00073, 00085
# files = ["00046", "00047", "00048", "00049", "00050", "00051", "00052", "00053", "00054", "00055", "00056", "00057", "00058","00059", "00060", "00061", "00062", "00063", "00064", "00065", "00066", "00067", "00068", "00069", "00070","00071", "00072","00073", "00074", "00075", "00076", "00077", "00078", "00079","00080", "00081", "00082", "00083", "00084","00085"] #Y
# exclude  44 13, 16, 19 22, 17, 18
files = ["00005", "00006", "00007", "00008", "00009", "00010", "00011","00012","00013", "00014", "00015","00016","00017","00018","00019",  "00020", "00021", "00022","00023", "00024", "00025", "00026", "00027", "00028", "00029", "00030", "00031", "00032", "00033", "00034", "00035", "00036", "00037", "00038", "00039", "00040", "00041", "00042", "00043","00044"] #x


# TEST
# files = ["00230","00232","00234","00236","00238","00241","00244","00246","00248","00228","00229","00231",]

# type = "aba" # reference after every scan
type = "ab" # reference at start 
fit_number = 60
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
x_fits = []
y_fits = []
Minus_curves = []

def gaussian(x, a, x0, sigma):
    return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))

# def func(x, a, x0, sigma): 
#     return a * np.exp(-(x - x0) ** 2/(2*sigma**2))

def lorentzian(x, a, x0,  gamma):
            return a * gamma**2 / ((x - x0)**2 + gamma**2)

def fit_gaussian(x, y):
    popt, _ = curve_fit(gaussian, x, y, p0=[1, np.mean(x), np.std(x)])
    return popt

def fit_lorentzian(x, y):
    popt, _ = curve_fit(lorentzian, x, y, p0=[np.mean(x), 1, np.std(x)])
    return popt


def func(x, a, b, c, offset):
    return a * np.exp(-0.5 * np.power((x-b) / c, 2.0)) + offset

def find_peak_start_end(x_curve, y_curve, exclusion_list, fit_no = 60):
    peak_index = np.argmax(y_curve)
    peak_z = x_curve[peak_index]
    if y_curve[peak_index] - y_curve[peak_index-1] > 0.1 or number in exclusion_list:
        exclude_points = 2
            # if the peak is an outlier point, find the next point
        print("Peak is an outlier point")
        print("Peak next point: ", y_curve[peak_index] - y_curve[peak_index-1])

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
    return start, end


def plot_fit(start, end, x, y, ax, color, label, alpha, offset=None):
    x_data = x[start:end]
    y_data = y[start:end]
    # print("X data: ", x_data)
        # axMinus.plot(x_data, y_data, 'ro', label="Data")





    print("offset: ", offset)
    if offset is not None:
    # Add an offset to the data
        # offset = abs(min(y_data)) + offset # Ensure all y_data values are positive
        #find mean of y curve without peak region
        offset = 0.025
        print("Offset: ", offset)
        y_data = y_data + offset
        # print("Y data: ", y_data)
        axPeak.plot(x_data, y_data-offset,"o", color = color ,alpha = alpha, label="data for peak")

    initial_guess = [0.1, np.mean(x_data), np.std(x_data)]

    # ax.plot(x_data, gaussian(x_data, *initial_guess_1), 'k-', label="Initial guess")
    print("Initial guess: ", initial_guess)
    # print("X data: ", x_data)
    # print("Y data: ", y_data)
    try:
        popt, pcov = curve_fit(gaussian, x_data, y_data,p0=initial_guess, maxfev=10000)
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

        if offset is None:
            ax.plot(x_fit, y_fit, 'b-', color=color, label=f'Gaussian fit\nHigh Point: {height:.3f}\nFWHM: {fwhm:.3f}\nCenter: {x0:.3f}')
        else:
            ax.plot(x_fit, y_fit-offset, 'b-', color=color, label=f'Gaussian fit\nHigh Point: {height:.3f}\nFWHM: {fwhm:.3f}\nCenter: {x0:.3f}')
            y_fit = y_fit - offset


        integral, error = integrate_gaussian(popt, x_fit,min(x_data), max(x_data))
        return x0, height, fwhm, error_x0, error_a, error_fwhm, integral, error, x_fit, y_fit
    except RuntimeError as e:
        print(f"Error in curve fitting: {e}")
        # return axFit, axResiduals, axDataMinusFit

        # print("Popt: ", popt)
        return None, None, None, None, None, None, None, None, 0, 0

def fit_and_plot(x_curve, df, y_curve, axData,axRef,axPeak,exclusion_list, number, label, color,fit_no = 60, offset=None):
    
    axData.plot(x_curve, y_curve, label="Minus Curve, pre Poly")

    start_1, end_1 = find_peak_start_end(x_curve, y_curve, exclusion_list, fit_no = fit_no)

    # Find the dip region
    dip_start = start_1  # Start index of the dip region
    dip_end = end_1  # End index of the dip region

    axData.plot(x_curve[dip_start:dip_end], y_curve[dip_start:dip_end], 'go', alpha=0.15, label="Dip region")
    # axPeak.plot(x_curve[dip_start:dip_end], y_curve[dip_start:dip_end], 'go', alpha=0.15, label="Dip region")

    # Remove the dip region from the x_curve and y_curve
    x_curve_ND = np.concatenate((x_curve[:dip_start], x_curve[dip_end:]))
    y_curve_ND = np.concatenate((df[:dip_start], df[dip_end:]))

    # ax.plot(x_curve_ND, y_curve_ND, 'bo', alpha=0.45, label="Data without dip")
    # Plot the modified curve without the dip
    # axRef.plot(x_curve_ND, y_curve_ND,color="purple" , alpha=0.45, label="Data without dip")

    #fit a polynomial to the curves
    poly_coeffs = np.polyfit(x_curve, df, 8)
    poly = np.polyval(poly_coeffs, x_curve)
    axRef.plot(x_curve, poly, 'k-',color = "cyan", label="Polynomial fit w. dip")


    poly_coeffs = np.polyfit(x_curve_ND, y_curve_ND, 8)
    poly = np.polyval(poly_coeffs, x_curve)
    axRef.plot(x_curve, poly, 'k-', label="Polynomial fit w/o dip")
    
    # Subtract the polynomial curve from y_curve
    y_curve = -(df - poly)


    axData.plot(x_curve, y_curve,"k-", alpha=1, label="Minus Curve with Poly")
    axPeak.plot(x_curve, y_curve, "k-", alpha=0.5, label="Minus Curve with Poly")
    #find peaks in new curve
    start_2, end_2 = find_peak_start_end(x_curve, y_curve, exclusion_list, fit_no = fit_no)

    x0_1, height_1, fwhm_1, error_x0_1, error_a_1, error_fwhm_1, integral_1, error_1, x_fit_1, y_fit_1 = plot_fit(start_1, end_1, x_curve, y_curve, axPeak, "orange", label, alpha = 0.7,offset=offset)
    x0_2, height_2, fwhm_2, error_x0_2, error_a_2, error_fwhm_2, integral_2, error_2, x_fit_2, y_fit_2 = plot_fit(start_2, end_2, x_curve, y_curve, axPeak, "blue", label, alpha = 0.4, offset=offset)

    x0 = x0_2
    height = height_2
    fwhm = fwhm_2
    error_x0 = error_x0_2
    error_a = error_a_2
    error_fwhm = error_fwhm_2
    integral = integral_2
    error = error_2
    x_fit = x_fit_2
    y_fit = y_fit_2


    return x0, height, fwhm, error_x0, error_a, error_fwhm, integral, error, x_fit, y_fit

def integrate_gaussian(params,x_fit, x_min, x_max):
        a, x0, sigma = params
        fitted_curve = gaussian(x_fit, a, x0, sigma)
        area = trapz(fitted_curve, dx=5)
        print("area =", area)
        # integral, error = quad(lambda x: gaussian(x, a, x0, sigma), x_min, x_max)
        return area ,0
        # return integral, error   

if type == "ab":
    
    on_atom_file_name = file_beginning_atom + on_atom_file + ".dat"
    atom_spectrum = Spectrum(path=path, fileName=on_atom_file_name, channel='OC M1 Freq. Shift (Hz)')
    atom_df = atom_spectrum.y
    atom_z_rel = atom_spectrum.x
    
    offset = 0

    # print(atom_df)

    for number in files:
        fig, [axRef,axData, axPeak] = plt.subplots(nrows=3, ncols=1, sharex=True)

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
        

        axRef.plot(z_rel, df, label=file_name)
        axRef.plot(atom_z_rel, atom_df, label=on_atom_file_name)
        axRef.plot(atom_z_rel[:25500],atom_poly[:25500], label=on_atom_file_name+ " polynomial")
        
        # axSmooth.plot(z_rel, smoothed_df, label=file_name)
        # axSmooth.plot(atom_z_rel, smoothed_atom_df, label=on_atom_file_name)


        
        old_Minus_curve = -(df - atom_df)
        # Minus_curve = -(df - atom_df)
        Minus_curve = -(df - atom_poly)
        # smoothed_minus = np.convolve(Minus_curve, np.ones(5)/5, mode='same')
        print("Lengths: ", len(z_rel), len(Minus_curve), len(df), len(atom_df))
        # print(len(z_rel), len(Minus_curve))
        # axMinus.plot(z_rel, Minus_curve, label="Minus with Poly")
        # axMinus.plot(z_rel,old_Minus_curve,alpha=00.75, label = "Minus without Poly")
        # axMinus.plot(z_rel, smoothed_minus, label="Smoothed")

        smoothed_minus = -(smoothed_df - smoothed_atom_df)
        smoothed_minus_poly = -(smoothed_df - smoothed_atom_poly)
        # axSmoothMinus.plot(z_rel, smoothed_minus, label="Smoothed raw")
        # axSmoothMinus.plot(z_rel, smoothed_minus_poly, alpha=00.75, label = "Smoothed Poly")

        # only find peaks in the first half of the data 

        x0, height, fwhm, error_x0, error_a, error_fwhm, integral, error, x_fit, y_fit = fit_and_plot(z_rel, df, Minus_curve, axData,axRef,axPeak, ["00186","00111","00112",], number, "0-1", "b", fit_no=fit_number, offset=offset)
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
        Minus_curves.append(Minus_curve)
        x_fits.append(x_fit)
        y_fits.append(y_fit)
        
        

        # # Plot the fitted Gaussian with offset
        # x_fit = np.linspace(min(x_data), max(x_data), 1000)
        # y_fit = gaussian(x_fit, *popt)
        # # print("Y fit: ", y_fit)
        # axSmoothMinus.plot(x_fit, y_fit-offset, 'b-', label=f'Gaussian fit\nHigh Point: {height:.3f}\nFWHM: {fwhm:.3f}\nCenter: {x0:.3f}')



        plt.xlabel('Relative Z (m)')
        plt.ylabel('Frequency Shift (Hz)')
        plt.title(f"Z-Spectroscopy BP {number}")
        axData.legend()
        axRef.legend()
        axPeak.legend()

        formatter = ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((-10, -10))
        axPeak.xaxis.set_major_formatter(formatter)
        # axSmoothMinus.yaxis.set_major_formatter(formatter)

                
        axPeak.xaxis.set_major_locator(MultipleLocator(1E-10))
        axPeak.yaxis.set_major_locator(MultipleLocator(0.05))

        # fig.append(axData)
        # fig.append(axMinus)
        # fig.append(axSmoothMinus)
        # plt.show()

        # time.sleep(5)

        # plt.close()
        
    plt.clf()
    # for i in range(len(dfs)):
    #     plt.plot(z_rels[i], dfs[i] + i/4, label=files[i])
        # plt.plot(x_fits[i],y_fits[i] + i, color = "black")


#plot all data on one graph with offsets between file packets

plt.title(f"Z-Spectroscopy BP All")
plt.xlabel('Relative Z (m)')
plt.ylabel('Frequency Shift (Hz)')

# plt.show()


print(len(z_rels), len(Minus_curves), len(x_fits), len(y_fits))

for i in range(len(z_rels)):
    plt.plot(z_rels[i], Minus_curves[i]+i/6, label=numbers_end[i])
    plt.plot(x_fits[i], y_fits[i]+i/6, color="Black")
plt.legend()
plt.show()


print("Numbers: ", numbers_end)
print("As: ", As)
print("Errors on As: ", As_err)
print("FWHMS: ", FWHMS)
print("Errors on FWHMS: ", FWHMS_err)
print("Centers: ", Centers)
print("Errors on Centers: ", Centers_err)
print("Integrals: ", integrals)
print("Integral Errors: ", integral_errs)


now = datetime.now()
time = now.strftime("%d-%m-%Y %H-%M-%S")

# Save the arrays to a .txt file
data = {
    'Notes': f'Fitted parabola on {time} ',
    'Numbers': numbers_end,
    'As': As,
    'Errors on As': As_err,
    'FWHMS': FWHMS,
    'Errors on FWHMS': FWHMS_err,
    'Centers': Centers,
    'Errors on Centers': Centers_err,
    'Integrals': integrals,
    'Integral Errors': integral_errs
}

with open('Data.txt', 'w') as file:
    for key, value in data.items():
        file.write(f'{key}: {value}\n')