
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as spy
from scipy.special import erf
from scipy.optimize import curve_fit
import numpy as np
from scipy.special import erfinv

from pathlib import Path

from PyQt5 import QtWidgets, QtCore
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar


def model(x, A, x0, sigma, C):
        
        '''

        THIS IS THE MODEL USED TO FIT THE ERF FUNCTION TO THE PROBABILITY DATA
        
        ''' 
        return A * (erf((x - x0) / (np.sqrt(2) * sigma))) + C


def fit_erf(x_data, y_data, y_err=None):

    '''
    
    Code to fit the model from above. Guesses were just what seemed sensible but also primarily they work so I haven't changed them much.
    
    '''

    x = np.asarray(x_data, dtype=float)
    y = np.asarray(y_data, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    order = np.argsort(x)
    x = x[order]
    y = y[order]

    

    # sensible initial guesses
    A0 = 0.5 * (np.nanmax(y) - np.nanmin(y))
    C0 = 0.5 * (np.nanmax(y) + np.nanmin(y))
    x0_0 = np.median(x)
    sigma0 = (np.nanmax(x) - np.nanmin(x)) / 6.0
    p0 = [A0, x0_0, max(sigma0, 1e-8), C0]

    # This just means if you feed it an error, it makes that sigma not the sigma guess
    if y_err is not None:
        if np.isscalar(y_err):
            sigma_arr = np.full_like(y, float(y_err))
        else:
            sigma_arr = np.asarray(y_err, dtype=float)
    else:
        sigma_arr = None

    bounds = ([-np.inf, -np.inf, 1e-12, -np.inf], [np.inf, np.inf, np.inf, np.inf])

    try:
        params, cov = curve_fit(
            model,
            x,
            y,
            p0=p0,
            sigma=sigma_arr,
            absolute_sigma=(sigma_arr is not None),
            bounds=bounds,
            maxfev=20000,
        )
    except Exception as e:
        raise RuntimeError(f"erf fit failed: {e}")

    y_fit = model(x, *params)

    A, x0, sigma, C = params
    factor = (x - x0) / (np.sqrt(2) * sigma)
    exp_factor = np.exp(-factor**2)
    
    # Jacobian of model wrt parameters [A, x0, sigma, C]

    #Found this on stack overflow and then added my function to pull out an error from the fit to try and propagate it. 
    dA = erf(factor)
    dx0 = A * (2 / (np.sqrt(2) * np.pi)) * exp_factor / (np.sqrt(2) * sigma)
    dsigma = A * (2 / (np.sqrt(2) * np.pi)) * exp_factor * (x - x0) / (np.sqrt(2) * sigma**2)
    dC = np.ones_like(x)
    
    jac = np.column_stack([dA, dx0, dsigma, dC])
    y_fit_var = np.sum(jac @ cov * jac, axis=1)
    y_fit_err = np.sqrt(np.abs(y_fit_var))


    return params, cov, x, y_fit, y_fit_err

def calculate_diffusion(fit, time, r_c, plot = False):

    '''
    
    Where the meat and potatos of the diffusion calculation is.
    
    Fit: This is the data you want to use to calculate so just feed in the output from fit_erf
    Time: This is the time away from the surface, typically 0.1s
    r_c: capture radius from monte carlo. Set this to lattice constant of Ag(111)

    '''

    


    x_fit, probabilities, fit_error = fit

    plt.plot(x_fit, probabilities, 'o', label='Data')

    erfinv_vals = erfinv(probabilities) + 0.002 # If I don't add this offset, we get a singularity in hopping rate from where the gradient goes near 0. This doesn't seem to affect the shape of the curve.
    
    deriv_erfinv = np.sqrt(np.pi) / 2 * np.exp(erfinv_vals**2)
    erfinv_err = deriv_erfinv * fit_error


    #Calculate diffusion coefficient and error propagation from fit.

    D = r_c**2 / (4*time*erfinv_vals**2)

    dD_derfinv = -2 * r_c**2 / (4 * time * erfinv_vals**3)
    D_err = np.abs(dD_derfinv) * erfinv_err

    

    # if any nans in D, print warning and skip those points
    if np.isnan(D).any():
        print("Warning: NaN values in diffusion coefficient, likely due to probabilities of 0 or 1. Skipping those points.")
        D_remove = D[~np.isnan(D)]
        x_fit = x_fit[~np.isnan(D)]

        D = D_remove


    #Change to nu

    nu = 2*D/((4.6e-10)**2)
    nu_err = 2*D_err/((4.6e-10)**2)
    

    gradient = np.gradient(np.log(nu), x_fit)
    grad_grad = np.gradient(gradient, x_fit)
    index = np.where(abs(grad_grad) > 0.1)[0]
    first = index[0]
    second = index[-1]
    
    print(grad_grad)



    def Phil_method(Temp, nu):
        kB      = 8.617333e-5
        (slope, intercept), cov = np.polyfit(1/Temp, np.log(nu), 1, cov=True)
        dE      = -kB * slope
        dE_err  = kB * np.sqrt(cov[0, 0])
        nu0     = np.exp(intercept)
        nu0_err = nu0 * np.sqrt(cov[1, 1])

        ln_fit  = slope / Temp + intercept
        ss_res  = np.sum((np.log(nu) - ln_fit)**2)
        ss_tot  = np.sum((np.log(nu) - np.mean(np.log(nu)))**2)
        R2      = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

        return dE, dE_err, nu0, nu0_err, R2, ln_fit

    #This was from the old one where we just took the highest and lowest points and took the gradient to calc barrier.
    #Keeping in here so that the code later on doesn't break. 

    nu_high = nu[second]
    nu_low = nu[first]

    T_high = x_fit[second]
    T_low = x_fit[first]
    print(f"{time} s - {D[0]:.2e} to {D[-1]:.2e} m²/s")
    print(f" Nu_high = {nu_high:.2e} Hz at T = {T_high:.2f} K")
    print(f" Nu_low = {nu_low:.2e} Hz at T = {T_low:.2f} K")

    ln_ratio = np.log(nu_high) - np.log(nu_low)

    ln_err = ln_ratio * np.sqrt((nu_err[-1]/nu_high)**2 + (nu_err[0]/nu_low)**2)
    print(f"ln(nu_high/nu_low) = {ln_ratio:.4f}")

    delta_E = (1.380649e-23 * ln_ratio / (1/T_low - 1/T_high))/ 1.60218e-22
    delta_E_err = (1.380649e-23* ln_err / (1/T_low - 1/T_high))  / 1.60218e-22
    print(f"Estimated diffusion barrier: {delta_E:.3e} meV ± {delta_E_err:.3e} meV") # VALUE LIKELY WRONG -> CHANGE ONCE DONE MORE ON NO CONSTANT nu_0


    delta_E_grad = (max(gradient)) / 8.617333262e-5
    print(f"Estimated diffusion barrier from gradient: {delta_E_grad:.3e} meV ± {delta_E_err:.3e} meV") # VALUE LIKELY WRONG -> CHANGE ONCE DONE MORE ON NO CONSTANT nu_0

    dE, dE_err, nu0, nu0_err, R2, ln_fit = Phil_method(x_fit, nu)
    print(f"Phil method: {dE:.3e} meV ± {dE_err:.3e} meV, nu0 = {nu0:.3e} Hz ± {nu0_err:.3e} Hz, R² = {R2:.4f}")

    ### Plot the plots of nu/T, ln(nu)/(1/t) and grad(ln(nu))/(1/T)

    if plot:

        fig_D, [Ax_nu, Ax_lnnu, Ax_D] = plt.subplots(3,1, figsize=(10, 15), sharex=False)

        # Ax_nu.errorbar(1/x_fit, nu, yerr=nu_err, fmt='o', label='Attempt Frequency (Hz)')
        Ax_nu.plot(x_fit, nu, label='Attempt Frequency (Hz)')
        Ax_nu_fit = Ax_nu.twinx()

        Ax_nu_fit.plot(x_fit, probabilities, "g", label='Fitted Probability')
        
        # Ax_nu.axvline((x_fit)[first], color='g', linestyle='--', label='Max grad ln(nu)')
        # Ax_nu.axvline((x_fit)[second], color='b', linestyle='--', label='Max grad ln(nu)')

        Ax_nu.set_xlabel('Temperature (K)')
        Ax_nu.set_ylabel('Hopping rate (Hz)')
        Ax_nu_fit.set_ylabel('Fitted Probability')
        # plt.legend()
        Ax_lnnu.scatter(1/x_fit, np.log(nu), label='ln(Hopping ratey)')

        

        Ax_lnnu.plot(1/x_fit, ln_fit, "r--", label='Fitted ln(Hopping rate)')
        Ax_lnnu_t = Ax_lnnu.twinx()
        Ax_lnnu_t.plot(1/x_fit, np.log(nu/x_fit), "g", label='Fitted Probability')
        Ax_lnnu_t
        # Ax_lnnu.axvline((1/x_fit)[first], color='g', linestyle='--', label='Max grad ln(nu)')
        # Ax_lnnu.axvline((1/x_fit)[second], color='b', linestyle='--', label='Max grad ln(nu)')
        Ax_lnnu.set_xlabel('1/Temperature (K)')
        Ax_lnnu.set_ylabel('ln(Hopping rate)')
        Ax_lnnu_t.set_ylabel('ln(Hopping rate/Temperature)')

        # Ax_D.errorbar(1/x_fit, D, yerr=D_err, fmt='o', label='Diffusion Coefficient (m²/s)')

        
        Ax_D.plot(1/x_fit, gradient)
        # ax_D_log = Ax_D.twinx()
        # ax_D_log.plot(1/x_fit, np.gradient(gradient), 'r--', label='log10(grad ln(nu))')
        # Ax_D.axvline((1/x_fit)[first], color='g', linestyle='--', label='Max grad ln(nu)')
        # Ax_D.axvline((1/x_fit)[second], color='b', linestyle='--', label='Max grad ln(nu)')

        # Ax_D.axhline(y = max(np.gradient(np.log(nu))), color = 'r', linestyle = '--', label='Max grad ln(nu)')
        # Ax_D.text(1/x_fit[np.argmax(np.gradient(np.log(nu)))], max(np.gradient(np.log(nu))), f'  {delta_E_grad:.2e} meV', va='bottom')
        # Ax_D.plot(x_fit, np.gradient(np.log(nu)), label='d/dT ln(Attempt Frequency)', color='orange')
        Ax_D.set_xlabel('1/Temperature (K)')
        Ax_D.set_ylabel('Gradient of ln(Hopping rate)')

        # plt.show()

    return D, delta_E, delta_E_err, nu


def main_plot_diffusion():


    #DATA from raw counting the probabilities. ALL OF THESE are datasets we have

    data_20_01 = [0,0,0,0,0,0,0,0.18,0.6,0.84,0.94,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
    data_20_5  = [0,0,0,0,0,0,0.2,0.5,0.94,0.94,0.98,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
    data_21_01 = [0,0,0,0,0,0,0,0.06,0.23,0.66,0.89,0.93,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
    data_21_1  = [0,0,0,0,0,0,0,0,0.08,0.38,0.674,0.84,0.96,0.98,0.98,1,1,1,1,1,1]
    data_22_01 = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.3,0.74,0.84,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
    data_22_05 = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.06,0.22,0.56,0.84,0.96,0.98,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]

    data_24_01 = [0,0,0,0,0,0,0,0,0,0.02,0.32,0.7,0.84,0.86,0.92,0.98,1,0.98,1,1,1,1,1,1,1]
    data_24_5  = [0,0,0,0,0,0,0,0,0,0,0,0,0.1,0.36,0.54,0.78,0.84,0.88,0.98,0.92,0.96,0.96,0.98,1,1]
    data_25_01 = [0,0,0,0,0,0,0,0,0,0.06,0.08,0.48,0.7,0.72,0.94,0.92,0.92,0.98,0.98,0.96,1,1,0.98,1,1,1,1,1,1,1,1,1,1,1]
    data_25_05 = [0,0,0,0,0,0,0,0,0,0,0,0.26,0.44,0.62,0.74, 0.88,0.92,0.94,0.98,0.96,0.98,1,0.98,1,0.96,1,0.98,1,0.98,1,0.98,0.96,0.98,1]
    data_26_01 = [0,0,0,0,0,0,0,0,0,0,0.06,0.16,0.54,0.84,0.8,0.88,0.88,1,1,1,1,1,1,1,1,1,1,1,1,1]
    data_26_1  = [0,0,0,0,0,0,0,0,0,0,0,0,0.08,0.26,0.5,0.66,0.8,0.9,0.94,0.92,0.94,0.96,0.96,0.98,1,0.96,1,1,0.98,1]

    data_27_01 = [0,0,0,0,0,0,0,0,0.04,0.12,0.48,0.64,0.92,0.98,0.96,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
    data_27_05 = [0,0,0,0,0,0,0,0,0,0.04,0.08,0.28,0.62,0.84,0.9,0.98,1,0.96,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
    data_28_01 = [0,0,0,0,0,0,0,0,0,0,0,0.04,0.08,0.5,0.76,0.86,0.96,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
    data_28_1  = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.2,0.4,0.66,0.88,0.94,0.9,0.96,0.98,0.98,0.98,1,0.98,1,0.98,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]

    
    data_31_1  = [0,0,0,0,0,0,0,0,0,0,0,0,0,0.06,0.02,0.12,0.22,0.22,0.34,0.44,0.54,0.54,0.64,0.88,0.76,0.78,0.8,0.84,0.82,0.82,0.88,0.98,0.86,0.9,0.92,0.84,0.86,0.94,0.94,0.9,0.92,0.9,0.94,0.94,1,1,1,1,1,1]
    data_32_01 = [0,0,0,0,0,0,0.04,0.22,0.42,0.7,0.84,0.96,0.96,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
    data_32_1  = [0,0,0,0,0,0,0,0.04,0.16,0.54,0.86,0.9,1,1,0.98,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
    data_33_01 = [0,0,0,0,0,0,0,0,0.04,0.12,0.48,0.78,0.86,0.84,0.92,0.9,0.94,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
    data_33_1  = [0,0,0,0,0,0,0,0,0,0.04,0.04,0.24,0.54,0.8,0.8,0.94,0.98,0.96,0.98,1,1,0.98,1,1,1,1,1,1,1,1,1,1]
    data_34_1  = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.04,0.16,0.32,0.36,0.54,0.63,0.82,0.9,0.92,0.98,1,0.94,0.96,0.96,1,0.98,0.96,0.98,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
    data_35_01 = [0,0,0.06,0.06,0.18,0.26,0.48,0.62,0.62,0.84,0.84,0.9,0.94,1,1,0.98,0.98,1,1,1,0.98]
    data_35_1  = [0,0,0,0,0,0,0,0,0.04,0.1,0.32,0.42,0.62,0.755,0.92,0.98,0.98,1,1,1,0.98]
    data_36_01 = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.08,0.29,0.38,0.53,0.33,0.9,1,0.92,0.98,0.98,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
    data_36_1  = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.02,0.22,0.51,0.57,0.7,0.89,0.88,0.92,1,0.94,0.98,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
    data_37_01 = [0.1,0.18,0.18,0.21,0.46,0.58,0.84,0.98,0.96,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
    data_37_1  = [0.06,0.04,0.06,0.1,0.1,0.21,0.62,0.78,0.96,0.94,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
    
    data_38_1  = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.05,0.08,0.08,0.16,0.18,0.32,0.48,0.52,0.64,0.7,1]


    data_31_01 = [0,0,0,0,0,0,0,0,0,0,0,0.06,0.1,0.26,0.36,0.48,0.76,0.76,0.78,0.84,0.96,0.92,0.94,0.98,0.98,0.98,0.98,0.98,0.98,1,1,0.98,0.98,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
    data_34_01 = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.04,0.18,0.38,0.68,0.88,0.94,0.94,0.86,0.9,1,0.98,0.98,0.98,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
    data_38_01 = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.02,0.02,0.14,0.32,0.7,0.72,0.84,0.9,1,1,1,1,1,1,1,1]


    #Set the order of which the data is plotted. Should match the order of the files in the directory.
    data = [data_24_01,data_24_5,data_25_01,data_25_05,data_26_01,data_26_1 ]

    
    #Times
    ts = [0.1, 5.0, 0.1, 0.5, 0.1, 1.0]
    
    symbol = ['o', 's', 'o', 's', 'o', 's']
    labels = ["D24 - 0.1s", "D24 - 5.0s", "D25 - 0.1s", "D25 - 0.5s", "D26 - 0.1s", "D26 - 1.0s"]
    
    def y_error_func(data):
        return np.sqrt(np.asarray(data) * (1 - np.asarray(data)) / 50)

    colours = ['#1f77b4','#1f77b4', "#ff7f0e",'#ff7f0e', '#2ca02c','#2ca02c']

    dir_path = Path(r"C:\Users\ppxfc1\OneDrive - The University of Nottingham\Desktop\PhD\DBTF\December data")
    files = [f.name for f in dir_path.iterdir() if f.is_file()]
    print(files)

    fits = []
    fits_smooth = []
    probabilities = []

    # fig, [ax, lnAx] = plt.subplots(2,1, figsize=(10, 6))
    fig, ax = plt.subplots(1,1, figsize=(10, 6))

    #Run the fit and plot each of the files

    for i, file in enumerate(files):
        arr = np.load(dir_path / file)

        # tempAx.plot(np.linspace(0, len(arr[0]), len(arr[0])), arr[0], label=labels[i], color=colours[i])
        y_error = y_error_func(data[i]) + 0.00002
        params, cov, x_fit, y_fit, y_fit_err = fit_erf(arr[0], data[i], y_err=None)
        
        fits.append([x_fit, y_fit, y_fit_err])


        #smooth that fit out baby
        # x_fit_smooth = np.linspace(min(arr[0]), max(arr[0]), 200)
        # y_fit_smooth = model(x_fit_smooth, *params)
        # y_fit_smooth_err = np.interp(x_fit_smooth, x_fit, y_fit_err)

        # fits_smooth.append([x_fit_smooth, y_fit_smooth, y_fit_smooth_err])
        ax.plot(x_fit, y_fit, '--', color=colours[i],  alpha=1)
        # ax.plot(x_fit_smooth, y_fit_smooth, '-', color=colours[i], alpha = 0.5, linewidth=2)
        ax.errorbar(arr[0], (np.array(data[i])),fmt=symbol[i], yerr=y_error, xerr=arr[1],alpha= 0.75, label=labels[i], color=colours[i])
        y_error_arr = np.full_like(data[i], y_error)
        probabilities.append([np.asarray(arr[0]), np.asarray(data[i]), np.asarray(y_error_arr)])

    ax.legend(fontsize=16)

    ax.set_ylabel("Probability of molecule found (%)")
    ax.set_xlabel("Temperature (K)")

    # tempAx.set_ylabel("Temperature (K)")
    # tempAx.set_xlabel("Measurement index")

    plt.show()

    barriers_e = []
    barriers_o = []
    barriers = []
    for i, fit in enumerate(probabilities):
        #Run the diffusion calculation for each of the fits, feel free to change to fits to run it on non-smoothed data
        print(f"\n File: {files[i]}")
        # mask = (fit[1] > 0.0) & (fit[1] < 0.95)
        # fit[0] = fit[0][mask]
        # fit[1] = fit[1][mask]
        # fit[2] = fit[2][mask]
        D, delta_e, delta_e_err, nu = calculate_diffusion(fit, time=ts[i], r_c=1.5e-9, plot=True)
        if i % 2 == 0:
            barriers_e.append(delta_e)
        else:
            barriers_o.append(delta_e)

        barriers.append(delta_e)



    plt.close('all')

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, barrier in enumerate(barriers_e):
        ax.scatter(i, barrier, color='blue', label='Even' if i == 0 else "")
        ax.scatter(i, barriers_o[i], color='orange', label='Odd' if i == 1 else "")
    ax.set_ylabel("Diffusion barrier (meV)")
    ax.set_xlabel("Dataset index")
    ax.legend()
    
    plt.show()
    print("--------------------")
    print("Diffusion barriers calculated:")
    print(f"Avg barrier: {np.mean(barriers):.3e} meV")
    print(f"STD of barrier: {np.std(barriers):.3e} meV")
    print(f"Min barrier: {np.min(barriers):.3e} meV -> Max barrier: {np.max(barriers):.3e} meV")
    print(f"Final value: {np.mean(barriers):.3e} ± {(np.mean(barriers) - np.min(barriers)):.3e} meV")
    print("--------------------")

    return 1

if __name__ == "__main__":
    main_plot_diffusion()
    # plot_force()



