# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 13:43:10 2024

@author: physicsuser
"""

import lmfit
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
# from scipy.integrate import trapz

class KPFMSpectrumAnalysis():
    
    def __init__(self, bias, df, fit_range=25):
        self.bias = bias
        self.df = df
        self.fit_range = fit_range

    
    def CalcVContact(self, aGuess=0.0, bGuess=0.0, cGuess=0.0, E_min = None, E_max = None, error=False):
        
        """
        Contact potential calculation. Involves performing a parabolic fit on 
        the KPFM spectra data, and finding the fit's minimum.
        
        Parameters
        ----------
        aGuess : float, optional
            Initial guess for the fitting parameter a. The default is 0.
        bGuess : float, optional
            Initial guess for the fitting parameter b. The default is 0.
        cGuess : float, optional
            Initial guess for the fitting parameter c. The default is 0.
        error: bool, optional
            Whether to report the estimated error on Vcontact. Found by 
            propagating the estimates error found for the fitting parameters,
            a, b, and c.
            
        where ax**2 + bx + c 
            
        Returns
        -------
        vContact : float
            Calculated contact potential.
        vContactErr: float, only if error==True
            estimated error on Vcontact
        
        The class instance will have added attributes, including the found
        fit and its residuals, so we can get a measure of the confidence to have
        in our result eg. using the PlotVContactCalculation method below.
        """

        self.E_min = E_min
        self.E_max = E_max
        self.ParabolaFit(exclude_min=E_min, exclude_max=E_max)
        self.ParabolaMinima()            
        if error == True: return self.vContact, self.vContactErr
        else: return self.vContact
    
    
    
    def _Parabola(self, x, a, b, c):
        return a*x**2 + b*x + c



    def _ParabolaModelResidual(self, params, x, y):
        """
        Function to minimize for a parabolic fit.

        """
        a, b, c = params['a'], params['b'], params['c'] # fitting parameters
        model = a*x**2  + b*x + c # objective function
        return (y - model) # residual
    
    
    
    def ParabolaFit(self, aGuess = 0.0, bGuess = 0.0, cGuess = 0.0, exclude_min=None, exclude_max=None):
        """
        Parameters
        ----------
        aGuess : float, optional
            Initial guess for the fitting parameter a. The default is 0.
        bGuess : float, optional
            Initial guess for the fitting parameter b. The default is 0.
        cGuess : float, optional
            Initial guess for the fitting parameter c. The default is 0.

        where ax**2 + bx + c 
        
        Returns
        -------
        fit : 1D array
            Parabolic fit to the KPFM spectra.
        fitInfo : Lmfit ModelResult instace. 
            Contains the found fitting parameters, residuals... See lmfitâ€™s 
            ModelResult documentation for more info.

        """

        
        x, y = self.bias, self.df

        

        # Filter data within the specified range
        if exclude_min is not None and exclude_max is not None:
            print('Excluding data points outside the range: ', exclude_min, exclude_max)
            mask = (x < exclude_min) | (x > exclude_max)
            x = x[mask]
            y = y[mask]

            self.bias = self.bias[mask]
            self.df = self.df[mask]

        parabola_params = lmfit.Parameters() # define a python dict to store the fitting parameters
        parabola_params.add('a', value=aGuess) # define the fitting parameters and an initial guess for its value. Here you can also define contraints, and other useful things
        parabola_params.add('b', value=bGuess) 
        parabola_params.add('c', value=cGuess)
        
        model = lmfit.Model(self._Parabola, independent_vars='x', param_names=['a','b','c'])
        fitInfo = model.fit(y, params=parabola_params, x=x)
        
        # The found fitting parameters are stored in the fitInfo object
        a, b, c = fitInfo.params['a'].value, fitInfo.params['b'].value, fitInfo.params['c'].value 
        
        # Evaluate the found fit for our x values
        fit = self._Parabola(x, a, b, c) 
        
        # calclate the fit's confidence band to 2 sigma, ie ~95%. fit +/- fitConfBand.
        fitConfBand = fitInfo.eval_uncertainty(params=fitInfo.params, sigma=2)
        
        self.fitConfBand = fitConfBand
        self.fit = fit
        self.fitInfo = fitInfo
    
        
        return fit, fitInfo
 

    
    def ParabolaMinima(self):
        """
        
        Returns
        -------
        x_min : float
            x value at the prabolic fit's minimum, ie. the calculated contact 
            potential.
        y_min : float
            y value at the prabolic fit's minimum, ie. the calculated minimum 
            frequency shift.
        xMinErr: float
            estimated error on x_min. Derived by propagating the estimated 
            errors found for the fitting parameters.
        yMinErr: float
            estimated error on y_min. Derived by propagating the estimatewd 
            errors found for the fitting parameters.

        """
        # Get the best-fitting paramenters 
        a, b, c = self.fitInfo.params['a'].value, self.fitInfo.params['b'].value, self.fitInfo.params['c'].value
        
        # Get the estimated standard error for the best-fitting paramenters 
        aErr, bErr, cErr = self.fitInfo.params['a'].stderr, self.fitInfo.params['b'].stderr, self.fitInfo.params['c'].stderr
        
        # Calculate the best-fit's minima
        xMin = -b / (2*a)
        yMin = c - b**2 / (4*a)
        
        # Calculate the error on the best-fit's minima
        xMinErr = 0.5 * np.sqrt(((bErr**2)*(a**2) + (b**2)*(aErr**2))/(a**4))
        yMinErr = 0.25*np.sqrt((b**4*aErr**2 + 4*b**2*a**2*bErr**2 + 16*cErr**2*a**4)/a**4)
        self.vContact = xMin
        self.dfAtVContact = yMin
        self.vContactErr = xMinErr
        self.dfAtVContactErr = yMinErr
        
        return xMin, yMin, xMinErr, yMinErr
    
    
    
    def CalcR(self, xSpectrumPos, ySpectrumPos, xAdatomCentre, yAdatomCentre):
        """
        For KPFM spectra taken aiming to characterize the shape of the potential
        around an adatom, calculate the distance from the adatom's centre.

        Parameters
        ----------
        xAdatomCentre : float
            x coordinate of the adatom's centre.
        yAdatomCentre : float
            x coordinate of the adatom's centre.

        Returns
        -------
        r : float
            distace from the adatom's centre to the spectra's position.

        """
        
        # transformation: origin @ nanonis origin -> origin @ adatom centre
        xDash = xSpectrumPos - xAdatomCentre
        yDash = ySpectrumPos - yAdatomCentre
        
        # evaluate r
        r = 1e9*np.sqrt((xDash)**2 + (yDash)**2)
        self.r = r
        return r
    
    def PeakFit(self,  peakType='Gaussian', amplitudeGuess=None, centerGuess=None,
                    sigmaGuess=None, amplitudeMaxLim=None, centerMaxLim=None,
                    sigmaMaxLim=None, amplitudeMinLim=None, centerMinLim=None,
                    sigmaMinLim=None):
    
        x_data = self.x_data.values
        y_data = self.y_data.values
        
        # https://lmfit.github.io/lmfit-py/builtin_models.html#lmfit.models
        if peakType == 'Gaussian':
            modelPeak = lmfit.models.GaussianModel() 
        elif peakType == 'Lorentzian':
            modelPeak = lmfit.models.LorentzianModel() 
        else: raise ValueError('peakType must be either Gaussian or Lorentzian')
        
        # inbuilt method of guessing the starting values for the fitting params
        paramsPeak = modelPeak.guess(y_data, x=x_data)
        
        paramsPeak['amplitude'].set(value=amplitudeGuess, vary=True, max=amplitudeMaxLim, min=amplitudeMinLim)
        paramsPeak['center'].set(value=centerGuess, vary=True, max=centerMaxLim, min=centerMinLim)
        paramsPeak['sigma'].set(value=sigmaGuess, vary=True, max=sigmaMaxLim, min=sigmaMinLim)
        
        modelBackground = lmfit.models.LinearModel()
        paramsBackground = modelBackground.guess(y_data, x=x_data)
        
        paramsBackground['intercept'].set(value=y_data.min(), vary=True)
        paramsBackground['slope'].set(value=0, vary=True)
        
        model = modelPeak + modelBackground
        params = paramsPeak + paramsBackground
        
        # make the fit
        fitInfo = model.fit(y_data, x=x_data, params=params)
        
        fit = fitInfo.best_fit
        
        # calclate the fit's confidence band to 2 sigma, ie ~95%. fit +/- fitConfBand.
        fitConfBand = fitInfo.eval_uncertainty(params=fitInfo.params, sigma=2)
        
        self.fitConfBand = fitConfBand
        self.fit = fit
        self.fitInfo = fitInfo
        self.area = fitInfo.params['amplitude']
        self.height = fitInfo.params['height']
        self.centre = fitInfo.params['center']
        self.meanAbsRes = np.mean(np.absolute(fitInfo.residual))
        
        return fit, fitInfo
    
    
    def PlotVContactCalculation(self, axFit=None, axResiduals=None, offset=True):
        """
        Use this method to visualise the quality of the data and the contact 
        potential calculation. 
        
        Returns
        -------
        Plot showing the spectra data, the parabolic fit, the fit's minima (ie. 
        the calculated contact potential), and the fit's residuals.

        """
        # if the contact potential has not yet been calculated, calculate it.
        if not hasattr(self, 'vContact'): self.CalcVContact()
        
        def lorentzian(x, x0, a, gamma):
            return a * gamma**2 / ((x - x0)**2 + gamma**2)
        
        

        def gaussian(x, x0, a, sigma):
            return a * np.exp(-((x - x0)**2) / (2 * sigma**2))
        

        def integrate_gaussian(params,x_fit, x_min, x_max):
            a, x0, sigma = params
            fitted_curve = gaussian(x_fit, a, x0, sigma)
            area = trapz(fitted_curve, dx=5)
            print("area =", area)
            # integral, error = quad(lambda x: gaussian(x, a, x0, sigma), x_min, x_max)
            return area ,0


        if axFit == None and axResiduals == None:
            fig, [axFit, axResiduals, axDataMinusFit] = plt.subplots(nrows=3, ncols=1, sharex=True)
        elif axFit == None and axResiduals != None:
            fig, [axFit, axDataMinusFit] = plt.subplots(nrows=2, ncols=1, sharex=True)
        elif axFit != None and axResiduals == None:
            fig, [axResiduals, axDataMinusFit] = plt.subplots(nrows=2, ncols=1, sharex=True)
        else:
            fig, axDataMinusFit = plt.subplots()

        axFit.plot(self.bias, self.df, label = 'data')
        axFit.plot(self.bias, self.fit, label = 'fit', color='red')
        axFit.plot(self.vContact, self.dfAtVContact, 'o', color='black', label = f'$V_{{Contact}}$, ~' + str(round(self.vContact, ndigits=2)) + r'V $\pm $ ' + str(round(self.vContactErr, ndigits=2)))
        axFit.errorbar(self.vContact, self.dfAtVContact, xerr=self.vContactErr, yerr=self.dfAtVContactErr, color='black')
        axFit.fill_between(self.bias, self.fit-self.fitConfBand,
                 self.fit+self.fitConfBand, color='red', alpha=0.2, label=r'confidence band, 2$\sigma$')
        
        axFit.set_ylabel(r'$\Delta$ f / Hz')
        axFit.legend()
        axFit.grid()

        axResiduals.plot(self.bias, self.fitInfo.residual, '.')
        axResiduals.set_ylabel('residuals / Hz')
        axResiduals.set_xlabel('bias / V')
        axResiduals.grid()

        
        data_minus_fit = -(self.df - self.fit)
        peak_index = np.argmax(data_minus_fit)
        peak_bias = self.bias[peak_index]

        # mask = (self.bias > 0)
        # self.bias = self.bias[mask]
        # data_minus_fit = data_minus_fit[mask]
        # #fit lorentzian

        fit_range = self.fit_range  # Number of points to include around the peak
        start = max(0, peak_index - fit_range)
        end = min(len(data_minus_fit), peak_index + fit_range)

        self.x_data = self.bias[start:end]
        self.y_data = data_minus_fit[start:end]
        if offset is not None:
        # Add an offset to the data
            offset = 0 # Ensure all y_data values are positive
            self.y_data = self.y_data + offset

        initial_guess = [peak_bias, max(self.y_data) - 0.1, 1]
        initial_guess = [peak_bias, np.mean(self.x_data), np.std(self.x_data)]
        try:
            popt, pcov = curve_fit(gaussian, self.x_data, self.y_data, p0=initial_guess, maxfev=4000)
        except RuntimeError as e:
            print(f"Error in curve fitting: {e}")
            return axFit, axResiduals, axDataMinusFit

        x0, a, gamma = popt
        fwhm = 2.355 * gamma

        perr = np.sqrt(np.diag(pcov))
        error_x0, error_a, error_gamma = perr
        error_fwhm = 2.355 * error_gamma

        fit, peak_fit = self.PeakFit(peakType='Gaussian', amplitudeGuess=x0, centerGuess=a, sigmaGuess=gamma,)

        # Plot the fitted Gaussian with offset
        lmfit_x = np.linspace(min(self.x_data), max(self.x_data), 60)
        x_fit = np.linspace(min(self.x_data), max(self.x_data), 1000)
        y_fit = gaussian(x_fit, *popt)

        
        print(f"Fitted parameters:")
        print(f"Center (x0): {x0} ± {error_x0}")
        print(f"Height (a): {a} ± {error_a}")
        print(f"FWHM: {fwhm} ± {error_fwhm}")

        # fitted parameters

        axDataMinusFit.plot(self.bias, data_minus_fit, label='data - fit', color='blue')
        smoothed_minus = np.convolve(data_minus_fit, np.ones(5)/5, mode='same')
        axDataMinusFit.plot(self.bias, smoothed_minus, label='smoothed data - fit', color='green')
        axDataMinusFit.plot(x_fit, np.zeroes(len(x_fit)), color='black')
        # axDataMinusFit.plot(x_fit, y_fit - offset, label='fitted Gaussian', color='red')  # Subtract the offset for plotting
        
        if self.bias[peak_index] < 0:
                print("No peak found in the data")
                return axFit, axResiduals, axDataMinusFit
        if offset is not None:
            y_fit = y_fit - offset
        axDataMinusFit.errorbar(x_fit, y_fit, label=f'Gaussian fit\nHeight: {a:.2f}\nFWHM: {fwhm:.2f}', color='red')
        # axDataMinusFit.plot(lmfit_x, fit, label='Gaussian fit LMFit', color='green')


        axDataMinusFit.set_ylabel('data - fit / Hz')
        axDataMinusFit.set_xlabel('bias / V')
        axDataMinusFit.legend()
        axDataMinusFit.grid()
        
        plt.subplots_adjust(wspace=0, hspace=0)
        
        return axFit, axResiduals, axDataMinusFit