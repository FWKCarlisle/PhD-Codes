# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 09:12:20 2024

@author: ppysa5


"""

import matplotlib.pyplot as plt
import numpy as np
import lmfit
from file_reader import Spectrum 
import warnings


class SpectrumAnalysis(Spectrum):
    """
    The Spectrum class encapsulates 1 spectra file. 
    It is a subclass of the spectra opening class
    so it inherits attributes found on the file's metadata (eg. xy position). 

    When performing some analysis to a spectrum, I want to:
        1. write the analysis within separate class 
        (eg. see KPFMSpectrumAnalysis class below)
        3. write a SpectrumAnalysis method that runs my analysis, so that the 
        output is stored as a SpectrumAnalysis attribute 
        (eg. see SpectrumAnalysis.KPFMAnalysis method)
    This way, I'll see, at glance, the different analysis that I or the
    group have written, without siving over too much info.
    """
    
    def __init__(self, channel, path, fileName):
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
        
        self.x, self.y = self.ReadSpectrum(path, fileName, channel)
        self.channel = channel
        
    
    
    def KPFMAnalysis(self, xAdatomCentre=None, yAdatomCentre=None, 
                     plotCalculation=False, axFit=None, axResiduals=None):
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
        
        kpfmAnalysis = KPFMSpectrumAnalysis(bias=self.x, df=self.y)
        
        self.vContact = kpfmAnalysis.CalcVContact()
        
        self.fit = kpfmAnalysis.fit
        self.residuals = kpfmAnalysis.fitInfo.residual
        self.dfAtVContact = kpfmAnalysis.dfAtVContact
        self.vContactErr = kpfmAnalysis.vContactErr
        self.dfAtVContactErr = kpfmAnalysis.dfAtVContactErr
        self.fitInfo = kpfmAnalysis.fitInfo
        self.meanAbsRes = kpfmAnalysis.meanAbsRes
        self.fitA = kpfmAnalysis.fitA
        
        if xAdatomCentre != None and yAdatomCentre != None:
            self.r = kpfmAnalysis.CalcR(self.x_pos, self.y_pos, xAdatomCentre, yAdatomCentre)
            
        if plotCalculation == True: 
            axFit, axResiduals = kpfmAnalysis.PlotVContactCalculation(axFit, axResiduals)
            return axFit, axResiduals
        
    
    def DIDVAnalysis(self, peakType='Gaussian', amplitudeGuess=None, centerGuess=None,
                    sigmaGuess=None, amplitudeMaxLim=None, centerMaxLim=None,
                    sigmaMaxLim=None, amplitudeMinLim=None, centerMinLim=None,
                    sigmaMinLim=None, plotCalculation = False):
        
        analysis = DIDVSpectrumAnalysis(self.x, self.y, self.channel)
        
        analysis.PeakFit(peakType, amplitudeGuess, centerGuess,
                        sigmaGuess, amplitudeMaxLim, centerMaxLim,
                        sigmaMaxLim, amplitudeMinLim, centerMinLim,
                        sigmaMinLim)
        
        self.fit = analysis.fit
        self.fitArea = analysis.area
        self.fitHeight = analysis.height
        self.fitCentre = analysis.centre
        self.meanAbsRes = analysis.meanAbsRes
    

        if plotCalculation == True: 
            axFit, axResiduals = analysis.PlotPeakFit()
            return axFit, axResiduals
        
#%%
# =============================================================================
# KPFM analysis
# =============================================================================

class KPFMSpectrumAnalysis():
    
    def __init__(self, bias, df):
        self.bias = bias
        self.df = df
    
    def CalcVContact(self, aGuess=0.0, bGuess=0.0, cGuess=0.0, error=False):
        
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
        self.ParabolaFit()
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
    
    
    
    def ParabolaFit(self, aGuess = 0.0, bGuess = 0.0, cGuess = 0.0):
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
            Contains the found fitting parameters, residuals... See lmfit’s 
            ModelResult documentation for more info.

        """
        x, y = self.bias, self.df
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
        self.meanAbsRes = np.mean(np.absolute(fitInfo.residual))
        self.fitA = a
        
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
    
    
    def PlotVContactCalculation(self, axFit=None, axResiduals=None):
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
        
        if axFit == None and axResiduals == None:
            fig, [axFit, axResiduals] = plt.subplots(nrows=2, ncols=1, sharex=True)
        elif axFit == None and axResiduals != None:
            fig, axFit = plt.subplots()
        elif axFit != None and axResiduals == None:
            fig, axResiduals = plt.subplots()

        axFit.plot(self.bias, self.df, label = 'data')
        axFit.plot(self.bias, self.fit, label = ' parabolic fit', color='red')
        axFit.plot(self.vContact, self.dfAtVContact, 'o', color='black', label = '$V_{Contact}$, ' + str(round(self.vContact, ndigits=2)) + r'V $\pm $ ' + str(round(self.vContactErr, ndigits=2)))
        axFit.errorbar(self.vContact, self.dfAtVContact, xerr=self.vContactErr, yerr=self.dfAtVContactErr, color='black')
        axFit.fill_between(self.bias, self.fit-self.fitConfBand,
                 self.fit+self.fitConfBand, color='red', alpha=0.2, label='confidence band, 2$\sigma$')
        
        axFit.set_ylabel('$\Delta$ f / Hz')
        axFit.legend()
        axFit.grid()

        axResiduals.plot(self.bias, self.fitInfo.residual, '.')
        axResiduals.set_ylabel('residuals / Hz')
        axResiduals.set_xlabel('bias / V')
        axResiduals.grid()
        
        plt.subplots_adjust(wspace=0, hspace=0)
        
        return axFit, axResiduals
    
    

#%%
# =============================================================================
# dI/dV analysis
# =============================================================================

class DIDVSpectrumAnalysis():
    
    def __init__(self, bias, y, channel):
        self.bias = bias
        if 'Demod' in channel: self.didv = y
        elif 'Current' in channel: self.didv = np.gradient(y, bias)
        else: raise ValueError('Demod or Current channel needed for DIDVSpectrumAnalysis')
    
    
    def PeakFit(self, peakType='Gaussian', amplitudeGuess=None, centerGuess=None,
                    sigmaGuess=None, amplitudeMaxLim=None, centerMaxLim=None,
                    sigmaMaxLim=None, amplitudeMinLim=None, centerMinLim=None,
                    sigmaMinLim=None):
        
        x = self.bias
        y = self.didv 
        
        # https://lmfit.github.io/lmfit-py/builtin_models.html#lmfit.models
        if peakType == 'Gaussian':
            modelPeak = lmfit.models.GaussianModel() 
        elif peakType == 'Lorentzian':
            modelPeak = lmfit.models.LorentzianModel() 
        else: raise ValueError('peakType must be either Gaussian or Lorentzian')
        
        # inbuilt method of guessing the starting values for the fitting params
        paramsPeak = modelPeak.guess(y, x=x)
        
        paramsPeak['amplitude'].set(value=amplitudeGuess, vary=True, max=amplitudeMaxLim, min=amplitudeMinLim)
        paramsPeak['center'].set(value=centerGuess, vary=True, max=centerMaxLim, min=centerMinLim)
        paramsPeak['sigma'].set(value=sigmaGuess, vary=True, max=sigmaMaxLim, min=sigmaMinLim)
        
        modelBackground = lmfit.models.LinearModel()
        paramsBackground = modelBackground.guess(y, x=x)
        
        paramsBackground['intercept'].set(value=y.min(), vary=True)
        paramsBackground['slope'].set(value=0, vary=True)
        
        model = modelPeak + modelBackground
        params = paramsPeak + paramsBackground
        
        # make the fit
        fitInfo = model.fit(y, x=x, params=params)
        
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



    def PlotPeakFit(self, axFit=None, axResiduals=None):
        """
        Use this method to visualise the quality of the data and the contact 
        potential calculation. 
        
        Returns
        -------
        Plot showing the spectra data, the parabolic fit, the fit's minima (ie. 
        the calculated contact potential), and the fit's residuals.

        """
        # if the fit has not yet been calculated, calculate it.
        if not hasattr(self, 'fit'): self.PeakFit()
        
        if axFit == None and axResiduals == None:
            fig, [axFit, axResiduals] = plt.subplots(nrows=2, ncols=1, sharex=True)
        elif axFit == None and axResiduals != None:
            fig, axFit = plt.subplots()
        elif axFit != None and axResiduals == None:
            fig, axResiduals = plt.subplots()
        
        
        axFit.plot(self.bias, self.didv, label = 'data')
        axFit.plot(self.bias, self.fit, label = 'linear + Gaussian fit', color='red')
        axFit.fill_between(self.bias, self.fit-self.fitConfBand,
                  self.fit+self.fitConfBand, color='red', alpha=0.2, label='2$\sigma$ conf band')
        
        # if 'Demod' in channel: yLabel = channel
        # elif 'Current' in channel: yLabel = 'numerical derivative of ' + channel
        axFit.set_ylabel('LI Demod 1 Y [AVG] (A)')
        axFit.legend()
        axFit.grid()

        axResiduals.plot(self.bias, self.fitInfo.residual, '.')
        axResiduals.set_ylabel('residuals (A)')
        axResiduals.set_xlabel('bias / V')
        axResiduals.grid()
        
        plt.subplots_adjust(wspace=0, hspace=0)
        
        return axFit, axResiduals


import os

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


path = r'D:\Results\2024\III-Vs\July 2024'
fileName = FindLatestFile(path)
channel = 'OC M1 Freq. Shift [AVG] (Hz)'
s = SpectrumAnalysis(channel, path, fileName)
s.KPFMAnalysis(plotCalculation=True)
# print(lmfit.fit_report(s.fit_info))

# # path = 'test_files'
# # fileName = 'dfVMap_InSb_20_00481.dat'
# # channel = 'LI Demod 1 Y [AVG] (A)'
# # s = SpectrumAnalysis(channel, path, fileName)
# # s.DIDVAnalysis(plotCalculation=True)
# # # print(lmfit.fit_report(s.fit_info))  
        