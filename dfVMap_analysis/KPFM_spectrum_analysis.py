# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 13:43:10 2024

@author: physicsuser
"""

import lmfit
import numpy as np
import matplotlib.pyplot as plt

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
            Contains the found fitting parameters, residuals... See lmfitâ€™s 
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
        axFit.plot(self.bias, self.fit, label = 'fit', color='red')
        axFit.plot(self.vContact, self.dfAtVContact, 'o', color='black', label = '$V_{Contact}$, ~' + str(round(self.vContact, ndigits=2)) + r'V $\pm $ ' + str(round(self.vContactErr, ndigits=2)))
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