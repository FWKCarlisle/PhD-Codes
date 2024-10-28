import matplotlib.pyplot as plt
import numpy as np
import lmfit
from read_spectra import output_data_spectra_dat
import sys

class Spectrum(output_data_spectra_dat):
    """
    The Spectrum class encapsulates 1 spectra file.
    It is a subclass of output_data_spectra_dat (Matt's spectra reading class)
    so it inherits attributes found on the file's metadata (eg. xy position).

    When performing some analysis to a spectrum, I like to:
        1. write the analysis within separate class
        (eg. KPFMSpectrumAnalysis)
        2. write a method that runs my analysis and stores only the
        objects of interest as Spectrum attributes (eg.
        Spectrum.KPFMAnalysis method)
    """

    def __init__(self, path, fileName):
        super().__init__()
        self.get_file(path, fileName) # load all data *and* metadata


    def ReadChannel(self, channel):
        """
        Read a channel's data.
        @param channel: channel name
        If the channel is not found, the available channels are
        printed. So, if you're not sure of the exact channel name, just
        type in nonsense.

        Note: channel = 'Index' is an option. May seem redundant, but may
        be useful in future to convert to a time channel, if we make
        note of sampling freq. TCP receiver sampling is limited to
        20 kHz. So, if measurement made using data logger through
        Matt's python_interface_nanonis.py, default is 20 kHz.
        Note: we may be able to play with TCP receiver to lower the 20kHz limit.
        @type channel: str
        @return: channel's data
        @rtype: arr
        """
        def CheckChannelExists(channel):
            if channel not in list(self.df):
                print('Choice of channel not found')
                self.show_method_fun()
                print('Index')
                sys.exit()

        if channel == 'Index':
            foo = self.give_data(0)[0] # load a random channel e.g. 0 to read its length
            x = list(range(len(foo)))
        else:
            if type(channel) == str: CheckChannelExists(channel)
            x = self.give_data(channel)[0]

        return x


    def KPFMAnalysis(self, xAdatomCentre=None, yAdatomCentre=None,
                     plotCalculation=False, axFit=None, axResiduals=None,
                     yChannel=None):
        """
        From KPFM spectra,  df(V), we want to calculate the Vcontact. That is a parabolic
         fit's, y=ax**2+bx+c, minima, b/(2a).

        Note: we can get a feel for the calculation's accuracy by:
            1. looking at the error on Vcontact by propagating the error quoted by lmfit
             for the fitting parameters b and a. *But* this is likely an underestimate
             because:
              - experimental variables play a role, eg. the quality of the AFM
                resonance, drift...
              - I don't know how lmfit calculates error. We have noticed that lmfit's
                errors appear surprisingly low.
            2. plotting it (setting plotCalculation=True)
            3. inspecting the fit's stats using lmfit.fit_report(self.fitInfo).

        @param xAdatomCentre: x coordinate of the artificial atom's centre.
        @type xAdatomCentre: float, optional
        @param yAdatomCentre: y coordinate of the artificial atom's centre.
        @type yAdatomCentre: float, optional
        If xAdatomCentre and yAdatomCentre are specified, each spectrum's distance
        from the artificial atom's centre will be calculated and added as self.r (float).
        @param plotCalculation: If True, plot of the spectrum, its found fit and its corresponding 2
        sigma conf band; the fit's minimum and its correspoding error bar derived by propagating
        the error on the fitting parameters. The default is False.
        @type plotCalculation: bool
        @param axFit:
        @type axFit:
        @param axResiduals:
        @type axResiduals:
        @param yChannel: Use if you want to analyse only one of the repeat sweeps e.g.
        'OC M1 Freq. Shift [00002] (Hz)'. Otherwise, the default is 'OC M1 Freq. Shift [AVG] (Hz)' or, in its absence,
        'OC M1 Freq. Shift (Hz)'.
        @type yChannel: str
        @return: if plotCalculation == True, the matplotlib fig and axs objects will be returned
        @rtype: a matplotlib Figure and two Axes objects
        The useful info from the analysis is added as Spectrum attributes: self.vContact (float), self.fit (arr),
        self.residuals (arr), self.dfAtVContact (float), self.vContactErr (float), self.dfAtVContactErr (float),
        self.fitInfo (Lmfit ModelResult instance), self.meanAbsRes (float), self.fitA (float)
        """
        # read the x and y channels
        self.x = self.ReadChannel('Bias calc (V)')

        if yChannel is not None:
            self.y = self.ReadChannel(yChannel)
        else:
            try: self.y = self.ReadChannel('OC M1 Freq. Shift [AVG] (Hz)')
            except: self.y = self.ReadChannel('OC M1 Freq. Shift [Hz]')

        # analyse
        kpfmAnalysis = KPFMSpectrumAnalysis(bias=self.x, df=self.y)

        if xAdatomCentre != None and yAdatomCentre != None:
            self.r = kpfmAnalysis.CalcR(self.x_pos, self.y_pos, xAdatomCentre, yAdatomCentre)

        # store useful variables
        self.vContact = kpfmAnalysis.CalcVContact()
        self.fit = kpfmAnalysis.fit
        self.residuals = kpfmAnalysis.fitInfo.residual
        self.dfAtVContact = kpfmAnalysis.dfAtVContact
        self.vContactErr = kpfmAnalysis.vContactErr
        self.dfAtVContactErr = kpfmAnalysis.dfAtVContactErr
        self.fitInfo = kpfmAnalysis.fitInfo
        self.meanAbsRes = kpfmAnalysis.meanAbsRes
        self.fitA = kpfmAnalysis.fitA


        if plotCalculation == True:
            fig, axFit, axResiduals = kpfmAnalysis.PlotVContactCalculation(axFit, axResiduals)
            return fig, axFit, axResiduals


    def DIDVAnalysis(self, peakType='Gaussian', amplitudeGuess=None, centerGuess=None,
                    sigmaGuess=None, amplitudeMaxLim=None, centerMaxLim=None,
                    sigmaMaxLim=None, amplitudeMinLim=None, centerMinLim=None,
                    sigmaMinLim=None, plotCalculation = False, yChannel=None):
        """
        DiDv spectrum analysis procedure. I.e. a peak + linear background fit
        @param peakType: accepts 'Gaussian' or 'Lorentzian'. Default is 'Gaussian'
        @type peakType: str, optional
        @param amplitudeGuess: peak amplitude guess
        @type amplitudeGuess: float, optional
        @param centerGuess: peak center guess
        @type centerGuess: float, optional
        @param sigmaGuess: peak sigma guess
        @type sigmaGuess: float, optional
        @param amplitudeMaxLim: restrict peak amplitude maximum limit
        @type amplitudeMaxLim: float, optional
        @param centerMaxLim: restrict peak center maximum limit
        @type centerMaxLim: float, optional
        @param sigmaMaxLim: restrict peak sigma maximum limit
        @type sigmaMaxLim: float, optional
        @param amplitudeMinLim: restrict peak amplitude minimum limit
        @type amplitudeMinLim: float, optional
        @param centerMinLim: restrict peak center minimum limit
        @type centerMinLim: float, optional
        @param sigmaMinLim: restrict peak sigma minimum limit
        @type sigmaMinLim: float, optional
        @param plotCalculation: whether to plot the calculation, default is False
        @type plotCalculation: bool
        @param yChannel: specify the y channel, Otherwise, the default is 'LI Demod 1 Y [AVG] (A)' or, in its absence,
        'LI Demod 1 Y (A)'.
        @return: The useful info from the analysis is added as Spectrum attributes: self.fit (arr),
        self.fitInfo (Lmfit ModelResult instance), self.fitArea (float), self.fitHeight (float), self.fitCentre (float),
        self.meanAbsRes (float)
        """
        # read the x and y channels
        self.x = self.ReadChannel('Bias calc (V)')

        if yChannel is not None:
            self.y = self.ReadChannel(yChannel)
            if 'Current' in yChannel:
                self.y = np.gradient(self.y, self.x)
                print('current channel provided for DIDV analysis so, first, took numerical derivative by y = np.gradient(y, x)')
        else:
            try: self.y = self.ReadChannel('LI Demod 1 Y [AVG] (A)')
            except: self.y = self.ReadChannel('LI Demod 1 Y (A)')

        # analyse
        analysis = DIDVSpectrumAnalysis(self.x, self.y)

        analysis.PeakFit(peakType, amplitudeGuess, centerGuess,
                        sigmaGuess, amplitudeMaxLim, centerMaxLim,
                        sigmaMaxLim, amplitudeMinLim, centerMinLim,
                        sigmaMinLim)

        # store useful variables
        self.fit = analysis.fit
        self.fitInfo = analysis.fitInfo
        self.fitArea = analysis.area
        self.fitHeight = analysis.height
        self.fitCentre = analysis.centre
        self.meanAbsRes = analysis.meanAbsRes

        if plotCalculation == True:
            fig, axFit, axResiduals = analysis.PlotPeakFit()
            return fig, axFit, axResiduals


    def ForceAnalysis(self, threshold=7.5e-12, plotCalculation=False):
        """
        work in progress
        @param threshold:
        @type threshold:
        @param plotCalculation:
        @type plotCalculation:
        """
        # read the needed channels
        self.x = self.ReadChannel('Index') # we might convert this to position or time
        self.y = self.ReadChannel('OC M1 Freq. Shift (Hz)')
        self.I = self.ReadChannel('Current (A)')

        # analyse
        analysis = ForceSpectrumAnalysis(self.y, self.I, threshold)
        analysis.FindEvent()

        # store useful variables
        self.eventMask = analysis.eventMask
        self.absDI = analysis.absDI
        self.threshold = analysis.threshold

        if plotCalculation == True:
            analysis.PlotEventCalculation()



        
#%%
# ======================================================================================================================
# KPFM analysis
# ======================================================================================================================

class KPFMSpectrumAnalysis():
    
    def __init__(self, bias, df):
        self.bias = bias
        self.df = df



    def CalcVContact(self, aGuess=0.0, bGuess=0.0, cGuess=0.0, error=False):
        """
        Contact potential calculation. Involves performing a parabolic fit, ax**2 + bx + c, on
        the KPFM spectra data, and finding the fit's minimum.
        @param aGuess: Initial guess for the fitting parameter a. The default is 0.
        @type aGuess: float, optional
        @param bGuess: Initial guess for the fitting parameter b. The default is 0.
        @type bGuess: float, optional
        @param cGuess: Initial guess for the fitting parameter c. The default is 0.
        @type cGuess: float, optional
        @param error: Whether to report the estimated error on Vcontact. Found by
        propagating the estimates error found for the fitting parameters, a, b, and c.
        @type error: float, optional
        @return: (Calculated Vcontact, estimated error on Vcontact if error==True)
        @rtype: (float, float)
        The class instance will have added attributes, including the found
        fit and its residuals, so we can get a measure of the confidence to have
        in our result eg. using the PlotVContactCalculation method below.
        """
        self.ParabolaFit(aGuess = aGuess, bGuess = bGuess, cGuess = cGuess)
        self.ParabolaMinima()            
        if error == True: return self.vContact, self.vContactErr
        else: return self.vContact

    

    def ParabolaFit(self, aGuess = 0.0, bGuess = 0.0, cGuess = 0.0):
        """
        Parabolic fit, ax**2 + bx + c.
        @param aGuess: Initial guess for the fitting parameter a. The default is 0.
        @type aGuess: float, optional
        @param bGuess: Initial guess for the fitting parameter b. The default is 0.
        @type bGuess: float, optional
        @param cGuess: Initial guess for the fitting parameter c. The default is 0.
        @type cGuess: float, optional
        @return: (fit, fitInfo)
        @rtype: (arr, Lmfit ModelResult instance which contains the found fitting parameters,
         residuals... See lmfit’s ModelResult documentation for more info.)
        """
        def Parabola(x, a, b, c):
            return a * x ** 2 + b * x + c

        x, y = self.bias, self.df
        parabola_params = lmfit.Parameters() # define a python dict to store the fitting parameters

        # define the fitting parameters and an initial guess for its value.
        # Here you can also define contraints, and other useful things.
        parabola_params.add('a', value=aGuess)
        parabola_params.add('b', value=bGuess) 
        parabola_params.add('c', value=cGuess)
        
        model = lmfit.Model(Parabola, independent_vars='x', param_names=['a','b','c'])
        fitInfo = model.fit(y, params=parabola_params, x=x)
        
        # The found fitting parameters are stored in the fitInfo object
        a, b, c = fitInfo.params['a'].value, fitInfo.params['b'].value, fitInfo.params['c'].value 
        
        # Evaluate the found fit for our x values
        fit = Parabola(x, a, b, c)
        
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
        The parabolic fit's minima.
        @return: (xMin, yMin, xMinErr, yMinErr) of the parabolic fit. errors derived from the fitting parameters'
        error, as calculated by lmfit
        Note: we suspect lmfit's errors are an underestimate!
        @rtype: (float, float, float, float)
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
        Pythagoras, to find the spectrum's distance from the artificial atom's centre.
        (Written here because I'm aiming to characterize artificial atoms' potential landscape).
        @param xSpectrumPos: x coordinate of spectrum.
        @type xSpectrumPos: float
        @param ySpectrumPos: y coordinate of spectrum.
        @type ySpectrumPos: float
        @param xAdatomCentre: x coordinate of the artificial atom's centre.
        @type xAdatomCentre: float
        @param yAdatomCentre: y coordinate of the artificial atom's centre.
        @type yAdatomCentre: float
        Note: all inputs are in expected in nanonis coordinates and SI units.
        @return: the radial distance of the spectrum from the artificial atom's centre.
        @rtype: float
        """
        # transformation: origin @ nanonis origin -> origin @ adatom centre
        xDash = xSpectrumPos - xAdatomCentre
        yDash = ySpectrumPos - yAdatomCentre
        
        # evaluate r
        r = np.sqrt((xDash)**2 + (yDash)**2)
        self.r = r
        return r
    
    
    def PlotVContactCalculation(self, axFit=None, axResiduals=None):
        """
        Visualise the self.ParabolaFit() and self.ParabolaMinima() calculation with a
        plot showing the spectrum data, the parabolic fit, the fit's minima (ie.
        the calculated contact potential), the fits 2 sigma confidence band, and the fit's residuals.
        @param axFit: axes for the fit's plot
        @type axFit: matplotlib Axes instance, optional
        @param axResiduals: axes for the residuals' plot
        @type axResiduals: matplotlib Axes instance, optional
        @return: figure and its axes
        @rtype: a matplotlib Figure and two Axes objects
        """
        # if the contact potential has not yet been calculated, calculate it.
        if not hasattr(self, 'vContact'): self.CalcVContact()
        
        if axFit == None and axResiduals == None:
            fig, [axFit, axResiduals] = plt.subplots(nrows=2, ncols=1, sharex=True)
        elif axFit == None and axResiduals != None:
            fig, axFit = plt.subplots()
        elif axFit != None and axResiduals == None:
            fig, axResiduals = plt.subplots()

        axFit.plot(self.bias, self.df, label = 'data', color='black')
        axFit.plot(self.bias, self.fit, label = ' parabolic fit', color='red')
        axFit.errorbar(self.vContact, self.dfAtVContact, xerr=self.vContactErr, yerr=self.dfAtVContactErr, color='black')
        axFit.fill_between(self.bias, self.fit-self.fitConfBand,
                 self.fit+self.fitConfBand, color='red', alpha=0.2, label='confidence band, 2$\sigma$')
        axFit.plot(self.vContact, self.dfAtVContact, "*", color='orange', markersize=10, label = '$V_{Contact}$, ' + str(round(self.vContact, ndigits=2)) + r'V $\pm $ ' + str(round(self.vContactErr, ndigits=2)))
        
        
        axFit.set_ylabel('$\Delta$ f / Hz')
        axFit.legend(bbox_to_anchor=(1, 1))
        axFit.grid()

        axResiduals.plot(self.bias, self.fitInfo.residual, '.', color='gray')
        axResiduals.set_ylabel('residuals / Hz')
        axResiduals.set_xlabel('bias / V')
        axResiduals.grid()
        
        plt.tight_layout()
        plt.subplots_adjust(wspace=0, hspace=0)
        
        return fig, axFit, axResiduals
    
    

#%%
# ======================================================================================================================
# dI/dV analysis
# ======================================================================================================================

class DIDVSpectrumAnalysis():
    
    def __init__(self, bias, y):
        """
        @param bias: spectrum's x channel, i.e. bias
        @type bias: arr
        @param y: spectrum's y channel, i.e. a demod channel or the numerical derivative of a current channel
        @type y: arr
        """
        self.bias = bias
        self.didv = y

    
    
    def PeakFit(self, peakType='Gaussian', amplitudeGuess=None, centerGuess=None,
                    sigmaGuess=None, amplitudeMaxLim=None, centerMaxLim=None,
                    sigmaMaxLim=None, amplitudeMinLim=None, centerMinLim=None,
                    sigmaMinLim=None):
        """
        Gaussian or Lorentzian peak + linear background fitting routine, using lmfit
        @param peakType: accepts 'Gaussian' or 'Lorentzian'. Default is 'Gaussian'
        @type peakType: str, optional
        @param amplitudeGuess: peak amplitude guess
        @type amplitudeGuess: float, optional
        @param centerGuess: peak center guess
        @type centerGuess: float, optional
        @param sigmaGuess: peak sigma guess
        @type sigmaGuess: float, optional
        @param amplitudeMaxLim: restrict peak amplitude maximum limit
        @type amplitudeMaxLim: float, optional
        @param centerMaxLim: restrict peak center maximum limit
        @type centerMaxLim: float, optional
        @param sigmaMaxLim: restrict peak sigma maximum limit
        @type sigmaMaxLim: float, optional
        @param amplitudeMinLim: restrict peak amplitude minimum limit
        @type amplitudeMinLim: float, optional
        @param centerMinLim: restrict peak center minimum limit
        @type centerMinLim: float, optional
        @param sigmaMinLim: restrict peak sigma minimum limit
        @type sigmaMinLim: float, optional
        @return: fit and fitInfo
        @rtype: arr and lmfit's ModelResult instance (https://lmfit.github.io/lmfit-py/model.html#lmfit.model.ModelResult)

        note: more inputs could be added to customise the fit further, and more peak types could be added.
        I just haven't needed to do so yet.
        see https://lmfit.github.io/lmfit-py/parameters.html
        and https://lmfit.github.io/lmfit-py/builtin_models.html#lmfit.models
        """
        x = self.bias
        y = self.didv 
        
        # see https://lmfit.github.io/lmfit-py/builtin_models.html#lmfit.models
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
        self.sigma = fitInfo.params['sigma']
        self.slope = fitInfo.params['slope']
        self.intercept = fitInfo.params['intercept']
        self.realHeight = fitInfo.params['height'] + fitInfo.params['slope']*fitInfo.params['center'] + fitInfo.params['intercept']
        self.meanAbsRes = np.mean(np.absolute(fitInfo.residual))
        
        return fit, fitInfo




    def PlotPeakFit(self, axFit=None, axResiduals=None):
        """
        Plot of the peak fitted (by self.PeakFit) to the spectrum.
        Note: if self.PeakFit has not been called beforehand, it will be called with the default inputs.
        @param axFit: axes for the fit's plot
        @type axFit: matplotlib Axes instance, optional
        @param axResiduals: axes for the residuals' plot
        @type axResiduals: matplotlib Axes instance, optional
        @return: figure and its axes
        @rtype: a matplotlib Figure and two Axes objects
        """
        # if the fit has not yet been calculated, calculate it with default inputs.
        if not hasattr(self, 'fit'): self.PeakFit()
        
        if axFit == None and axResiduals == None:
            fig, [axFit, axResiduals] = plt.subplots(nrows=2, ncols=1, sharex=True)
        elif axFit == None and axResiduals != None:
            fig, axFit = plt.subplots()
        elif axFit != None and axResiduals == None:
            fig, axResiduals = plt.subplots()
        
        
        axFit.plot(self.bias, self.didv, label = 'data', color='black')
        axFit.plot(self.bias, self.fit, label = 'peak + linear fit', color='red')
        axFit.fill_between(self.bias, self.fit-self.fitConfBand,
                  self.fit+self.fitConfBand, color='red', alpha=0.2, label='2$\sigma$ conf band')
        
        axFit.plot(self.bias,
                   self.slope*self.bias+self.intercept,
                   '--', color='tab:blue', label = 'linear term')
        axFit.plot(self.bias,
                   self.area/(np.sqrt(2*np.pi)*self.sigma) * np.exp(-(self.bias-self.centre)**2/(2*self.sigma**2))
                   , '--', color='navy', label = 'peak term')
        
        axFit.plot(self.centre, self.realHeight, 'o', color='orange', label = 'peak + linear height')
        axFit.plot(self.centre, self.height, "*", markersize=10, color='orange', label = 'peak height')

        axFit.set_ylabel('dI/dV')
        axFit.legend(bbox_to_anchor=(1, 1))
        axFit.grid()

        axResiduals.plot(self.bias, self.fitInfo.residual, '.', color='gray')
        axResiduals.set_ylabel('residuals (A)')
        axResiduals.set_xlabel('bias / V')
        axResiduals.grid()
        
        plt.tight_layout()
        plt.subplots_adjust(wspace=0, hspace=0)
        
        return fig, axFit, axResiduals


# ======================================================================================================================
# Manipulation Force Analysis
# ======================================================================================================================

class ForceSpectrumAnalysis():

    def __init__(self, df, I, threshold=7.5e-12):
        """
        @param df: frequency shift channel
        @type df: arr
        @param I: current channel
        @type I: arr
        @param threshold: cutoff for the derivative of current to be identified as large, and therefore due to an event
        @type threshold: float
        """
        self.df = df
        self.I = I
        self.threshold = threshold


    def FindEvent(self):
        """
        finding events. I define an event as the point right before a large dI or a number of consecutive large dI.
        self.eventMask will be True where an event was identified.
        The details of this method are best seen visually, by self.PlotEventCalculation().
        """
        dI = np.gradient(self.I)
        absDI = np.abs(dI)

        # 1
        bigDIMask = absDI > self.threshold # true where big dI, false elsewhere

        # 2 (1, shifted by one, backward)
        bigDIShiftedMask = bigDIMask[1:] # remove first data point
        bigDIShiftedMask = np.append(bigDIShiftedMask, False) # add a last data point so len(bigDIMask) == len(bigDIShiftedMask)

        # 3 (1 XOR 2)
        bigDIStartNEndMask = np.logical_xor(bigDIShiftedMask, bigDIMask)

        # 4 (3's absDI vals < threshold)
        bigDIStartNEnd = absDI.copy()
        bigDIStartNEnd[np.logical_not(bigDIStartNEndMask)] = np.nan # masked absDI
        bigDIStartMask = bigDIStartNEnd < self.threshold

        self.absDI = absDI
        self.bigDIMask = bigDIMask
        self.bigDIShiftedMask = bigDIShiftedMask
        self.bigDIStartNEndMask = bigDIStartNEndMask
        self.eventMask = bigDIStartMask



    def PlotEventCalculation(self):
        """
        Visualise the self.FindEvent() calculation.
        """
        # if not yet calculated, calculate it.
        if not hasattr(self, 'bigDIMask'): self.FindEvent()

        plt.style.use('dark_background') # easier to highlight different bright colours on the data trace

        fig, ax = plt.subplots(3,4, sharex=True, sharey='row', figsize=(10,8))

        for mask, color, label, col in zip([self.bigDIMask, self.bigDIShiftedMask,
                     self.bigDIStartNEndMask,self.eventMask],
                     ['red', 'cyan','gold', 'lime'],
                     ['1. abs(dI) > threhold', '2. red, shifted back by 1', '3. red XOR blue',
                      '4. yellow < threshold'], range(4)):

            # background
            ax[0, col].plot(self.df, '.', color='darkslategray', markersize=3)
            ax[1, col].plot(self.I, '.', color='darkslategray', markersize=3)
            ax[2, col].plot(self.absDI, '.', color='darkslategray', markersize=3)
            ax[2, col].plot([0, len(self.I)],
                            [self.threshold, self.threshold], '--', color='white', lw=1)

            # foreground
            y = np.copy(self.df)
            y[~mask] = np.nan
            ax[0, col].plot(y, '.', color=color, markersize=3)

            y = np.copy(self.I)
            y[~mask] = np.nan
            ax[1, col].plot(y, '.', color=color, markersize=3, label=label)

            y = np.copy(self.absDI)
            y[~mask] = np.nan
            ax[2, col].plot(y, '.', color=color, markersize=3)

        ax[0, 0].set_ylabel('df (Hz)')
        ax[1, 0].set_ylabel('I (A)')
        ax[2, 0].set_ylabel('abs(dI) (A)')
        ax[2, 0].set_xlabel('Index')

        plt.subplots_adjust(wspace=0, hspace=0)
        fig.legend(loc='lower center', bbox_transform=fig.transFigure, ncol=4)

        plt.style.use('default') # revert back to the default white style setting



#%%
"""

# ======================================================================================================================
# Example Use
# ======================================================================================================================

path = r'data/test_data'

fileName = 'dfVMap_InSb_9_00449.dat'
s = Spectrum(path, fileName)
s.KPFMAnalysis(plotCalculation=True)
print(lmfit.fit_report(s.fitInfo))

fileName = 'dfVMap_InSb_20_00481.dat'
s = Spectrum(path, fileName)
fig, axFit, axRes = s.DIDVAnalysis(plotCalculation=True)
axFit.set_title('Analysis of average dI/dV (default)', size=7.5)
print(lmfit.fit_report(s.fitInfo))

# note that the default is the average y demod channel, but you can specify any other demod, or even current channel
fig, axFit, axRes = s.DIDVAnalysis(yChannel='LI Demod 1 Y [00003] (A)', plotCalculation=True) # of one of the sweeps
axFit.set_title('Analysis of only one of the dI/dV sweeps', size=7.5)

fig, axFit, axRes = s.DIDVAnalysis(yChannel='Current [AVG] (A)', plotCalculation=True) # or of the current channel
axFit.set_title('Analysis of average I(V)', size=7.5)


fileName = 'force_in_manipulation_test_5_012.dat'
s = Spectrum(path, fileName)
s.ForceAnalysis(plotCalculation=True)


plt.show()

"""


