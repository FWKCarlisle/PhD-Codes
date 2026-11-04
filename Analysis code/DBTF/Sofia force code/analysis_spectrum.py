import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import lmfit
import sys
import warnings
from statsmodels.stats.stattools import durbin_watson
import scipy.optimize
from scipy.ndimage import gaussian_filter1d

from read_spectra import output_data_spectra_dat


mpl.use('Qt5Agg')

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

    def __init__(self, path, file_name):
        super().__init__()
        self.get_file(path, file_name) # load all data *and* metadata
        self.file_name = file_name
        


    def read_channel(self, channel):
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

        if channel == 'Index':
            foo = self.give_data(0)[0] # load a random channel e.g. 0 to read its length
            x = list(range(len(foo)))
            """
            Can be later converted to time, if we made note of sampling freq. 
            TCP receiver sampling is limited to 20 kHz. So, if measurement made using data logger 
            through Matt's python_interface_nanonis.py, default is 20 kHz. 
            Note: we may be able to play with TCP receiver to lower the 20kHz limit, if needed.
            """
        elif not self.does_channel_exist(channel):
            sys.exit()
        else:
            x = self.give_data(channel)[0]

        return np.array(x)


    def does_channel_exist(self, channel):
        exist = False

        if type(channel) == str:
            if channel in self.list_of_methods:
                exist = True

        if type(channel) == int:
            if channel < len(self.list_of_methods):
                warnings.warn('define spectra channels as str (not int) to avoid loading an unwanted channel when the recorded channels change.')
                exist = True

        if exist == False:
            print('Choice of channel (', channel,') not found for file: ', self.file_name)
            self.show_method_fun()
            print('Index')

        return exist


    def kpfm_analysis(self, xAdatomCentre=None, yAdatomCentre=None,
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
        self.x = self.read_channel('Bias calc (V)')

        if yChannel is not None:
            self.y = self.read_channel(yChannel)
        else:
            try: self.y = self.read_channel('OC M1 Freq. Shift [AVG] (Hz)')
            except: self.y = self.read_channel('OC M1 Freq. Shift (Hz)')

        # analyse
        analysis = KPFMSpectrumAnalysis(bias=self.x, df=self.y)

        if xAdatomCentre != None and yAdatomCentre != None:
            self.r = analysis.CalcR(self.x_pos, self.y_pos, xAdatomCentre, yAdatomCentre)

        # store useful variables
        self.vContact = analysis.CalcVContact()
        self.x_filt = gaussian_filter1d(self.x, 4)
        self.y_filt = gaussian_filter1d(self.y, 4)
        self.biasAtMaxDf = self.x_filt[np.argmax(self.y_filt)]

        #self.vContact = analysis.CalcCPDFromDDFDV() # test

        self.fit = analysis.fit
        self.residuals = analysis.fitInfo.residual
        self.dfAtVContact = analysis.dfAtVContact
        self.vContactErr = analysis.vContactErr
        self.dfAtVContactErr = analysis.dfAtVContactErr
        self.fitInfo = analysis.fitInfo
        self.meanAbsRes = analysis.meanAbsRes
        self.fitA = analysis.fitA
        self.fitB = analysis.fitB
        self.fitC = analysis.fitC
        self.dw_parabola = analysis.dw_parabola


        try: self.control = self.read_channel('OC M1 Excitation [AVG] (V)')
        except:  self.control = self.read_channel('OC M1 Excitation (V)')
        self.meanExcitation = np.mean(self.control)
        analysis.excitationLinearFit(self.x, self.control, axFit, axResiduals)
        self.dw_excitation = analysis.dw_excitation

        try: self.control = self.read_channel('OC D1 Amplitude [AVG] (m)')
        except:  self.control = self.read_channel('OC D1 Amplitude (m)')

        self.meanAmplitude = np.mean(self.control)
        analysis.excitationLinearFit(self.x, self.control, axFit, axResiduals)
        self.dw_amplitude = analysis.dw_excitation

        try: current = self.read_channel('Current [AVG] (A)')
        except:  current = self.read_channel('Current (A)')

        self.maxAbsCurrent = max(np.absolute(current))

        if plotCalculation == True:
            fig, axFit, axResiduals = analysis.PlotVContactCalculation(axFit, axResiduals)
            return fig, axFit, axResiduals


    def didv_analysis(self, peakType='Gaussian', centerVary=True, amplitudeGuess=None, centerGuess=None,
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
        self.x = self.read_channel('Bias calc (V)')

        if yChannel is not None:
            self.y = self.read_channel(yChannel)
            if 'Current' in yChannel:
                self.y = np.gradient(self.y, self.x)
                print('current channel provided for DIDV analysis so, first, took numerical derivative by y = np.gradient(y, x)')
        else:
            try: self.y = self.read_channel('LI Demod 1 Y [AVG] (A)')
            except: self.y = self.read_channel('LI Demod 1 Y (A)')




        # analyse
        mask = np.logical_and(self.x < -0.03, self.x > -0.08)
        self.y = self.y[mask]

        self.x = self.x[mask]


        if peakType == 'arctan':
            try:
                self.curr = self.read_channel('Current [AVG] (A)')
            except:
                self.curr = self.read_channel('Current (A)')

            self.curr = self.curr[mask]

            analysis = IVSpectrumAnalysis(self.x, self.curr)
            analysis.IVArctanFit(center_guess=centerGuess, area_guess=amplitudeGuess, sigma_guess=sigmaGuess,
                    center_vary=centerVary, area_vary=True, sigma_vary=True,
                    center_min_lim=centerMinLim, area_min_lim=amplitudeMinLim, sigma_min_lim=sigmaMinLim,
                    center_max_lim=centerMaxLim, area_max_lim=amplitudeMaxLim, sigma_max_lim=sigmaMaxLim,
                    background_intercept_guess=None, background_slope_guess=0,
                    background_intercept_vary=True, background_slope_vary=False,
                    background_intercept_min_lim=None, background_slope_min_lim=0,
                    background_intercept_max_lim=None, background_slope_max_lim=None)
            self.fitK = analysis.K
            self.iv_fit = analysis.iv_fit
            self.didv_fit = analysis.didv_fit
            self.didv = np.gradient(self.curr, self.x)


        elif peakType == 'Gaussian' or peakType == 'Lorenztian':
            analysis = DIDVSpectrumAnalysis(self.x, self.y)
            analysis.PeakFit(peakType=peakType, centerVary=centerVary, amplitudeGuess=amplitudeGuess, centerGuess=centerGuess,
                            sigmaGuess=sigmaGuess, amplitudeMaxLim=amplitudeMaxLim, centerMaxLim=centerMaxLim,
                            sigmaMaxLim=sigmaMaxLim, amplitudeMinLim=amplitudeMinLim, centerMinLim=centerMinLim,
                            sigmaMinLim=sigmaMinLim)




        # store useful variables
        self.fit = analysis.fit

        self.fitInfo = analysis.fitInfo
        self.fitArea = analysis.area
        self.fitHeight = analysis.height
        self.fitCentre = analysis.centre
        self.meanAbsRes = analysis.meanAbsRes
        self.fitDW = durbin_watson(analysis.fitInfo.residual)
        self.fitSigma = analysis.sigma
        self.fitSlope = analysis.slope
        self.fitIntercept = analysis.intercept
        try: current = self.read_channel('Current [AVG] (A)')
        except:
            print('Choice of channel ( Current [AVG] (A) ) not found, trying ( Current (A) )')
            current = self.read_channel('Current (A)')

        self.maxAbsCurrent = max(np.absolute(current))

        if plotCalculation == True:
            if peakType == 'arctan':
                fig, axDIDV, axIV, axResiduals = analysis.PlotIVArctanFit()
                return fig, axDIDV, axIV, axResiduals
            elif peakType == 'Gaussian' or peakType == 'Lorenztian':
                fig, axFit, axResiduals = analysis.PlotPeakFit(peakType=peakType)
                return fig, axFit, axResiduals


    def iz_analysis(self, yChannel=None, plotCalculation=False, bias=None):
        # read the x and y channels
        self.x = self.read_channel('Z rel (m)')

        if yChannel is not None:
            self.y = self.read_channel(yChannel)
        else:
            try:
                self.y = self.read_channel('Current [AVG] (A)')
            except:
                self.y = self.read_channel('Current (A)')

        #self.y = self.y[self.x<-75e-12]
        #self.x = self.x[self.x < -75e-12]

        analysis = IZSpenctrumAnalysis(self.x, self.y)
        analysis.calc_decay_lecgth()

        self.decay_length = analysis.decay_length
        self.maxAbsCurrent = max(np.absolute(self.y))
        self.meanAbsRes = analysis.meanAbsRes
        self.fit_intercept = analysis.fit_intercept
        self.log_curr = analysis.log_curr
        self.fit = analysis.fit


        if plotCalculation == True:
            fig, axFit, axResiduals = analysis.plot_linear_fit()
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
        self.x = self.read_channel('Index') [1:-1] # we might convert this to position or time
        self.y = self.read_channel('OC M1 Freq. Shift (Hz)') [1:-1]
        self.I = self.read_channel('Current (A)') [1:-1]


        try:
            i = 1
            while True:
                var_name, var_val = (self.meta_data.split(f'Comment{i:02}\t')[1].split('\n')[0][0:-1]).split('\t')
                setattr(self, var_name, var_val)
                i = i + 1
        except:
            pass

        z = self.read_channel('Z (m)') [1:-1]
        self.z_av = np.average(z)
        v = self.read_channel('Bias (V)') [1:-1]
        self.v_av = np.average(v)

        x = self.read_channel('X (m)')[1:-1]
        y = self.read_channel('Y (m)')[1:-1]

        x_av = np.average(x)
        y_av = np.average(y)
        xDash = x - x_av
        yDash = y - y_av
        r = np.sqrt((xDash) ** 2 + (yDash) ** 2)
        self.r = r

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


    def CalcCPDFromDDFDV(self):
        def Line(x, m, c):
            return m * x + c

        y = np.gradient(self.df, self.bias)
        x = self.bias

        model = lmfit.models.LinearModel()

        params = model.guess(y, x=x)

        params['intercept'].set(value=y.min(), vary=True)
        params['slope'].set(value=0, vary=True)

        # make the fit
        fitInfo = model.fit(y, x=x, params=params)

        slope = fitInfo.params['slope']
        intercept = fitInfo.params['intercept']

        vContact = intercept




        self.vContact = vContact
        return vContact


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
        self.fitB = b
        self.fitC = c

        dw = durbin_watson(fitInfo.residual)

        self.dw_parabola = dw

        return fit, fitInfo

    def excitationLinearFit(self, x_control, y_control, ax1, ax2):
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
        x, y = x_control, y_control

        model = lmfit.models.LinearModel()
        params = model.guess(y, x=x)

        params['intercept'].set(value=np.average(y), vary=True)
        params['slope'].set(value=0, vary=False)

        fitInfo = model.fit(y, x=x, params=params)

        fit = fitInfo.best_fit

        #self.meanAbsResControl = np.mean(np.absolute(fitInfo.residual))
        #c = fitInfo.params['intercept'].value
        dw_excitation = durbin_watson(fitInfo.residual)

        self.dw_excitation = dw_excitation


        """
        #fig, ax = plt.subplots(2, sharex=True)
        ax1.plot(x, y-c, 'k')
        ax1.plot(x, fit-c, c='tab:red', label='flat line fit')
        #ax1.plot(x, fitInfo.residual, c='gray')
        ax2.plot(x, np.absolute(fitInfo.residual), '.', c='gray')
        ax2.scatter(np.mean(x), self.meanAbsResControl, s=50, marker='*', c='tab:orange', zorder=2, label='mean abs res')
        ax2.set_title('DW stat: ' + str(round(dw,3)))
        ax1.grid()
        ax2.grid()
        print(lmfit.fit_report(fitInfo))

        """

    
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
                 self.fit+self.fitConfBand, color='red', alpha=0.2, label=r'confidence band, 2$\sigma$')
        axFit.plot(self.vContact, self.dfAtVContact, "*", color='orange', markersize=10, label = '$V_{Contact}$, ' + str(round(self.vContact, ndigits=2)) + r'V $\pm $ ' + str(round(self.vContactErr, ndigits=2)))
        
        
        axFit.set_ylabel(r'$\Delta$ f / Hz')
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
    
    def __init__(self, bias, didv):
        """
        @param bias: spectrum's x channel, i.e. bias
        @type bias: arr
        @param y: spectrum's y channel, i.e. a demod channel or the numerical derivative of a current channel
        @type y: arr
        """
        self.bias = bias
        self.didv = didv

    
    
    def PeakFit(self, peakType='Gaussian', centerVary=True,amplitudeGuess=None, centerGuess=None,
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
        paramsPeak['center'].set(value=centerGuess, vary=centerVary, max=centerMaxLim, min=centerMinLim)
        paramsPeak['sigma'].set(value=sigmaGuess, vary=True, max=sigmaMaxLim, min=sigmaMinLim)
        
        modelBackground = lmfit.models.LinearModel()
        paramsBackground = modelBackground.guess(y, x=x)
        
        paramsBackground['intercept'].set(value=y.min(), vary=True)
        paramsBackground['slope'].set(value=0, min=0, vary=True)
        
        model = modelPeak + modelBackground
        params = paramsPeak + paramsBackground
        
        # make the fit
        fitInfo = model.fit(y, x=x, params=params)
        
        fit = fitInfo.best_fit
        
        # calclate the fit's confidence band to 2 sigma, ie ~95%. fit +/- fitConfBand.
        fitConfBand = fitInfo.eval_uncertainty(params=fitInfo.params, sigma=2)

        self.fitConfBand = fitConfBand
        self.fit =  fit
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




    def PlotPeakFit(self, peakType='Gaussian', axFit=None, axResiduals=None):
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
                  self.fit+self.fitConfBand, color='red', alpha=0.2, label=r'2$\sigma$ conf band')
        
        axFit.plot(self.bias,
                   self.slope*self.bias+self.intercept,
                   '--', color='tab:blue', label = 'linear term')
        if peakType=='Gaussian':
            axFit.plot(self.bias,
                       self.area/(np.sqrt(2*np.pi)*self.sigma) * np.exp(-(self.bias-self.centre)**2/(2*self.sigma**2))
                       , '--', color='navy', label = 'peak term')
        elif peakType=='Lorentzian':
            axFit.plot(self.bias,
                       (self.area / np.pi) * (self.sigma/((self.bias - self.centre) ** 2 + self.sigma ** 2))
                       , '--', color='navy', label='peak term')
        
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



class IVSpectrumAnalysis():
    def __init__(self, bias, curr):
        """
        @param bias: spectrum's x channel, i.e. bias
        @type bias: arr
        @param y: spectrum's y channel, i.e. a demod channel or the numerical derivative of a current channel
        @type y: arr
        """
        self.bias = bias
        self.curr = curr
        self.didv = np.gradient(curr, bias)

    def IVArctanFit(self, center_guess=None, area_guess=None, sigma_guess=None,
                    center_vary=True, area_vary=True, sigma_vary=True,
                    center_min_lim=None, area_min_lim=None, sigma_min_lim=None,
                    center_max_lim=None, area_max_lim=None, sigma_max_lim=None,
                    background_intercept_guess=None, background_slope_guess=None,
                    background_intercept_vary=True, background_slope_vary=True,
                    background_intercept_min_lim=None, background_slope_min_lim=None,
                    background_intercept_max_lim=None, background_slope_max_lim=None):
        def arctan_func(x, center, area, sigma, slope, intercept, K):
            return area/np.pi * np.arctan((x-center) / sigma) + (0.5 * slope) * x ** 2 + intercept * x + K

        x = self.bias
        y = self.curr

        didv_smoothed = gaussian_filter1d(self.didv, sigma=3, truncate=4)
        didv_analysis = DIDVSpectrumAnalysis(self.bias[15:-15], didv_smoothed[15:-15])

        didv_analysis.PeakFit(peakType='Lorentzian', centerVary=center_vary,amplitudeGuess=area_guess, centerGuess=center_guess,
                    sigmaGuess=sigma_guess, amplitudeMaxLim=area_max_lim, centerMaxLim=center_max_lim,
                    sigmaMaxLim=sigma_max_lim, amplitudeMinLim=area_min_lim, centerMinLim=center_min_lim,
                    sigmaMinLim=sigma_min_lim)

        params = lmfit.Parameters()  # define a python dict to store the fitting parameters

        # define the fitting parameters and an initial guess for its value.
        # Here you can also define contraints, and other useful things.
        params.add('center',
                          value=didv_analysis.centre,
                          vary=center_vary,
                          max=center_max_lim,
                          min=center_min_lim)
        params.add('area',
                          value=didv_analysis.area,
                          vary=area_vary,
                          max=area_max_lim,
                          min=area_min_lim)
        params.add('sigma',
                          value=didv_analysis.sigma,
                          vary=sigma_vary,
                          max=sigma_max_lim,
                          min=sigma_min_lim)

        params.add('intercept',
                          value=didv_analysis.intercept,
                          vary=True,
                          max=background_intercept_max_lim,
                          min=background_intercept_min_lim)
        params.add('slope',
                          value=didv_analysis.slope,
                          vary=True,
                          max=background_slope_max_lim,
                          min=background_slope_min_lim)

        params.add('K',
                   value=0.0,
                   vary=True)

        model = lmfit.Model(arctan_func, independent_vars='x', param_names=['center', 'area', 'sigma', 'intercept','slope', 'K'])



        # make the fit
        fitInfo = model.fit(y, x=x, params=params)

        fit = fitInfo.best_fit

        # calclate the fit's confidence band to 2 sigma, ie ~95%. fit +/- fitConfBand.
        fitConfBand = fitInfo.eval_uncertainty(params=fitInfo.params, sigma=2)

        self.iv_fitConfBand = fitConfBand
        self.iv_fit = fit
        self.fit = fit
        self.didv_fit = np.gradient(fit, x)
        self.iv_fitInfo = fitInfo
        self.fitInfo = fitInfo
        self.area = fitInfo.params['area']
        self.height = fitInfo.params['area'] / (np.pi * fitInfo.params['sigma'])
        self.centre = fitInfo.params['center']
        self.sigma = fitInfo.params['sigma']
        self.slope = fitInfo.params['slope']
        self.intercept = fitInfo.params['intercept']
        self.K = fitInfo.params['K']
        self.realHeight = self.height + fitInfo.params['slope'] * fitInfo.params['center'] + fitInfo.params['intercept']
        self.meanAbsRes = np.mean(np.absolute(fitInfo.residual))

        return fit, fitInfo

    def PlotIVArctanFit(self, fig=None, axDIDV=None, axIVFit=None, axResiduals=None):
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
        if not hasattr(self, 'iv_fit'):
            raise ValueError('IVArctanFit needs to be called before attempting to plot its result')

        if fig==None and axDIDV==None and axIVFit==None and axResiduals==None:
            fig, [axDIDV, axIVFit, axResiduals] = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(7,6))
        elif fig!=None and axDIDV!=None and axIVFit!=None and axResiduals!=None:
            pass
        else:
            raise ValueError('either all fig, axDIDV, axIVFit, axResiduals are left undefined as None or they all are defined')

        axDIDV.plot(self.bias, self.didv, label='derivative of data', color='black')
        axDIDV.plot(self.bias, self.didv_fit, label='derivative of fit', color='red')

        axIVFit.plot(self.bias, self.curr, label='data', color='black')
        axIVFit.plot(self.bias, self.iv_fit, label='fit, arctan + parabolic model', color='red')
        axIVFit.fill_between(self.bias, self.iv_fit - self.iv_fitConfBand,
                           self.iv_fit + self.iv_fitConfBand, color='red', alpha=0.2, label=r'2$\sigma$ conf band')

        axIVFit.plot(self.bias,
                   0.5* self.slope * self.bias**2 + self.intercept * self.bias + self.K,
                   '--', color='tab:blue', label='parabolic term')

        axIVFit.plot(self.bias,
                   -1/np.pi * self.area * np.arctan((self.centre - self.bias) / self.sigma)
                   , '--', color='navy', label='arctan term')

        axDIDV.plot(self.bias,
                     self.slope * self.bias + self.intercept ,
                     '--', color='tab:blue', label='derivative of parabolic term')

        axDIDV.plot(self.bias,
                   (self.area / np.pi) * (self.sigma / ((self.bias - self.centre) ** 2 + self.sigma ** 2))
                   , '--', color='navy', label='derivative of arctan term')

        axDIDV.plot([self.centre-self.sigma, self.centre+self.sigma], [self.height/2,self.height/2], '-|', color='orange', label='fwhm = 2$\sigma$')
        axDIDV.plot(self.centre, self.realHeight, 'o', color='orange', label='arctan + linear derived height')
        axDIDV.plot(self.centre, self.height, "*", markersize=10, color='orange', label='arctan derived height')

        axDIDV.set_ylabel('dI/dV')
        axDIDV.legend(bbox_to_anchor=(1, 1))
        axDIDV.grid()

        axIVFit.set_ylabel('I (A)')
        axIVFit.legend(bbox_to_anchor=(1, 1))
        axIVFit.grid()

        axResiduals.plot(self.bias, self.iv_fitInfo.residual, '.', color='gray')
        axResiduals.set_ylabel('residuals (A)')
        axResiduals.set_xlabel('bias / V')
        axResiduals.grid()

        plt.tight_layout()
        plt.subplots_adjust(wspace=0, hspace=0)

        return fig, axDIDV, axIVFit, axResiduals

class IZSpenctrumAnalysis():

    def __init__(self, z, curr):

        self.z = z
        self.curr = curr

    def calc_decay_lecgth(self):
        x = self.z
        curr = np.abs(self.curr)

        #x = x[curr > 0.1e-12]
        #curr = curr[curr>0.1e-12]

        y = np.log(curr)

        self.x, self.y = x, y

        model = lmfit.models.LinearModel()
        params = model.guess(y, x=x)

        fitInfo = model.fit(y, x=x, params=params)

        fit = fitInfo.best_fit

        self.decay_length = -0.5 * fitInfo.params['slope']
        self.fit_intercept = fitInfo.params['intercept']
        self.meanAbsRes = np.mean(np.absolute(fitInfo.residual))
        self.log_curr = y
        self.z = x
        self.fit = fit
        self.fitInfo = fitInfo

        return self.decay_length

    def calc_work_function(self, bias):
        if not hasattr(self, 'decay_length'):
            self.calc_decay_lecgth()

        hbar = 1.05e-34
        e_mass = 9.12e-31
        e_charge = 1.6e-19

        potential = (self.decay_length * hbar)**2 / (2 * e_mass * e_charge)
        self.fermi_potential = potential + bias

        return self.fermi_potential


    def plot_linear_fit(self, axFit=None, axResiduals=None):
        # if the fit has not yet been calculated, calculate it with default inputs.
        if not hasattr(self, 'fit'): self.calc_decay_lecgth()

        if axFit == None and axResiduals == None:
            fig, [axFit, axResiduals] = plt.subplots(figsize=(3,4), nrows=2, ncols=1, sharex=True)
        elif axFit == None and axResiduals != None:
            fig, axFit = plt.subplots()
        elif axFit != None and axResiduals == None:
            fig, axResiduals = plt.subplots()

        axFit.plot(self.x, self.y, label='data', color='black')
        axFit.plot(self.x, self.fit, label=' linear fit', color='red')

        axFit.set_ylabel('log Current (lnA)')
        #axFit.legend(bbox_to_anchor=(1, 1))
        axFit.grid()

        axResiduals.plot(self.x, self.fitInfo.residual, '.', color='gray')
        axResiduals.set_ylabel('residuals (A)')
        axResiduals.set_xlabel('z rel (m)')
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
# ======================================================================================================================
# Examples :)
# ======================================================================================================================

if __name__ == "__main__":

    path = r'C:\Users\Usuario\OneDrive - The University of Nottingham\PhD\data\2025-07-17'

    fileName = 'dimer_4_map_didv_00365.dat'
    s = Spectrum(path, fileName)
    curr = s.read_channel('Current [AVG] (A)')

    s.didv_analysis(yChannel='Current [AVG] (A)', peakType='arctan', centerGuess=-0.06, plotCalculation=True)
    plt.show()


    path = r'C:\Users\Usuario\OneDrive - The University of Nottingham\PhD\data\2025-05-16'

    fileName = '3_1_map_iz_1_00001.dat'
    s = Spectrum(path, fileName)
    s.iz_analysis(bias = -60e-3, plotCalculation=True)
    plt.show()

    path = r'C:\Users\Usuario\OneDrive - The University of Nottingham\PhD\data\2024-07-16'

    fileName = 'dfVMap_InSb_69_00001.dat'
    s = Spectrum(path, fileName)
    s.kpfm_analysis(plotCalculation=True)

    print(lmfit.fit_report(s.fitInfo)) # lmfit's fit report

    bias = s.read_channel('Bias calc (V)')

    yChannels = ['Current [AVG] (A)', 'OC M1 Excitation [AVG] (V)', 'OC D1 Amplitude [AVG] (m)']

    fig, ax = plt.subplots(len(yChannels),1, figsize=(4,6), sharex=True)

    for ax_i, chan, in zip(ax, yChannels):
        y = s.read_channel(chan)

        ax_i.plot(bias, y, 'k.')

        ax_i.set_xlabel('bias (V)')
        ax_i.set_ylabel(chan)
        ax_i.grid()

    plt.tight_layout()
    plt.show()


    path = r'C:\Users\Usuario\OneDrive - The University of Nottingham\PhD\data\2024-07-12-2'

    fileName = 'dfVMap_InSb_63_00001.dat'
    s = Spectrum(path, fileName)

    s.kpfm_analysis(plotCalculation=False, axFit=ax[0,1], axResiduals=ax[1,1])
    #print(lmfit.fit_report(s.fitInfo))


    ax[0,0].set_ylabel('Excitation (V), rigidly shifted so 0 @ fit')
    ax[1,0].set_ylabel('Abs Res (V)')
    ax[1,0].set_xlabel('Bias (V)')
    ax[1,1].set_xlabel('Bias (V)')
    handles, labels = ax[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')
    handles, labels = ax[1,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right')
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.2, hspace=0.1)
    plt.show()

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



    # didvfit: the analytically differentiated fermi function as an array
    # Vupscale: the upscaled bias range to plot the fit. has 1000pts as default
    # h: the peak height
    # centre: peak centre
    # fwhm: the full width half maximum as defined by the name
    # residuals: difference between IV fermi fit and IV data
    # final return from the function is the raw differentiated data using np.gradient


    path = r'C:\Users\Usuario\OneDrive - The University of Nottingham\PhD\data\test_manip'
    name = r'manip_13_32_042.dat'
    s = Spectrum(path, name)
    s.ForceAnalysis(threshold=5e-11, plotCalculation=True)
    plt.show()
