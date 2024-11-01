from spectrum_analysis import Spectrum
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import math
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import lmfit
from file_reader import MyFiles


    
class SpectraSet():
    
    def __init__(self, path, fileNames, xDriftCorrected=None, yDriftCorrected=None):
        """
        A list of Spectrum instances.
        Encapsulates a dataset, to be analysed / visualised together.
        @param path: path to spectra files
        @type path: str
        @param fileNames: spectra file names
        @type fileNames: list of str
        @param xDriftCorrected: x coordinates of spectra, compensated for drift. To use for analysis / plotting
        in place of the xy pos in the spectra's metadata.
        @type xDriftCorrected: arr, optional
        @param yDriftCorrected: y coordinates of spectra, compensated for drift. To use for analysis / plotting
        in place of the xy pos in the spectra's metadata.
        @type yDriftCorrected: arr, optional
        Note: The list's order is set by the order of fileNames provided.
        Every object in this class will follow this order.
        """
        spectraData = []

        for file in fileNames:
            # create a Spectrum instance for each spectra file
            s = Spectrum(path, file)

            # store each Spectrum instance in a list
            spectraData.append(s)

        self.spectraData = spectraData
        self.fileNames = fileNames

        self.xComp = xDriftCorrected
        self.yComp = yDriftCorrected




    def SliceAttribute(self, attribute):
        """
        List of the sliced attribute out of each Spectrum instance.
        @param attribute: attribute to be sliced
        @type attribute: str
        @return: sliced attribute, in order of self.fileNames
        @rtype:list
        """
        return [getattr(s, attribute) for s in self.spectraData]



    def ReadChannel(self, channel):
        """
        A list of the channel's data from each Spectrum instance.
        Note: The channel's data is, itself, a list. So we form a list of lists.
        Note: The list's order is set by the order of fileNames provided.
        @param channel: channel name
        @type channel: str
        If the channel is not found, the available channels are
        printed. So, if you're not sure of the exact channel name, just
        type in nonsense.
        Note: channel = 'Index' is an option. May seem redundant, but may
        be useful in future to convert to a time channel, if we make
        note of sampling freq. TCP receiver sampling is limited to
        20 kHz. So, if measurement made using data logger through
        Matt's python_interface_nanonis.py, default is 20 kHz.
        Note: we may be able to play with TCP receiver to lower the 20kHz limit.
        @return: the channel's data, of each Spectrum instance.
        @rtype: list of lists
        """
        #for s in self.spectraData:
            #s.y = s.ReadChannel(channel)

        return [s.ReadChannel(channel) for s in self.spectraData]



    # ==================================================================================================================
    # Individual Analysis
    # ==================================================================================================================

    def KPFMAnalysis(self, xAdatomCentre=None, yAdatomCentre=None,
                     yChannel='OC M1 Freq. Shift [AVG] (Hz)'):
        """
        Runs the KPFM analysis on each Spectrum instance.
        See Spectrum.KPFMAnalysis in spectrum_analysis.py for details.
        @param xAdatomCentre: x coordinate of the artificial atom's centre.
        @type xAdatomCentre: float, optional
        @param yAdatomCentre: y coordinate of the artificial atom's centre.
        @type yAdatomCentre: float, optional
        If xAdatomCentre and yAdatomCentre are specified, each spectrum's distance
        from the artificial atom's centre will be calculated and added as Spectrum.r (float).
        @param yChannel: Use if you want to analyse only one of the repeat sweeps e.g.
        'OC M1 Freq. Shift [00002] (Hz)'. Otherwise, the default is 'OC M1 Freq. Shift [AVG] (Hz)' or, in its absence,
        'OC M1 Freq. Shift (Hz)'.
        @type yChannel: str
        @return: The useful info from the analysis is added as attributes to each Spectrum instance.
        See Spectrum.KPFMAnalysis in spectrum_analysis.py for details.
        (Remember self is a list of Spectrum instances)
        """

        for s in self.spectraData:
            s.KPFMAnalysis(xAdatomCentre, yAdatomCentre, plotCalculation=False,
                         yChannel=yChannel) # plotCalculation=False bc. it's best to use the self.PlotStacked method.


            


    def DIDVAnalysis(self, peakType='Gaussian', amplitudeGuess=None, centerGuess=None,
                    sigmaGuess=None, amplitudeMaxLim=None, centerMaxLim=None,
                    sigmaMaxLim=None, amplitudeMinLim=None, centerMinLim=None,
                    sigmaMinLim=None, yChannel='LI Demod 1 Y [AVG] (A)'):
        """
        Runs the dI/dV analysis on each Spectrum instance.
        See Spectrum.DIDVAnalysis in spectrum_analysis.py for details.
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
        @param yChannel: specify the y channel, Otherwise, the default is 'LI Demod 1 Y [AVG] (A)' or, in its absence,
        'LI Demod 1 Y (A)'.
        @return: The useful info from the analysis is added as attributes to each Spectrum instance.
        See Spectrum.DIDVAnalysis in spectrum_analysis.py for details.
        (Remember self is a list of Spectrum instances)
        """
        for s in self.spectraData:
            s.DIDVAnalysis(peakType, amplitudeGuess, centerGuess,
                            sigmaGuess, amplitudeMaxLim, centerMaxLim,
                            sigmaMaxLim, amplitudeMinLim, centerMinLim,
                            sigmaMinLim, plotCalculation = False, yChannel=yChannel)
                            # plotCalculation=False bc. it's best to use the self.PlotStacked method.



    def ForceAnalysis(self):
        """
        work in progress
        """
        for s in self.spectraData:
            s.ForceAnalysis(plotCalculation=False) # plotCalculation=False bc. it's best to use the self.PlotStacked method.


    
     
    # ==================================================================================================================
    # Group Analysis
    # ==================================================================================================================

    def KPFM2DFit(self, r, vContact, DGuess=0, CGuess=-1, r0Guess=1):
        def OurModel(r, D, C, r0):
            return D + (C) / (r - r0)
        
        fitParams = lmfit.Parameters() # define a dict to store the fitting parameters
        fitParams.add('D', value=DGuess) # define the fitting parameters and an initial guess for their value
        fitParams.add('C', value=CGuess) 
        fitParams.add('r0', value=r0Guess)
        
        # specify which variables are independent (r) and which are fitting parameters (D, C, r0) in the model function (self._Vcontact_model)
        model = lmfit.Model(OurModel, independent_vars='r', param_names=['D','C','r0'])
        
        # perform the fit
        fitInfo =model.fit(data=vContact, params=fitParams, r=r)
        fit = fitInfo.best_fit
        
        # calculate the fit's confidence bound to 2 sigma, ie ~95%
        fitConfBound = fitInfo.eval_uncertainty(params=fitInfo.params, sigma=2)
        
        return fit, fitInfo, fitConfBound



    # ==================================================================================================================
    # General Visualisation Methods
    # ==================================================================================================================


    # def PlotStacked(self):
        
      
    def ScatterMap(self, z, x='x_pos', y='y_pos', theta=0, ax=None, fig=None, 
                   cmap='seismic', cmapScale = 'linear', colourbarLabel = None):            
        
        def rotate(x, y, xPivot=0, yPivot=0, rot=0):
            rot = -np.deg2rad(rot)
            
            x = x - xPivot
            y = y - yPivot
            
            # carteesian -> polar
            r = np.sqrt(x**2 + y**2)
            theta = np.arctan2(y, x) # element-wise arctan
            
            # rotation
            theta = theta + rot
            
            # polar -> carteesian
            x = (r*np.cos(theta)) + xPivot
            y = (r*np.sin(theta)) + yPivot
    
            return x, y
        
        
        if type(z) is str: 
            z = self.SliceAttribute(z)
            if colourbarLabel is None: colourbarLabel = z
        
        if type(x) is str: 
            x = self.SliceAttribute(x)
        if type(y) is str: 
            y = self.SliceAttribute(y)
        
        
        if ax is None: fig, ax = plt.subplots()
        
        x, y = rotate(x, y, rot=-theta, xPivot = np.average(x), yPivot=np.average(y) )
            
        if cmapScale == 'log': cmapScale = colors.LogNorm()
        if cmapScale == 'linear': cmapScale =colors.Normalize()
        plot = ax.scatter(x, y, c=np.array(z), cmap=cmap, norm=cmapScale,
                          marker="s", linewidths=3)
        
        fig.colorbar(plot, label=colourbarLabel)
        
    
        ax.set_ylabel('y / m')
        ax.set_xlabel('x / m')
        
        plt.show()



    def Map(self, z, x, y, masks=None, theta=0, ax=None, fig=None,
            cmap='spring', colourbarLabel=None, cmapScale='linear'):

        if type(z) is str: z = self.SliceAttribute(z)
        map = DriftCorrectedMap(z, x, y, angle=theta, xPivot=None, yPivot=None, mask=masks)
        map.Plot(ax, cmapScale, cmap, colourbarLabel, scalebar=3e-9)



    # ==================================================================================================================
    # Specific Visualisation Methods
    # ==================================================================================================================

    def Plot2D(self, y, xGrid, yGrid, xCentre=None, yCentre=None,
               yLabel=None, fig=None, ax=None, rMin=-np.inf,
               rMax=np.inf, thetaMin=-np.inf, thetaMax=np.inf, color='black',
               fit=False):

        def CalcR(xSpectrumPos, ySpectrumPos, xAdatomCentre, yAdatomCentre):
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
            r = 1e9 * np.sqrt((xDash) ** 2 + (yDash) ** 2)
            theta = np.arctan2(yDash, xDash)  # element-wise arctan
            theta = np.rad2deg(theta)

            return r, theta

        if type(y) is str:
            y = self.SliceAttribute(y)
            if yLabel is None: yLabel = y

        if xCentre is None: xCentre = np.average(xGrid)
        if yCentre is None: yCentre = np.average(yGrid)
        r, theta = CalcR(xGrid, yGrid, xCentre, yCentre)

        r = np.array(r)
        y = np.array(y)

        if -180 > thetaMin > 180: raise ValueError('-180 > thetaMin > 180')
        if -180 > thetaMax > 180: raise ValueError('-180 > thetaMax > 180')

        condition = (r >= rMin) & (r <= rMax) & (theta >= thetaMin) & (theta <= thetaMax)

        yDash = y[condition]
        rDash = r[condition]

        # order arrays for nice plotting
        orderedIdx = np.argsort(rDash)
        yDash = yDash[orderedIdx]
        rDash = rDash[orderedIdx]

        if ax is None: fig, ax = plt.subplots()
        ax.plot(rDash, yDash, '.', c=color)

        if fit == True:
            # fit
            fit, fitInfo, fitConfBound = self.KPFM2DFit(r=rDash, vContact=yDash)

            rFit = np.linspace(min(rDash), max(rDash), len(rDash))
            ax.plot(rDash, fit, color=color, label=r'$D+\frac{C}{r-r_0}$ fit')
            # ax.fill_between(rDash, fit-fitConfBound,
            #          fit+fitConfBound, color=color, alpha=0.15, label=r'$2\sigma$ conf. interval')

            ax.legend(fontsize=7, loc='lower right')

        ax.set_xlabel('r (nm)', fontsize='x-large')
        ax.set_ylabel(yLabel, fontsize='x-large')

        fig, ax = plt.subplots()
        ax.scatter(xGrid, yGrid, c=color, alpha=0.2)
        xGrid = xGrid[condition]
        yGrid = yGrid[condition]
        ax.scatter(xGrid, yGrid, c=color)
        ax.set_axis_off()




class PlotStacked():

    def __init__(self):
        print('')

    def TuneCmap(self, ax, cmap='gist_rainbow', minVal=None, maxVal=None,
                 nDivisions=None, cmapScale='linear',
                 colourbarLabel=None, plotCBar=False, location='right',
                 shrink=0, axForCBarPlot=None):

        if nDivisions is None: Ndivisions = len(self.spectraData)
        if minVal is None: minVal = 0
        if maxVal is None: maxVal = Ndivisions

        if cmapScale == 'linear':
            scale = np.linspace(minVal, maxVal, nDivisions)
        if cmapScale == 'log':
            scale = np.logspace(minVal, maxVal, nDivisions)

        dummy_xyz = [minVal, maxVal]
        dummie_cax = ax.scatter(dummy_xyz, dummy_xyz, dummy_xyz,
                                cmap=cmap, norm=cmapScale)
        ax.cla()
        cmap = plt.get_cmap(cmap)

        if plotCBar == True:
            plt.colorbar(dummie_cax, label=colourbarLabel, location=location,
                         shrink=shrink, ax=axForCBarPlot)

        return cmap, scale


    def PlotStacked(self, x='x', y='y', ncols=1, ax=None, alpha=1, xLabel='',
                    yLabel='', color='black', cmap=None, marker='-'):

        # form / format axes
        if ax is None:
            spectraPerAx = math.ceil(len(self.spectraData) / ncols)
            nrows = spectraPerAx
            fig, ax = plt.subplots(nrows, ncols, sharex=True, sharey=True,
                                   frameon=False, figsize=(24, 9))
            for col in range(ncols):
                for row in range(nrows):
                    ax[row, col].axis('off')
        else:
            if type(ax) is np.ndarray:
                nrows, ncols = np.shape(ax)
            else:
                ncols = 1

        if ncols == 1: ax = [[ax]]

        # form colour range
        if cmap is not None:
            cmap = self.TuneCmap(ax[0, 0], cmap=cmap, Ndivisions=len(self.spectraData),
                                 colourbarLabel='spectrum number',
                                 plotCBar=True, axForCBarPlot=ax[:, ncols - 1])

        # slice data, if needed
        if type(x) is str: x = self.SliceAttribute(x)
        if type(y) is str: y = self.SliceAttribute(y)

        # plot
        spectraCount = 0

        for col in range(ncols):
            for row in range(nrows):
                if spectraCount < len(self.spectraData):

                    if cmap is not None: color = cmap(spectraCount)
                    ax[row, col].plot(x[spectraCount], y[spectraCount],
                                      color=color,
                                      alpha=alpha)

                    spectraCount = spectraCount + 1
                    print(spectraCount)

        plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, hspace=0)
        ax[0, 0].set_ylabel(yLabel)
        ax[0, 0].set_xlabel(xLabel)

        return ax

    def PlotSpectraPos(self, x='x_pos', y='y_pos', fig=None, ax=None):
        if ax is None: fig, ax = plt.subplots(figsize=(10, 10))
        x = self.SliceAttribute(x)
        y = self.SliceAttribute(y)
        ax.scatter(x, y, color='white')
        for i in range(len(x)):
            ax.annotate(i, (x[i], y[i]), fontsize='xx-small')

        ax.set_ylabel('y / m')
        ax.set_xlabel('x / m')


class DriftCorrectedMap():

    def __init__(self, z, x, y, angle=0, xPivot=None, yPivot=None, mask=None):
        if angle != 0:
            if xPivot is None: xPivot = np.average(x)
            if yPivot is None: yPivot = np.average(y)
            x, y = self.Rotate(x, y, xPivot, yPivot, -angle) # note: - angle bc. we want to rotate back to 0 deg.

        self.z = np.array(z)
        self.x = np.array(x)
        self.y = np.array(y)

        if mask is not None:
            mask = np.array(mask)
        self.mask = mask




    def Rotate(self, x, y, xPivot=0, yPivot=0, rot=0):
        rot = -np.deg2rad(rot)

        x = x - xPivot
        y = y - yPivot

        # carteesian -> polar
        r = np.sqrt(x ** 2 + y ** 2)
        theta = np.arctan2(y, x)  # element-wise arctan

        # rotation
        theta = theta + rot

        # polar -> carteesian
        x = (r * np.cos(theta)) + xPivot
        y = (r * np.sin(theta)) + yPivot

        return x, y



    def CoordTransform(self, z, x, y):
        # remove duplicates
        x, y = x.round(decimals=14), y.round(decimals=14)
        xDash = list(set(x))
        yDash = list(set(y))

        xDash.sort()
        yDash.sort()

        zDash = np.empty((len(xDash), len(yDash)))
        zDash.fill(np.nan)

        for s in range(len(z)):
            i = xDash.index(x[s])
            j = yDash.index(y[s])
            zDash[i, j] = z[s]

        zDash = np.array(zDash).T
        xDash, yDash = np.array(xDash), np.array(yDash)

        return zDash, xDash, yDash



    def DrawOutline(self, ax, z, x, y):

        z = np.isnan(z)  # bool array, if True at nan, false otherwise

        # upsample using Kronecker product so outline follows square px edges
        z = np.kron(z, np.ones((100, 100), dtype=bool))

        # edge border
        z[0, :], z[-1, :], z[:, 0], z[:, -1] = True, True, True, True

        # upsample x
        xStep = x[1] - x[0]
        x = np.linspace(min(x) - xStep / 2, max(x) + xStep / 2, num=len(x) * 100)
        # upsample y
        yStep = y[1] - y[0]
        y = np.linspace(min(y) - yStep / 2, max(y) + yStep / 2, num=len(y) * 100)

        ax.contour(x, y, z, linewidths=3, colors='black', zorder=-1) # zorder = -1 to send to the very back



    def Plot(self, ax=None, cmapScale='linear', cmap='spring', colourbarLabel=None, scalebar=None):
        if ax is None: fig, ax = plt.subplots()

        if cmapScale == 'log': cmapScale = colors.LogNorm(vmin=np.min(self.z),
                                                          vmax=np.max(self.z))
        if cmapScale == 'linear': cmapScale = colors.Normalize(vmin=np.min(self.z),
                                                               vmax=np.max(self.z))

        if self.mask is None:
            z, x, y = self.CoordTransform(self.z, self.x, self.y)

            plot = ax.pcolormesh(x, y, z, norm=cmapScale, cmap=cmap)
            self.DrawOutline(ax, z, x, y)

        else:
            for i in set(self.mask):
                m = np.isin(self.mask, i)
                zm, xm, ym = self.z[m], self.x[m], self.y[m]

                zm, xm, ym = self.CoordTransform(zm, xm, ym)

                plot = ax.pcolormesh(xm, ym, zm, norm=cmapScale, cmap=cmap)
                self.DrawOutline(ax, zm, xm, ym)

        fig.colorbar(plot, label=colourbarLabel)

        if scalebar is not None:
            scalebar = AnchoredSizeBar(ax.transData,
                                       scalebar, str(scalebar)+' m', 'lower right',
                                       frameon=False)
            ax.add_artist(scalebar)

        ax.set_axis_off()
        ax.use_sticky_edges = False
        ax.margins(0.05)
        ax.set_aspect('equal')




if __name__ == '__main__':

    path = r'data\test_data'

    # %%
    # =============================================================================
    # dI/dV data
    # =============================================================================

    fileNames = MyFiles().DirFilter(path, baseName='dfVMap_InSb_20_')

    xGrid = np.load(path + r'\xGrid_20.npy')
    yGrid = np.load(path + r'\yGrid_20.npy')

    theta = 2.4

    # %% Analysis
    spectra = SpectraSet(path, fileNames)

    spectra.DIDVAnalysis(centerGuess=-0.09, sigmaGuess=0.01, amplitudeMaxLim=1e-11, amplitudeMinLim=0)

    # %% 2D plot
    xCentre = np.average(xGrid)
    yCentre = np.average(yGrid)

    spectra.Plot2D('fitHeight', xGrid, yGrid, xCentre, yCentre,
                yLabel='dI/dV fitted peak height')

    # %% stacked plot
    #ax = spectra.PlotStacked(ncols=20, legendLabel='data')
    #spectra.PlotStacked(y='fit', ax=ax, color='red', legendLabel='fit')

    # %% real spasitions of spectra (to see drift)
    spectra.PlotSpectraPos()

    # %% Maps
    spectra.Map('meanAbsRes', xGrid, yGrid, theta=theta, colourbarLabel='mean absolute residuals',
                cmapScale='linear', cmap='bone_r')

    spectra.Map('fitCentre', xGrid, yGrid, theta=theta, colourbarLabel='fitted peak centre',
                cmapScale='linear', cmap='bone_r')

    spectra.Map('fitArea', xGrid, yGrid, theta=theta, colourbarLabel='dI/dV fitted peak area',
                cmapScale='log', cmap='BuPu')

    spectra.Map('fitHeight', xGrid, yGrid, theta=theta, colourbarLabel='dI/dV fitted peak height',
                cmapScale='log', cmap='BuPu')

    # %% Scatter Maps to check that maps are done correctly

    spectra.ScatterMap('meanAbsRes', xGrid, yGrid, colourbarLabel='mean abs(dI/dV fitted peak residuals)',
                    cmapScale='linear', cmap='bone_r')

    spectra.ScatterMap('fitArea', xGrid, yGrid, colourbarLabel='dI/dV fitted peak area',
                    cmapScale='log', cmap='BuPu')

    spectra.ScatterMap('fitHeight', xGrid, yGrid, colourbarLabel='dI/dV fitted peak height',
                    cmapScale='log', cmap='BuPu')

    # %%
    # =============================================================================
    # df(V) data
    # =============================================================================

    fileNames = MyFiles().DirFilter(path, baseName='dfVMap_InSb_40_')

    xGrid = np.load(path + r'\xGrid_40.npy')
    yGrid = np.load(path + r'\yGrid_40.npy')

    masks = np.load(path + r'\40masks.npy')

    theta = 1.6

    # %% Analysis
    spectra = SpectraSet(path, fileNames)

    spectra.KPFMAnalysis()

    # %% 2D plot + 1/r fit, excluding data inside ring
    xCentre = np.average(xGrid)
    yCentre = np.average(yGrid)

    fig, ax = plt.subplots()
    spectra.Plot2D('vContact', xGrid, yGrid, xCentre, yCentre,
                yLabel='Contact Potential (V)', rMin=2, fit=True)

    # %% 2D plot + 1/r fit, for different angle slices
    fig, ax = plt.subplots()
    spectra.Plot2D('vContact', xGrid, yGrid, xCentre=xCentre, yCentre=yCentre,
                yLabel='Contact Potential (V)', rMin=2,
                thetaMin=25, thetaMax=110, color='red', fig=fig, ax=ax, fit=True)

    spectra.Plot2D('vContact', xGrid, yGrid, xCentre=xCentre, yCentre=yCentre,
                yLabel='Contact Potential (V)', rMin=2,
                thetaMin=-155, thetaMax=-70, color='blue', ax=ax, fig=fig, fit=True)

    # %% stacked plot
    #ax = spectra.PlotStacked(ncols=20)
    #spectra.PlotStacked(y='fit', ax=ax, color='red')

    # %% real spectra positions (to see drift)
    spectra.PlotSpectraPos()

    # %% Maps

    spectra.Map('vContact', xGrid, yGrid, masks=masks, theta=theta, colourbarLabel='Contact Potential (V)',
                cmapScale='linear', cmap='inferno')

    spectra.Map('meanAbsRes', xGrid, yGrid, masks=masks, theta=theta, colourbarLabel='mean absolute residuals (V)',
                cmapScale='linear', cmap='bone_r')

    spectra.Map('fitA', xGrid, yGrid, masks=masks, theta=theta, colourbarLabel='a fit param for $ax^2+bx+c$',
                cmapScale='linear', cmap='bone_r')

    # %% Scatter Maps to check maps OK
    spectra.ScatterMap('vContact', xGrid, yGrid, colourbarLabel='Contact Potential Diff',
                    cmapScale='linear', cmap='inferno')

    spectra.ScatterMap('vContact', xGrid, yGrid, colourbarLabel='Contact Potential Diff',
                    cmapScale='linear', cmap='inferno')

    spectra.ScatterMap('meanAbsRes', xGrid, yGrid, colourbarLabel='mean absolute residuals (V)',
                    cmapScale='linear', cmap='bone_r')

    # %%
    # =============================================================================
    # I/V data (to make current maps for both dI/dV and df(V) datasets)
    # =============================================================================

    # %%

    spectra = SpectraSet(path, fileNames)

    current = spectra.ReadChannel('Current [AVG] (A)')

    currentMax = [max(c) for c in current]

    # %% Map
    spectra.Map(currentMax, xGrid, yGrid, masks=masks, theta=theta, colourbarLabel='max current (A)',
                cmapScale='log', cmap='bone_r')

    # %% Scatter
    spectra.ScatterMap(currentMax, xGrid, yGrid, colourbarLabel='max current (A)',
                    cmapScale='log', cmap='bone_r')
    plt.show()