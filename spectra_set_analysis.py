from spectrum_analysis import SpectrumAnalysis
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from file_reader import MyFiles
from file_reader import Sxm
import matplotlib.colors as colors
from matplotlib.widgets import Slider
import math
from matplotlib.collections import LineCollection
from scipy.signal import convolve2d
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import lmfit


    
    
class SpectraSetAnalysis():
    
    def __init__(self, channel, path, fileNames):

            spectraSet = []
            
            for file in fileNames:
                # create a Spectrum instace for each spectra file
                s = SpectrumAnalysis(channel, path, file)
             
                # store each SpectrumAnalysis instance in a list
                spectraSet.append(s)
            
            self.fileNames = fileNames
            self.spectraSet = spectraSet
            
        
        
    def KPFMAnalysis(self, xAdatomCentre=None, yAdatomCentre=None, 
                 plotCalculation=False, axFit=None, axResiduals=None):
    
        for s in self.spectraSet:
            s.KPFMAnalysis(xAdatomCentre, yAdatomCentre, 
                         plotCalculation, axFit, axResiduals)
            
            
            
    def DIDVAnalysis(self, peakType='Gaussian', amplitudeGuess=None, centerGuess=None,
                    sigmaGuess=None, amplitudeMaxLim=None, centerMaxLim=None,
                    sigmaMaxLim=None, amplitudeMinLim=None, centerMinLim=None,
                    sigmaMinLim=None, plotCalculation = False):
        
        for s in self.spectraSet:
            s.DIDVAnalysis(peakType, amplitudeGuess, centerGuess,
                            sigmaGuess, amplitudeMaxLim, centerMaxLim,
                            sigmaMaxLim, amplitudeMinLim, centerMinLim,
                            sigmaMinLim, plotCalculation)


    
    def SliceAttribute(self, attribute):
        return [getattr(s, attribute) for s in self.spectraSet]
    
     
    # =========================================================================
    # Potential landscape in 2D fit 
    # =========================================================================
        
    def _Vcontact_model(self, r, D, C, r0):
        return D + (C)/(r-r0)
    
    
    
    def _Vcontact_model_residual(self, params, r, data):
        D, C, r0 = params['D'], params['C'], params['r0'] 
        model = D + (C)/(r-r0)
        return (data - model)


    def _SliceDataToFit(self, attribute, filesToFit):
        
        data = [getattr(self.spectraSet[spectrumNum], attribute) for spectrumNum 
                in range(len(self.spectraSet))]
        dataInFit = [data[self.fileNames.index(f)] for f in filesToFit]
        
        return dataInFit


    def VLandscape2DFit(self, r, vContact, DGuess=0, CGuess=-1, r0Guess=1):
        
        fitParams = lmfit.Parameters() # define a dict to store the fitting parameters
        fitParams.add('D', value=DGuess) # define the fitting parameters and an initial guess for their value
        fitParams.add('C', value=CGuess) 
        fitParams.add('r0', value=r0Guess)
        
        # specify which variables are independent (r) and which are fitting parameters (D, C, r0) in the model function (self._Vcontact_model)
        model = lmfit.Model(self._Vcontact_model, independent_vars='r', param_names=['D','C','r0'])
        
        # perform the fit
        fitInfo =model.fit(data=vContact, params=fitParams, r=r)
        fit = fitInfo.best_fit
        
        # calclate the fit's confidence bound to 2 sigma, ie ~95%
        fitConfBound = fitInfo.eval_uncertainty(params=fitInfo.params, sigma=2)
        
        return  fit, fitInfo, fitConfBound   
    
    # =========================================================================
    # Data visualisation 
    # =========================================================================
    
    def TuneCmap(self, ax, cmap='gist_rainbow', minVal=None, maxVal=None, 
                 nDivisions=None, cmapScale='linear', 
                 colourbarLabel=None, plotCBar=False, location='right',
                 shrink=0, axForCBarPlot=None):
            
            if nDivisions is None: Ndivisions = len(self.spectraSet)
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
                plt.colorbar(dummie_cax, label = colourbarLabel, location=location,
                                              shrink=shrink, ax=axForCBarPlot)
                
            return cmap, scale
        
        
        
    def PlotStacked(self, x='x', y='y', ncols=1, ax=None, alpha=1, xLabel='',
                    yLabel='', color='black', cmap=None, legendLabel='', marker='-'):
            
        if ax is None:
            spectraPerAx = math.ceil(len(self.spectraSet) / ncols)
            nrows = spectraPerAx
            fig, ax = plt.subplots(nrows, ncols, sharex=True, sharey=True,
                                   frameon=False, figsize=(24,9))
            for col in range(ncols):          
                for row in range(nrows):
                    ax[row, col].axis('off')
        else: 
            if type(ax) is np.ndarray: 
                nrows, ncols = np.shape(ax)
            else: ncols = 1
        
        if ncols == 1: ax = [[ax]]
        
        if cmap is not None:
            cmap = self.TuneCmap(ax[0,0], cmap=cmap, Ndivisions=len(self.spectraSet), 
                                    colourbarLabel='spectrum number', 
                                    plotCBar=True, axForCBarPlot=ax[:,ncols-1])
        
        if type(x) is str: x = self.SliceAttribute(x)
        if type(y) is str: y = self.SliceAttribute(y)
    
        spectraCount = 0
       
        for col in range(ncols):          
            for row in range(nrows):
                if spectraCount < len(self.spectraSet):
                    
                    if cmap is not None: color = cmap(spectraCount)
                    ax[row, col].plot(x[spectraCount], y[spectraCount],
                                      color=color,
                                      alpha=alpha)
                    
                    spectraCount = spectraCount + 1
                    print(spectraCount)

    
        plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, hspace=0)
        ax[0,0].set_ylabel(yLabel)
        ax[0,0].set_xlabel(xLabel)
   
        return ax
        
        
    def Map(self, z, x, y, masks=None, theta=0, ax=None, fig=None,
            cmap='spring', colourbarLabel = None, cmapScale='linear'):
                  
       
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
            
            x, y = x.round(decimals=14), y.round(decimals=14)
            
            return x, y
        
        
        def coordTransform(z, x, y):
            
            
            
            xDash = list(set(x)) # remove duplicates
            yDash = list(set(y))
  
            xDash.sort()
            yDash.sort()

            zDash = np.empty((len(xDash), len(yDash)))
            zDash.fill(np.nan)

            for s in range(len(z)):
                i = xDash.index(x[s])
                j = yDash.index(y[s])
                zDash[i,j] = z[s]
                
            zDash = np.array(zDash).T 
            xDash, yDash = np.array(xDash), np.array(yDash)
            return zDash, xDash, yDash
        
        
        def outline(ax, z, x, y):
            ax.set_xlim(min(x) - 0.05*(max(x)-min(x)), max(x) + 0.05*abs(max(x)-min(x)))
            ax.set_ylim(min(y) - 0.05*(max(x)-min(x)), max(y) + 0.05*abs(max(x)-min(x)))
           
            z = np.isnan(z) # bool array, if True at nan, false otherwise
            
            # upsample using Kronecker product so outline follows square px edges
            z = np.kron(z, np.ones((100,100), dtype=bool))
            
            # edge border 
            z[0,:], z[-1,:], z[:,0], z[:,-1] = True, True, True, True
            
            
            xStep = x[1]-x[0]
            yStep = y[1]-y[0]
            
            # upsample x
            x = np.linspace(min(x) - xStep/2, max(x) + xStep/2, num=len(x)*100)
            # upsample y
            y = np.linspace(min(y) - yStep/2, max(y) + yStep/2, num=len(y)*100)
            
            ax.contour(x,y,z, linewidths=2, colors='black')
            
            
   
        
        def NaNBoundary(ax, z, x, y):
            z = np.isnan(z) # bool array, if True at nan, false otherwise
            
            # upsample using Kronecker product
            z = np.kron(z, np.ones((5,5), dtype=bool))
            
            xStep = x[1]-x[0]
            yStep = y[1]-y[0]
            
            # upsample x
            x = np.linspace(min(x) - xStep/2, max(x) + xStep/2, num=len(x)*5)
            # upsample y
            y = np.linspace(min(y) - yStep/2, max(y) + yStep/2, num=len(y)*5)
            
            # convolve
            z = np.array(z, dtype=int)
            k = np.ones((3,3), dtype=int)
            zExt = convolve2d(z, k,'same')
            
            # z = np.array(z, dtype=bool) ^ np.array(zExt, dtype=bool)
    
            # ax.pcolormesh(x, y, z, cmap='binary')
            ax.contour(x,y,np.array(zExt, dtype=bool), linewidths=2)
            ax.set_axis_off()

        if type(z) is str:
            z = self.SliceAttribute(z)
            if colourbarLabel is None: colourbarLabel == z
            
            
        
        
        
            
            
        if ax is None: fig, ax = plt.subplots()
        if cmapScale == 'log': cmapScale = colors.LogNorm(vmin=np.min(z),
                                                          vmax=np.max(z))
        if cmapScale == 'linear': cmapScale =colors.Normalize(vmin=np.min(z),
                                                          vmax=np.max(z))
        
        if masks is None:
        
            x, y = rotate(x, y, rot=-theta, xPivot = np.average(x), yPivot=np.average(y) )
            z, x, y = coordTransform(z, x, y)
            plot = ax.pcolormesh(x, y, z, norm=cmapScale, cmap=cmap)
            outline(ax, z, x, y)
        
        else:
            
            z, x, y = np.array(z), np.array(x), np.array(y)
            for i in set(masks):
                m = np.isin(masks, i)
                zm, xm, ym = z[m], x[m], y[m]
                xm, ym = rotate(xm, ym, rot=-theta, xPivot = np.average(x), yPivot=np.average(y) )
                zm, xm, ym = coordTransform(zm, xm, ym)
                plot = ax.pcolormesh(xm, ym, zm, norm=cmapScale, cmap=cmap)
       
        
        fig.colorbar(plot, label=colourbarLabel)

        scalebar = AnchoredSizeBar(ax.transData,
                           3e-9, '3 nm', 'lower right',
                           frameon=False)

        ax.add_artist(scalebar)
        ax.set_aspect('equal')
        ax.set_axis_off()
        
    
    def PlotSpectraPos(self, x='x_pos', y='y_pos', fig=None, ax=None):
        if ax is None: fig, ax = plt.subplots(figsize=(10,10))
        x = self.SliceAttribute(x)
        y = self.SliceAttribute(y)
        ax.scatter(x,y,color='white')
        for i in range(len(x)):
            ax.annotate(i, (x[i], y[i]), fontsize='xx-small')
        
        ax.set_ylabel('y / m')
        ax.set_xlabel('x / m')
      
        
      
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
        
    
    def Plot2D(self, y, xGrid, yGrid, xCentre=None, yCentre=None, 
               yLabel=None, fig=None, ax=None, rMin=-np.inf,
               rMax=np.inf, thetaMin=-np.inf, thetaMax=np.inf, color='black'):
        
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
            r = 1e9*np.sqrt((xDash)**2 + (yDash)**2)
            theta = np.arctan2(yDash, xDash) # element-wise arctan
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
        
        
        if ax is None: fig, ax = plt.subplots()
        ax.plot(rDash, yDash, '.', c=color)
        
        # # fit 
        # fit, fitInfo, fitConfBound = self.VLandscape2DFit(r=rDash, vContact=yDash)
        
        # rFit = np.linspace(min(rDash), max(rDash), len(rDash))
        # ax.plot(rDash, fit, color=color, label=r'$D+\frac{C}{r-r_0}$ fit' )
        # # ax.fill_between(rDash, fit-fitConfBound,
        # #          fit+fitConfBound, color=color, alpha=0.15, label=r'$2\sigma$ conf. interval')
        # ax.legend(fontsize = 7, loc='lower right')
        
        ax.set_xlabel('r (nm)', fontsize='x-large')
        ax.set_ylabel(yLabel, fontsize='x-large')
        
        
        fig, ax = plt.subplots()
        ax.scatter(xGrid, yGrid, c=color, alpha=0.2)
        xGrid = xGrid[condition]
        yGrid = yGrid[condition]
        ax.scatter(xGrid, yGrid, c=color)
        ax.set_axis_off()
        
        
        
#%%      


path = r'test_files'
fileNames = MyFiles().DirFilter(path, baseName='dfVMap_InSb_31_')
fileNames = fileNames[:-1]
# fileNames = fileNames[:-11]


xGrid = np.load(path + r'\xGrid_31.npy')
yGrid = np.load(path + r'\yGrid_31.npy')
# print(max(yGrid), min())
xGrid =xGrid[0:156]
yGrid = yGrid[0:156]

masks = np.load(path + r'\dfVMap_InSb_31__masks.npy')
masks = masks[:156]

theta = 1.6

fig, ax = plt.subplots()
ax.scatter(xGrid, yGrid)
#%%
channel = 'LI Demod 1 Y [AVG] (A)'
spectra = SpectraSetAnalysis(channel, path, fileNames)


spectra.DIDVAnalysis(centerGuess=-0.09, sigmaGuess=0.01, amplitudeMaxLim=1e-11)
# %% pres plot

fig, ax = plt.subplots(ncols=3, figsize=(20,5))
# SXM plot 
sxm = Sxm()
im = sxm.ReadSxm('test_files', 'InSb(110)_0637.sxm')

sxm.Plot(fig=fig, ax=ax[0], cmapLabel='z (m)')
ax[0].add_artist(AnchoredSizeBar(ax[0].transData,
                   3e-9, '3 nm', 'lower right',
                   frameon=False,
                   color='yellow'))

# Map OI
spectra.Map('fitHeight', xGrid, yGrid, theta=theta, fig=fig, ax=ax[1], colourbarLabel='dI/dV fitted peak height',
            cmapScale='log', cmap='BuPu')
 
# residuals
spectra.Map('meanAbsRes', xGrid, yGrid, theta=theta, fig=fig, ax=ax[2], colourbarLabel='mean absolute residuals',
            cmapScale='linear', cmap='bone_r')


ax[0].set_xlim(ax[1].get_xlim())
ax[0].set_ylim(ax[1].get_ylim())
ax[0].set_aspect('equal')
for i in range(3):
    ax[i].set_axis_off()

#%%
xCentre = 22.55e-9
yCentre = 233.5e-9

fig, ax = plt.subplots()
spectra.Plot2D('fitHeight', xGrid, yGrid,
               yLabel='dI/dV fitted peak height')


#%%
ax = spectra.PlotStacked(ncols=20, legendLabel='data')
spectra.PlotStacked(y='fit', ax=ax, color='red', legendLabel='fit')
#%%

spectra.PlotSpectraPos()
#%%
spectra.Map('meanAbsRes', xGrid, yGrid, theta=theta, colourbarLabel='mean absolute residuals',
            cmapScale='linear', cmap='bone_r')

spectra.Map('fitCentre', xGrid, yGrid, theta=theta, colourbarLabel='fitted peak centre',
            cmapScale='linear', cmap='bone_r')

# spectra.Map('fitArea', xGrid, yGrid, theta=theta, colourbarLabel='dI/dV fitted peak area',
#             cmapScale='log', cmap='BuPu')
#%%
spectra.Map('fitHeight', xGrid, yGrid, theta=theta, colourbarLabel='dI/dV fitted peak height',
            cmapScale='log', cmap='BuPu')
spectra.ScatterMap('fitHeight', xGrid, yGrid, theta=theta, colourbarLabel='dI/dV fitted peak height',
            cmapScale='log', cmap='BuPu')


# #%%
# spectra.ScatterMap('meanAbsRes', xGrid, yGrid, colourbarLabel='mean abs(dI/dV fitted peak residuals)',
#             cmapScale='linear', cmap='plasma')

# spectra.ScatterMap('fitArea', xGrid, yGrid, colourbarLabel='dI/dV fitted peak area',
#             cmapScale='log', cmap='plasma')

# spectra.ScatterMap('fitHeight', xGrid, yGrid, colourbarLabel='dI/dV fitted peak height',
#             cmapScale='log', cmap='plasma')


#%%




channel = 'OC M1 Freq. Shift [AVG] (Hz)'
spectra = SpectraSetAnalysis(channel, path, fileNames)


spectra.KPFMAnalysis()
# %% pres plot

fig, ax = plt.subplots(ncols=3, figsize=(20,5))
# SXM plot 
sxm = Sxm()
im = sxm.ReadSxm('test_files', 'InSb(110)_4091.sxm')

sxm.Plot(fig=fig, ax=ax[0], cmapLabel='z (m)')
ax[0].add_artist(AnchoredSizeBar(ax[0].transData,
                   3e-9, '3 nm', 'lower right',
                   frameon=False,
                   color='yellow'))

# Map OI
spectra.Map('vContact', xGrid, yGrid, fig=fig, ax=ax[1], theta=theta, colourbarLabel='Contact Potential (V)',
            cmapScale='linear', cmap='inferno')
 
# residuals
spectra.Map('meanAbsRes', xGrid, yGrid, fig=fig, ax=ax[2], theta=theta, colourbarLabel='mean absolute residuals (V)',
            cmapScale='linear', cmap='bone_r')

# ax[1].get_xlim(), ax[1].get_xlim()


ax[0].set_xlim(ax[1].get_xlim())
ax[0].set_ylim(ax[1].get_ylim())
ax[0].set_aspect('equal')
for i in range(3):
    ax[i].set_axis_off()



#%%
xCentre = -0.1504e-9
yCentre = 702.9e-9

fig, ax = plt.subplots()
spectra.Plot2D('vContact', xGrid, yGrid, xCentre=xCentre, yCentre=yCentre,
               yLabel='Contact Potential (V)', rMin=2,
               thetaMin=25, thetaMax=110, color='red', fig=fig,ax=ax)
spectra.Plot2D('vContact', xGrid, yGrid, xCentre=xCentre, yCentre=yCentre,
               yLabel='Contact Potential (V)', rMin=2,
               thetaMin=-155, thetaMax=-70, color='blue', ax=ax, fig=fig)

#%%


ax = spectra.PlotStacked(ncols=20)
spectra.PlotStacked(y='fit', ax=ax, color='red')
#%%

fileName = 'InSb(110)_0780.sxm'
sxm = Sxm()
im = sxm.ReadSxm(path, fileName)
fig, ax = plt.subplots()
ax = sxm.Plot()

spectra.PlotSpectraPos(fig=fig, ax=ax)


# ax.scatter(xGrid,yGrid)
# for i in range(len(xGrid)):
#     ax.annotate(i, (xGrid[i], yGrid[i]))



#%%
spectra.PlotSpectraPos()

spectra.ScatterMap(masks, xGrid, yGrid, colourbarLabel='Contact Potential Diff',
            cmapScale='linear', cmap='inferno')
#%%
spectra.Map('vContact', xGrid, yGrid, masks=masks, theta=theta, colourbarLabel='Contact Potential (V)',
            cmapScale='linear', cmap='inferno')
#%%
spectra.Map('meanAbsRes', xGrid, yGrid, masks=masks, theta=theta, colourbarLabel='mean absolute residuals (V)',
            cmapScale='linear', cmap='bone_r')
#%%
spectra.Map('fitA', xGrid, yGrid, masks=masks, theta=theta, colourbarLabel='a fit param for $ax^2+bx+c$',
            cmapScale='linear', cmap='bone_r')

#%%
spectra.ScatterMap('vContact', xGrid, yGrid, colourbarLabel='Contact Potential Diff',
            cmapScale='linear', cmap='inferno')

spectra.ScatterMap('meanAbsRes', xGrid, yGrid, colourbarLabel='mean absolute residuals (V)',
            cmapScale='linear', cmap='bone_r')

#%%
channel = 'Current [AVG] (A)'
spectra = SpectraSetAnalysis(channel, path, fileNames)
current = spectra.SliceAttribute('y')
currentMax = [max(c) for c in current]

spectra.Map(currentMax, xGrid, yGrid, masks=masks, theta=theta, colourbarLabel='max current (A)',
            cmapScale='log', cmap='bone_r')

#%%

spectra.ScatterMap(currentMax, xGrid, yGrid, colourbarLabel='max current (A)',
            cmapScale='log', cmap='bone_r')
