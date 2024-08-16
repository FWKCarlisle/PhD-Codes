# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 10:23:03 2024

@author: ppysa5
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 11:39:07 2024

@author: physicsuser
"""

from read_spectra import output_data_spectra_dat
import os
import lmfit
from lmfit import minimize
import matplotlib.pyplot as plt
import math
import numpy as np
import matplotlib as mpl
  
class DfV2Vcontact(output_data_spectra_dat):
    
    def __init__(self, path, filename, channel):
        super().__init__()
        self.bias, self.df = self.OpenSpectra(path, filename, channel)
        
    
    def OpenSpectra(self, path, filename, channel):
        self.get_file(path, filename)   
        if channel not in list(self.df): # if channel not in file, print list of possible channels and stop running the script
            print('Choice of channel not found in ' + filename)
            self.show_method_fun()
            exit()
        
        x = self.give_data(0)[0] 
        y = self.give_data(channel)[0]

        return x, y # bias, df
    
    
    
    def _Parabola(self, x, a, b, c):
        return a*x**2 + b*x + c



    def _ParabolaModelResidual(self, params, x, y):
        a, b, c = params['a'], params['b'], params['c'] # fitting parameters
        model = a*x**2  + b*x + c # objective function
        return (y - model) # residual
    
    
    
    def ParabolaFit(self, aGuess = 0, bGuess = 0, cGuess = 0):
        x, y = self.bias, self.df
        parabola_params = lmfit.Parameters() # define a python dict to store the fitting parameters
        parabola_params.add('a', value=0) # define the fitting parameters and an initial guess for its value. Here you can also define contraints, and other useful things
        parabola_params.add('b', value=0) 
        parabola_params.add('c', value=0)
        
        fitInfo = minimize(self._ParabolaModelResidual, parabola_params, args=(x, y))
        a, b, c = fitInfo.params['a'].value, fitInfo.params['b'].value, fitInfo.params['c'].value # get the optimised fitting parameters from the "parabola_model_output" object
        fit = self._Parabola(x, a, b, c) # using x and the fitting parameters, calculate the parabolic fit
        
        self.fit = fit
        self.fitInfo = fitInfo
        
        return fit, fitInfo
 
    
 
    def ParabolaMinima(self, fitInfo):
        a, b, c = fitInfo.params['a'].value, fitInfo.params['b'].value, fitInfo.params['c'].value
        x_min = -b / (2*a)
        y_min = c - b**2 / (4*a)
        
        self.vContact = x_min
        self.dfAtVContact = y_min
        
        return x_min, y_min



    def CalcVContact(self):
        self.ParabolaFit(self.bias, self.df)
        self.ParabolaMinima(self.fitInfo)            
        
        return self.vContact
    
    
    
    def CalcR(self, xAdatomCentre, yAdatomCentre):
        self.xAdatomCentre = xAdatomCentre
        self.yAdatomCentre = yAdatomCentre
        
        # transformation: origin @ nanonis origin -> origin @ adatom centre
        xDash = self.x_pos - xAdatomCentre
        yDash = self.y_pos - yAdatomCentre
        
        # evaluate r
        self.r = np.sqrt((xDash)**2 + (yDash)**2)
        return self.r
    
    
class VcontactFit():
    
    def __init__(self, r, vContact):
        self.r = r
        self.vContact = vContact
    
    def _Vcontact_model(self, r, D, C, r_0):
        return D + (C)/(r-r_0)
    

    def _Vcontact_model_residual(self, params, x, data):
        D, C, r_0 = params['D'], params['C'], params['r_0'] 
        model = D + (C)/(x-r_0)
        return (data - model)


    def VcontactVsNormRFit(self):
        self.rNorm = (self.r - np.min(self.r)) / (np.max(self.r) - np.min(self.r)) # normalise r to avoid very small number problems
        Vcontact_model_params = lmfit.Parameters() # define a dict to store the fitting parameters
        Vcontact_model_params.add('D', value=0) # define the fitting parameters and an initial guess for their value
        Vcontact_model_params.add('C', value=-0.1) 
        Vcontact_model_params.add('r_0', value=-0.1)

        fitInfo = minimize(self._Vcontact_model_residual, Vcontact_model_params, args=(self.rNorm, self.vContact))
        D, C, r_0 = fitInfo.params['D'].value, fitInfo.params['C'].value, fitInfo.params['r_0'].value
        fit = self._Vcontact_model(self.rNorm, D, C, r_0 )
        
        self.fit = fit
        self.fitInfo = fitInfo
        
        return fit, fitInfo
    
    
    
#%%

def DefineColourbar(fig, ax, Ndivisions, colourbarLabel = None):
    cmap = mpl.cm.get_cmap('hsv', Ndivisions)
    c = np.arange(0, Ndivisions-1)
    dummie_cax = ax.scatter(c, c, c=c, cmap=cmap)
    ax.cla()
    fig.colorbar(dummie_cax, label = colourbarLabel)
    return cmap
    


def CheckSpectra(numOfSubplotCols, dataSet):
    fig, ax = plt.subplots(ncols=numOfSubplotCols)
    cmap = DefineColourbar(fig, ax[0], Ndivisions = len(dataSet), colourbarLabel = 'spectra number')
    spectraPerAxis = math.ceil(len(dataSet) / numOfSubplotCols)

    count = 0
    for col in range(numOfSubplotCols):        
        if col == 0:
            ax[col].set_ylabel('df/ Arb units')
            ax[col].set_xlabel('bias/V')
            ax[col].set_title('check spectra')
        ax[col].set_yticklabels([])
        
        plotShift = 0
        for i in range(spectraPerAxis):
            spectraNum = i + count
            
            if spectraNum < len(dataSet):
                
                
                dfRelative = dataSet[spectraNum].df - dataSet[spectraNum].dfAtVContact + plotShift
                fitRelative = dataSet[spectraNum].fit - dataSet[spectraNum].dfAtVContact + plotShift
                
                ax[col].plot(dataSet[spectraNum].bias, dfRelative, color=cmap(spectraNum), alpha = 0.7)
                ax[col].plot(dataSet[spectraNum].bias, fitRelative, color=cmap(spectraNum), alpha= 0.3)
                ax[col].scatter(dataSet[spectraNum].vContact, plotShift, color=cmap(spectraNum))
                
                plotShift = plotShift + np.ptp(dataSet[spectraNum].df) 
                
        count = count + spectraPerAxis

    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    
    
def plot_fitNresiduals(x, y, fit, residuals, xlabel, ylabel):
    fig1, axs1 = plt.subplots(nrows=2, ncols=1)
    axs1[0].plot(x, y, '.')
    axs1[0].plot(x, fit, alpha = 0.75)
    axs1[0].set_xlabel(xlabel)
    axs1[0].set_ylabel(ylabel)
    axs1[1].grid()
    axs1[1].plot(x, residuals, '.')
    axs1[1].set_xlabel(xlabel)
    axs1[1].set_ylabel('residuals')   
    plt.tight_layout()
#%%
path = r'D:\Results\2024\III-Vs'
baseName = 'dfVMap_InSb_20_'
filenames = [f.name for f in os.scandir(path) if f.is_file() and f.name.startswith(baseName)]

# filenames = filenames[0:70]
dataSet = []
vContact = []
r = []
for f in filenames:

    vContactCalculation = DfV2Vcontact(path, filename = f, channel = 'OC M1 Freq. Shift [AVG] (Hz)')
    vContact.append(vContactCalculation.CalcVContact())
    dataSet.append(vContactCalculation)
    r.append(vContactCalculation.CalcR(xAdatomCentre = -661.4e-9, yAdatomCentre = -300.1e-9))

#%%
# CheckSpectra(15, dataSet)

#%% 2D plot 
# vContactFit = VcontactFit(r, vContact)
# fit, fitInfo = vContactFit.VcontactVsNormRFit()
# residuals = fitInfo.residual


# plot_fitNresiduals(r, vContact, fit, residuals, xlabel = '~relative distance r (m)', ylabel = 'Bias at df(V) minima (V)') # plot spectra, fit and residuals
#%% 2D plot 
x = np.load(r'C:\Users\physicsuser\Documents\Python Scripts\dfVMap_InSb_20__initial_x.npy')
y = np.load(r'C:\Users\physicsuser\Documents\Python Scripts\dfVMap_InSb_20__initial_y.npy')
z = []
res = []
for spectraNum in range(len(dataSet)):
    z.append(dataSet[spectraNum].vContact)
    # x.append(dataSet[spectraNum].x_pos)
    # y.append(dataSet[spectraNum].y_pos)
    resList = dataSet[spectraNum].fitInfo.residual
    meanAbsRes = np.average(abs(resList))
    res.append(meanAbsRes)
    

#%%
def DefineColourbar2(fig, ax, minVal, maxVal, nDivisions, colourbarLabel = None):
    cmap = mpl.cm.get_cmap('seismic', nDivisions)
    c = np.linspace(minVal, maxVal, num = nDivisions)
    dummie_cax = ax.scatter(c, c, c=c, cmap=cmap)
    ax.cla()
    fig.colorbar(dummie_cax, label = colourbarLabel)
    return cmap


    
fig, ax = plt.subplots()
cmap=DefineColourbar2(fig, ax, min(z), max(z), 100,  colourbarLabel = '$V_{contact}$')

ax.scatter(x, y, color=cmap((z-min(z))/(max(z)-min(z))))

# for i in range(len(x)):
#     ax.annotate(i, (x[i],y[i]))

# fig, ax = plt.subplots()
# cmap=DefineColourbar2(fig, ax, min(res), max(res), 100,  colourbarLabel = '$meanAbsRes$')

# ax.scatter(x, y, color=cmap((res-min(res))/(max(res)-min(res))))
#%%
x = np.load(r'C:\Users\physicsuser\Documents\Python Scripts\dfVMap_InSb_20__initial_x.npy')
y = np.load(r'C:\Users\physicsuser\Documents\Python Scripts\dfVMap_InSb_20__initial_y.npy')

xDash = list(set(x))
yDash = list(set(y))

xDash.sort()
yDash.sort()

zDash = np.empty((len(xDash), len(yDash)))
zDash.fill(np.nan)

for spectraNum in range(len(dataSet)):
    i = xDash.index(x[spectraNum])
    j = yDash.index(y[spectraNum])
    zDash[i,j] = z[spectraNum]
    
z = np.array(z)
fig,ax=plt.subplots()
plot = ax.pcolormesh(xDash, yDash, zDash.T,  cmap='spring')
fig.colorbar(plot, label='$V_{contact}$')
ax.set_ylabel('y / m')
ax.set_xlabel('x / m')

