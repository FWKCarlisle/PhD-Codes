
import numpy as np
from numpy.polynomial import polynomial as P
import re
import io
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import struct
import datetime
import os
import glob
from scipy import integrate
from pathlib import Path
from scipy.signal import savgol_filter
from scipy.interpolate import make_lsq_spline, BSpline
from scipy.optimize import curve_fit
from scipy.interpolate import make_interp_spline
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import UnivariateSpline



class output_data_spectra_dat():
    
    def get_file(self, pathname, filename, show_methods=0):

        #Find get the file name
        self.filename_path = pathname + "/" + filename
        
        #Now we want to get the data and read it. 
        self.load_file(show_methods)
        
        if show_methods == 1:
            self.show_method_fun()
    
    def convert_time_to_epoch(self, time):
        dt = datetime.datetime.strptime(str(time), '%d.%m.%Y %H:%M:%S')
        return dt.timestamp()

    def load_file(self, show_methods):

        def read_metadata_info(meta_data):
            """
            Note: some files will not have this info e.g. data logger files
            Can be extended/edited to access more/different metadata info.
            """

            # Extract position information:
            try: self.x_pos = float(meta_data.split('X (m)\t')[1].split('\n')[0][0:-1])
            except: pass
            try: self.y_pos = float(meta_data.split('Y (m)\t')[1].split('\n')[0][0:-1])
            except: pass
            try: self.z_pos = float(meta_data.split('Z (m)\t')[1].split('\n')[0][0:-1])
            except: pass

            # Extract the time information:
            try: self.start_time = self.convert_time_to_epoch(
                meta_data[meta_data.index('Start time') + 11:meta_data.index('Start time') + 11 + 19])
            except: pass

            # Extract comment
            try: self.comment = meta_data.split('Comment10\t')[1].split('\n')[0][0:-1]
            except: pass

        #Open the file. The data and headings start after the '[data]' section.
        f = open(self.filename_path, "r")
        f.seek(0)
        file_all_data = f.read()
        f.close()
        
        #Split at the '[data]' section. The second in the list is our data with the headers.
        file_data = re.split(r"\[DATA\]\n",file_all_data)[1]
        meta_data = re.split(r"\[DATA\]\n", file_all_data)[0]

        read_metadata_info(meta_data)

        #Load as df to make life easy
        self.df = pd.read_csv(io.StringIO(file_data),delimiter = '\t')
        
        self.list_of_methods = list(self.df)
     
    def show_method_fun(self):
        
        print('Possible methods to use are:')
        
        for count, i in enumerate(self.list_of_methods):
            print(str(count) +') '+ i)
        
    def give_data(self, method_number):
        
        #if type(method_number) is int:
            #Do nothing, as this is the expected format
        if type(method_number) is str:
            #Convert to number but use the lookup table
            for count, i in enumerate(self.list_of_methods):
                if method_number in i:
                    method_number = count
                    break
            if method_number is not count:
                #We did not convert to an int value
                raise Exception("This input is not found in the list, check spelling is correct!")
                
        elif type(method_number) is not int: 
            raise Exception("Must be either and int or str variable")
        
        name_to_export = self.list_of_methods[method_number]
        
        exported_data = (self.df[name_to_export],self.list_of_methods[method_number])
        
        return exported_data
    



m=output_data_spectra_dat()
stiffness = 1800 #N/m
resfreq = 31515 #Hz

def getdf(pname,fname,fnum,plot):
    
    filename=str(fname) + str('{num:05d}'.format(num=fnum)) + '.dat'
    
    m.get_file(pathname,filename,show_methods=0)
    m.load_file(show_methods=0)
    
    df=np.array(m.give_data(method_number='OC M1 Freq. Shift (Hz)')[0])    
    df = df[1:-1]
    
    xdata=np.array(m.give_data(method_number='X (m)')[0])
    xdata=xdata[1:-1]
    
    ydata=np.array(m.give_data(method_number='Y (m)')[0])
    ydata=ydata[1:-1]
    
    
    x = np.sqrt(xdata**2+ydata**2)
    
    xshift=x-np.mean(x)
    
    z=np.mean(m.give_data(method_number='Z (m)')[0])
    #print(z)
    
    I=np.array(m.give_data(method_number='Current (A)')[0]) 
    I=I[1:-1]
    
    #excitation = np.array(m.give_data(method_number='OC M1 Excitation (V)')[0])
    excitation=np.array(m.give_data(method_number='OC M1 Excitation')[0]) 
    excitation = excitation[1:-1]
    kz=(2*stiffness/resfreq)*df
    if plot==True:
        plt.plot(x-np.mean(x),I,marker='o',markersize=3,alpha=fnum/len(files))
    
    return x,df,kz,z,I,excitation

def get_info(pname,fname):
    filename=str(fname) + str('{num:05d}'.format(num=3)) + '.dat'
    m.get_file(pathname,filename,show_methods=0)
    m.load_file(show_methods=0)
    bias = np.array(m.give_data(method_number='Bias (V)')[0])[10]
    return bias
    
def checklengths(path,files):
    array=[]
    for i in range(0,len(files)):
        m.get_file(path,str(files[i]),show_methods=0)
        m.load_file(show_methods=0)
        datacheck=np.array(m.give_data(method_number='OC M1 Freq. Shift (Hz)')[0])    
        datacheck = datacheck[1:-1]
        array.append(len(datacheck))
        
    return array
    
def makearrays(pathname,files,moveindex):
    dfarr=[]
    kzarr=[]
    xarr=[]
    zarr=[]
    Iarr=[]
    excarr = []
    minlength = min(checklengths(pathname,files))
    
    for i in range(1,len(files)+1):
        x,df,kz,z,I,excitation = getdf(pathname,fname,i,False)
        dfarr.append(df[2:minlength-2])
        kzarr.append(kz[2:minlength-2])
        xarr.append(x[2:minlength-2])
        zarr.append(z)
        Iarr.append(I[2:minlength-2])
        excarr.append(excitation[2:minlength-2])
    
        
    dfarr=np.array(dfarr)[1:]
    kzarr=np.array(kzarr)[1:]
    xarr=np.array(xarr)[1:]*10**9
    zarr=np.array(zarr)[1:]*10**9
    Iarr=np.array(Iarr)[1:]
    excarr = np.array(excarr)[1:]
    
    
    
    zarr_cal = -np.linspace(zarr[0], zarr[0]+5e-3*(len(zarr)-1), len(zarr))
    
    
    return dfarr,kzarr,xarr,zarr,Iarr,excarr

#dfarr, kzarr, xarr, zarr, Iarr = makearrays(r'/home/ppyop1/PhD_stuff/DBTF/manip_data_2025/manips/Manip 01')

def integrateKz(kzarr,z,plot,moveindex):
    Fzarr=[]
    if moveindex == None:
        moveindex = len(kzarr)
    for i in range(0,len(kzarr[0])):
        Fzx=[]
        for j in range(0,moveindex):
            Fzx.append(np.trapezoid(kzarr[0:j][0:,i],-z[0:j]))
        Fzarr.append(np.array(Fzx))
    
    Fzarr=np.array(Fzarr)
    Fzarr = Fzarr.transpose() #transpose back to same structure as kz and df
    
    if plot==True:
        for i in range(len(Fzarr[:,0])):
            plt.plot(Fzarr[i],alpha=i/len(Fzarr[:,0]))
    
    return Fzarr

def subtract_wdw(x,F,xmin,xmax):
    mask = (x >= xmin) & (x <= xmax)
    #mask = x[:int(len(xshift/20))]
    Fsub = np.zeros_like(F)

    for i in range(2,len(Fsub[:,0])):
        Fsub[i]=F[i]-np.mean(F[i][:15])
        
    return Fsub

def getpot(x,F,z,plot):
    Uarr=[]
    for i in range(0,len(F[0])):
        U=[]
        for j in range(0,len(F[:,0])):
            #print(len(F[0:j][:,i]))
            U.append(np.trapezoid(F[0:j][:,i],-z[0:j]))
        Uarr.append(np.array(U))
    
    Uarr=np.array(Uarr)
    Uarr = Uarr.transpose() #transpose back to same structure as kz and df
    #Uarr = Uarr*10**-18 #since we are in nN units, convert back (nN and nm multiplied together btw)
    UarreV = (Uarr * 6.242e18) #convert to eV
    
    if plot == True:
        for i in range(len(UarreV[:,0])):
            plt.plot(xarr[i],UarreV[i],alpha=i/len(UarreV[:,0]))
    
    return Uarr, UarreV


def fitting(xshift,y,fittype,plot,whereknots,eV):
    Ufit=[]
    
    for i in range(len(y[:,0])):
        if fittype == 'savgol':
            smooth = savgol_filter(y[i], 31,2,mode='interp')
            Ufit.append(smooth)
            if plot==True:
                plt.plot(xshift,y[i])
                plt.plot(xshift[1:-1],smooth,c='k')
                
        if fittype == 'lsq_spline':
            smooth = savgol_filter(y[i], 100,9,mode='interp')
            if whereknots == 'DBTF':
                t = [-3e-9, -2.5e-9, -2e-9, -1.5e-9, -1e-9, -0.5e-9, 0.0, 0.5e-9, 1e-9, 1.5e-9, 2e-9, 2.5e-9, 3e-9]
            
            if whereknots == 'Cs':
                t=np.linspace(0e-9,1.5e-9,num=100)
            k = 3
            #print(xshift[0])
            #print(xshift[-1])
            t = np.r_[(xshift[0],)*(k+1),t,(xshift[-1],)*(k+1)]
            #t = np.r_[(t[0],)*(k+1),t,(t[-1],)*(k+1)]
            print(t)
        
            spl = make_lsq_spline(xshift, y[i], t, k)
            
            if plot==True:
                plt.plot(xshift,spl(xshift),c='k')
                plt.plot(xshift,Uarr[i])
            Ufit.append(spl(xshift))
            
            
                
        if fittype == 'interp_spline':
            spl_i = make_interp_spline(xshift, y[i])
            if plot == True:
                plt.plot(xshift,y[i])
                plt.plot(xshift,spl_i(xshift),c='k')
########################### GAUSSIAN FITTING #################################                
        if fittype == 'gaussian':
            spl = gaussian_filter1d(y[i],10,mode='mirror')
            Ufit.append(spl)
###############################################################################
        if fittype == 'spline':
            spl = UnivariateSpline(xshift, y[i], s=1)
            Ufit.append(spl(xshift))
            
        if fittype == None:
            Ufit.append(y[i])
################################################################################
    Ufit=np.array(Ufit) 
    UfiteV=np.array(Ufit*6.242e18)
    return Ufit,UfiteV

def gauss_smooth(data,order):
    for i in range(len(data[:,0])):
        data[i]=gaussian_filter1d(data[i],order)
        return data
        

def diffpot(x,y,yerr,plot):
    Flateral=[]
    Flerr=[]
    for i in range(len(y[:,0])):
        Fl = np.gradient(y[i],x[i],edge_order=2)
        Flateral.append(Fl) #note we convert U back to joules from eV
        if plot ==True:
            plt.plot(x[i],Fl)
        
        #errors - propogate from error on potential (Uerr)
        if yerr != None:
            error = np.sqrt(2)*yerr[i]/abs(x[i][10]-x[i][11])
            Flerr.append(error)
        
    Flateral = np.array(Flateral)
    Flerr = np.array(Flerr)
    
    return Flateral,Flerr

#Flateral = diffpot(xarr[0],Ufit,True)

def findforce(x,F,moveindex):
    if moveindex == None:
        moveindex = len(F[:,0])-1
    #maxneg = np.nanmin(F[moveindex])
    #maxpos = np.nanmax(F[moveindex])
    maxpos = np.mean(F[moveindex][np.nanargmax(F[moveindex])-3:np.nanargmax(F[moveindex])+3])
    maxneg = np.mean(F[moveindex][np.nanargmin(F[moveindex])-3:np.nanargmin(F[moveindex])+3])

    maxposerror = max(abs(F[moveindex-1][x>0]))
    maxnegerror = max(abs(F[moveindex-1][x<0]))
    
    return maxneg,maxpos,maxnegerror,maxposerror

def find_vertical_force(F,moveindex):
    if moveindex == None:
        moveindex = len(F[:,0])-1
    return np.mean(F[moveindex][np.argmin(F[moveindex])-10:np.argmin(F[moveindex])+10])

def plot_single(ax,xarr,y,moveindex,colour):
    if colour == 'default':
        cmap = plt.get_cmap('Blues')
    if colour == 'k':
        cmap=plt.get_cmap('Greys')
    #plot that shit
    if moveindex == None:
        moveindex = len(y[:,0])-1
    for j in range(moveindex+1):
        #x=xshiftarr[j]
        norm = mcolors.Normalize(vmin=1, vmax=moveindex - 1)
        colour = cmap(norm(j))
        if j >= moveindex:
            colour='tab:red'
        ax.plot(xarr,y[j],c=colour,lw=1)
    
    
def remove_edges(a):
    cols_with_data = np.any(~np.isnan(a), axis=0)

    # Index of the last column with data
    last_data_col = np.where(cols_with_data)[0][-2]
    first_data_col = np.where(cols_with_data)[0][1]
    print(last_data_col)
    a[:, last_data_col:] = np.nan
    a[:, :first_data_col] = np.nan

    return a



pathname = r"C:\Users\ppxfc1\OneDrive - The University of Nottingham\Desktop\PhD\Code\PhD-Codes\Analysis code\DBTF\Manip_3_500uV"
# pathname = r"C:\Users\ppxfc1\OneDrive - The University of Nottingham\Desktop\PhD\DBTF\manips\manip 8"
# fname = r'manip_DBTF_8_'
fname = r'Move_DBTF@Pb_500uV_2_'
#fname = r'down_manip_34_0_-0.4_'
#fname = r'manip_DBTF_3_'
moveindex = 48

##################################


folder = Path(pathname)
#files = [f for f in folder.iterdir() if f.is_file() and str(fname) in f.name]
files = [f.name for f in folder.iterdir() if f.is_file() and fname in f.name]



dfarr,kzarr,xarr,zarr,Iarr,excarr = makearrays(pathname,files,False)

#### subtract off df background if needed ####
dfarr = dfarr-np.mean(dfarr[0])
kzarr = kzarr-np.mean(kzarr[0])

xshiftarr = []

for i in range(len(xarr[:,0])):
    xshift = xarr[i]-np.mean(xarr[i])
    ### put onto a uniform grid, this helps with any spiking seen due to non uniformity in xdata. this can happen when TCP is slightly slow
    x_uniform = np.linspace(min(xshift), max(xshift), len(xshift))
    xshiftarr.append(x_uniform)
    
#kzarr = subtract_wdw(xshiftarr[5],kzarr,xshift[1],xshift[50])

Fzarr = integrateKz(kzarr,zarr,False,None)
Fzarr_subtract = subtract_wdw(xshiftarr[5],Fzarr,xshift[1],xshift[50])
Uarr,UarreV = getpot(xshiftarr,Fzarr_subtract,zarr,False)


##########################################################
Ufit,UfiteV = fitting(xshiftarr,Uarr,'savgol',False,'Cs',False)
###########################################################

Flateral,Ferr = diffpot(xshiftarr,Ufit,None,False)
Flateral_chop = remove_edges(Flateral)
#chop off ends of Flateral (ignoring nans)
maxneg,maxpos,maxnegerror,maxposerror = findforce(xshift,Flateral_chop,moveindex)
max_vertical = find_vertical_force(Fzarr_subtract,moveindex)
bias = get_info(pathname,fname)

fig,ax=plt.subplots(3,2,figsize=(15,10))
plt.rcParams.update({'font.size': 15})

#current
plot_single(ax[0][0],xshift,Iarr*10**12,moveindex,'default')
ax[0][0].set_ylabel('Current (pA)')
#df
plot_single(ax[0][1],xshift,dfarr,moveindex,'default')
ax[0][1].set_ylabel('Frequency shift (Hz)')
#excitation
# plot_single(ax[2][0],xshift,excarr*1000,moveindex,'default')
# ax[2][0].set_ylabel('Excitation (mV)')
#vertical force w/out background
plot_single(ax[2][1],xshift,Fzarr_subtract*10**3,moveindex,'default')
ax[2][1].set_ylabel('Vertical Force (pN)\n without background')
ax[2][1].hlines(max_vertical*10**3,xmin=xshift[1],xmax=xshift[-1],ls='dashed',color='k',alpha=0.5)
ax[2][1].text(xshift[1],max_vertical*10**3,str(round(max_vertical*10**3,2))+'pN',fontsize=12)
#vertical force with background
plot_single(ax[1][1],xshift,Fzarr*10**3,moveindex,'default')
ax[1][1].set_ylabel('Vertical Force (pN)')
#Potential fit
plot_single(ax[1][0],xshift,UfiteV*10**-18*1000,moveindex,'k')
#Potential raw
plot_single(ax[1][0],xshift,UarreV*10**-18*1000,moveindex,'default')
ax[1][0].set_ylabel('Potential (meV)')
#lateral force
plot_single(ax[2][0],xshift,Flateral_chop*10**3,moveindex,'default')
ax[2][0].set_ylabel('Lateral Force (pN)')
#dashed lines showing max lateral force
ax[2][0].hlines(y=maxneg*10**3,xmin=xshift[1],xmax=xshift[-1],ls='dashed',color='k',alpha=0.5)
ax[2][0].hlines(y=maxpos*10**3,xmin=xshift[1],xmax=xshift[-1],ls='dashed',color='k',alpha=0.5)
ax[2][0].text(0,maxneg*10**3+0.05,str(round(maxneg*10**3,2))+'pN',fontsize=12)
ax[2][0].text(0,maxpos*10**3-0.1,str(round(maxpos*10**3,2))+'pN',fontsize=12)

plt.suptitle(str(fname) + '\n'+str(round(bias,3))+'V',fontsize=20)
#plt.savefig(r'/home/ppyop1/PhD_stuff/cs_force_measurements/'+str(fname))

plt.show()


