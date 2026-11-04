
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
    



def getdf(pname,fname,fnum,plot):
    # try:
    #     filename=str(fname) + str('{num:05d}'.format(num=fnum)) + '.dat'
    # except:
    # print("File not found, trying to find file with same name and number in the folder...")
    files = [f.name for f in Path(pname).iterdir() if f.is_file()]
    # print(files)
    filename = files[fnum - 1]
    # print(filename)
    m.get_file(pname,filename,show_methods=0)
    m.load_file(show_methods=0)
    
    df=np.array(m.give_data(method_number='OC M1 Freq. Shift (Hz)')[0])    
    df = df[1:-1]
    
    xdata=np.array(m.give_data(method_number='X (m)')[0])
    xdata=xdata[1:-1]
    
    ydata=np.array(m.give_data(method_number='Y (m)')[0])
    ydata=ydata[1:-1]

    bias = np.array(m.give_data(method_number='Bias (V)')[0])[10]
    
    
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
    
    return x,df,kz,z,I,excitation, bias

def get_info(pname,fname):
    filename=str(fname) + str('{num:05d}'.format(num=3)) + '.dat'
    m.get_file(pname,filename,show_methods=0)
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
    
def makearrays(pathname,fname,files,moveindex):
    dfarr=[]
    kzarr=[]
    xarr=[]
    zarr=[]
    Iarr=[]
    excarr = []
    minlength = min(checklengths(pathname,files))
    
    if moveindex == None:
        nfiles = len(files)
    else: 
        nfiles = moveindex


    
    for i in range(2,nfiles+1):
        print("File number: ", pathname + "_" + str(i))
        x,df,kz,z,I,excitation, bias = getdf(pathname,fname,i,False)
        # print(z)
        dfarr.append(df[2:minlength-2])
        kzarr.append(kz[2:minlength-2])
        xarr.append(x[2:minlength-2])
        zarr.append(z)
        Iarr.append(I[2:minlength-2])
        excarr.append(excitation[2:minlength-2])
    
        
    dfarr=np.array(dfarr)
    kzarr=np.array(kzarr)
    xarr=np.array(xarr)*10**9
    zarr=np.array(zarr)*10**9
    Iarr=np.array(Iarr)
    excarr = np.array(excarr)
    
    
    
    zarr_cal = -np.linspace(zarr[0], zarr[0]+5e-3*(len(zarr)-1), len(zarr))
    
    # dfarr = dfarr[:moveindex+1] 
    # kzarr = kzarr[:moveindex+1]
    # xarr = xarr[:moveindex+1]
    # zarr = zarr[:moveindex+1]
    # Iarr = Iarr[:moveindex+1]
    # excarr = excarr[:moveindex+1]


    return dfarr,kzarr,xarr,zarr,Iarr,excarr, bias

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
    # print("Shapes: Z ", F.shape, " -z ", z.shape)
    for i in range(0,len(F[0])):
        U=[]
        for j in range(0,len(F[:,0])):
            #print(len(F[0:j][:,i]))
            # print("Shapes: Z ", F.shape, " -z ", z.shape)
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
            smooth = savgol_filter(y[i], 35,1,mode='interp')
            # smooth = y[i]
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

    effective_moveindex = len(F) - 1 if moveindex is None else min(moveindex, len(F) - 1)


    maxpos = np.mean(F[effective_moveindex][np.nanargmax(F[effective_moveindex])-3:np.nanargmax(F[effective_moveindex])+3])
    maxneg = np.mean(F[effective_moveindex][np.nanargmin(F[effective_moveindex])-3:np.nanargmin(F[effective_moveindex])+3])

    maxposerror = max(abs(F[effective_moveindex-1][x>0]))
    maxnegerror = max(abs(F[effective_moveindex-1][x<0]))
    
    return maxneg,maxpos,maxnegerror,maxposerror

def find_vertical_force(F,moveindex):
    if moveindex == None:
        moveindex = len(F[:,0])-1

    effective_moveindex = len(F) - 1 if moveindex is None else min(moveindex, len(F) - 1)

    
    idx = np.nanargmax(np.abs(F[effective_moveindex]))
    max_vertical = F[effective_moveindex][idx]


    # return np.mean(F[effective_moveindex][np.argmax(F[effective_moveindex])-10:np.argmax(F[effective_moveindex])+10])
    return max_vertical

def plot_single(ax,xarr,y,moveindex,colour,cutoff=None, label=None):

    # print("Plotting: ", label, " length of y: ", len(y), " moveindex: ", moveindex)
    if colour == 'default':
        cmap = plt.get_cmap('Blues')
    if colour == 'k':
        cmap=plt.get_cmap('Greys')
    #plot that shit
    if moveindex == None:
        moveindex = len(y[:,0])-1

    for j in range(len(y)):
        #x=xshiftarr[j]
        if j%1 == 0 or  j == len(y)-1:
            norm = mcolors.Normalize(vmin=1, vmax=moveindex - 1)
            colour = cmap(norm(j))
            if j >= len(y)-1:
                colour='tab:red'
                if cutoff is not None:
                    y[j][cutoff:] = np.nan
                    # xarr[cutoff:] = np.nan
            ax.plot(xarr,y[j],color=colour, lw=1)
            ax.set_title(label, fontsize=8)
            ax.legend(fontsize=6)
    
    
def remove_edges(a):
    cols_with_data = np.any(~np.isnan(a), axis=0)

    # Index of the last column with data
    last_data_col = np.where(cols_with_data)[0][-2]
    first_data_col = np.where(cols_with_data)[0][1]
    
    a[:, last_data_col:] = np.nan
    a[:, :first_data_col] = np.nan

    return a

def feven_deconv(zk, df, k, f0, A0, sgwin=6, sgdegree=3, tozero=0):
    """
    Sader–Jarvis inversion (converted from Philipp Rahe's Matlab Sader-Jarvis code, Ver 8, 2020) to recover conservative
    tip–sample force from frequency-gradient data.

    Parameters
    ----------
    zk : array
        z positions (m)
    df : array
        Frequency shift data (Hz).
    k : float
        Spring constant (N/m).
    f0 : float
        Resonant frequency (Hz).
    A0 : float
        Oscillation amplitude (m).
    sgwin, sgdegree : int
        Savitzky–Golay smoothing window length and polynomial degree.
        sgwin must be odd and > sgdegree for smoothing to be applied.
    tozero : int
        Number of far-z points over which to subtract mean from ktscap.
    """

    # print("Starting Sader–Jarvis Feven deconvolution...")
    # print(f"Shapes of inputs: zk={zk.shape}, df={df.shape}, k={k}, f0={f0}, A0={A0}")
    
    zk = np.asarray(-zk, dtype=float)
    ktscap = -2*k*df/f0
    ktscap = np.asarray(ktscap, dtype=float)
    N = len(zk)

    # --- Shift so far tail averages to zero ---
    if tozero > 0 and tozero < N:
        ktscap = ktscap - np.mean(ktscap[-tozero:])

    # # --- Optionally smooth & compute derivative ---
    # if (sgwin > 1) and (sgdegree >= 1) and (sgwin > sgdegree) and (sgwin % 2 == 1):
    #     # Smoothed derivative via Savitzky–Golay filter
    #     ddfdz = savgol_filter(ktscap, sgwin, sgdegree, deriv=1, delta=np.mean(np.diff(zk)))
    # else:
    # Central differences (fallback)
    ddfdz = np.zeros_like(ktscap)
    ddfdz[0] = (ktscap[1] - ktscap[0]) / (zk[1] - zk[0])
    ddfdz[1:-1] = (ktscap[2:] - ktscap[:-2]) / (zk[2:] - zk[:-2])
    ddfdz[-1] = (ktscap[-1] - ktscap[-2]) / (zk[-1] - zk[-2])

    # --- Local spacing array (central differences) ---
    ddz = np.zeros_like(zk)
    if zk[1] <= zk[0]:
        ddz[0] = zk[0] - zk[1]
    else:
        ddz[0] = zk[1] - zk[0]
    
    ddz[1:-1] = (zk[2:] - zk[:-2]) / 2.0
    ddz[-1] = zk[-1] - zk[-2]

    Feven = np.zeros(N-2)

    # --- Main loop over z_j ---
    for j in range(N-2):
        # Forward distances from current basepoint
        dz_seg = zk[(j+1):] - zk[j]
       
        
        mask = dz_seg > 0  # only forward, positive separation
        dz_seg = dz_seg[mask]
        kts_seg = ktscap[(j+1):][mask]
        ddf_seg = ddfdz[(j+1):][mask]

        if len(dz_seg) == 0:
            Feven[j] = np.nan
            continue

        # --- MAIN SADER–JARVIS INTEGRAL ---
        # This is the nasty integral in the SJ formula:
        #  F_even(z_j) = -∫_{z_j}^∞ [ (1 + sqrt(A0/(64π(u-z_j)))) ktscap(u)
        #                             - sqrt(A0^3/(2(u-z_j))) d(ktscap)/du ]
        #                 du
        # The kernel has a 1/sqrt(u - z_j) singularity at the lower limit,
        # so u=z_j is excluded here and handled by corr0..corr2 below.
        # print(np.shape(dz_seg), np.shape(kts_seg), np.shape(ddf_seg))
        int_term = integrate.trapezoid(
            (1 + np.sqrt(A0/(64*np.pi*dz_seg))) * kts_seg
            - np.sqrt(A0**3/(2*dz_seg)) * ddf_seg,
            dz_seg + zk[j]
        )

        # print(f"Integral term for j={j}: {int_term:.3e} N")

        # Corrections for singular part at u = z_j
        corr0 = -ktscap[j] * ddz[j]
        corr1 = -np.sqrt(A0/(16*np.pi)) * ktscap[j] * np.sqrt(abs(ddz[j]))
        corr2 =  np.sqrt(4*A0**3/2) * ddfdz[j] * np.sqrt(abs(ddz[j]))

        # Combine
        Feven[j] = int_term + corr0 + corr1 + corr2
        # print(f"Feven[{j}] = {Feven[j]:.3f} N")

    zFeven = zk[:N-2]
    return zFeven, Feven

def remove_background(F):
    F_new = np.zeros_like(F)
    for i, F_array in enumerate(F):
        F_new[i] = F_array - np.mean(F_array[:30])  # Subtract mean of first 30 points as background
    return F_new


def manip_analysis(pathname,fname,moveindex):

    folder = Path(pathname)
    #files = [f for f in folder.iterdir() if f.is_file() and str(fname) in f.name]
    files = [f.name for f in folder.iterdir() if f.is_file() and fname in f.name]
    



    dfarr,kzarr,xarr,zarr,Iarr,excarr, bias = makearrays(pathname,fname,files,moveindex)

    #### subtract off df background if needed ####
    # dfarr = remove_background(dfarr)
    # kzarr = remove_background(kzarr)

    xshiftarr = []

    for i in range(len(xarr[:,0])):
        xshift = xarr[i]-np.mean(xarr[i])
        ### put onto a uniform grid, this helps with any spiking seen due to non uniformity in xdata. this can happen when TCP is slightly slow
        x_uniform = np.linspace(min(xshift), max(xshift), len(xshift))
        xshiftarr.append(x_uniform)
        
    #kzarr = subtract_wdw(xshiftarr[5],kzarr,xshift[1],xshift[50])

    # for i in range(len(dfarr)):
    #     print(f"Plotting df curve {i+1}/{len(dfarr)} with shape {dfarr[i].shape} and xshift shape {xshiftarr[i].shape}")
    #     plt.plot(xshiftarr[0],dfarr[i],alpha=i/len(dfarr))
    # plt.show()


    print(f"Shapes of arrays: dfarr={dfarr.shape}, kzarr={kzarr.shape}, xshiftarr={len(xshiftarr)}, zarr={zarr.shape}, Iarr={Iarr.shape}, excarr={excarr.shape}")

    # print(f"\nDiagnostic check on zarr:")
    # print(f"zarr values: {zarr}")
    # print(f"zarr differences: {np.diff(zarr)}")
    # print(f"Is zarr monotonic? {np.all(np.diff(zarr) < 0) or np.all(np.diff(zarr) > 0)}")

    F_SJ_arr = []
    Z_SJ_arr = []

    
    # plt.plot(xshiftarr[0],dfarr[-1],alpha=1)

    # plt.plot(zarr)
    # plt.show()


    # for i in range(len(dfarr)):
        #  dfarr[i] = savgol_filter(dfarr[i], 55, 1, mode='interp')
    df_sj_arr = np.zeros_like(dfarr)
    for i,df in enumerate(dfarr):
        df_sj_arr[i] = savgol_filter(df, 9, 1, mode='interp')

    for i in range(np.shape(df_sj_arr)[1]):

        df_SJ = -np.abs(df_sj_arr[:, i])

        Z_SJ, F_SJ = feven_deconv(zarr, df_SJ, stiffness, resfreq, 100e-12, sgwin=9, sgdegree=3, tozero=0)

        F_SJ_arr.append(F_SJ)
        Z_SJ_arr.append(Z_SJ)
    
    #is there any nans in the force arrays?
    for i, F_SJ in enumerate(F_SJ_arr):
        if np.isnan(F_SJ).any():
            print(f"Warning: NaN values found in F_SJ_arr at index {i}")
        else:
            print(f"No NaN values in F_SJ_arr at index {i}")

    print(f"Completed Sader–Jarvis deconvolution for all curves. Shapes: F_SJ_arr={np.shape(F_SJ_arr)}, Z_SJ_arr={np.shape(Z_SJ_arr)}")

    # F_SJ_arr = np.array(np.transpose(F_SJ_arr))*-1
    # Z_SJ_arr = np.array(np.transpose(Z_SJ_arr))







    # print("Shapes of Z arrays: ", Z_SJ_arr.shape, zarr.shape)
    # for Z_arr in Z_SJ_arr:
    #     plt.plot(Z_arr)
    # plt.plot(zarr)
    # plt.show()

    # for i in range(len(F_SJ_arr[:,0])):
    #     plt.plot(Z_SJ_arr[i],F_SJ_arr[i],alpha=i/len(F_SJ_arr[:,0]))
    # plt.xlabel('z (nm)')
    # plt.ylabel('F (N)')
    # plt.title('Sader–Jarvis deconvolution results')
    # plt.show()

    # for i in range(len(F_SJ_arr)):
    #     plt.plot(xshiftarr[0],F_SJ_arr[i],alpha=i/len(dfarr[:,0]))
    #     plt.show()

    Fzarr = integrateKz(kzarr,zarr,False,None)

    # F_SJ_arr = np.flip(F_SJ_arr, axis=0)  # Flip the array along the first axis (curves)
    # F_SJ_arr = np.flip(F_SJ_arr, axis=0)
    

    Fzarr = np.array(Fzarr)
    F_SJ_arr = np.array(F_SJ_arr)
    Z_SJ_arr = np.array(Z_SJ_arr)[1]

    for i,arr in enumerate(F_SJ_arr):
        F_SJ_arr[i] = arr - arr[0]

    # F_SJ_arr = np.flipud(F_SJ_arr)
    # print(f"Shapes before plotting: F_SJ_arr={F_SJ_arr.shape}, Fzarr={Fzarr.shape}")

    # fig, [firstAx, TenthAx, Lastax] = plt.subplots(1, 3, figsize=(15, 5))
    # firstAx.plot(F_SJ_arr[5], c='r')
    # firstAx.plot(Fzarr[:,5], c='b')  # Flip the array along the first axis (curves)
    # TenthAx.plot(F_SJ_arr[10], c='r')
    # Lastax.plot(F_SJ_arr[-1], c='r')
    # TenthAx.plot(Fzarr[:,10], c='b')
    # Lastax.plot(Fzarr[:,-1], c='b')
    # plt.show()

    # fig, [firstAx, TenthAx, Lastax] = plt.subplots(1, 3, figsize=(15, 5))
    # firstAx.plot(F_SJ_arr[:,5], c='r')
    # firstAx.plot(Fzarr[5], c='b')  # Flip the array along the first axis (curves)
    # TenthAx.plot(F_SJ_arr[:,10], c='r')
    # Lastax.plot(F_SJ_arr[:,-1], c='r')
    # TenthAx.plot(Fzarr[10], c='b')
    # Lastax.plot(Fzarr[-1], c='b')
    # plt.show()

    # plt.plot(F_SJ_arr[150])
    # plt.plot(Fzarr[:,150])
    # plt.show()




    Fzarr_subtract = subtract_wdw(xshiftarr[0],Fzarr,xshift[1],xshift[50])



    F_SJ_arr = np.transpose(F_SJ_arr)

    FSJ_arr = subtract_wdw(xshiftarr[0],F_SJ_arr,xshift[1],xshift[50])
    # print(Z_SJ_arr.shape)
    # print(Z_SJ_arr.shape, zarr.shape)
    # print(len(xshiftarr))

    xshiftarr = np.array(xshiftarr)

   

    # print("Shapes: ", FSJ_arr.shape, Fzarr_subtract.shape,Z_SJ_arr.shape, xshiftarr.shape)
    Uarr_SJ,UarreV_SJ = getpot(xshiftarr,FSJ_arr,Z_SJ_arr,False)
    Uarr_SJ, UarreV_SJ = np.array(Uarr_SJ), np.array(UarreV_SJ)
    Ufit_SJ,UfiteV_SJ = fitting(xshiftarr,Uarr_SJ,'savgol',False,'Cs',False)

    Uarr_oli, UarreV_oli = getpot(xshiftarr,Fzarr_subtract,zarr,False)
    Ufit_oli,UfiteV_oli = fitting(xshiftarr,Uarr_oli,'savgol',False,'Cs',False)

    FLateral_SJ, Ferr_SJ = diffpot(xshiftarr,Ufit_SJ,None,False)
    FLateral_oli, Ferr_oli = diffpot(xshiftarr,Ufit_oli,None,False)

    FLateral_SJ_chop = remove_edges(FLateral_SJ)
    FLateral_oli_chop = remove_edges(FLateral_oli)

    # F_SJ_arr = remove_background(F_SJ_arr)

    # fig, ax = plt.subplots(3, 3, figsize=(12, 5))
    # plot_single(ax[0][0],xshiftarr[0],dfarr,moveindex,'default', label='Frequency shift (Hz)')
    # ax[0][0].set_xlabel('Lateral position (nm)')
    # ax[0][0].set_ylabel('Frequency shift (Hz)')

    # plot_single(ax[0][1],xshiftarr[0],kzarr,moveindex,'default', label='Frequency gradient (N/m)')
    # ax[0][1].set_xlabel('Lateral position (nm)')
    # ax[0][1].set_ylabel('Frequency gradient (N/m)')

    # plot_single(ax[1][1],xshiftarr[0],Fzarr_subtract*10**3,moveindex,'default', label='Vertical Force (pN)')
    # ax[1][0].set_xlabel('Lateral position (nm)')
    # ax[1][0].set_ylabel('Vertical Force (pN)')

    # plot_single(ax[1][0],xshiftarr[0],FSJ_arr*10**3,moveindex,'default', label='Vertical Force (pN) - Sader–Jarvis')
    # ax[1][1].set_xlabel('Lateral position (nm)')
    # ax[1][1].set_ylabel('Vertical Force (pN)')

    # plot_single(ax[2][0],xshiftarr[0],UarreV_SJ*10**-18*1000,moveindex,'default', label='Potential (meV) - Sader–Jarvis')
    # plot_single(ax[2][0],xshiftarr[0],UfiteV_SJ*10**-18*1000,moveindex,'default', label='Potential (meV) - Sader–Jarvis')
    # plot_single(ax[2][1],xshiftarr[0],UfiteV_oli*10**-18*1000,moveindex,'default', label='Potential (meV) - Integrated Fz')
    # ax[2][0].set_xlabel('Lateral position (nm)')
    # ax[2][0].set_ylabel('Potential_SJ (meV)')
    # ax[2][1].set_xlabel('Lateral position (nm)')
    # ax[2][1].set_ylabel('Potential_oli (meV)')


    # # print(F_SJ_arr[:,0].shape, xshiftarr[0].shape)
    # # plot_single(ax[1][2],zarr,F_SJ_arr[:,0],moveindex,'default', label='Current (A)')
    # ax[1][2].plot(zarr[:-2]*10**9,FSJ_arr[:,150]*10**3, label='Vertical Force vs Z (SJ)', c='tab:orange')
    # ax[1][2].plot(zarr*10**9,Fzarr_subtract[:,150]*10**3, label='Vertical Force vs Z (Integrated)', c='tab:blue')

    # plt.show()







    return xshiftarr, dfarr, df_sj_arr, kzarr, zarr, Iarr, excarr,bias,  Fzarr_subtract, F_SJ_arr, Z_SJ_arr, Uarr_SJ, UarreV_SJ,Ufit_SJ,UfiteV_SJ, Uarr_oli,UarreV_oli, Ufit_oli, UfiteV_oli, FLateral_SJ_chop, FLateral_oli_chop,


def plot_manips(pathname,fname,moveindex, plot=False):
    xshiftarr, dfarr, df_sj_arr, kzarr, zarr, Iarr, excarr, bias, Fzarr_subtract, F_SJ_arr, Z_SJ_arr, Uarr_SJ, UarreV_SJ, Ufit_SJ, UfiteV_SJ, Uarr_oli,UarreV_oli, Ufit_oli, UfiteV_oli, FLateral_SJ_chop, FLateral_oli_chop = manip_analysis(pathname,fname,moveindex)

    plotting_x = xshiftarr[0] #just take the first one, they should all be the same after interpolation onto uniform grid
    try:
        maxneg,maxpos,maxnegerror,maxposerror = findforce(plotting_x,FLateral_SJ_chop,moveindex)
        maxneg_oli,maxpos_oli,maxnegerror_oli,maxposerror_oli = findforce(plotting_x,FLateral_oli_chop,moveindex)
    except Exception as e:
        print(f"Error finding forces: {e}")
        maxneg = maxpos = maxnegerror = maxposerror = np.nan
        maxneg_oli = maxpos_oli = maxnegerror_oli = maxposerror_oli = np.nan

    F_SJ_arr_bck = remove_background(F_SJ_arr)
    
    max_vertical = find_vertical_force(F_SJ_arr_bck,len(F_SJ_arr_bck)-2)
    max_vertical_oli = find_vertical_force(Fzarr_subtract,len(Fzarr_subtract)-2)
    max_potential_SJ = find_vertical_force(UarreV_SJ,len(UarreV_SJ)-2)
    max_potential_oli = find_vertical_force(UarreV_oli,len(UarreV_oli)-2)

    print(f"Max vertical force (Sader–Jarvis): {max_vertical*10**3:.2f} pN")
    print(f"Max vertical force (Integrated Fz): {max_vertical_oli*10**3:.2f} pN")
    print(f"Max potential (Sader–Jarvis): {max_potential_SJ*10**-18*1000:.2f} meV")
    print(f"Max potential (Integrated Fz): {max_potential_oli*10**-18*1000:.2f} meV")
    print(f"Max lateral force (Sader–Jarvis): {maxpos*10**3:.2f} pN (positive), {maxneg*10**3:.2f} pN (negative)"
          f" with errors ±{maxposerror*10**3:.2f} pN (positive), ±{maxnegerror*10**3:.2f} pN (negative)")
    print(f"Max lateral force (Integrated Fz): {maxpos_oli*10**3:.2f} pN (positive), {maxneg_oli*10**3:.2f} pN (negative)"
          f" with errors ±{maxposerror_oli*10**3:.2f} pN (positive), ±{maxnegerror_oli*10**3:.2f} pN (negative)")
    # bias = get_info(pathname,fname)

    ##Abstract figure
    # fig,ax = plt.subplots(1,3, figsize=(10,10))
    # plt.rcParams.update({'font.size': 19})

    

    # plot_single(ax[0],plotting_x,dfarr,moveindex,'default', label='Frequency shift (Hz)')
    # ax[0].set_xlabel('Lateral position (nm)')
    # ax[0].set_ylabel('Frequency shift (Hz)')

    # plot_single(ax[1],plotting_x,F_SJ_arr_bck*10**3,moveindex,'default', label='Vertical Force (pN) - Sader–Jarvis')
    # ax[1].set_xlabel('Lateral position (nm)')
    # ax[1].set_ylabel('Vertical Force (pN)')

    # plot_single(ax[2], plotting_x, FLateral_SJ_chop*10**3,moveindex,'default', label='Lateral Force (pN) - Sader–Jarvis')
    # ax[2].set_xlabel('Lateral position (nm)')
    # ax[2].set_ylabel('Lateral Force (pN)')

    # ax[1].hlines(max_vertical*10**3,xmin=plotting_x[1],xmax=plotting_x[-1],ls='dashed',color='g',alpha=0.5)
    # ax[1].text(plotting_x[1],(max_vertical*10**3)*0.92,str(round(max_vertical*10**3,2))+'pN',fontsize=12)

    # ax[2].hlines(y=maxneg*10**3,xmin=plotting_x[1],xmax=plotting_x[-1],ls='dashed',color='k',alpha=0.5)
    # ax[2].hlines(y=maxpos*10**3,xmin=plotting_x[1],xmax=plotting_x[-1],ls='dashed',color='k',alpha=0.5)
    # ax[2].text(-1,maxneg*10**3+0.1,str(round(maxneg*10**3,2))+'pN',fontsize=12)
    # ax[2].text(-1.7,maxpos*10**3-0.6,str(round(maxpos*10**3,2))+'pN',fontsize=12)
    
    if plot == True:
        fig,ax=plt.subplots(2,3, figsize=(14,6))
        plt.rcParams.update({'font.size': 19})

        #current
        plot_single(ax[0][0],plotting_x,Iarr*10**12,moveindex,'default', label='Current')
        ax[0][0].set_xlabel('Lateral position (nm)')
        ax[0][0].set_ylabel('Current (pA)')

        # #df
        plot_single(ax[0][1],plotting_x,dfarr,moveindex,'default', label='Frequency shift (Hz)')
        plot_single(ax[0][1],plotting_x,df_sj_arr,moveindex,'k', label='Frequency shift (Hz) - Sader–Jarvis')
        ax[0][1].set_xlabel('Lateral position (nm)')
        ax[0][1].set_ylabel('Frequency shift (Hz)')
        #Stiffness
        plot_single(ax[0][2],plotting_x,kzarr,moveindex,'default', label='Stiffness (N/m)')
        ax[0][2].set_xlabel('Lateral position (nm)')
        ax[0][2].set_ylabel('Stiffness (N/m)')


        #Oli stuff

        # plot_single(ax[1][0],plotting_x,Fzarr_subtract*10**3,moveindex,'default', label='Vertical Force (pN) - Integrated Fz')
        # plot_single(ax[1][1],plotting_x,UarreV_oli*10**-18*1000,moveindex,'default', label='Vertical Force (pN) - Sader–Jarvis')
        # plot_single(ax[1][1],plotting_x,UfiteV_oli*10**-18*1000,moveindex,'k', label='Potential fit')
        # plot_single(ax[1][2],plotting_x,FLateral_oli_chop*10**3,moveindex,'default', label='Lateral Force (pN) - Integrated Fz')

        # ax[1][0].hlines(max_vertical_oli*10**3,xmin=plotting_x[1],xmax=plotting_x[-1],ls='dashed',color='g',alpha=0.5)
        # ax[1][0].hlines(np.max(Fzarr_subtract)*10**3,xmin=plotting_x[1],xmax=plotting_x[-1],ls='dashed',color='r',alpha=0.5)
        # # ax[2][1].hlines(np.max(F_SJ_arr_bck[-1])*10**3,xmin=plotting_x[1],xmax=plotting_x[-1],ls='dashed',color='b',alpha=0.5)
        # ax[1][0].text(plotting_x[1],(max_vertical_oli*10**3)*0.92,str(round(max_vertical_oli*10**3,2))+'pN',fontsize=12)
        # ax[1][0].set_xlabel('Lateral position (nm)')
        # ax[1][0].set_ylabel('Vertical Force (pN)')

        # ax[1][1].hlines(max_potential_oli*10**-18*1000,xmin=plotting_x[1],xmax=plotting_x[-1],ls='dashed',color='g',alpha=0.5)
        # ax[1][1].text(plotting_x[1],(max_potential_oli*10**-18*1000)*0.92,str(round(max_potential_oli*10**-18*1000,2))+'meV',fontsize=12)
        # # ax[1][1].hlines(np.max(Fzarr_subtract)*10**3,xmin=plotting_x[1],xmax=plotting_x[-1],ls='dashed',color='r',alpha=0.5)

        # ax[1][2].hlines(y=maxneg_oli*10**3,xmin=plotting_x[1],xmax=plotting_x[-1],ls='dashed',color='k',alpha=0.5)
        # ax[1][2].hlines(y=maxpos_oli*10**3,xmin=plotting_x[1],xmax=plotting_x[-1],ls='dashed',color='k',alpha=0.5)
        # ax[1][2].text(-1,maxneg_oli*10**3+0.1,str(round(maxneg_oli*10**3,2))+'pN',fontsize=12)
        # ax[1][2].text(-1.7,maxpos_oli*10**3-0.6,str(round(maxpos_oli*10**3,2))+'pN',fontsize=12)
        # ax[1][2].set_xlabel('Lateral position (nm)')
        # ax[1][2].set_ylabel('Lateral Force (pN)')


        #SJ stuff

        plot_single(ax[-1][0],plotting_x,F_SJ_arr_bck*10**3,moveindex,'default', label='Vertical Force (pN) - Sader–Jarvis')
        plot_single(ax[-1][1],plotting_x,UarreV_SJ*10**-18*1000,moveindex,'default', label='Potential (meV) - Sader–Jarvis')
        plot_single(ax[-1][1],plotting_x,UfiteV_SJ*10**-18*1000,moveindex,'k', label='Potential fit')
        plot_single(ax[-1][2],plotting_x,FLateral_SJ_chop*10**3,moveindex,'default', label='Lateral Force (pN) - Sader–Jarvis')

        ax[-1][1].hlines(max_potential_SJ*10**-18*1000,xmin=plotting_x[1],xmax=plotting_x[-1],ls='dashed',color='g',alpha=0.5)
        ax[-1][1].text(plotting_x[1],(max_potential_SJ*10**-18*1000)*0.92,str(round(max_potential_SJ*10**-18*1000,2))+'meV',fontsize=12)
        #vertical force w/out background
        # plot_single(ax[1][0],plotting_x,F_SJ_arr_bck*10**3,moveindex,'default', 'Vertical Force (pN)')

        ax[-1][0].hlines(max_vertical*10**3,xmin=plotting_x[1],xmax=plotting_x[-1],ls='dashed',color='g',alpha=0.5)
        ax[-1][0].hlines(np.max(F_SJ_arr_bck[-2])*10**3,xmin=plotting_x[1],xmax=plotting_x[-1],ls='dashed',color='r',alpha=0.5)
        # ax[-1][1].hlines(np.max(F_SJ_arr_bck[-1])*10**3,xmin=plotting_x[1],xmax=plotting_x[-1],ls='dashed',color='b',alpha=0.5)
        ax[-1][0].text(plotting_x[1],(max_vertical*10**3)*0.92,str(round(max_vertical*10**3,2))+'pN',fontsize=12)
        ax[-1][0].set_xlabel('Lateral position (nm)')
        ax[-1][0].set_ylabel('Vertical Force (pN)')

        # #vertical force with background
        # plot_single(ax[1][1],plotting_x,F_SJ_arr*10**3,moveindex,'default', label='Vertical Force (pN)')

        # Potential fit
        # plot_single(ax[0][2],plotting_x,UfiteV_SJ*10**-18*1000,moveindex,'k', label='Potential fit')

        # #Potential raw
        # plot_single(ax[0][2],plotting_x,UarreV_SJ*10**-18*1000,moveindex,'default', label='Potential (meV)')
        # #lateral force
        # plot_single(ax[1][2],plotting_x,FLateral_SJ_chop*10**3,moveindex,'default', label='Lateral Force (pN)')

        #dashed lines showing max lateral force
        ax[-1][2].hlines(y=maxneg*10**3,xmin=plotting_x[1],xmax=plotting_x[-1],ls='dashed',color='k',alpha=0.5)
        ax[-1][2].hlines(y=maxpos*10**3,xmin=plotting_x[1],xmax=plotting_x[-1],ls='dashed',color='k',alpha=0.5)
        ax[-1][2].text(-1,maxneg*10**3+0.1,str(round(maxneg*10**3,2))+'pN',fontsize=12)
        ax[-1][2].text(-1.7,maxpos*10**3-0.6,str(round(maxpos*10**3,2))+'pN',fontsize=12)
        ax[-1][2].set_xlabel('Lateral position (nm)')
        ax[-1][2].set_ylabel('Lateral Force (pN)')

        plt.suptitle(pathname.split("\\")[-1] + " - Bias: " + str(bias) + " V - Move Index: " + str(moveindex), fontsize=16)

    return plotting_x, Iarr, dfarr, FLateral_SJ_chop, [np.abs(maxneg)*10**3, np.abs(maxpos)*10**3], UarreV_SJ, UfiteV_SJ, excarr
#######################################




# pathname_1 = r"C:\Users\ppxfc1\OneDrive - The University of Nottingham\Desktop\PhD\DBTF\manips\Pb\Manip_14_500mV"
# pathname = r"C:\Users\ppxfc1\OneDrive - The University of Nottingham\Desktop\PhD\DBTF\manips\Cu\Manip_9_0"
pathname_1 = r"C:\Users\ppxfc1\OneDrive - The University of Nottingham\Desktop\PhD\DBTF\manips\Ag\manip 5"
pathname_2 = r"C:\Users\ppxfc1\OneDrive - The University of Nottingham\Desktop\PhD\DBTF\manips\Ag\manip 10"
pathname_3 = r"C:\Users\ppxfc1\OneDrive - The University of Nottingham\Desktop\PhD\DBTF\manips\Ag\manip 8"
pathname_4 = r"C:\Users\ppxfc1\OneDrive - The University of Nottingham\Desktop\PhD\DBTF\manips\Ag\manip 14"


# fname_1 = r'manip_DBTF_1_'
# fname_1 = r'manip_DBTF_1_'
# fname = r"Move_DBTF_Cu_0mV_"
fname_1 = r'manip_DBTF_5_'
fname_2 = r'manip_DBTF_10_'
fname_3 = r'manip_DBTF_8_'
fname_4 = r'manip_DBTF_14_'
#fname = r'down_manip_34_0_-0.4_'
#fname = r'manip_DBTF_3_'
moveindex_5 = 14
moveindex_10 = 41
moveindex_8 = 60
moveindex_14 = 53

m=output_data_spectra_dat()
stiffness = 900 #N/m
resfreq = 31515 #Hz

plotting_x_5, Iarr_5, dfarr_5, FLateral_SJ_chop_5, Values_5, UarreV_SJ_5, UfiteV_SJ_5, excarr_5 = plot_manips(pathname_1,fname_1,moveindex_5, plot = False)

plotting_x_10, Iarr_10, dfarr_10, FLateral_SJ_chop_10, Values_10, UarreV_SJ_10, UfiteV_SJ_10, excarr_10 = plot_manips(pathname_2,fname_2,moveindex_10, plot = False)

plotting_x_8, Iarr_8, dfarr_8, FLateral_SJ_chop_8, Values_8, UarreV_SJ_8, UfiteV_SJ_8, excarr_8 = plot_manips(pathname_3,fname_3,moveindex_8, plot = False)

plotting_x_14, Iarr_14, dfarr_14, FLateral_SJ_chop_14, Values_14, UarreV_SJ_14, UfiteV_SJ_14, excarr_14 = plot_manips(pathname_4,fname_4,moveindex_14, plot = False)




# xs = [plotting_x_14, plotting_x_10, plotting_x_8, plotting_x_5]
# Is = [Iarr_14, Iarr_10, Iarr_8, Iarr_5]
# dfs = [dfarr_14, dfarr_2, dfarr_8, dfarr_1]
# FLs = [FLateral_SJ_chop_14, FLateral_SJ_chop_2, FLateral_SJ_chop_8, FLateral_SJ_chop_1]
# values = [Values_14[0], Values_2[0], Values_8[0], Values_1[0], Values_14[1], Values_2[1], Values_8[1], Values_1[1]]
# UarreV_SJs = [UarreV_SJ_14, UarreV_SJ_2, UarreV_SJ_8, UarreV_SJ_1]
# labels = ['Manip 14', 'Manip 2', 'Manip 8', 'Manip 1']

# mean = np.mean(values)
# error = np.std(values)

# print(f"Mean of max lateral forces: {mean:.2f} pN")
# print(f"Standard deviation: {error:.2f} pN")





# pathname_2 = r"C:\Users\ppxfc1\OneDrive - The University of Nottingham\Desktop\PhD\DBTF\manips\Ag\manip 14"
# fname_2 = r'manip_DBTF_14_'
# moveindex_2 = 53

# plotting_x_14, Iarr_14, dfarr_14, FLateral_SJ_chop_14 = plot_manips(pathname_2,fname_2,moveindex_2, plot = True )

# plt.show()

plt.rcParams.update({'font.size': 20})
# fig, [[I14, df14, FLateral14],[I11, df11, FLateral11] ] = plt.subplots(2, 3, figsize=(10, 5))
fig, I14 = plt.subplots(1, 1, figsize=(10, 8))

# x_min, x_max = -2.5, 2.5
y_min, y_max = -4, 13




plotting, UarreV_SJ = plotting_x_14, UarreV_SJ_14

moveindex = moveindex_14

plot_single(I14,plotting,FLateral_SJ_chop_14*10**3,moveindex, 'default', cutoff = 500)
# plot_single(I14,plotting,((UfiteV_SJ_14*10**-18*1000)),moveindex, 'k', cutoff = 500)
I14.set_xlabel('Lateral position (nm)')
I14.set_ylabel('Potential (meV)')

# I14.set_xlim(x_min, x_max)
# I14.set_ylim(y_min, y_max)

fig, Uarr5 = plt.subplots(1, 1, figsize=(10, 8))
plot_single(Uarr5,plotting_x_5,FLateral_SJ_chop_5*10**3,moveindex_5, 'default', cutoff = 500)
# plot_single(Uarr5,plotting_x_5,((UfiteV_SJ_5*10**-18*1000)),moveindex_5, 'k', cutoff = 500)
Uarr5.set_xlabel('Lateral position (nm)')
Uarr5.set_ylabel('Potential (meV)')

# Uarr5.set_xlim(x_min, x_max)
# Uarr5.set_ylim(y_min, y_max)

fig, uArr10 = plt.subplots(1, 1, figsize=(10, 8))
plot_single(uArr10,plotting_x_10,FLateral_SJ_chop_10*10**3,moveindex_10, 'default', cutoff = 500)
# plot_single(uArr10,plotting_x_10,((UfiteV_SJ_10*10**-18*1000)),moveindex_10, 'k', cutoff = 500)
uArr10.set_xlabel('Lateral position (nm)')
uArr10.set_ylabel('Potential (meV)')

# uArr10.set_xlim(x_min, x_max)
# uArr10.set_ylim(y_min, y_max)

# Flateral16.set_ylim(-5.5, 5)





# plot_single(I14,plotting_x_14,((Iarr_14-np.min(Iarr_14))*10**12),moveindex_2, 'default')
# plot_single(df14,plotting_x_14,dfarr_14,moveindex_2,'default', cutoff = 500)
# plot_single(FLateral14,plotting_x_14,FLateral_SJ_chop_14*10**3,moveindex_2,'default', cutoff = 217)


# plot_single(I11,plotting_x_11,((-1*Iarr_11-np.min(-1*Iarr_11))*10**12),moveindex_1, 'default',cutoff = 500)
# plot_single(df11,plotting_x_11,dfarr_11,moveindex_1,'default', cutoff = 500)
# plot_single(FLateral11,plotting_x_11,FLateral_SJ_chop_11*10**3,moveindex_1,'default', cutoff = 197)
# 
# I14.set_xlim(-2.85, 2.85)
# I11.set_xlim(-2.85, 2.85)
# df14.set_xlim(-2.85, 2.85)
# df11.set_xlim(-2.85, 2.85)
# FLateral14.set_xlim(-2.85, 2.55)
# FLateral11.set_xlim(-2.85, 2.55)

# I14.set_ylim(0,0.85)
# I11.set_ylim(0,0.85)
# df14.set_ylim(-3.1, 0.25)
# df11.set_ylim(-3.1, 0.25)
# FLateral14.set_ylim(-2.7, 3.1)
# # FLateral11.set_ylim(-2.7, 3.1)

# df11.set_xlabel('Lateral position (nm)')
# I11.set_xlabel('Lateral position (nm)')
# FLateral11.set_xlabel('Lateral position (nm)')

# I14.set_ylabel('Current (pA)')
# df14.set_ylabel('Frequency shift (Hz)')
# FLateral14.set_ylabel('Lateral Force (pN)')

# I11.set_ylabel('Current (pA)')
# df11.set_ylabel('Frequency shift (Hz)')
# FLateral11.set_ylabel('Lateral Force (pN)')






plt.show()


