from read_spectra_new import output_data_spectra_dat
from KPFM_spectrum_analysis import KPFMSpectrumAnalysis
import scipy
import numpy as np
import os
import matplotlib.pyplot as plt
# from scipy.integrate import trapz
# print(scipy.__version__)

class Spectrum(output_data_spectra_dat):
    
    def __init__(self, path, fileName, channel):
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
        
        self.x, self.y = self._OpenSpectra(path, fileName, channel)
        
    
    
    def _OpenSpectra(self, path, fileName, channel):
        """
        Parameters
        ----------
        path : str
            path to spectra file.
        filename : str
            spectra file (dat file).
        channel : str
            Channel to read.
            

        Returns
        -------
        x : 1D array
            spectra x data. For KPFM it'll be bias in Volts.
        y : 1D array
            spectra y data. For KPFM it'll be freq shift in Hz.

        """
        self.get_file(path, fileName)  
        
        # if channel not in file, print list of possible channels
        if channel not in list(self.df): 
            print('Choice of channel not found in ' + fileName)
            self.show_method_fun()
         
        
        x = self.give_data(0)[0] 
        y = self.give_data(channel)[0]

        return x, y 
    
    
    
    # =========================================================================
    # KPFM analysis
    # =========================================================================
    
    def KPFMAnalysis(self, fit_range=25, xAdatomCentre=None, yAdatomCentre=None, 
                     plotCalculation=False, axFit=None, axResiduals=None, e_min=None, e_max=None, ):
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
        
        kpfmAnalysis = KPFMSpectrumAnalysis(bias=self.x, df=self.y, fit_range=fit_range)
        self.E_min = e_min
        self.E_max = e_max
        self.vContact = kpfmAnalysis.CalcVContact(E_min=e_min, E_max=e_max)
        
        self.fit = kpfmAnalysis.fit
        self.residuals = kpfmAnalysis.fitInfo.residual
        self.dfAtVContact = kpfmAnalysis.dfAtVContact
        self.vContactErr = kpfmAnalysis.vContactErr
        self.dfAtVContactErr = kpfmAnalysis.dfAtVContactErr
        self.fitInfo = kpfmAnalysis.fitInfo
        self.bias = kpfmAnalysis.bias
        
        if xAdatomCentre != None and yAdatomCentre != None:
            self.r = kpfmAnalysis.CalcR(self.x_pos, self.y_pos, xAdatomCentre, yAdatomCentre)
            
        if plotCalculation == True: 
            axFit, axResiduals, axDataMinusFit = kpfmAnalysis.PlotVContactCalculation(axFit, axResiduals)
            return axFit, axResiduals, axDataMinusFit
        
### open "C:\Users\ppxfc1\OneDrive - The University of Nottingham\Desktop\PhD\Code\PhD-Codes\seder-jarvis\df_sample.txt"
def c_term(df, z, A, j):
        # print("c_term - j", j)
        c_term_1 = df[j]*(z[j+1] - z[j])
        c_term_2 = 2*(np.sqrt(A)/(8*np.sqrt(np.pi)))*df[j]*np.sqrt(z[j+1] - z[j])
        c_term_3 = -2*((A**(3/2))/np.sqrt(2))*((df[j+1] - df[j])/(z[j+1] - z[j]))*np.sqrt(z[j+1] - z[j])
        c_term = c_term_1 + c_term_2 + c_term_3
        return c_term

def g_l (z, df,A,  l, j):
    # print("g_l - l", l)
    # print("g_l - j", j)
    g_k = (1+(np.sqrt(A)/(8*np.sqrt(np.pi*(z[l] - z[j])))))*df[l] - ((A**(3/2))/(np.sqrt(2*(z[l]-z[j]))))*((df[l+1]-df[l])/(z[l+1]-z[l]))
    return g_k

def sum_term_calc(z, df, A, j, N ):
    # print("Sum_term_calc - j", j)
    i = j+1
    sum_term = 0
    while i < N-2:
        sum_term =+ (z[i+1]-z[i])*((g_l(z,df,A,i+1,j)+g_l(z,df,A,i,j))/2)
        i += 1 

    return sum_term

def F_j(z, df, A, j, f_0, N):
    # print("F_J - j", j)
    sum_term = sum_term_calc(z, df, A, j, N)
    coef_term = c_term(df, z, A, j)
    F_j = (2*k_spring/f_0)*(coef_term + sum_term)
    return F_j

def calc_force_array (z, df, A, f_0, N):
    forces = np.zeros(N)
    for i in range(N-1):
        forces[i] = F_j(z, df, A, i, f_0, N)
    return forces

def calc_force_trapz (z, df, A, k, f_0, abs_YN = True):
    print(f_0)
    Omega = df/f_0
    dOmega_dz = np.diff(Omega)/np.diff(z)

    z = z[:-1]
    # Delta_f = Delta_f[:-1]
    Omega = Omega[:-1]
    # print(len(z), len(Omega), len(dOmega_dz))
    force = np.zeros(len(z) - 2)
    # print("Omega ", Omega)
    for j in range(len(z) - 2):
        # start at j+1 due to pole at t=z
        t = z[j+1:]
        
        # adjust length of Omega and dOmega_dz to length of t
        Omega_tmp = Omega[j+1:]
        dOmega_dz_tmp = dOmega_dz[j+1:]
        
        # # calculate integral Eq.(9) in [1]
        # print("testing sqrts 1", (t - z[j]))
        # print("testing sqrts 2", (t - z[j]))

        ### Abs to stop negative values, added 11:05, 29/10/24
        # abs_YN = True
        if abs_YN:

            integral = np.trapz((1 + np.sqrt(A) / (8 * np.sqrt(np.pi * abs(t - z[j])))) * Omega_tmp - 
                            A**(3/2) / np.sqrt(2 * abs(t - z[j])) * dOmega_dz_tmp, t)
            
            # correction terms for t=z from [2]
            corr1 = Omega[j] * (z[j+1] - z[j])
            corr2 = 2 * (np.sqrt(A) / (8 * np.sqrt(np.pi))) * Omega[j] * np.sqrt(abs(z[j+1] - z[j]))
            corr3 = (-2) * (A**(3/2) / np.sqrt(2)) * dOmega_dz[j] * np.sqrt(abs(z[j+1] - z[j]))

        else:
            
            inner = (1 + np.sqrt(A) / (8 * np.sqrt(np.pi * (t - z[j])))) * Omega_tmp - A**(3/2) / np.sqrt(2 * (t - z[j])) * dOmega_dz_tmp
        
            integral = np.trapz(inner, t)
            
            # correction terms for t=z from [2]
            corr1 = Omega[j] * (z[j+1] - z[j])
            corr2 = 2 * (np.sqrt(A) / (8 * np.sqrt(np.pi))) * Omega[j] * np.sqrt((z[j+1] - z[j]))
            corr3 = (-2) * (A**(3/2) / np.sqrt(2)) * dOmega_dz[j] * np.sqrt((z[j+1] - z[j]))

        
        #### THIS IS ALL DEBUG TESTING TO FIND OUT WHAT IS DIFFERENT BETWEEN PY AND MATLAB
        # if j%15 == 0:
            # print("j=",j, " Inner first = ", Omega_tmp)
            # print("j =", j, "of" , corr1, " ",  corr2, " ", corr3," ", integral) 

        force[j] = 2 * k * (corr1 + corr2 + corr3 + integral)
    return force
# and read the data
if __name__ == "__main__":
    with open("C:\\Users\\ppxfc1\\OneDrive - The University of Nottingham\\Desktop\\PhD\\Code\\PhD-Codes\\seder-jarvis\\df_sample.txt") as f:
        data = f.readlines()
        new_data = []
        for line in data:
            line = line.split()
            new_data.append([float(i) for i in line])
    data = np.array(new_data)

    atom_z_rel = data[:, 0]
    atom_df = data[:, 1]

    amplitude = 1e-10 #in pm
    k_spring = 2000 #in N/m
    frequency_res = 20000 #in Hz


    

    # forces = calc_force_array(atom_z_rel, atom_df, amplitude, frequency_res, len(atom_z_rel))
    # forces_altered = calc_force_array(atom_z_rel - amplitude, atom_df, amplitude, frequency_res, len(atom_z_rel))
    forces_trapz = calc_force_trapz(atom_z_rel, atom_df, amplitude, k_spring, frequency_res)

    # forces_nN = forces * 1e9
    # forces_altered_nN = forces_altered * 1e9
    forces_trapz_nN = forces_trapz * 1e9

    forces_matlab_nN = 1e-10 * np.array([-0.3870,-0.3858,-0.4025,   -0.4025,   -0.4121,   -0.4210,   -0.4379,   -0.4448,   -0.4674,   -0.4784,   -0.5070,   -0.5162,   -0.5431,   -0.5604,   -0.5684,   -0.5640,   -0.5696,   -0.5491,   -0.5266,   -0.5007,   -0.4607,   -0.4200,   -0.3825,   -0.3461,   -0.2987,   -0.2685,   -0.2311,   -0.2020,   -0.1904, -0.1504,   -0.1424,   -0.1213,   -0.1029,   -0.1011,   -0.0815,   -0.0670,   -0.0710,   -0.0491,   -0.0586,   -0.0451,   -0.0306,   -0.0420,   -0.0318,   -0.0269,   -0.0296,   -0.0202,   -0.0303,   -0.0099,   -0.0224,   -0.0106,   -0.0178,   -0.0097,   -0.0181,   -0.0112,-0.0067,   -0.0047,   -0.0078,   -0.0033,   -0.0082,   -0.0096,   -0.0098,   -0.0070,    0.0044,   -0.0067,   -0.0077,   -0.0001,   -0.0063,    0.0029,-0.0060,   -0.0037,   -0.0070,-0.0030,    0.0030,   -0.0067,    0.0077,   -0.0060,   -0.0018,    0.0070,   -0.0061,    0.0011,   -0.0040,    0.0004,   -0.0056,   -0.0057,    0.0033,    0.0036,   -0.0046,   -0.0044,   -0.0033,    0.0031,   -0.0001,    0.0003,   -0.0029,    0.0031,   -0.0057,    0.0017,0.0040, -0.0045])


    fig, [dfAx, fAx, diffAx] = plt.subplots(3, 1)

    dfAx.plot(atom_z_rel, atom_df, label='df')
    # fAx.plot(atom_z_rel, forces_nN, label='Forces')
    # fAx.plot(atom_z_rel, forces_altered_nN, alpha = 0.5, label='Forces altered')
    plt.xlabel('Z position (nm)')
    dfAx.set_ylabel('df (Hz)')
    fAx.set_ylabel('Force (nN)')
    # plt.title('df vs Z position')
    plt.tick_params(direction='in')


    mathematica_x = []
    mathematica_force = []

    with open(r"C:\Users\ppxfc1\OneDrive - The University of Nottingham\Desktop\PhD\Code\PhD-Codes\seder-jarvis\Force_SJ.csv") as file:
        for i in range(10):
            file.readline()
        for line in file:
            # print(line)
            line = line.split(",")
            mathematica_x.append(float(line[0]))
            mathematica_force.append(float(line[1]))

    mathematica_x = np.array(mathematica_x)
    mathematica_force = np.array(mathematica_force)
    # if len(mathematica_force) == len(forces):
    #     dif_forces = mathematica_force - forces
    #     diffAx.plot(atom_z_rel, dif_forces, label='Mathematica F - Forces')
    #     df_forces = atom_df - mathematica_force
    mathematica_force = np.append(mathematica_force, mathematica_force[-3:])
    mathematica_x = np.append(mathematica_x, mathematica_x[-3:])
    forces_matlab = np.append(forces_matlab_nN, forces_matlab_nN[-3:]) * 1e9
    forces_trapz_nN = np.append(forces_trapz_nN, forces_trapz_nN[-3:])




    mathematica_force_nN = mathematica_force * 1e9 
    fAx.plot(mathematica_x, mathematica_force_nN, alpha=0.5, label='Mathematica force')
    fAx.plot(atom_z_rel, forces_matlab, alpha=0.5, label='Matlab force')
    fAx.plot(atom_z_rel, forces_trapz_nN, alpha=0.5, label='Forces trapz')
    # mathematica.plot(mathematica_/x, mathematica_force
    # mathematica.set_ylabel('Force (nN)')

    print(atom_z_rel)
    print(atom_df)


    # print(len(forces), len(mathematica_force_nN))
    diffAx.set_ylabel('Difference in force (nN)')
    diffAx.set_xlabel('Z position (nm)')
    # diffAx.plot(atom_z_rel, df_forces, label='df - Mathematica F')
    # diffAx.plot(atom_z_rel, df_forces2, label='df - Forces')



    #repeat the last 2 lines for the mathematica force

    diff_forces = mathematica_force_nN - forces_trapz_nN
    diff_matlab = mathematica_force_nN - forces_matlab
    diffAx.plot(atom_z_rel, diff_forces, label='Mathematica F - Forces')
    diffAx.plot(atom_z_rel, diff_matlab, label='Mathematica F - Matlab F')

    diffAx.legend()
    dfAx.legend()
    fAx.legend()

    plt.show()

