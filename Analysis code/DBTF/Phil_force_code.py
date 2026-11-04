"""
Python version of Philipp Rahe's Matlab Sader-Jarvis code Ver 08, 14/04/2020
with simulated Lennard-Jones potential and small oscillation amplitude approximation
for testing
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.integrate import trapezoid

# ---------------------------------
# 1. True Lennard–Jones parameters
# ---------------------------------

eps   = 1.602e-19   # J  (~1 eV depth). Stick to SI units -- fewer conversion factors to worry about
sigma = 1e-9    # m  -> well min ~ 1.0 nm

# Sensor parameters
k  = 1800.0        # N/m (i.e. use traditional convention in qPlus community and assume this is some type of universal, error-free physical constant.) :-)
f0 = 28763.0       # Use experimental resonant frequency (because why not?)
A  = 50e-12        # m  (oscillation amplitude)

# z vector
z = np.linspace(0.96e-9, 3.0e-9, 800)  # Set z range so that potential ranges between ~ -1 eV at minimum and +1 eV at maximum

# L-J potential and force
U_true = 4*eps*((sigma/z)**12 - (sigma/z)**6)
F_true = -np.gradient(U_true, z)  # Force = -dU/dz

# -----------------------------------------------------------------
# 2. Simulate Δf(z) from F(z)
# FM-AFM expression: see Eqn 1 of Sader & Jarvis APL 84 1801 (2004)
# -----------------------------------------------------------------

def delta_f_from_F(z_array, F_array, A, k, f0):
    """Simulate Δf(z) from F(z) using full FM-AFM expression."""
    df = np.zeros_like(z_array)
    for i, z0 in enumerate(z_array):
        th = np.linspace(0, np.pi, 400)
        integrand = F_array_interp(z0 + A*(1 + np.cos(th))) * np.cos(th)
        df[i] = -f0/(np.pi*k*A) * trapezoid(integrand, th)
    return df

# Linear interpolation helper for F(z)
def F_array_interp(x):
    return np.interp(x, z, F_true, left=F_true[0], right=0.0)

df_sim = delta_f_from_F(z, F_true, A, k, f0)

# --------------------------------
# 3. Small amplitude approximation
# --------------------------------

def force_small_amplitude(z, df, k, f0):
    dFdz = -(2*k/f0) * df
    F = trapezoid_cum(dFdz, z)
    # Anchor far tail to zero
    i_anchor = np.argmin(np.abs(df))
    F -= F[i_anchor]
    return F

def trapezoid_cum(y, x): # cumulative integration
    out = np.zeros_like(y)
    out[1:] = np.cumsum(0.5*(y[:-1] + y[1:]) * np.diff(x))
    return out

F_SA = force_small_amplitude(z, df_sim, k, f0)

# ---------------------------------------------------------------------------------------------------------
# 4. Sader–Jarvis (Feven) algorithm: Python version translated from P. Rahe Matlab code Ver 08, 14/04/2020
# ---------------------------------------------------------------------------------------------------------

def ktscap_from_df(df, k, f0):
    """Rewrite Δf(z) as force gradient."""
    return -2*k*df/f0  # N/m

from scipy.signal import savgol_filter
from scipy.integrate import trapezoid
import numpy as np

def feven_deconv(zk, df, k, f0, A0, sgwin=9, sgdegree=3, tozero=50):
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

    print("Starting Sader–Jarvis Feven deconvolution...")
    print(f"Shapes of inputs: zk={zk.shape}, df={df.shape}, k={k}, f0={f0}, A0={A0}")
    
    zk = np.asarray(zk, dtype=float)
    ktscap = -2*k*df/f0
    ktscap = np.asarray(ktscap, dtype=float)
    N = len(zk)

    # --- Shift so far tail averages to zero ---
    if tozero > 0 and tozero < N:
        ktscap = ktscap - np.mean(ktscap[-tozero:])

    # --- Optionally smooth & compute derivative ---
    if (sgwin > 1) and (sgdegree >= 1) and (sgwin > sgdegree) and (sgwin % 2 == 1):
        # Smoothed derivative via Savitzky–Golay filter
        ddfdz = savgol_filter(ktscap, sgwin, sgdegree, deriv=1, delta=np.mean(np.diff(zk)))
    else:
        # Central differences (fallback)
        if sgwin != 0:
            print("Warning: invalid Savitzky–Golay parameters, using central differences.")
        ddfdz = np.zeros_like(ktscap)
        ddfdz[0] = (ktscap[1] - ktscap[0]) / (zk[1] - zk[0])
        ddfdz[1:-1] = (ktscap[2:] - ktscap[:-2]) / (zk[2:] - zk[:-2])
        ddfdz[-1] = (ktscap[-1] - ktscap[-2]) / (zk[-1] - zk[-2])

    # --- Local spacing array (central differences) ---
    ddz = np.zeros_like(zk)
    ddz[0] = zk[1] - zk[0]
    ddz[1:-1] = (zk[2:] - zk[:-2]) / 2.0
    ddz[-1] = zk[-1] - zk[-2]

    Feven = np.zeros(N-2)

    # --- Main loop over z_j ---
    for j in range(N-2):
        # Forward distances from current basepoint
        dz_seg = zk[(j+1):] - zk[j]
        
        # dz_seg *= -1
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
        
        int_term = -trapezoid(
            (1 + np.sqrt(A0/(64*np.pi*dz_seg))) * kts_seg
            - np.sqrt(A0**3/(2*dz_seg)) * ddf_seg,
            dz_seg + zk[j]
        )

        # Corrections for singular part at u = z_j
        corr0 = -ktscap[j] * ddz[j]
        corr1 = -np.sqrt(A0/(16*np.pi)) * ktscap[j] * np.sqrt(ddz[j])
        corr2 =  np.sqrt(4*A0**3/2) * ddfdz[j] * np.sqrt(ddz[j])

        # Combine
        Feven[j] = int_term + corr0 + corr1 + corr2
        print(f"Feven[{j}] = {Feven[j]:.3f} N")

    zFeven = zk[:N-2]
    
    return zFeven, Feven
if __name__ == "__main__":
    ktscap_sim = ktscap_from_df(df_sim, k, f0)




    z_SJ, F_SJ = feven_deconv(z, df_sim, k, f0, 50e-12)

    # -------------------
    # 5. Plot comparison
    # -------------------
    plt.figure(figsize=(7,5))
    plt.plot(z*1e9, F_true*1e9, 'k-', label="True LJ force")
    plt.plot(z*1e9, F_SA*1e9, 'g--', label="Small-amplitude approx.")
    plt.plot(z_SJ*1e9, F_SJ*1e9, 'r-.', label="Sader–Jarvis Feven")
    plt.axhline(0, color='k', lw=0.5)
    plt.xlabel("z (nm)")
    plt.ylabel("F(z) [nN]")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Plot potential for reference
    plt.figure(figsize=(6,4))
    plt.plot(z*1e9, U_true/1.602e-19, 'b-')
    plt.axhline(0, color='k', lw=0.5)
    plt.xlabel("z (nm)")
    plt.ylabel("U(z) [eV]")
    plt.title("True Lennard–Jones potential")
    plt.tight_layout()
    plt.show()
