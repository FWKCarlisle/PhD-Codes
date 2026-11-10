import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

# =====================================================
# USER-ADJUSTABLE PARAMETERS
# =====================================================

EA = 0.005                # Activation energy (eV)
A = 1.0e13               # Arrhenius prefactor (s^-1)
kB = 8.617333262e-5      # Boltzmann constant (eV/K)

beta_min = 0.001          # Minimum heating rate (K/s)
beta_max = 0.01        # Maximum heating rate (K/s)
n_points = 1000           # Number of data points

# =====================================================
# CALCULATE PEAK TEMPERATURE (Tm)
# =====================================================

def calculate_Tm(beta, EA, A, kB):

    # Equation 7.6:
    # ln(Tm^2 / beta) = ln(EA / (kB*A)) + EA/(kB*Tm)

    def equation(Tm):
        return (
            np.log(Tm**2 / beta)
            - np.log(EA / (kB * A))
            - EA / (kB * Tm)
        )

    # Numerically solve for Tm
    lower = 1.0
    upper = 500.0

    while equation(upper) < 0:
        upper *= 2

    return brentq(equation, lower, upper)

plt.figure(figsize=(8, 6))
# =====================================================
# GENERATE HEATING RATES AND CALCULATE Tm
# =====================================================
def plot_beta_curves(EA, A, kB, beta_min, beta_max, n_points):
    beta_values = np.logspace(
        np.log10(beta_min),
        np.log10(beta_max),
        n_points
    )

    Tm_values = np.array([
        calculate_Tm(beta, EA, A, kB)
        for beta in beta_values
    ])

    Tm_values = Tm_values - Tm_values[0]  # Normalize to the first Tm value
    plt.plot(
        beta_values*1e3,
        Tm_values,
        linewidth=2,
        label=f'A = {A} s^-1'
    )

    return beta_values, Tm_values


Ea_values = [ 0.005, 0.01, 0.02,0.03, 0.04, 0.05]
for A in [1e6,1e7,1e8,1e9,1e10,1e11,1e12, 1e13, 1e14, 1e15, 1e16]:
    print(f"Calculating Tm for A = {A} s^-1...")
    beta_values, Tm_values = plot_beta_curves(EA, A, kB, beta_min, beta_max, n_points)

# for Ea in Ea_values:
#     print(f"Calculating Tm for Ea = {Ea} eV...")
#     beta_values, Tm_values = plot_beta_curves(Ea, A, kB, beta_min, beta_max, n_points)


plt.xlabel(r'Heating rate, $\beta$ (mK s$^{-1}$)', fontsize=12)
plt.ylabel(r'Middle temp shift, $T_m$ (K)', fontsize=12)

plt.title('TPD Peak Temperature vs Heating Rate', fontsize=14)

plt.grid(True, which='both', linestyle='--', alpha=0.5)

plt.legend()

plt.tight_layout()
plt.show()


# =====================================================
# PRINT CALCULATED VALUES
# =====================================================

print("\nHeating Rate (K/s)    Peak Temperature (K)")
print("------------------------------------------")

for beta, Tm in zip(beta_values[::20], Tm_values[::20]):
    print(f"{beta:12.4f}          {Tm:10.2f}")