import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

def gaussian(x, a, x0, sigma):
    return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))

def fit_gaussian(x, y):
    popt, _ = curve_fit(gaussian, x, y, p0=[1, np.mean(x), np.std(x)])
    return popt


x = np.linspace(0, 10, 100)
# y = gaussian(x, 1, 5, 2) + np.random.normal(0, 0.1, x.size)
y = gaussian(x, 1, 5, 2) + np.random.normal(0, 0.1, x.size)

peaks, _ = find_peaks(y, height=0.5)
popt = fit_gaussian(x[peaks], y[peaks])
print(popt)
popt_1 = [1.00764663e+00, 4.58549776e-10, 4.86833789e-10]
plt.plot(x, y)
plt.xlim(0, 10E-10)
plt.plot(x, gaussian(x, *popt), 'r-')
plt.plot(x, gaussian(x, *popt_1), 'g-')

plt.show()