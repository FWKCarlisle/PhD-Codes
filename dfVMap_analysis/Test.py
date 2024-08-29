import numpy as np 
from scipy.optimize import curve_fit 
import matplotlib.pyplot as plt

# Let's create a function to model and create data 
def func(x, a, x0, sigma): 
    return a*np.exp(-(x-x0)**2/(2*sigma**2)) 
  

def lorentzian(x, x0, a, gamma):
            return a * gamma**2 / ((x - x0)**2 + gamma**2)

# Generating clean data 
x = np.linspace(0, 10, 100) 
y = lorentzian(x, 1, 5, 2) 
  
# Adding noise to the data 
yn = y + 0.2 * np.random.normal(size=len(x)) 
  
# Plot out the current state of the data and model 
fig, [ax, axMinus,axSmoothMinus] = plt.subplots(nrows=3, ncols=1, sharex=True)
ax = fig.add_subplot(111) 
ax.plot(x, y, c='k', label='Function') 
ax.scatter(x, yn) 
  
# Executing curve_fit on noisy data 
popt, pcov = curve_fit(func, x, yn) 
  
#popt returns the best fit values for parameters of the given model (func) 
print (popt) 
  
ym = func(x, popt[0], popt[1], popt[2]) 
ax.plot(x, ym, c='r', label='Best fit') 
ax.legend() 
plt.show()
# fig.savefig('model_fit.png')
