import matplotlib.pyplot as plt
import numpy as np



# distances_1 =  np.array([0,0,0.52,0.80,1.11,1.47,2.09,3.22])
# Contact_potential_1 = np.array([0.18,0.17,0.18,0.19,0.2,0.22,0.21,0.22])
# CP_errors_1 = np.array([0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01])
# d_errors_1 = np.array([0,0,0.01,0.01,0.01,0.01,0.01,0.01])

# distances_2 =  np.array([1.97,1.30,0.66,0,-0.68,-1.31,-1.98])
# Contact_potential_2 = np.array([0.2,0.2,0.21,0.2,0.2,0.19,0.19])
# CP_errors_2 = np.array([0.01,0.01,0.01,0.01,0.01,0.01,0.01])
# d_errors_2 = np.array([0.02,0.02,0.01,0,0.01,0.02,0.02])

# distances_3 =  np.array([2.03,1.37,0.74,0,-0.56,-1.17,-1.83])
# Contact_potential_3 = np.array([0.21,0.21,0.19,0.18,0.2,0.2,0.18])
# CP_errors_3 = np.array([0.01,0.01,0.01,0.01,0.01,0.01,0.01])
# d_errors_3 = np.array([0.01,0.01,0.01,0,0.01,0.01,0.01])
number =                       [153 ,157 ,154 ,160 ,164 , 156 , 158 , 162]
distances_4 =         np.array([0   ,0   ,0.94,1.89,2.82,-0.97,-1.91,-2.88])
Contact_potential_4 = np.array([0.1 ,0.13,0.13,0.14,0.22, 0.13, 0.16, 0.22])
cut_off_potenital_4 = np.array([0.07,0.3 ,0.09,0.05,0.19, 0.27, 0.17, 0.18])
CP_errors_4 =         np.array([0.01,0.01,0.01,0.01,0.01,0.01, 0.01 ,  0.01])
Cutoff_CP_errors_4 =  np.array([0.09,0.09,0.09,0.09,0.09,0.09, 0.09 ,  0.09])
# d_errors_4 = np.array([])


# Dip_start = []
# Dip_end = []
# Dip_depth = []
# Dip_width = Dip_end - Dip_start

def calculate_best_fit_line(x, y):
    slope, intercept = np.polyfit(x, y, 1)
    best_fit = slope*x + intercept

    coefficients = np.polyfit(x, y, 2)  # 2 indicates a quadratic fit
    parabola = np.polyval(coefficients, x)
    return parabola

#plot the contact potential against the distance with error bars


# best_fit_1 = calculate_best_fit_line(distances_1, Contact_potential_1)
# best_fit_2 = calculate_best_fit_line(distances_2, Contact_potential_2)
# best_fit_3 = calculate_best_fit_line(distances_3, Contact_potential_3)

# plt.errorbar(distances_1, Contact_potential_1,xerr=d_errors_1, yerr=CP_errors_1, color='blue', fmt='o')
# plt.plot(distances_1, best_fit_1, 'r', color='red', label='Best fit line')

# plt.errorbar(distances_2, Contact_potential_2,xerr=d_errors_2, yerr=CP_errors_2, color='red', fmt='o')
# plt.plot(distances_2, best_fit_2, 'r', color='magenta', label='Best fit line')

# plt.errorbar(distances_3, Contact_potential_3,xerr=d_errors_3, yerr=CP_errors_3, color='green', fmt='o')
# plt.plot(distances_3, best_fit_3, 'r', color='#2CA02C', label='Best fit line')

# plt.errorbar(distances_4, Contact_potential_4, xerr=0, yerr=CP_errors_4, color='blue', fmt='o', label='Contact potential')
plt.errorbar(distances_4, cut_off_potenital_4, xerr=0, yerr=Cutoff_CP_errors_4, color='red', fmt='o', label='Cut off potential')



plt.xlabel('Distance (nm)')
plt.ylabel('Contact potential (V)')
plt.title('Contact potential against distance')
plt.show()
