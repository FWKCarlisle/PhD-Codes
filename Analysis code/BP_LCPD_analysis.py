import matplotlib.pyplot as plt
import numpy as np


#Spatial data from 16/08/24
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
# number =                       [153 ,157 ,154 ,160 ,164 , 156 , 158 , 162]
# distances_4 =         np.array([0   ,0   ,0.94,1.89,2.82,-0.97,-1.91,-2.88])
# Contact_potential_4 = np.array([0.1 ,0.13,0.13,0.14,0.22, 0.13, 0.16, 0.22])
# cut_off_potenital_4 = np.array([0.07,0.3 ,0.09,0.05,0.19, 0.27, 0.17, 0.18])
# CP_errors_4 =         np.array([0.01,0.01,0.01,0.01,0.01,0.01, 0.01 ,  0.01])
# Cutoff_CP_errors_4 =  np.array([0.09,0.09,0.09,0.09,0.09,0.09, 0.09 ,  0.09])
# d_errors_4 = np.array([])

#Spatial data from 19/08/24
# number =                       [449,451,453,455,457,459,461,463,465,467,469]
# file_name =                    [268,270,272,274,276,278,280,282,284,286,288]
# distances_5 =         np.array([-1.04,-2.16,-3.34,-4.41,-5.59,1.16,2.39,3.44,4.45,3.44,5.64])
# d_errors_5 =          np.array([0.03,0.05,0.05,0.05,0.05,0.05,0.06,0.06,0.06,0.06])
# Contact_potential_5 = np.array([0.19 ,0.18,0.17  ,0.2  ,0.2  ,0.2  ,0.19 ,0.205  ,0.21,0.21])
# CP_errors_5 =         np.array([0.005,0.005,0.004,0.003,0.003,0.004,0.003,0.006,0.003,0.004])
# max_residual_5 =      np.array([0.1489,0.1155,0,0,0,0.1035,0,0,0.0795,0,0.1702])
# max_bias_5 =          np.array([0.28  ,0.47  ,0,0,0,0.435,0,0,0.62,0,0.37])


number =                       [495,496,497,498,499,500,501,503,504,505,506,507]
file_name =                    [314,315,316,317,318,319,320,322,323,324,325,326]
distances_5 =         np.array([-0.34,-0.72,-1.07,-1.40,-1.77,-2.08,-2.441,0.33,0.71,1.03,1.38,1.73,2.08])
d_errors_5 =          np.array([0.06,0.06,0.06,0.06,0.06,0.06,0.06,0.06,0.06,0.06,0.06,0.06])
Contact_potential_5 = np.array([])
CP_errors_5 =         np.array([])
max_residual_5 =      np.array([])
max_bias_5 =          np.array([0.3120001, 0.36800009, 0.4320001, 0.51200002, 0.67200011, 0.76000005, 0, 0.31999999, 0.37599999, 0.45600003, 0.59200007, 0.6880001,0])

# cut_off_potenital_5 = np.array([])
# Cutoff_CP_errors_5 =  np.array([])






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
# plt.errorbar(distances_4, cut_off_potenital_4, xerr=0, yerr=Cutoff_CP_errors_4, color='red', fmt='o', label='Cut off potential')
# plt.errorbar(distances_5, Contact_potential_5, xerr=d_errors_5, yerr=CP_errors_5, color='blue', fmt='o', label='Contact potential')
print(len(distances_5),len(max_residual_5),len(max_bias_5))
# plt.scatter(distances_5, max_residual_5, color='red', label='Max residual')
plt.scatter(distances_5, max_bias_5, color='green', label='Max bias')

plt.xlabel('Distance (nm)')
plt.ylabel('Max Bias (V)')
plt.xlim(-3,3)
plt.title('Contact potential against distance')
plt.show()
