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


# number =                       [495,496,497,498,499,500,501,503,504,505,506,507]
# file_name =                    [314,315,316,317,318,319,320,322,323,324,325,326]
# distances_5 =         np.array([-0.34,-0.72,-1.07,-1.40,-1.77,-2.08,-2.441,0.33,0.71,1.03,1.38,1.73,2.08]) ### This is the data for slide 39 in scratch pad
# d_errors_5 =          np.array([0.06,0.06,0.06,0.06,0.06,0.06,0.06,0.06,0.06,0.06,0.06,0.06])
# max_bias_5 =          np.array([0.3120001, 0.36800009, 0.4320001, 0.51200002, 0.67200011, 0.76000005, 0, 0.31999999, 0.37599999, 0.45600003, 0.59200007, 0.6880001,0])

# number =                       [471,473,475,477,478, 479,481,483,485] #Number on image
# file_name =                    [289,290,292,294,296, 298,300,302,304] #File name
# distances_5 =         np.array([0, 1.20,2.29,3.40,4.08,-1.15,-2.24,-3.42,-4.57]) #Distance measured in nm
# d_errors_5 =          np.array([0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01,0.01]) #Error in distance
# max_bias_5 =          np.array([0.31499994, 0.48000002, 0.78999996, 0, 0.39499998,0, 0, 0]) #Well position
# #Well depth
# well_depths_5 =       np.array([0.13853357325280397, 0.14387903522809178, 0.12032916771994409, 0.10920208083530394, 0.07038546859238941, 0.1296870079292956, 0.04337245457452322, 0.05068743582084121, 0.055250692408798735])
# #'Rough depth error
# depth_errors_5 =      np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05])

#DATA for spatial 6 (?)               
# number =                       [517   , 518   , 519   , 520   , 521   , 522   , 523   , 524   , 525   , 526   , 527   , 528   , 529   , 530   ] #Number on image
# file_name =                    [336   , 337   , 338   , 339   , 340   , 341   , 342   , 343   , 344   , 345   , 346   , 347   , 348   , 349   ] #File name 
# distances_5 =         np.array([0     , 0.44  , 0.93  , 1.85  , 1.39  , 2.33  , 2.79  ,-0.14  ,-0.55  ,-1.03  ,-1.48  ,-1.92  ,-2.40  ,-2.75  ]) #Distance measured in nm
# d_errors_5 =          np.array([0.02  , 0.02  , 0.02  , 0.02  , 0.02  , 0.02  , 0.02  , 0.02  , 0.02  , 0.02  , 0.02  , 0.02  , 0.02  , 0.03  ]) #Error in distance
# max_bias_5 =          np.array([0.3360, 0.3573, 0.5333, 0.4107, 0.6507, 0.7467, 0.6453, 0.3307, 0.3413, 0.4320, 0.5333, 0.7360, 0.7253, 0.4427]) #Well position
# well_depths_5 =       np.array([0.1744, 0.1640, 0.0986, 0.1596, 0.1012, 0.0508, 0.0438, 0.1857, 0.1886, 0.1767, 0.1204, 0.1022, 0.0689, 0.0448]) #Well depth
# depth_errors_5 =      np.array([0.05, 0.05, 0.05,0.05, 0.05, 0.05,0.05, 0.05, 0.05,0.05, 0.05, 0.05,0.05, 0.05]) #'Rough depth error

#Data for spatial 8
# number =                       [] #Number on image
# file_name =                    [390  , 391 , 392 , 393 , 394 , 395 , 396 , 397 , 398 , 399 ] #File name 
# distances_5 =         np.array([0    , 30  , 60  , 90  , 120 , 150 , 0   , -30 , -60 ,-90  ]) #Distance measured in nm
# d_errors_5 =          np.array([0.02 , 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02]) #Error in distance
# LCPD_5 =          np.array([-0.14,-0.17,-0.18,-0.20,-0.16,-0.18,-0.18,-0.13,-0.15,-0.15]) #Well position
# max_bias_5 =          np.array([0.37666667, 0.40000004, 0.41666669, 0.41000003, 0.42666668, 0.44, 0.38, 0.37333333, 0.36666667, 0.32333338])
# well_depths_5 =       np.array([0.09 , 0.12   , 0.15, 0.16, 0.17, 0.15, 0.11, 0.06, 0.09, 0.08]) #Well depth
# FWHM_5 =              np.array([0.10 , 0.09   , 0.08, 0.08, 0.07, 0.06, 0.09, 0.09, 0.11, 0.15])#abs value of FWHM
# depth_errors_5 =      np.array([0.01 , 0.01, 0.01, 0.01, 0.01, 0.01,0.01 , 0.01, 0.01, 0.01]) #'Rough depth error
# FWHM_errors_5 =       np.array([0.01 , 0.01, 0.005, 0.005, 0.005, 0.005,0.005 , 0.01, 0.01, 0.01]) #'Rough depth error

#data for spatial 10 (dF/dZ)
# # LR
# number      =          [881   , 879  , 877  , 875  , 873  , 871  , 869  , 867  , 865  , 864  , 866 , 868 ,   870 , 872 , 874  , 876 , 878 , 880 , 882]
# file_number =          [ 225  , 223  , 221  , 219  , 217  , 215  , 213  , 211  , 209  , 208  , 210 , 212 ,   214 , 216 , 218  , 220 , 222 , 224 , 226]
# distances   = np.array([-2.43 , -2.15, -1.88, -1.62, -1.35, -1.07, -0.82, -0.55,-0.26 , 0    , 0.25,0.53 ,  0.80 ,1.08 ,1.35  ,1.62 ,1.88 ,2.15 , 2.43])
# d_errors    = np.array([0.01  , 0.01 , 0.01 , 0.01 , 0.01 , 0.01 , 0.01 , 0.01 , 0.01 , 0.01 , 0   , 0.01,   0.01, 0.01, 0.01 , 0.01, 0.01,0.01 ,0.01])
# dip_height  = np.array([1.800 ,0     , 0.182, 1.334,2.404 , 3.500, 4.182, 4.528, 4.658, 4.765,4.688, 4.476, 4.182,3.198, 2.255,1.401,1.401,1.835,4.551 ])
# dip_h_errs  = np.array([0.110 ,0     , 0.051, 0.009,0.023 , 0.018, 0.015, 0.012, 0.014, 0.015,0.013, 0.016, 0.037,0.048, 0.110,0.028,0.028,0.124,0.040])
# dip_width   = np.array([4.352 ,0     , 2.478, 2.060,2.735 , 2.667, 2.559, 2.213, 2.362, 2.274,2.064, 2.366, 3.464,4.849, 5.417,3.045,3.045,6.238,3.731])
# dip_w_errs  = np.array([0.502 ,0     , 0.096, 0.026,0.072 , 0.059, 0.049, 0.035, 0.043, 0.046,0.036, 0.049, 0.134,0.263, 0.472,0.115,0.115,0.767,0.170])
# dip_position= np.array([0.069 ,0     , 0.182, 0.186,0.148 , 0.145, 0.155, 0.179, 0.186, 0.192,0.198, 0.186, 0.138,0.128, 1.121,0.149,0.149,0.112,0.113])
# dip_p_errs  = np.array([0.001 ,0     , 0.096, 0.001,0.002 , 0.002, 0.002, 0.002, 0.002, 0.003,0.003, 0.002, 0.002,0.001, 0.002,0.002,0.002,0.002,0.002])
# peak_YN     = np.array([0     ,0     , 0.5  , 1    ,1     , 1    , 1    , 1    , 1    , 1    ,1    , 1    , 1    ,0.5  , 0    ,0.5  ,0    ,0    ,0])

# UD1
# number      =          [  860  ,857   ,859   ,   853,   851,   849,   847,   845,   843, 842  ,  844 ,  846,  848,  850,  852,  854,  856,  858, 861]
# file_number =          [  204  ,201   ,203   ,   197,   195,   193,   191,   189,   187, 186  ,  188 ,  190,  192,  194,  196,  198,  200,  202, 205]
# distances   = np.array([ -2.16 , -1.95, -1.73, -1.48, -1.24, -0.99, -0.75, -0.50,-0.25 ,  0   ,0.25  , 0.47, 0.74, 0.98, 1.24,1.48 , 1.72, 1.96, 2.18  ])
# d_errors    = np.array([ 0.01  , 0.01 , 0.01 , 0.01 , 0.01 , 0.01 , 0.01 , 0.01 , 0.01 , 0.01 , 0    , 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,0.01 ,0.01])
# dip_height  = np.array([ 0     ,-0.637,1.259 ,1.214 ,2.894 ,3.775 ,4.277 , 4.486, 4.650, 4.773, 4.683,4.483,4.115,3.507,2.596,1.645,0.734,4.804,4.703])
# dip_h_errs  = np.array([0      , 0.604,0.012 ,0.305 ,0.024 ,0.020 ,0.013 ,0.013 , 0.019, 0.015, 0.015,0.017,0.017,0.014,0.014,0.011,0.019,0.097,0.032])
# dip_width   = np.array([0      , 5.573,2.204 ,5.534 ,3.184 ,3.404 ,2.386 ,2.342 , 2.611, 2.452,2.410 ,2.453,2.634,2.906,3.125,2.849,2.412,6.532,4.382])
# dip_w_errs  = np.array([0      , 1.067,0.039 ,0.572 ,0.096 ,0.088 ,0.039 ,0.041 , 0.063, 0.045,0.047 ,0.050,0.050,0.050,0.057,0.038,0.071,0.703,0.172])
# dip_position= np.array([0      , 0.141,0.159 ,0.161 ,0.140 ,0.151 ,0.167 ,0.184 , 0.182, 0.198,0.201 ,0.198,0.201,0.191,0.200,0.216,0.230,0.120,0.131])
# dip_p_errs  = np.array([0      ,0.011 ,0.002 ,0.005 ,0.002 ,0.002 ,0.002 ,0.002 , 0.002, 0.002,0.002 ,0.002,0.002,0.002,0.002,0.001,0.002,0.002,0.001])
# peak_YN     = np.array([0      ,0     ,1     ,0.5   ,1     ,1     ,1     , 1    ,1     , 1    , 1    ,  1  ,1    ,1,1,1,1,0,0])

# UD2
number      =          [903,901,899,896,893,891,889,887,885,884,886,888,890,892,894,897,900,902,904]
file_number =          []
distances   = np.array([-2.22,-1.97,-1.74,-1.48,-1.24,-0.98,-0.76,-0.50,-0.25,0,0.22,0.46,0.74,0.97,1.24,1.48,1.71,1.96,2.22])
d_errors    = np.array([0.01 , 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0,0.01 , 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01 ])
dip_height  = np.array([4.566,-0.470,1.020,1.843,2.692,3.625,4.166,4.510,4.711,4.707,4.702,4.461,4.226,3.850,2.736,2.736,1.239,1.239,0])
dip_h_errs  = np.array([0.039,0.392,0.013 ,0.016,0.016,0.019,0.022,0.011,0.012,0.013,0.014,0.013,0.017,0.024,0.032,0.032,0.022,0.022,0])
dip_width   = np.array([3.257,3.386,2.288 ,2.604,2.742,2.797,2.501,2.092,2.116,2.180,2.101,2.138,2.320,2.715,2.982,2.982,2.189,2.189,0])
dip_w_errs  = np.array([0.148,0.554,0.041 ,0.052,0.056,0.065,0.062,0.031,0.033,0.037,0.038,0.037,0.047,0.077,0.115,0.115,0.065,0.065,0])
dip_position= np.array([0.110,0.184,0.204 ,0.187,0.166,0.163,0.175,0.192,0.188,0.175,0.172,0.162,0.146,0.131,0.118,0.118,0.143,0.143,0])
dip_p_errs  = np.array([0.002,0.019,0.002 ,0.002,0.002,0.002,0.002,0.002,0.002,0.002,0.002,0.002,0.002,0.002,0.002,0.002,0.003,0.003,0])
peak_YN     = np.array([0.5  ,0    , 1    ,1    ,1    ,1    ,1    ,1    ,1    ,1    ,1    ,1    ,1    ,1    ,0.5  ,0, 1, 0    ,0       ])

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
# print(len(distances_5),len(max_residual_5),len(max_bias_5))
# plt.scatter(distances_5, max_residual_5, color='red', label='Max residual')
# print(len(distances_5),len(well_depths_5))
# plt.errorbar(distances_5, well_depths_5,yerr=depth_errors_5, fmt= 'o', color='green', label='Max bias')
# plt.errorbar(distances_5, well_depths_5, xerr=d_errors_5, yerr=depth_errors_5, color='green', fmt='o', label='Well depth')
print(len(distances),len(dip_height),len(dip_position),len(dip_width))
# plt.errorbar(distances, dip_height, xerr=d_errors, yerr=dip_h_errs, color= 'blue', fmt='o', label='Dip height')
# plt.errorbar(distances, dip_width, xerr=d_errors, yerr=dip_w_errs, color= 'blue', fmt='o', label='Dip width')
plt.errorbar(distances, dip_position, xerr=d_errors, yerr=dip_p_errs, color= 'green', fmt='o', label='Dip position')


# plt.plot(distances, peak_YN, color='red', label='visable Peak or not')

plt.xlabel('Distance (nm)')
plt.ylabel('Dip height (Hz) ')
# plt.xlim(-155,155)
plt.legend()
plt.title('dI/dV Curve analysis graphs')
plt.show()
