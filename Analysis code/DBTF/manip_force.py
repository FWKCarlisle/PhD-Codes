# -*- coding: utf-8 -*-
"""
Created on Wed Jul  9 13:52:34 2025

@author: Unisoku


"""


import time
import numpy as np
import warnings
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import tqdm
import traceback
from numpy import unravel_index
import sys

import python_interface_nanonis_v9
from coord_trans import NanoCoordTrans

from spectra_mapping import Manipulation_force_measurement
from spectra_mapping import TipParams
from spectra_mapping import BiasSpecParams
from spectra_mapping import TrackablePos
from spectra_mapping import Pos
from spectra_mapping import TCP

def Px2NanoCoord(xPx, yPx, sxm_width_in_px, sxm_width_in_m, sxm_x_centre,
                 sxm_y_centre, sxm_angle):  
    
    
    def rotate(x, y, xPivot=0, yPivot=0, rot=0):
        rot = -np.deg2rad(rot)
        
        x = x - xPivot
        y = y - yPivot
        
        # carteesian -> polar
        r = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x) # element-wise arctan 
        
        # rotation
        theta = theta + rot
        
        # polar -> carteesian
        x = (r*np.cos(theta)) + xPivot
        y = (r*np.sin(theta)) + yPivot

        return x, y

    
    # origin trasformation: upper left corner -> centre of scan
    xPx = xPx - (sxm_width_in_px / 2)
    yPx = yPx - (sxm_width_in_px / 2)
    
    
    # px -> metre lengths
    xM = (sxm_width_in_m / sxm_width_in_px) * xPx
    yM = (sxm_width_in_m / sxm_width_in_px) * yPx
    
    # origin trasformation: scan centre -> nanonis origin
    xM = xM + sxm_x_centre
    yM = yM + sxm_y_centre
    
    # angle trasformation
    xM, yM = rotate(xM, yM, xPivot=sxm_x_centre, yPivot=sxm_y_centre,
                    rot=sxm_angle)
    return xM, yM

# define params
name = 'Move_DBTF@Pb_500mV_1_'

data_log_avg=200
q_factor=23552
amplitude_stp=100e-12
rot=-9
f0 = 31522 

idle_params = TipParams(v=50e-3,
                        i=None,
                        z_delta=0,
                        speed=5e-9,
                        timeVarRamp=0.5,
                        stepsVarRamp=10)

calibration_tip_params = TipParams(v=50e-3,
                        i=50e-12,
                        z_delta=None,
                        speed=5e-9,
                        timeVarRamp=0.5,
                        stepsVarRamp=10)

end_tip_params = TipParams(v=900e-3,
                        i=60e-12,
                        z_delta=None,
                        speed=10e-9,
                        timeVarRamp=0.5,
                        stepsVarRamp=10)


manipulation_params = TipParams(v=500e-3,
                                i=None,
                                z_delta=400e-12,
                                speed=0.3e-9,
                                timeVarRamp=0.5,
                                stepsVarRamp=10) # initial params

z_manip_rel_steps = np.linspace(start=manipulation_params.z_delta,
                                stop=-300e-12,
                                num=71,
                                endpoint=True) 

origin = TrackablePos(None, None, None)


path_rel = [Pos(0, -5e-9, None), Pos(0, 5e-9, None)] # in metres from the origin  (y-axis)
#path_rel = [Pos(-5e-9, 0, None), Pos(5e-9, 0, None)] # in metres from the origin (x-axis)

path_rot = rot
for i in range(len(path_rel)):
    trans = NanoCoordTrans(path_rel[i].x, path_rel[i].y)
    trans.Rot(path_rot, xPivot=0, yPivot=0)
    path_rel[i].x, path_rel[i].y = trans.x, trans.y
    

try:
    my_tcp = TCP()

    experiment = Manipulation_force_measurement(my_tcp)
    origin.point_out_xyz(my_tcp)
    
    idle_params.set(my_tcp, z_origin=origin.z)

    for z_rel in z_manip_rel_steps:
        
        manipulation_params.z_delta = z_rel
        
        experiment.execute(manipulation_params=manipulation_params,
                    idle_params=idle_params,
                    origin=origin,
                    calibrate_origin=False,
                    calibration_tip_params=calibration_tip_params,
                    path_rel=path_rel,
                    file_basename=name,
                    data_log_avg=data_log_avg,
                    q_factor=q_factor,
                    f0=f0,
                    amplitude_stp=amplitude_stp,
                    time_for_atom_tracking=10.0,
                    i_max=3e-9)


except Exception as e:
    print(e)
    print(traceback.format_exc())
finally:
    end_tip_params.set(my_tcp)
    my_tcp.close_socket()
