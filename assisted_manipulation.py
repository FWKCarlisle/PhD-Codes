# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 17:29:31 2024

@author: physicsuser
"""

import time 
import numpy as np
import warnings
import sys

import python_interface_nanonis_v7 as python_nano

class extended_python_nano(python_nano.python_Nanonis_TCP):
    
    def __init__(self):
        super().__init__() 
    
    def Current_SetPoint_set_slow(self, current, time_to_take_seconds=10, num_of_steps=50):
        """
        (current, time_to_take_seconds=10, num_of_steps=50)
        Use this function to slowly change the current setpoint from the current value to a new value. Useful for atom manipulation.
        
    
        Parameters
        ----------
        current : set current setppoint applied (float)
            Applies this value to the z controller.
        time_to_take_seconds : float or int, optional(set to 10s)
            How long the system will take to reach the current set
        num_of_steps : int, optional(set to 50 values between current and final)
            The default is 50. the number of steps to take betwen current and final setpoint. 
    
        Returns
        -------
        None.
    
        """
        
        #Use this to change the bias slowly from the current value to a different one.
        if time_to_take_seconds < 5:
            print('Might take longer than 5 seconds...Due to request time')
        #Get current bias and setup an array of values that will act as intermediate value on the way.
        current_start = self.Current_GetSetPoint()
        
        # ensure that the Chosen manipulation setpoint is not 0A nor requires a sweep through 0A
        if current_start * current <= 0:
            self.close_socket()
            raise ValueError('Chosen manipulation setpoint requires a sweep through 0A. Change this.')
        print('Moving current setpoint from '+str(current_start)+' A to '+str(current)+' A in '+str(time_to_take_seconds)+'s')
        
        current_values = np.linspace(current_start, current, num_of_steps)
        sleep_time = time_to_take_seconds / num_of_steps
        
        for i in current_values: 
            self.Current_SetPoint(i)
            time.sleep(sleep_time)
            
        time.sleep(3)
        print('Completed slow current setpoint move - Current SetPoint = ' + str(self.Current_GetSetPoint())+' A')


scanningV = 200e-3
scanningI = 10e-12
manipulationV = 50e-3
manipulationI = 31e-12
time_to_take_seconds = 5

yes = ['yes', 'y', '', 'YES', 'Yes']
no = ['no', 'n', 'NO', 'No']

nano = extended_python_nano()

while input('Do you want to manipulate?: ') in yes:
    # user place tip on atom to be manipulated
    if input('Place tip tip at atom to be manipulated. Done?: ') in yes: 
        print(r'okay :)')
    
    # change params, if required
    if input('Continue with previous manipulation parameters: ') in no: 
        manipulationV = float(input('manipulationV in mV: ')) * 10**(-3)
        manipulationI = float(input('manipulationI in pA: ')) * 10**(-12)
    
    # check user inputs are safe
    if manipulationV < 20e-3: 
        print('manipulationV is too low')
        nano.close_socket()
        sys.exit()
    
    if 0 >= manipulationI > 1200e-12: 
        print('manipulationI is too low/high')
        nano.close_socket()
        sys.exit()
    
    # change current and bias slowly
    nano.Bias_set_slow(manipulationV, time_to_take_seconds=time_to_take_seconds,
                       num_of_steps=100)
    nano.Current_SetPoint_set_slow(manipulationI,
                                   time_to_take_seconds=time_to_take_seconds,
                                   num_of_steps=100)
    
    # move tip 
    if input('Move tip. Done?: ') in yes: 
        print(r'okay :)')
        
    # change current and bias slowly
    
    nano.Current_SetPoint_set_slow(scanningI, 
                                   time_to_take_seconds=25,
                       num_of_steps=100)
    nano.Bias_set_slow(scanningV, time_to_take_seconds=time_to_take_seconds,
                       num_of_steps=100)
    
    print('you can scan now')

nano.close_socket()