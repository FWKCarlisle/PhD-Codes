# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 10:27:27 2024

@author: Sofia Alonso Perez
"""
import python_interface_nanonis_v7 as python_nano
import time 
import numpy as np
import sys
import traceback


bias = 1e-3 # V

xA, yA = 0, 0
xB, yB = 0, 0
speed = 0
zInitial = 0

class ForceInManipulation(python_nano.python_Nanonis_TCP):
    
    def __init__(self, bias, z_delta, speed, xA_delta, yA_delta, xB_delta, yB_delta, 
                 rotTraf, offDelay, tunnelI, tunnelV, baseName, dfAmpStp):
        
        super().__init__() 
        
        self.bias = bias
        self.z_delta = z_delta
        self.speed = speed
        self.xA_delta = xA_delta
        self.yA_delta = yA_delta
        self.xB_delta = xB_delta
        self.yB_delta = yB_delta
        self.rotTraf = rotTraf
        
        self.offDelay = offDelay
        
        self.tunnelI = tunnelI
        self.tunnelV = tunnelV
        
        self.baseName = baseName
        self.dfAmpStp = dfAmpStp
        
        
       
    def Run(self):
        self.Bias_set_slow(self.tunnelV, 3, 50)
        self.Current_SetPoint_set_slow(self.tunnelI, 3, 50)
        self.Z_feedback_set('on') # turn z controller on
        
        
        if input('place tip on atom. Done?: ') in  ['yes', 'y', '', 'YES', 'Yes']:
            print('starting')
        
        count = 0
        for v in self.bias:
            for z in self.z_delta:
                
                try:
                    self.AttemptManipulation(z, v, self.xA_delta, self.yA_delta,
                                        self.xB_delta, self.yB_delta, self.speed)
                except Exception as e:
                    print(e)
                    self.baseName = self.baseName + str(count) + '_'
                    count = count + 1
                    break
    
    def AttemptManipulation(self, z_delta, V, xA_delta, yA_delta, xB_delta, yB_delta, speed):
        fileComment = 'z = ' + str(z_delta) + ', V = ' + str(V) +', xA_delta = ' + str(xA_delta) + ', yA_delta = ' + str(yA_delta) +', xB_delta = ' + str(xB_delta) + ', yB_delta = ' + str(yB_delta) + ', speed = ' + str(speed) +'df_amp_stp = ' + str(self.dfAmpStp)+'tunnelI = ' + str(self.tunnelI)+'tunnelV = ' + str(self.tunnelV)
        print(fileComment)
        
        self.Data_LoggerPropsSet(self.baseName, fileComment, averaging=50)
        
        print('changing speed')
        self.XY_tip_move_set_speed(speed, 'custom')
        
        x0, y0 =  self.XY_tip_pos()
        print('fine tune atom pos, ie. ref point (0,0)')
        x0, y0 = self.SitePosFineTune(x0, y0)
        
        z0_before = self.GetTimeAvZ()
        
        print("constant height mode")
        self.Z_feedback_set('off') # turn z controller off
        time.sleep(self.offDelay)
        
        xA = x0 + xA_delta
        yA = y0 + yA_delta
        xB = x0 + xB_delta
        yB = y0 + yB_delta
        
        xA, yA = self.Rotate(xA, yA, x0, y0, rot=self.rotTraf)
        xB, yB = self.Rotate(xB, yB, x0, y0, rot=self.rotTraf)
        
        
        print('move tip to (xA, yA)')
        self.XY_tip_set_pos(xA, yA)
        self.WaitWhileTipMoves(xA, yA, errTol=100e-12)
        
        
        
        print('set z = ', z0_before + z_delta)
        self.Z_set_slow(z0_before + z_delta, 3, 50)
        time.sleep(0.1)
        print('changing bias')
        self.Bias_set_slow(V, 3, 50)
        

        
        print('record I and df channels')
        self.Data_LoggerStart()
        time.sleep(0.1)
        
        print('move tip to (xB, yB)')
        self.XY_tip_set_pos(xB, yB)
        self.WaitWhileTipMoves(xB, yB, errTol=100e-12)
        
        self.Data_LoggerStop()
    
        print('tunnelling mode')
        self.Bias_set_slow(self.tunnelV, 3, 50)
        self.Z_feedback_set('on') # turn z controller on
        time.sleep(0.1)
        
        print('moving to atom')
        self.XY_tip_set_pos(x0, y0)
        self.WaitWhileTipMoves(x0, y0, errTol=100e-12)
        
        z0_after = self.GetTimeAvZ()
        
        if z0_after-z0_before < -50e-12:
            print('atom moved. scan and stop.')
            width = max(xA_delta, yA_delta, xB_delta, yB_delta) + 3e-9
            self.Scan_FrameSet(centreX=x0, centreY=y0,
                               widthX=width, widthY=width, angle=0)
            
            self.Scan_Action('start', 'up')
            # while scanning -> wait
            time.sleep(1)
            self.Scan_Check_Hold()  
            time.sleep(3)
            
            # # load scan 
            # _, scan = self.Scan_FrameDataGrab(30) 
            
            # x0_guess, y0_guess = self.Px2NanoCoord(xPx=np.where(scan==np.max(scan))[0][0],
            #                                        yPx=np.where(scan==np.max(scan))[1][0],
            #                                        widthInPx=np.shape(scan)[0],
            #                                        widthInM=width, xCentre=x0,
            #                                        yCentre=y0, angle=0)
            
            
            x0_guess, y0_guess = self.SitePosFineTune(x0, y0, 360)
            
            # print('moving to atom')
            # self.XY_tip_set_pos(x0_guess, y0_guess)
            # self.WaitWhileTipMoves(x0_guess, y0_guess, errTol=100e-12)
            
            
            # x0_guess, y0_guess = self.SitePosFineTune(x0_guess, y0_guess)
            
            z0_guess = self.GetTimeAvZ()
            if z0_guess-z0_before > -50e-12:
                raise ValueError('skip z loop straight to next voltage')
            else:
                print('did not find atom, stop.')
                self.close_socket()
                sys.exit()
            
        else: print('atom did not move')
        
        
    def SitePosFineTune(self, x, y, atomTrackTime=20):
        
        # move the tip to the atom to manipulate ie. the first path point
        self.XY_tip_set_pos(x,y)
        self.WaitWhileTipMoves(x,y, 100e-12)
        
        # quick atom tracking to find the site's centre
        print("atom tracking")
        self.AtomTrack_CtrlSet(1, 1) # initialise atom traking
        time.sleep(atomTrackTime) 
        self.AtomTrack_CtrlSet(0, 0) # end atom tracking
        time.sleep(0.1)
        
        x, y = self.XY_tip_pos() 
        
        return x, y
    
    
    def CalcTimeManipulation(self, xA, yA, xB, yB, speed):
        dist = np.sqrt((xB - xA)**2 + (yB - yA)**2)
        time = dist / speed
        return time
    
    
    def WaitWhileTipMoves(self, xEnd, yEnd, errTol):
        print('tip moving. waiting...')
        
        while True:
            x, y = self.XY_tip_pos()
            if (abs(x - xEnd) > errTol
                or abs(y - yEnd) > errTol):
                time.sleep(1)
            else:
                break
        print('tip move finished')
        
        
    def GetTimeAvZ(self):
        # Z calibration: 
        # take a time average of Z, rather than taking a instantaneus Z reading.
        # Use the "Off Delay functionality on nanonis' Z controller window
        print("Using off delay time average to get a reading for z")
        
        self.Z_feedback_set('off') # turn z controller off
        print("z controller off")
        
        time.sleep(self.offDelay) # wait for time average (and a bit longer to avoind timing errors)
        time.sleep(0.2)
        
        z = self.Z_pos_get() 
        self.Z_feedback_set('on') # turn z controller off
        print("z controller on")
        
        return z
    
    
    
        
        
    
    
    def Current_SetPoint_set_slow(self, current, time_to_take_seconds, num_of_steps):
        """
        Copied idea from python_nano method Bias_SetPoint_set_slow. 
        
        Use this function to slowly change the current setpoint from the current
        value to a new value.
        
    
        Parameters
        ----------
        current : set current setppoint applied (float)
            Applies this value to the z controller.
        time_to_take_seconds : float or int
            How long the system will take to reach the current set
        num_of_steps : int
            The number of steps to take betwen current and final setpoint. 
    
    
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
        print('Moving current setpoint from '+str(current_start)+' A to '+ str(current)+' A in '+str(time_to_take_seconds)+'s')
        
        current_values = np.linspace(current_start, current, num_of_steps)
        sleep_time = time_to_take_seconds / num_of_steps
        
        for i in current_values: 
            self.Current_SetPoint(i)
            time.sleep(sleep_time)
            
        time.sleep(1)
        print('Completed slow current setpoint move - Current SetPoint = ' + str(self.Current_GetSetPoint())+' A')
    
    
    def Z_set_slow(self, z, time_to_take_seconds, num_of_steps):
        """
        Copied idea from python_nano method Bias_SetPoint_set_slow. 
        
        Use this function to slowly change the current setpoint from the current
        value to a new value.
        
    
        Parameters
        ----------
        current : set current setppoint applied (float)
            Applies this value to the z controller.
        time_to_take_seconds : float or int
            How long the system will take to reach the current set
        num_of_steps : int
            The number of steps to take betwen current and final setpoint. 
    
    
        """
        
        #Use this to change the bias slowly from the current value to a different one.
        if time_to_take_seconds < 5:
            print('Might take longer than 5 seconds...Due to request time')
        #Get current bias and setup an array of values that will act as intermediate value on the way.
        z_start = self.Z_pos_get()
        
        z_values = np.linspace(z_start, z, num_of_steps)
        sleep_time = time_to_take_seconds / num_of_steps
        
        for i in z_values: 
            self.Z_pos_set(i)
            time.sleep(sleep_time)
            
        time.sleep(1)
        print('Completed slow z move - z = ' + str(self.Z_pos_get())+' m')
        
    
    def Rotate(self, x, y, xPivot=0, yPivot=0, rot=0):
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

    
    def Px2NanoCoord(self, xPx, yPx, widthInPx, widthInM, xCentre, yCentre, angle):  
        
        def Px2M(px):
            return  (widthInM / widthInPx) * px 
        
        # origin trasformation: upper left corner -> centre of scan
        xPx = xPx - (widthInPx / 2)
        yPx = yPx - (widthInPx / 2)
        
        # px -> metre lengths
        xM = Px2M(xPx)
        yM = Px2M(yPx)
        
        # origin trasformation: scan centre -> nanonis origin
        xM = xM + xCentre
        yM = yM + yCentre
        
        # angle trasformation
        xM, yM = self.Rotate(xM, yM, xPivot=xCentre, yPivot=yCentre,
                        rot=angle)
        return xM, yM
        
#%%
a = 0.648e-9
b = 0.458e-9



bias =  np.arange(50e-3, 420e-3, 20e-3)
z_delta = np.arange(0, -0.25e-9, -10e-12)
speed = 200e-12
xA_delta = -5*a
yA_delta = 0
xB_delta = 5*a
yB_delta = 0
rotTraf = 2.2

offDelay = 1
tunnelI = 10e-12
tunnelV = 200e-3
baseName = 'force_in_manipulation_run4_'
dfAmpStp = 120e-12

e = ForceInManipulation(bias, z_delta, speed, xA_delta, yA_delta, xB_delta, yB_delta, 
             rotTraf, offDelay, tunnelI, tunnelV, baseName, dfAmpStp)

e.Run()