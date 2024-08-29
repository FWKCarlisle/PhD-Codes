# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 16:53:11 2024

@author: ppysa5
"""

# -*- coding: utf-8 -*-




import python_interface_nanonis_v7 as python_nano
import numpy as np
import time
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt 
from datetime import datetime

matplotlib.rc('image', cmap='gray')
#%% 


class Experiment_dfVmap(python_nano.python_Nanonis_TCP): # define child class for our experiment
    
    def __init__(self, nPointsPerCalibration, zRelForMoving,
                 zRelForSpectra, VForMoving, VForSpectra, timeForAtomTracking,
                 offDelay, fileBaseName,
                 xGrid, yGrid, angle=0
                 ):
        
        super().__init__() 
        

        self.nPointsPerCalibration = nPointsPerCalibration
        self.zRelForMoving = zRelForMoving
        self.zRelForSpectra = zRelForSpectra
        self.VForMoving = VForMoving
        self.VForSpectra = VForSpectra
        self.timeForAtomTracking = timeForAtomTracking
        self.offDelay = offDelay
        self.fileBaseName = fileBaseName
        self.angle = angle
        self.pointCount = 0
        self.calibrationCount = -1
        
        
        yes = ['yes', 'y', '', 'YES', 'Yes']
        print('Choose which atom to track.')
        if input('Place the tip on the chosen atom. Done?: ') in yes:
            self.xt, self.yt = self.XY_tip_pos()
            self.CalibrateXYZ(firstCalibration=True)
            
        self.masks = None
        self.x = xGrid + self.xt
        self.y = yGrid + self.yt
        
        
        self.zChanNum = int(30)
        # # find the Z (m) channel 
        # _, self.recordedChanels, xPxPerLine, yPxPerLine = self.Scan_BufferGet()
        # for chanelNum in self.recordedChanels:
        #     chanName, _ = self.Scan_FrameDataGrab(chanelNum) 
        #     if chanName == 'Z (m)':
        #         self.zChanNum = chanelNum
                
        
    
        
        
    def CalibrateXYZ(self, firstCalibration=False):
        self.Z_feedback_set('on') # turn z controller on
        print("check z controller is on")
        time.sleep(1) # wait
        
        # test ================================================================
        if firstCalibration == False:
            xScanCentre, yScanCentre, scanWidth, = self.xt, self.yt, 1e-9
            # self.TestStatus(xScanCentre, yScanCentre, scanWidth, pxPerLine=256) ### Comment out to get rid of image after spectra
            
        # =====================================================================
        
        # previously measured atom centre is now just an estimate, from where to start our new calibration
        xt_estim = self.xt 
        yt_estim = self.yt
        
        
        self.XY_tip_set_pos(xt_estim, yt_estim) # move tip to estimated centre
        print("check tip was moved to tracking atom")
        time.sleep(0.5) # wait
        
        # initialise atom traking
        self.AtomTrack_CtrlSet('modulation','on') # Turn on Atom Tracking Modulation
        self.AtomTrack_CtrlSet('controller','on') # Turn on Atom Tracking Controller
        print("check atom tracking is on")
        
        time.sleep(self.timeForAtomTracking) # wait
        
        # end atom tracking
        self.AtomTrack_CtrlSet('controller','off')
        self.AtomTrack_CtrlSet('modulation','off')
        print("check atom tracking is off")
        time.sleep(0.5) # wait
        
        self.xt, self.yt = self.XY_tip_pos() # update xt and yt to calibrated vals
        
        if firstCalibration == False:
            
            dxt = self.xt - xt_estim # xdrift
            dyt = self.yt - yt_estim # ydrift
            
            self.x = self.x + dxt # calibrate x
            self.y = self.y + dyt # calibrate y
            print("xy have been calibrated. dx = ", dxt, "dy = ", dyt)
        
        # Z calibration: 
            
        # take a time average of Z, rather than taking a instantaneus Z reading.
        # Use the "Off Delay functionality on nanonis' Z controller window
        zNew = self.GetTimeAvZ()
        self.z = zNew # update z to calibrated val
        
        dz = zNew - self.z #zdrift
        print("z has been calibrated. dz = ", dz)
    
           
        self.calibrationCount = self.calibrationCount + 1
        print("calibration round ", self.calibrationCount, " complete")
        
        
        # # test ================================================================
        # if firstCalibration == False:
        #     xScanCentre, yScanCentre, scanWidth, = self.xt, self.yt, 10e-9
        #     self.TestStatus(xScanCentre, yScanCentre, scanWidth, pxPerLine=256)
            
        # # =====================================================================
        
        
        
    def Spectra(self):
        
        self.Z_feedback_set('off') # turn z controller off
        print("check z controller is off")
        
        xOut = self.x 
        yOut = self.y 
        
        
        while self.pointCount < (self.nPointsPerCalibration * self.calibrationCount):
            if self.pointCount < len(self.x):
                time.sleep(self.offDelay + 0.5)
                
                self.Z_pos_set(self.z + self.zRelForMoving) # lift tip
                print("check tip is at moving pos")
                time.sleep(1) # wait
                
                self.XY_tip_set_pos(xOut[self.pointCount], yOut[self.pointCount])
                self.WaitWhileTipMoves(xOut[self.pointCount], yOut[self.pointCount], 0.1e-12)
                time.sleep(0.5) # wait
                print("check tip was moved")
                
                
                # change bias slowly
                self.Bias_set_slow(self.VForSpectra, time_to_take_seconds=1,
                                   num_of_steps=100)
                
                self.Z_pos_set(self.z + self.zRelForSpectra)
                print("check tip is at spectra taking pos")
                time.sleep(2) # wait
                
                self.Z_SpectraStart(self.fileBaseName)
                print("check spectra is done")
                time.sleep(0.5) # wait
              
                
                self.Z_pos_set(self.z + self.zRelForMoving) # lift tip
                print("check tip is at moving pos")
                time.sleep(2) # wait
                
                # change bias slowly
                self.Bias_set_slow(self.VForMoving, time_to_take_seconds=1,
                                   num_of_steps=100)
                time.sleep(1) # wait
                
                self.pointCount = self.pointCount + 1
                print("spectra number ", self.pointCount, "/", len(self.x), " complete")
            else:
                self.pointCount = (self.nPointsPerCalibration * self.calibrationCount)
    
    
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
        
        time.sleep(self.offDelay) # wait for time average (and a bit longer to avoind timing errors)
        time.sleep(0.1)
        
        self.Z_feedback_set('off') # turn z controller off
        print("check z controller is off")
        
        time.sleep(self.offDelay) # wait for time average (and a bit longer to avoind timing errors)
        time.sleep(0.1)
        
        z = self.Z_pos_get() 
        self.Z_feedback_set('on') # turn z controller off
        print("check z controller is on")
        
        return z
        
    
    def TestStatus(self, xScanCentre, yScanCentre, scanWidth, pxPerLine):
        
        print('testing status')
        
        self.Z_feedback_set('on') # turn z controller on
        print("check z controller is on")
        time.sleep(1) # wait
        
        self.Scan_FrameSet(centreX=xScanCentre, centreY=yScanCentre,
                           widthX=scanWidth,  widthY=scanWidth, angle=self.angle)
        
        # start scan
        self.Scan_Action('start', 'up') 

        # while scanning -> wait
        self.Scan_Check_Hold()
        self.Scan_Check_Hold()        
        
        # load scan 
        _, scan = self.Scan_FrameDataGrab(self.zChanNum) 
        
        #plot
        fig, ax = plt.subplots()
        ax.imshow(scan, extent=[xScanCentre - scanWidth/2,
                                    xScanCentre + scanWidth/2,
                                    yScanCentre - scanWidth/2,
                                    yScanCentre + scanWidth/2],
                  origin='lower')

        ax.scatter(self.xt, self.yt, marker="P")
        
        
        if self.masks is None: 
            print(len(self.x), len(self.y))
            ax.scatter(self.x, self.y)
        else:
            ax.scatter(self.x, self.y, c=np.array(self.masks), marker="s",
                   cmap='inferno')
        
        plt.pause(0.5)
        plt.show(block = False)

        time.sleep(5)

        plt.close()

        
        
        
    def RunExperiment(self):
        # change bias slowly
        self.Bias_set_slow(self.VForMoving, time_to_take_seconds=2,
                           num_of_steps=100)
        
        
        
        # save sampling grid, for analysis of results
        timeStamp = datetime.now().strftime("%d-%m-%Y %H-%M-%S")
        np.save('xGrid_'+timeStamp, self.x)
        np.save('yGrid_'+timeStamp, self.y)
        
        # test ================================================================
        xScanCentre = 0.5 * ( np.max(self.x) + np.min(self.x) )
        yScanCentre = 0.5 * ( np.max(self.y) + np.min(self.y) )
        
        xGridWidth= np.max(self.x) - np.min(self.x) 
        yGridWidth= np.max(self.y) - np.min(self.y) 
        gridWidth = max([xGridWidth,yGridWidth])
        
        scanWidth = gridWidth + (0.25 * gridWidth)
        
        scanWidth = 7e-9
        
        self.TestStatus(xScanCentre, yScanCentre, scanWidth, pxPerLine=64)
     
        # =====================================================================
        
        # save the spectra grid, for the analysis of the experiment
        np.save(self.fileBaseName + '_initial_x', np.asarray(self.x))
        np.save(self.fileBaseName + '_initial_y', np.asarray(self.y))
        np.save(self.fileBaseName + '_masks', np.asarray(self.masks))
        
        while self.pointCount < len(self.x):
            self.CalibrateXYZ()
            
            self.Spectra()
            
        self.Z_feedback_set('on') # turn z controller on
        print("check z controller is on")
        


        # test ================================================================
        self.TestStatus(xScanCentre, yScanCentre, scanWidth, pxPerLine=64)
     
        # =====================================================================
        
        # re-save sampling grid
        timeStamp = datetime.now().strftime("%d-%m-%Y %H-%M-%S")
        np.save('xGrid_'+timeStamp, self.x)
        np.save('yGrid_'+timeStamp, self.y)
        
        self.Z_feedback_set('on') # turn z controller on
        print("check z controller is on")
        time.sleep(1) # wait
      
        
        self.XY_tip_set_pos(self.xt , self.yt) # move tip to estimated centre
        print("check tip was moved to tracking atom")
     
        
        time.sleep(2) # wait
        print("Experiment complete")
        self.close_socket()
        
        
        


#%% Inputs

nPointsPerCalibration = 1
zRelForMoving = 0e-9
zRelForSpectra = 0e-9
VForMoving = -0.8
VForSpectra = 0.6
# VForScanning = 

timeForAtomTracking = 20
offDelay = 1
fileBaseName = 'dfzMap_1_'


xGrid = np.linspace(-5e-9, 5e-9, 2)
yGrid = np.zeros(2)


angle = -171  # Angle in degrees
rotation_matrix = np.array([[np.cos(np.radians(angle)), -np.sin(np.radians(angle))],
                            [np.sin(np.radians(angle)), np.cos(np.radians(angle))]])
rotated_points = np.dot(rotation_matrix, np.vstack((xGrid, yGrid)))
xGrid = rotated_points[0, :]
yGrid = rotated_points[1, :]

print(xGrid, yGrid)
# xGrid = np.linspace(-5e-9, 5e-9, 2) [-5,5]
# yGrid = np.zeros(2) [0,0]

#%%
e = Experiment_dfVmap(nPointsPerCalibration, zRelForMoving,
              zRelForSpectra, VForMoving, VForSpectra, timeForAtomTracking,
              offDelay, fileBaseName,
              xGrid, yGrid,angle=0)

e.RunExperiment()





