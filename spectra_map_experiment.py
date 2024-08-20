# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 20:15:46 2024

@author: physicsuser
"""



import python_interface_nanonis_v7 as python_nano
import numpy as np
import time
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt 
from datetime import datetime

matplotlib.rc('image', cmap='gray')
#%% 

class SamplingGrid():
    
    def __init__(self, sideLength, PointsPerLine, xCentre, yCentre):
        """
        Parameters
        ----------
        sideLength : float
            side length of the square grid in metres
        PointsPerLine : int
            number of sampling points within the side length 
        xCentre : float
            x centre coordinate in metres
        yCentre : float
            y centre coordinate in metres
            
        """
        self.xCentre = xCentre
        self.yCentre = yCentre
        
        xMin, xMax = xCentre - 0.5 * sideLength, xCentre + 0.5 * sideLength 
        yMin, yMax = yCentre - 0.5 * sideLength, yCentre + 0.5 * sideLength
        
        self.xTicks = np.linspace(xMin, xMax, num=PointsPerLine)
        self.yTicks = np.linspace(yMin, yMax, num=PointsPerLine)

    

    def _SamplingOrderTune(self, x2D, y2D):
        """
        To rule out any possible systematic errors related to the sampling
        order. Ie. can change whether we sample top -> bottom, bottom -> top,
        left -> right...
        
        Here, we also ensure that drift is minimised by sneaking through the 
        sampling lines (If this doesn't make sense see example below).

        Parameters
        ----------
        x2D : 2D array
            x sampling grid coordinates.
        y2D : TYPE
            DESCRIPTION.

        Returns
        -------
        x2D : TYPE
            DESCRIPTION.
        y2D : TYPE
            DESCRIPTION.

        """
        # overall sampling direction:
        # x2D, y2D = x2D.T, y2D.T  
        # y2D = np.flipud(y2D) 
        
        # snaking sampling order:
        # Use a numpy left right flip function on every other row
        # Every other row is sliced as [start = 0 : end = -1 : step = 2]
        x2D[0:-1:2], y2D[0:-1:2] = np.fliplr(x2D[0:-1:2]), np.fliplr(y2D[0:-1:2]) 
        
        return x2D, y2D
    
    
    
    def SquareGrid(self, xTicks=None, yTicks=None):
        """
        Returns
        -------
        x : 1D numpy array
            x coordinates of sampling points in metres
        y : 1D numpy array
            y coordinates of sampling points in metres
        
        Note - x and y are ordered snaking upwards to minimise drift. (See example below).
        """
        if xTicks is None: xTicks = self.xTicks
        if yTicks is None: yTicks = self.yTicks
        
        x2D, y2D = np.meshgrid(xTicks, yTicks)
        
        x2D, y2D = self._SamplingOrderTune(x2D, y2D)
        
        # flatten the 2D arrays, to 1D 
        x = x2D.ravel()
        y = y2D.ravel()
        
        return x, y
        
    
    
    def CircleGrid(self, xTicks=None, yTicks=None):
        """
            number of sampling points within the length of the circle's diameter
            will be only an approximation bc we want sampling points aranged 
            in a square manner (so we get sqare pixels in our eg. KPFM spectra
            map) and a round border. 
        

        Returns
        -------
        x : 1D numpy array
            x coordinates of sampling points in metres
        y : 1D numpy array
            y coordinates of sampling points in metres
        
        Note - x and y are ordered snaking upwards to minimise drift. (See example below).
        """
        # 1. define a basic square grid:
            
        x, y = self.SquareGrid(xTicks, yTicks)
        
        
        # 2. remove unwanted sampling points:
        
        # coordinate trasformation to polar, with origin at (xCentre, yCentre)
        xDash = x - self.xCentre 
        yDash = y - self.yCentre
        r = np.sqrt(xDash**2 + yDash**2)
        
        # In this new coordinate system, sampling points with r>rmax are unwanted
        diameter = self.xTicks[-1] - self.xTicks[0]
        rmax = 0.5 * diameter 
        x = x[r <= rmax]
        y = y[r <= rmax]
        
        return x, y
    
    
    
    def RemoveSamplingAboveAtoms(self, x, y, 
                                 xAtoms, yAtoms, forbiddenDiameter,
                                 masks=None):
        """
        Parameters
        ----------
        x : 1D array like
            x coordinates of sampling points in metres.
        y : 1D array like
            y coordinates of sampling points in metres.
        xAtoms : 1D array like
            x coordinates of atoms positions in metres.
        yAtoms : 1D array like
            y coordinates of atoms positions in metres.
        forbiddenDiameter : float
            diameter, in metres, around atom centre to avoid sampling over.

        Returns
        -------
        xGrid : 1D array
            x coordinates of sampling points, with points above atoms removed.
        yGrid : 1D array
            x coordinates of sampling points, with points above atoms removed.

        """
        # need to use polar coords to define forbidden regions
        for i in range(len(xAtoms)):
            # move the origin to the atom centre
            xDash = x - xAtoms[i]
            yDash = y - yAtoms[i]
            
            # now the forbidden region is simply anything with r <= forbidden radius
            r = np.sqrt(xDash**2 + yDash**2)
            forbiddenRad = forbiddenDiameter / 2
            x = x[r >= forbiddenRad]
            y = y[r >= forbiddenRad]
            if masks is not None: 
                masks = masks[r >= forbiddenRad]
            
        return x, y, masks
    
    
    
    
                       
    
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
    
    
    def UpsamplingSteps(self, x, y, upsampling, radSteps):
            
        def upsample(x, y, spacing, upsampling,
                     minLim, maxLim):
            
            x, y = np.array(x), np.array(y)
            
            r = np.sqrt((x-self.xCentre)**2 + (y-self.yCentre)**2)
            xNew, yNew = [], []
            
            idxExt = 0 
            for i in np.nonzero((r >= minLim) & (r < maxLim))[0]:
                i = i + idxExt
                
                # per old xy, n new xy ticks slightly irregularly spaced
                xTicksExt = np.linspace(x[i]-0.5*spacing, x[i]+0.5*spacing,
                                   upsampling*2 + 1)[1:-1:2]
                yTicksExt = np.linspace(y[i]-0.5*spacing, y[i]+0.5*spacing,
                                   upsampling*2 + 1)[1:-1:2]
                
                xExt, yExt = self.SquareGrid(xTicksExt, yTicksExt)
                xExt, yExt = np.array(xExt), np.array(yExt)
                
                
                x, y = np.delete(x, i), np.delete(y, i)
                x, y = np.insert(x, i, xExt), np.insert(y, i, yExt)
                
                xNew.extend(xExt)
                yNew.extend(yExt)
                
                idxExt = idxExt + len(xExt) - 1
            
            xNew, yNew = np.array(xNew), np.array(yNew)
            spacingNew = xTicksExt[0] - xTicksExt[1]
            
            return x, y, xNew, yNew, spacingNew
        
        spacing = self.xTicks[1] - self.xTicks[0] # assuming uniform xy spacing
                
        maxLim = radSteps
        xHist = []
        yHist = []
        
        for i in range(len(upsampling)):

            x, y, xNew, yNew, spacing = upsample(x, y, spacing, upsampling[i],
                          0, maxLim[i])
            
            xHist.append(xNew)
            yHist.append(yNew)
           
        masks = np.zeros(len(x))
        
        for i in range(len(upsampling)):
            
            xMask = np.isin(x, xHist[i])
            yMask = np.isin(y, yHist[i])
            mask = np.logical_and(xMask, yMask)
            fig, ax = plt.subplots()
            ax.scatter(x[mask], y[mask])
            
            masks = np.add(masks, (i+1)*mask.astype(int))
            
        return x, y, masks




class Experiment_dfVmap(python_nano.python_Nanonis_TCP): # define child class for our experiment
    
    def __init__(self, nPointsPerCalibration, zRelForMoving,
                 zRelForSpectra, VForMoving, VForSpectra, timeForAtomTracking,
                 offDelay, fileBaseName,
                 xAtomsVec, yAtomsVec, a, b, theta, gridDiameter, 
                 gridPointsPerDiameter, gridAtomDiameterToAvoid, 
                 upsampling=None, radSteps=None):
        
        super().__init__() 
        

        self.nPointsPerCalibration = nPointsPerCalibration
        self.zRelForMoving = zRelForMoving
        self.zRelForSpectra = zRelForSpectra
        self.VForMoving = VForMoving
        self.VForSpectra = VForSpectra
        self.timeForAtomTracking = timeForAtomTracking
        self.offDelay = offDelay
        self.fileBaseName = fileBaseName
        
        self.xAtomsVec = xAtomsVec
        self.yAtomsVec = yAtomsVec
        self.a = a
        self.b = b
        self.theta = theta
        self.gridDiameter = gridDiameter
        self.gridPointsPerDiameter = gridPointsPerDiameter
        self.gridAtomDiameterToAvoid = gridAtomDiameterToAvoid
        self.upsampling = upsampling
        self.radSteps = radSteps
        
        
        self.zChanNum = int(30)
        # # find the Z (m) channel 
        # _, self.recordedChanels, xPxPerLine, yPxPerLine = self.Scan_BufferGet()
        # for chanelNum in self.recordedChanels:
        #     chanName, _ = self.Scan_FrameDataGrab(chanelNum) 
        #     if chanName == 'Z (m)':
        #         self.zChanNum = chanelNum
                
        self.pointCount = 0
        self.calibrationCount = -1
        
        
        
    def RealAtomPos(self):
        
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
        
        x = np.array(self.xAtomsVec, dtype=float)
        y = np.array(self.yAtomsVec, dtype=float)
        
        x = self.a * x 
        y = self.b * y
        
        x, y = rotate(x, y, rot=self.theta)
        
        # yes = ['yes', 'y', '', 'YES', 'Yes']
        # print('Choose which atom to track.')
        # if input('Place the tip on the chosen atom. Done?: ') in yes:
        #     self.xt, self.yt = self.XY_tip_pos()
        
        self.xt, self.yt = self.XY_tip_pos()
    
        self.CalibrateXYZ(firstCalibration=True)
        
        x = x + self.xt
        y = y + self.yt
        
        return x, y



    def Grid(self, xAtoms, yAtoms, upsampling=None, radSteps=None):
        
        xCentre = np.average(xAtoms)
        yCentre = np.average(yAtoms)
        
        sg = SamplingGrid(self.gridDiameter, self.gridPointsPerDiameter,
                                           xCentre, yCentre)

        xGrid, yGrid = sg.CircleGrid()
        
        if upsampling is None: masks = None
        else: xGrid, yGrid, masks = sg.UpsamplingSteps(xGrid, yGrid,
                                                     upsampling, radSteps)
        
        xGrid, yGrid = sg.Rotate(xGrid, yGrid, xPivot=xCentre, yPivot=yCentre,
                                 rot=self.theta)

        xGrid, yGrid, masks = sg.RemoveSamplingAboveAtoms(xGrid, yGrid, xAtoms, yAtoms,
                                                   self.gridAtomDiameterToAvoid,
                                                   masks=masks)
        
        return xGrid, yGrid, masks
    
        
        
    def CalibrateXYZ(self, firstCalibration=False):
        self.Z_feedback_set('on') # turn z controller on
        print("check z controller is on")
        time.sleep(1) # wait
        
        # test ================================================================
        if firstCalibration == False:
            xScanCentre, yScanCentre, scanWidth, = self.xt, self.yt, 1e-9
            self.TestStatus(xScanCentre, yScanCentre, scanWidth, pxPerLine=256)
            
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
                self.WaitWhileTipMoves(xOut[self.pointCount], yOut[self.pointCount], 20e-12)
                time.sleep(0.5) # wait
                print("check tip was moved")
                
                
                # change bias slowly
                self.Bias_set_slow(self.VForSpectra, time_to_take_seconds=1,
                                   num_of_steps=100)
                
                self.Z_pos_set(self.z + self.zRelForSpectra)
                print("check tip is at spectra taking pos")
                time.sleep(2) # wait
                
                self.Bias_SpectraStart(self.fileBaseName)
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
        
        self.Z_feedback_set('off') # turn z controller off
        print("check z controller is off")
        
        time.sleep(self.offDelay) # wait for time average (and a bit longer to avoind timing errors)
        time.sleep(1)
        z = self.Z_pos_get() 
        self.Z_feedback_set('on') # turn z controller off
        print("check z controller is on")
        
        # z = self.Z_pos_get() 
        
        return z
        
    
    def TestStatus(self, xScanCentre, yScanCentre, scanWidth, pxPerLine):
        
        print('testing status')
        
        self.Z_feedback_set('on') # turn z controller on
        print("check z controller is on")
        time.sleep(1) # wait
        
        self.Scan_FrameSet(centreX=xScanCentre, centreY=yScanCentre,
                           widthX=scanWidth,  widthY=scanWidth, angle=0)
        
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
            ax.scatter(self.x, self.y)
        else:
            ax.scatter(self.x, self.y, c=np.array(self.masks), marker="s",
                   cmap='inferno')
        
        plt.pause(0.5)
        plt.show()
        
        
        
    def RunExperiment(self):
        # change bias slowly
        self.Bias_set_slow(self.VForMoving, time_to_take_seconds=2,
                           num_of_steps=100)
        
        xAtoms, yAtoms = self.RealAtomPos()
        
        self.x, self.y, self.masks = self.Grid(xAtoms, yAtoms,
                                               self.upsampling, self.radSteps)
        
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
        
        
        # self.TestStatus(xScanCentre, yScanCentre, scanWidth, pxPerLine=512)
     
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
        # self.TestStatus(xScanCentre, yScanCentre, scanWidth, pxPerLine=512)
     
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

nPointsPerCalibration = 5
zRelForMoving = 0.2e-9
zRelForSpectra = 0e-9
VForMoving = 0.2
VForSpectra = 0.6

timeForAtomTracking = 20
offDelay = 1
fileBaseName = 'dfVMap_InSb_69_'


# xAtomsVec = np.array([0, 4, -2, 6, 0, 4], dtype=float)
# yAtomsVec = np.array([0, 0, -5, -5, -10, -10], dtype=float)
xAtomsVec = np.array([0, -3, 3, -4, 4, -3, 3, 0], dtype=float) # change this 
yAtomsVec = np.array([0, -1, -1, -5, -5, -9, -9, -10], dtype=float) # change this 

# xAtomsVec = np.array([0, -2, -4, -6, -7, -6, -4, -2, 0, 1], dtype=float)
# yAtomsVec = np.array([0, -2, -2, 0, 3, 6, 8, 8, 6, 3], dtype=float)

# xAtomsVec = np.array([0, 3, 5, 7, 9, 0, 2, 4, 6, 9], dtype=float)
# yAtomsVec = np.array([0, -1, 1, 3, 5, 4, 6, 8, 10, 9], dtype=float)

a = 0.648e-9
b = 0.458e-9
theta = 2.2
gridDiameter = 55e-9
gridPointsPerDiameter = 10
gridAtomDiameterToAvoid = 3e-9
upsampling = [4,5]
radSteps = [10e-9,2e-9]
# upsampling = None
# radSteps = None

e = Experiment_dfVmap(nPointsPerCalibration, zRelForMoving,
              zRelForSpectra, VForMoving, VForSpectra, timeForAtomTracking,
              offDelay, fileBaseName,
              xAtomsVec, yAtomsVec, a, b, theta, gridDiameter, 
              gridPointsPerDiameter, gridAtomDiameterToAvoid, upsampling, radSteps)

e.RunExperiment()

#%% Run the experiment

# zRelForSpectra = [ 0e-9, -0.05e-9, -0.1e-9]
# fileBaseName = ['dfVMap_InSb_55_',
#                 'dfVMap_InSb_56_',
#                 'dfVMap_InSb_57_']

# for z, name in zip(zRelForSpectra, fileBaseName):


#     e = Experiment_dfVmap(nPointsPerCalibration, zRelForMoving,
#                   z, VForMoving, VForSpectra, timeForAtomTracking,
#                   offDelay, name,
#                   xAtomsVec, yAtomsVec, a, b, theta, gridDiameter, 
#                   gridPointsPerDiameter, gridAtomDiameterToAvoid, upsampling, radSteps)
    
#     e.RunExperiment()
    
#     time.sleep(1)



