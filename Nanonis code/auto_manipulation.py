# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 16:53:45 2024

@author: physicsuser
"""
import matplotlib
import matplotlib.pyplot as plt

import time 
import numpy as np
import warnings
import sys

import python_interface_nanonis_v7 as python_nano
import find_atoms
from path_finding import AStarSquareGrid

matplotlib.use('Agg')


class AutoAssembly(python_nano.python_Nanonis_TCP, find_atoms.CircleFinder):
    
    
    def __init__(self, 
                 scanningV, scanningI, manipulationV, manipulationI,
                 atomDiameter, dangerDiameter,
                 dangerPerimeterIndent, errorInPos):
        
        super().__init__() 
        try: 
            self.scanningV = scanningV
            self.scanningI = scanningI
            self.manipulationV = manipulationV
            self.manipulationI = manipulationI
            self.atomDiameter = atomDiameter
            self.dangerDiameter = dangerDiameter
            self.errorInPos = errorInPos
            
            # define the dimensions and location of our scan window
            self.xScanCentre, self.yScanCentre, self.xScanSize, self.yScanSize, _ = self.Scan_FrameGet()
            _, recordedChanels, self.xPxPerLine, self.yPxPerLine = self.Scan_BufferGet()        
            
            if self.xScanSize != self.yScanSize: 
                raise ValueError('sxm must be square')
    
                
            if self.xPxPerLine != self.yPxPerLine: 
                raise ValueError('sxm must have square pixels')
    
                
            # set the manipation limits
            self.xManipulationMin = self.xScanCentre - self.xScanSize/2 + dangerPerimeterIndent
            self.xManipulationMax = self.xScanCentre + self.xScanSize/2 - dangerPerimeterIndent
            self.yManipulationMin = self.yScanCentre - self.yScanSize/2 + dangerPerimeterIndent
            self.yManipulationMax = self.yScanCentre + self.yScanSize/2 - dangerPerimeterIndent

                
            # find the Z (m) channel 
            for chanelNum in recordedChanels:
                chanName, _ = self.Scan_FrameDataGrab(chanelNum) 
                if chanName == 'Z (m)':
                    self.zChanNum = chanelNum
                    
        except Exception as e: 
            print(e)
            self.close_socket()
            sys.exit()
                
    def Run(self, substrate, gridName, path2DesiredGrid):
        try:
            # check circle finder works well
            # load scan 
            _, scan = self.Scan_FrameDataGrab(self.zChanNum) 
            
            # # define atom grid 
            # # (important note: this is a grid of *ALL* atoms found, some of these 
            # # we wont want to move eg. if they are already in a desired position)
            # # This is useful because we need to define *ALL* danger zones.
            # self.xAtomGrid, self.yAtomGrid = self.Scan2AtomGrid(scan,
            #                                                     sigmaForGaussianFilter=1,
            #                                                     confidence=0.4,
            #                                                     showPreprocessing = True,
            #                                                     plotFoundAtoms = True)
            
            # import the desired grid
            self.xDesiredGrid, self.yDesiredGrid = self.LoadDesiredGrid(substrate,
                                                                        gridName,
                                                                        path2DesiredGrid)
            
            # traslate the desired grid to the user-defined tip position
            self.xDesiredGrid, self.yDesiredGrid = self.ChangeDesiredGridPos()
            
            # Grab scanning params ================================================
            
            # initiate scanning parameters
            # change bias slowly... (to prevent a 'bias pulse')
            self.Bias_set_slow(self.scanningV, time_to_take_seconds=5)
           
            # change setpoint current slowly
            self.Current_SetPoint_set_slow(self.scanningI)
    
            # start scan
            self.Scan_Action('start', 'up') 
            time.sleep(10)
            # while scanning -> wait
            self.Scan_Check_Hold()
            # self.Scan_Check_Hold()
            
            # load scan 
            _, scan = self.Scan_FrameDataGrab(self.zChanNum) 
            
            # define atom grid 
            # (important note: this is a grid of *ALL* atoms found, some of these we wont want to move eg. if they are already in a desired position)
            # This is useful because we need to define *ALL* danger zones.
            self.xAtomGrid, self.yAtomGrid = self.Scan2AtomGrid(scan, 
                                                                sigmaForGaussianFilter=2,
                                                                confidence=0.4,
                                                                showPreprocessing = True, 
                                                                plotFoundAtoms = True)
            
            # check for atoms in desired positions
            self.indxOfAvailableAtoms, self.indxOfAvailableDesired = self.AvailableAtomsNDesiredPos()
            
            self.iterationCount = 0
            
            while len(self.indxOfAvailableDesired) != 0 and len(self.indxOfAvailableAtoms) != 0:
                print('manipulation attempt ', self.iterationCount)
                
                # find path
                xTipPos, yTipPos = self.Path(0)
                
                # plot
                self.PlotGrids(scan, xTipPos, yTipPos)
                
                # move the tip to the atom to manipulate ie. the first path point
                self.XY_tip_set_pos(xTipPos[0],yTipPos[0])
                time.sleep(2)
                
                # quick atom tracking to find the atom's centre
                self.AtomTrack_CtrlSet(1, 1) # initialise atom traking
                print("check atom tracking is on")
                
                time.sleep(20)
    
                self.AtomTrack_CtrlSet(0, 0) # end atom tracking
                
                # Grab manipulation params ========================================
                
                # initiate manipulation parameters
                # change bias slowly... (to prevent a 'bias pulse')
                self.Bias_set_slow(self.manipulationV, time_to_take_seconds=10)
                
                # change setpoint current slowly
                self.Current_SetPoint_set_slow(self.manipulationI, time_to_take_seconds=10)
                
                # move the tip (with, hopefully, the atom)
                for i in range(len(xTipPos)):
                    self.XY_tip_set_pos(xTipPos[i],yTipPos[i])
                    time.sleep(0.2)
                
                # Grab scanning params ============================================
                
                # initiate scanning parameters
                # change bias slowly... (to prevent a 'bias pulse')
                self.Bias_set_slow(self.scanningV, time_to_take_seconds=10)
               
                # change setpoint current slowly
                self.Current_SetPoint_set_slow(self.scanningI, time_to_take_seconds=10)
                
                # traslate the desired grid to the user-defined tip position
                # self.xDesiredGrid, self.yDesiredGrid = self.ChangeDesiredGridPos()
    
                # start scan
                self.Scan_Action('start', 'up') 
                time.sleep(10)
                # while scanning -> wait
                self.Scan_Check_Hold()
                # self.Scan_Check_Hold()
                
                # load scan 
                _, scan = self.Scan_FrameDataGrab(self.zChanNum) 
                
                # define atom grid 
                # (important note: this is a grid of *ALL* atoms found, some of these we wont want to move eg. if they are already in a desired position)
                # This is useful because we need to define *ALL* danger zones.
                self.xAtomGrid, self.yAtomGrid = self.Scan2AtomGrid(scan, 
                                                                    sigmaForGaussianFilter=2,
                                                                    confidence=0.4,
                                                                    showPreprocessing=True,
                                                                    plotFoundAtoms=True)
                
                # check for atoms in desired positions
                self.indxOfAvailableAtoms, self.indxOfAvailableDesired = self.AvailableAtomsNDesiredPos()
                
                self.iterationCount = self.iterationCount + 1
                
            print("Finished!")
            # iniciate manipulation parameters
            # initiate scanning parameters
            # change bias slowly... (to prevent a 'bias pulse')
            self.Bias_set_slow(self.scanningV, time_to_take_seconds=10)
           
            # change setpoint current slowly
            self.Current_SetPoint_set_slow(self.scanningI, time_to_take_seconds=10)
            
            self.close_socket()
            sys.exit()
            
        except Exception as e: 
            print(e)
            self.close_socket()
            sys.exit()
        
    def Scan2AtomGrid(self, scan, sigmaForGaussianFilter =1.5, confidence=0.2,
                      showPreprocessing = False, plotFoundAtoms = False):
        # find atoms
        atomRadiusInPx = self.RadiiOfInterest(self.atomDiameter/2, 0, self.xPxPerLine, self.xScanSize)
        
        self.Estimate(scan, atomRadiusInPx, sigmaForGaussianFilter, confidence, showPreprocessing)
        if plotFoundAtoms == True: self.PlotAllAtoms(labelAtoms = False)
        
        # create grid
        xAtomGridInPx, yAtomGridInPx = self.AtomGrid()
        
        # px coord syst -> nanonis cood syst
        xAtomGrid, yAtomGrid = self.Px2MetresCoordinates(xAtomGridInPx, yAtomGridInPx, 
                                                       self.xPxPerLine, self.xScanSize,
                                                       self.xScanCentre, self.yScanCentre)
        
        return xAtomGrid, yAtomGrid
    
    
    def Current_SetPoint_set_slow(self, current, time_to_take_seconds=5, num_of_steps=50):
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



    def LoadDesiredGrid(self, substrate, gridName, path = r'desired_grids'):
        """
        load a grid of desired atom positions (previously defined) and translate it to the current scan centre.
        Desired atom grids are defined using desiredGrid.py
        There will be several grids that will be used again and again, so this method saves us from
        having to redefine a previously-used one, just because we are at a different 
        position in space. 
        Parameters
        ----------
        substrate : string 
            Used to retireve the grid's filename. The idea here is that we name all desired
            grids following a standard name structure.
        gridName : string
            Used to retireve the grid's filename. The idea here is that we name all desired
            grids following a standard name structure.
        path : string, optional
            DESCRIPTION. The default is r'desired_grids'.

        Returns
        -------
        xDesiredGrid : 1D numpy array
            x coordinates of the desired atom positions.
        yDesiredGrid : 1D numpy array
            y coordinates of the desired atom positions.

        """
        
        xDesiredGrid = np.load(path + r'\xdg_' + substrate + '_' + gridName + '.npy')
        yDesiredGrid = np.load(path + r'\ydg_' + substrate + '_' + gridName + '.npy')

        # translate grid: nanonis origin -> scan centre
        xDesiredGrid = xDesiredGrid + self.xScanCentre
        yDesiredGrid = yDesiredGrid + self.yScanCentre
    
        return xDesiredGrid, yDesiredGrid
           
    
    
    def ChangeDesiredGridPos(self):
        """
        This method allows the user to traslate the whole grid of desired atom positions
        by defining the position of the first grid point as the STM tip position. When loading
        a desired grid (using the LoadDesiredGrid method), the grid will appear around our scan centre.
        This might not align well with the positions where atoms will lie on, given the underlying 
        subtrate. In this case, the manipulation will likely be suboptimal.
        

        Returns
        -------
        xDesiredGrid : 1D numpy array
            x coordinates of the translated desired atom positions.
        yDesiredGrid : 1D numpy array
            y coordinates of the translated desired atom positions.

        """
        
        yes = ['yes', 'y', '', 'YES', 'Yes']
        
        if input('place the tip at the new grid position. Done?: ') in yes:
            
            # translate grid: scan centre -> nanonis origin
            xDesiredGrid = self.xDesiredGrid - self.xScanCentre
            yDesiredGrid = self.yDesiredGrid - self.yScanCentre
            
            # translate grid origin: scan centre of where the grid was created -> first grid point
            xDesiredGrid = xDesiredGrid - xDesiredGrid[0]
            yDesiredGrid = yDesiredGrid - yDesiredGrid[0]
            
            # translate grid: nanonis origin -> tip position
            xNew, yNew = self.XY_tip_pos()
            xDesiredGrid = xDesiredGrid + xNew
            yDesiredGrid = yDesiredGrid + yNew
            
        return xDesiredGrid, yDesiredGrid    

    
    
    def AvailableAtomsNDesiredPos(self):
        """
        

        Returns
        -------
        indxOfAvailableAtoms: list
            index of available atoms for manipulation in self.xAtomGrid and self.yAtomGrid
        xDesiredGrid : 1D numpy array
            x coordinates of our desired positions, where fulfilled positions are excluded. 
        yDesiredGrid : 1D numpy array
            x coordinates of our desired positions, where fulfilled positions are excluded.

        """
        indxOfAvailableAtoms = []
        indxOfAvailableDesired = []
        
        for atomIndx in range(len(self.xAtomGrid)):
            xDelta = self.xDesiredGrid - self.xAtomGrid[atomIndx]
            yDelta = self.yDesiredGrid - self.yAtomGrid[atomIndx]
            rDelta = np.sqrt(xDelta**2 + yDelta**2 ) # array of distance from the (atomIndx)th atom to each of the desired positions. Size of this array = size of desired grid
            print('rDelta: ', rDelta)
            print('err tolerance: ', self.errorInPos)
            
            # if the (atomIndx)th atom is within our safe manipulation area
            if (self.xAtomGrid[atomIndx] >= self.xManipulationMin
            and self.xAtomGrid[atomIndx] <= self.xManipulationMax 
            and self.yAtomGrid[atomIndx] >= self.yManipulationMin 
            and self.yAtomGrid[atomIndx] <= self.yManipulationMax):
                
                # if the (atomIndx)th atom is not in any of the desired positions
                if np.all(rDelta >= self.errorInPos): 
                    indxOfAvailableAtoms.append(atomIndx)
                    print('pass')
                
        for desiredIndx in range(len(self.xDesiredGrid)):
            xDelta = self.xDesiredGrid[desiredIndx] - self.xAtomGrid
            yDelta = self.yDesiredGrid[desiredIndx] - self.yAtomGrid
            rDelta = np.sqrt(xDelta**2 + yDelta**2 )
            
            if (self.xDesiredGrid[desiredIndx] >= self.xManipulationMin
            and self.xDesiredGrid[desiredIndx] <= self.xManipulationMax 
            and self.yDesiredGrid[desiredIndx] >= self.yManipulationMin 
            and self.yDesiredGrid[desiredIndx] <= self.yManipulationMax):
               
                if np.all(rDelta >= self.errorInPos): 
                    indxOfAvailableDesired.append(desiredIndx)
                
        print(indxOfAvailableAtoms, indxOfAvailableDesired)
        return indxOfAvailableAtoms, indxOfAvailableDesired
        


    def PlotGrids(self, scanData, xTipPos, yTipPos):
        
        fig, ax = plt.subplots(nrows=1, ncols=1)
        
        # scan
        ax.imshow(scanData,alpha = 0.3, extent=[self.xScanCentre - self.xScanSize/2,
                                    self.xScanCentre + self.xScanSize/2,
                                    self.yScanCentre - self.yScanSize/2,
                                    self.yScanCentre + self.yScanSize/2])
        
        # danger zones
        ax.pcolor(self.xPathGrid, self.yPathGrid, self.zPathGrid, alpha = 0.3, cmap = 'Wistia')
        
        # atom and desired grid
        ax.scatter(xTipPos, yTipPos, label = 'tip path')
        ax.scatter(self.xAtomGrid[self.indxOfAvailableAtoms], self.yAtomGrid[self.indxOfAvailableAtoms], c = 'red', label = 'Available atoms for manipulation')
        ax.scatter(self.xDesiredGrid[self.indxOfAvailableDesired], self.yDesiredGrid[[self.indxOfAvailableDesired]], c = 'green', label = 'Remaining desired atom positions')
        ax.legend()
        
        plt.savefig('testManipulationSetup' + str(self.iterationCount))
        
        
    def ChooseAtomForManipulation(self, desiredNumber):
        xAvailableDesiredGrid = self.xDesiredGrid[self.indxOfAvailableDesired]
        yAvailableDesiredGrid = self.yDesiredGrid[self.indxOfAvailableDesired]
        xDesired, yDesired = xAvailableDesiredGrid[desiredNumber], yAvailableDesiredGrid[desiredNumber]  
        xDelta = self.xAtomGrid - xDesired
        yDelta = self.yAtomGrid - yDesired
        rDelta = np.sqrt(xDelta**2 + yDelta**2)
        chosenAtomNumber = np.where(rDelta == rDelta.min())
        return chosenAtomNumber
    

    def PathGrid(self, atomNumToManipulate):
        
        x = np.linspace(self.xManipulationMin, self.xManipulationMax, num=256)
        y = np.linspace(self.yManipulationMin, self.yManipulationMax, num=256)
        
        # allowed points = 0; forbidden points = 1
        # start with a grid of all allowed points
        z = np.zeros((256,256))
        # mix with a grid of all non-allowed points, wherever a condition is met
        nonAllowed = z + 1
        
        # need to use polar coords to define the condition
        for atomNumber in range(len(self.xAtomGrid)):
            if atomNumber != atomNumToManipulate:
                # move the origin to the atom centre
                x2D, y2D = np.meshgrid(x,y)
                xDash = x2D - self.xAtomGrid[atomNumber]
                yDash = y2D - self.yAtomGrid[atomNumber]
                
                # now the forbidden region is simply anything with r <= danger radius
                rDash = np.sqrt(xDash**2 + yDash**2)
                
                z = np.where(rDash <= self.dangerDiameter/2, nonAllowed, z)
            
        xPathGrid = x
        yPathGrid = y
        zPathGrid = z
        
        return xPathGrid, yPathGrid, zPathGrid
        

    def Path(self, desiredGridNum):
           
        
        bestCost = float('inf')
        bestAtomForManipulation = None

        bestXTipPosIndxList = None
        bestYTipPosIndxList = None
        
        # define path grid
        xPathGrid, yPathGrid, zPathGrid = self.PathGrid(0)
        
        xAvailableDesiredGrid = self.xDesiredGrid[self.indxOfAvailableDesired]
        yAvailableDesiredGrid = self.yDesiredGrid[self.indxOfAvailableDesired]
              
        xEnd = xAvailableDesiredGrid[desiredGridNum] # Nanonis Coordinates
        yEnd = yAvailableDesiredGrid[desiredGridNum] # Nanonis Coordinates
        
        # tranformation: Nanonis coords -> index of position in path grids
        # xPathGrid, yPathGrid gives us the info needed to make this transformation
        # find the index of the nearest value
        xEndIndx = np.argmin(abs(xPathGrid - xEnd))
        yEndIndx = np.argmin(abs(yPathGrid - yEnd))
    
        end = (xEndIndx, yEndIndx)
        
        for atomNum in self.indxOfAvailableAtoms:
                 
            
            xStart = self.xAtomGrid[atomNum] # Nanonis Coordinates 
            yStart = self.yAtomGrid[atomNum] # Nanonis Coordinates 
            
            xStartIndx = np.argmin(abs(xPathGrid - xStart))
            yStartIndx = np.argmin(abs(yPathGrid - yStart))
            
            # define path grid
            xPathGrid, yPathGrid, zPathGrid = self.PathGrid(atomNum)
            
            start = (xStartIndx, yStartIndx)
            
            pathfinder = AStarSquareGrid(zPathGrid)
            
            xTipPosIndxList, yTipPosIndxList = pathfinder.find_path(start, end)
            
            if xTipPosIndxList is not None and len(xTipPosIndxList) < bestCost:
                bestCost = len(xTipPosIndxList)
                bestAtomForManipulation = atomNum
                
                bestXTipPosIndxList = xTipPosIndxList
                bestYTipPosIndxList = yTipPosIndxList
                
                self.xPathGrid, self.yPathGrid, self.zPathGrid = xPathGrid, yPathGrid, zPathGrid 
                
        if bestXTipPosIndxList is not None:
            
            # xTipPos = [self.xAtomGrid[bestAtomForManipulation]] + self.xPathGrid[bestXTipPosIndxList] + [xEnd]
            # yTipPos = [self.yAtomGrid[bestAtomForManipulation]] + self.yPathGrid[bestYTipPosIndxList] + [yEnd]
            xTipPos = self.xPathGrid[bestXTipPosIndxList] 
            yTipPos = self.yPathGrid[bestYTipPosIndxList]
            
            xTipPos = np.insert(xTipPos, 0, self.xAtomGrid[bestAtomForManipulation])
            yTipPos = np.insert(yTipPos, 0, self.yAtomGrid[bestAtomForManipulation])
            
            xTipPos = np.insert(xTipPos, -1, xEnd)
            yTipPos = np.insert(yTipPos, -1, yEnd)            
            
            return xTipPos, yTipPos
            
        else:
            print("No path found. Close socket.")
            # iniciate manipulation parameters
            # initiate scanning parameters
            # change bias slowly... (to prevent a 'bias pulse')
            self.Bias_set_slow(self.scanningV)
           
            # change setpoint current slowly
            self.Current_SetPoint_set_slow(self.scanningI)
            
            self.close_socket()
            sys.exit()
        
        



        
assembly = AutoAssembly(scanningV = 200e-3, scanningI = 4e-12, manipulationV = 30e-3, 
                        manipulationI = 350e-12, atomDiameter = 1e-9, 
                        dangerDiameter = 1.2e-9, dangerPerimeterIndent = 0.65e-9, errorInPos=1.2e-9)

assembly.Run('InSb(110)', gridName = '8_atoms', path2DesiredGrid = r'desired_grids')
assembly.close_socket()