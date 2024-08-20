# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 12:50:01 2024

@author: Sofia Alonso Perez
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time 
import numpy as np
import sys
import traceback

import python_interface_nanonis_v7 as python_nano
from find_atoms import AtomFinder
from path_finding import AStarSquareGrid


class AutoAssembly(python_nano.python_Nanonis_TCP):
    
    
    def __init__(self, substrate, gridName):
        
        super().__init__() 
        
        try: 
            self.ReadConfig()

        except Exception: 
            traceback.print_exc()
            self.close_socket()
            sys.exit()
            
        # load targets & translate    
        xt, yt = self.LoadTargets(substrate, gridName)
        self.xTargets, self.yTargets = self.TranslateTargets(xt, yt)
        
        self.count = 0
        while True:
            self.AssemblyAttempt()
            self.count += 1
            
        
        
    def AssemblyAttempt(self):
                
            
        try:
            # start scan
            self.Scan_Action('start', 'up') 
            # while scanning -> wait
            time.sleep(1)
            self.Scan_Check_Hold()
            time.sleep(4)
            
            # load scan 
            _, scan = self.Scan_FrameDataGrab(30) 
            
            self.ReadConfig()
            
            # find atoms
            atoms = AtomFinder(scan, self.xCentre, self.yCentre, self.width, 
                         self.angle, self.atomD)
            atoms.Run(self.atomFinderSigma, self.atomFinderConf)                              
            xAtoms, yAtoms = atoms.xCentresInM, atoms.yCentresInM
            
            # check availability 
            xFreeAtoms, yFreeAtoms, xFreeTargets, yFreeTargets = self.FreeAtomsNTargets(xAtoms,
                                                                                        yAtoms,
                                                                                        self.xTargets,
                                                                                        self.yTargets)
            
            if len(xFreeAtoms) == 0: raise ValueError('no free atoms left')
            if len(xFreeTargets) == 0: raise ValueError('no free targets left')
            
            # find best path
            xPath, yPath = self.BestPath(xAtoms, yAtoms, xFreeAtoms, yFreeAtoms,
                         xFreeTargets, yFreeTargets)
        
            # plot status 
            self.PlotStatus(scan, xAtoms, yAtoms, xFreeAtoms, yFreeAtoms,
                           xFreeTargets, yFreeTargets, xPath, yPath)
            
            # move the tip to the atom to manipulate ie. the first path point
            self.XY_tip_set_pos(xPath[0],yPath[0])
            # wait for tip to arrive
            self.WaitWhileTipMoves(xPath[0],yPath[0])
            
            # atom tracking to find the atom's centre
            print('atom tracking')
            self.AtomTrack_CtrlSet(1, 1) # initialise atom traking         
            time.sleep(20)
            self.AtomTrack_CtrlSet(0, 0) # end atom tracking
            
            # initiate manipulation parameters
            # change bias slowly... (to prevent a 'bias pulse')
            print('manipulation mode. attempt ', self.count)
            self.Bias_set_slow(self.manipV, 
                               time_to_take_seconds=self.sweepTime,
                               num_of_steps=self.sweepSteps)
            # change setpoint current slowly
            self.Current_SetPoint_set_slow(self.manipI, 
                               time_to_take_seconds=self.sweepTime,
                               num_of_steps=self.sweepSteps)
            
            # move the tip (with, hopefully, the atom)
            for x, y in zip(xPath, yPath):
                self.XY_tip_set_pos(x, y)
                # wait for tip to arrive
                self.WaitWhileTipMoves(x, y)
            
            print('scan mode')
            # initiate scanning parameters
            # change setpoint current slowly
            self.Current_SetPoint_set_slow(self.scanI, 
                               time_to_take_seconds=self.sweepTime,
                               num_of_steps=self.sweepSteps)
            # change bias slowly... (to prevent a 'bias pulse')
            self.Bias_set_slow(self.scanV, 
                               time_to_take_seconds=self.sweepTime,
                               num_of_steps=self.sweepSteps)
            

        except Exception: 
            traceback.print_exc()
            self.Quit()
        
        
        
    def Quit(self):
        """
        Safe stop to script, by returning to the non-perturbative scanning 
        I & V vals and closing the TCP socket. 

        """
        print('Finished! Safely quitting...')
        
        # initiate scanning parameters
        
        # # change bias slowly... (to prevent a 'bias pulse')
        # self.Bias_set_slow(self.scanV, 
        #                    time_to_take_seconds=self.sweepTime,
        #                    num_of_steps=self.sweepSteps)
       
        # # change setpoint current slowly
        # self.Current_SetPoint_set_slow(self.scanI, 
        #                    time_to_take_seconds=self.sweepTime,
        #                    num_of_steps=self.sweepSteps)
        
        self.close_socket()
        sys.exit()
        


    def ReadConfig(self):
        """
        Many params are needed to run an automated assembly. Some we need to 
        read from nanonis' (scan frame & z channel num). The rest are defined 
        on the cofiguration file, auto_assembly_config.txt.
        
        This function is called before all manipulation attempts, so these
        params can be redefined during runtime. For params in the config text 
        file, change and *and* save. 

        """
        
        # Read nanonis' config params needed:
        
        self.xCentre, self.yCentre, self.width, length, self.angle = self.Scan_FrameGet()
        _, recordedChanels, self.PxPerWidth, PxPerLength = self.Scan_BufferGet()  
        
        # print(recordedChanels)
        # for chanelNum in recordedChanels: # find the Z (m) channel number
        #     chanName, _ = self.Scan_FrameDataGrab(chanelNum) 
        #     print(chanName)
        #     if chanName == 'Z (m)':
        # self.zChanNum = int(30.0)
                
        if (self.width != length
            or self.PxPerWidth != PxPerLength): # check scan frame config is square
            raise ValueError('Scan frame must be square')


        # Read config file:
            
        config = {}
        
        with open('auto_assembly_config.txt', 'r') as file:
            for line in file:
                if '=' in line:
                    key, value = line.strip().split(' = ')
                    config[key] = float(value)
                    
                    
                    
        # check params are safe
        if config['scanningI'] == 0: ValueError('unsafe config: scanI == 0')
        if config['manipulationI'] == 0: ValueError('unsafe config: manipI == 0')
        if np.abs(config['scanningI']) > 0.5e-9: ValueError('unsafe config: np.abs(self.scanI) > 0.5e-9')
        if np.abs(config['manipulationI']) > 2e-9: ValueError('unsafe config: np.abs(self.manipI) > 2e-9')
        if np.abs(config['scanningV']) > 5: ValueError('unsafe config: np.abs(self.scanV) > 5')
        if np.abs(config['manipulationV']) > 5: ValueError('unsafe config: np.abs(self.manipV) > 5') 
        if config['scanningI'] * config['manipulationI'] <= 0: ValueError('unsafe config: self.scanI * self.manipI <= 0')
        if config['scanningV'] * config['manipulationV'] <= 0: ValueError('unsafe config: self.scanV * self.manipV <= 0')        


        self.manipV = config['manipulationV']
        self.manipI = config['manipulationI']
        self.sweepTime = int(config['IVsweepTime'])
        self.sweepSteps = int(config['IVsweepSteps'])
        self.scanI = config['scanningI']
        self.scanV = config['scanningV']
        self.errTol = config['errorTolerance']
        self.atomD = config['atomDiameter']
        self.dangerD = config['dangerDiameter']
        self.dangerIndent = config['dangerPerimeterIndent']
        self.atomFinderSigma = config['atomFinderSigma']
        self.atomFinderConf = config['atomFinderConf']
        
        print('Read config file: ', config)
        
       
        
        
        
        
    def WaitWhileTipMoves(self, xEnd, yEnd):
        print('tip moving. waiting...')
        
        while True:
            x, y = self.XY_tip_pos()
            if (abs(x - xEnd) > self.width/self.PxPerWidth
                and abs(x - yEnd) > self.width/self.PxPerWidth):
                time.sleep(0.5)
            break
        print('tip move finished')
    

    def LoadTargets(self, substrate, gridName, path = r'desired_grids'):
        """
        load an arr of target atom positions (previously defined) and translate 
        it to the current scan centre.
        Target arrs are defined using make_target_grid.py
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
            The default is r'desired_grids'.

        Returns
        -------
        xTarget : 1D numpy array
            x coordinates of target atom positions.
        yTarget : 1D numpy array
            y coordinates of target atom positions.

        """
        
        xTargets = np.load(path + r'\xdg_' + substrate + '_' + gridName + '.npy')
        yTargets = np.load(path + r'\ydg_' + substrate + '_' + gridName + '.npy')

        # translate grid: nanonis origin -> scan centre
        xTargets = xTargets + self.xCentre
        yTargets = yTargets + self.yCentre
        
        # rotate to self.angle
        if self.angle != 0:
            xTargets, yTargets = self.Rotate(xTargets, yTargets, xPivot=self.xCentre,
                               yPivot=self.yCentre, rot= self.angle) 
            
        return xTargets, yTargets
    
    
    
    def TranslateTargets(self, xTargets, yTargets):
        """
        This method allows the user to traslate the whole grid of target atom positions
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
            xTargets = xTargets - self.xCentre
            yTargets = yTargets - self.yCentre
            
            # translate grid origin: scan centre -> first grid point
            xTargets = xTargets - xTargets[0]
            yTargets = yTargets - yTargets[0]
            
            # translate grid: nanonis origin -> tip position
            xTip, yTip = self.XY_tip_pos()
            xTargets = xTargets + xTip
            yTargets = yTargets + yTip
            
        return xTargets, yTargets 
    
    
    
    def Rotate(self, x, y, xPivot=0, yPivot=0, rot=0):

        rot = -np.deg2rad(rot) # -ve bc. this is as done by nanonis
        
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

    
    
    def FreeAtomsNTargets(self, xAtoms, yAtoms, xTargets, yTargets):
        """
        Discart atoms and targets outside of the manipulation limits within the 
        scan frame, or within the tolerance error of eachother.
        
        Returns
        -------
        freeAtomIdx: list
            index of available atoms for manipulation in self.xAtoms and self.yAtoms
        freeTargetIdx: list
            index of available targets in self.xTarget and self.yTarget

        """
        
        def InRange(x, y):
            idx = []
            
            # rotate by -self.angle
            if self.angle != 0:
                x, y = self.Rotate(x, y, xPivot=self.xCentre,
                                   yPivot=self.yCentre, rot= -self.angle)
                
            # set the range limits
            xRangeMin = self.xCentre - self.width/2 + self.dangerIndent
            xRangeMax = self.xCentre + self.width/2 - self.dangerIndent
            yRangeMin = self.yCentre - self.width/2 + self.dangerIndent
            yRangeMax = self.yCentre + self.width/2 - self.dangerIndent
            
            for i in range(len(x)):                
                # if within range
                if (x[i] >= xRangeMin
                and x[i] <= xRangeMax 
                and y[i] >= yRangeMin 
                and y[i] <= yRangeMax):
                    idx.append(i)
                    
            # rotate back to self.angle
            if self.angle != 0:
                x, y = self.Rotate(x, y, xPivot=self.xCentre,
                                   yPivot=self.yCentre, rot= self.angle)      

            x, y = x[idx], y[idx]
            
            return x, y
        
        
        
        def Unmatched(x, y, xRef, yRef):
            idx = []
            
            for i in range(len(x)):
                xDelta = xRef - x[i]
                yDelta = yRef - y[i]
                # array of distance from the (atomIndx)th atom to each of the desired positions. Size of this array = size of desired grid
                rDelta = np.sqrt(xDelta**2 + yDelta**2 )
                
                if np.all(rDelta >= self.errTol): 
                    idx.append(i)
            
            x, y = x[idx], y[idx]
            
            return x, y
        
        
        xInRangeAtoms, yInRangeAtoms = InRange(xAtoms, yAtoms)
        xInRangeTargets, yInRangeTargets = InRange(xTargets, yTargets)
        
        xFreeAtoms, yFreeAtoms = Unmatched(xInRangeAtoms, yInRangeAtoms,
                                           xInRangeTargets, yInRangeTargets)
        
        xFreeTargets, yFreeTargets = Unmatched(xInRangeTargets, yInRangeTargets,
                                               xInRangeAtoms, yInRangeAtoms)
        
        print('out of ', len(xAtoms), 'atoms', len(xFreeAtoms), 'are available.')
        print('out of ', len(xTargets), 'targets', len(xFreeTargets), 'are available.')
        
        return xFreeAtoms, yFreeAtoms, xFreeTargets, yFreeTargets
 
    
        
    def BestPath(self, xAtoms, yAtoms, xFreeAtoms, yFreeAtoms,
                 xFreeTargets, yFreeTargets):
        
         
        """
        Due to the input format required by the path finder algorithm, 
        two coord tranfs are made, the path is found, the two coord tranfs 
        are reversed. The method is divided by sections, depending on the 
        coordinate space we're dealing with. This will hopefully make it clear 
        enough. 
        
        Input format: a flat space of weghted values, the start and end positions
        in this space, given as x and y indices.
        
        Rewrite this more clearly in future. 
        """
        
        bestCost = float('inf')
        bestPath = None
        # bestAtom = None
        # bestTarget = None
        
        
        # deg = self.angle -> deg = 0 space ===================================
        if self.angle != 0:
            xAtoms, yAtoms = self.Rotate(xAtoms, yAtoms, xPivot=self.xCentre,
                               yPivot=self.yCentre, rot= -self.angle)
            
            xFreeAtoms, yFreeAtoms = self.Rotate(xFreeAtoms, yFreeAtoms, xPivot=self.xCentre,
                               yPivot=self.yCentre, rot= -self.angle)
            
            xFreeTargets, yFreeTargets = self.Rotate(xFreeTargets, yFreeTargets, xPivot=self.xCentre,
                               yPivot=self.yCentre, rot= -self.angle)
        
        # Define the path space 
        # limits
        xRangeMin = self.xCentre - self.width/2 + self.dangerIndent
        xRangeMax = self.xCentre + self.width/2 - self.dangerIndent
        yRangeMin = self.yCentre - self.width/2 + self.dangerIndent
        yRangeMax = self.yCentre + self.width/2 - self.dangerIndent
        
        # num of ticks in space is set to the num of px in scan 
        # this was an arbritary choice, can be changed
        x = np.linspace(xRangeMin, xRangeMax, num=self.PxPerWidth)
        y = np.linspace(yRangeMin, yRangeMax, num=self.PxPerWidth)

        
        # iterate over free atoms & targets, to find the best possible path 
        for xStart, yStart in zip(xFreeAtoms, yFreeAtoms):

            # define weighted space
            # allowed points = 0; forbidden points = 1
            # start with a grid of all allowed points
            weigtedSpace = np.zeros((self.PxPerWidth, self.PxPerWidth))
            
            # mix with a grid of all non-allowed points, where condition is met
            # condition = around free *and* occuped, except for around start atom
            ones = weigtedSpace + 1
            
            # need to use polar coords to define the condition
            for xA, yA in zip(xAtoms, yAtoms): 
                if xA != xStart and yA != yStart:
                    # move the origin to the atom centre
                    x2D, y2D = np.meshgrid(x,y)
                    xDash = x2D - xA
                    yDash = y2D - yA
                    # forbidden region is simply anything with r <= danger radius
                    rDash = np.sqrt(xDash**2 + yDash**2)
                    
                    weigtedSpace = np.where(rDash <= self.dangerD/2, ones, weigtedSpace)
                        
        # metres -> idx space =================================================
            # find the index of the nearest value in x & y
            start = (np.argmin(abs(x - xStart)), np.argmin(abs(y - yStart)))
            
            for xEnd, yEnd in zip(xFreeTargets, yFreeTargets):

                end = (np.argmin(abs(x - xEnd)), np.argmin(abs(y - yEnd)))

                # run path finder
                xPath, yPath = AStarSquareGrid(weigtedSpace).find_path(start, end)
                
                if xPath is not None and len(xPath) < bestCost:
                    bestCost = len(xPath)
                    bestPath = (xPath, yPath)
                    
        # idx -> metres space =================================================
                    # bestAtom = (xStart, yStart)
                    # bestTarget = (xEnd, yEnd)
                    
        if bestPath[0] is None: raise ValueError('No path found')
        else:
            
            # transform to metres by slicing off x & y space the corresponding coord
            xPath = x[bestPath[0]] 
            yPath = y[bestPath[1]]
            
            # add the start and end points, as b4 we approximated to closest px
            # xStart = x[bestAtom[0]] 
            # yStart = y[bestAtom[1]]
            # xPath = np.insert(xPath, 0, xStart)
            # yPath = np.insert(yPath, 0, yStart)
            
            # xEnd = x[bestTarget[0]] 
            # yEnd = y[bestTarget[0]] 
            # xPath = np.insert(xPath, -1, xEnd)
            # yPath = np.insert(yPath, -1, yEnd)         
            
        # deg = 0 -> deg = self.angle space ===================================
        if self.angle != 0:
            xPath, yPath = self.Rotate(xPath, yPath, xPivot=self.xCentre,
                               yPivot=self.yCentre, rot= self.angle)
         
        return xPath, yPath
    
    
    
    def PlotStatus(self, scan, xAtoms, yAtoms, xFreeAtoms, yFreeAtoms,
                   xFreeTargets, yFreeTargets, xPath, yPath):
         
        if self.angle != 0:
            xAtoms, yAtoms = self.Rotate(xAtoms, yAtoms, xPivot=self.xCentre,
                               yPivot=self.yCentre, rot= -self.angle)
            
            xFreeAtoms, yFreeAtoms = self.Rotate(xFreeAtoms, yFreeAtoms, xPivot=self.xCentre,
                               yPivot=self.yCentre, rot= -self.angle)
            
            xFreeTargets, yFreeTargets = self.Rotate(xFreeTargets, yFreeTargets, xPivot=self.xCentre,
                               yPivot=self.yCentre, rot= -self.angle)
            
            xPath, yPath = self.Rotate(xPath, yPath, xPivot=self.xCentre,
                               yPivot=self.yCentre, rot= -self.angle)
            
        fig, ax = plt.subplots(figsize=(8, 5))
        
        # scan 
        ax.imshow(scan, cmap='gray', extent=[self.xCentre - self.width/2,
                                    self.xCentre + self.width/2,
                                    self.yCentre - self.width/2,
                                    self.yCentre + self.width/2], origin='lower')

        ax.plot(xPath, yPath, label = 'Best path')
        ax.scatter(xFreeTargets, yFreeTargets, c = 'green', label = 'Free targets')
        ax.scatter(xFreeAtoms, yFreeAtoms, c = 'red', label = 'Free atoms')
        
        for x, y in zip(xAtoms, yAtoms):
            # First define a circle with centre (0, 0) using polar coords.
            # Ie., theta ranging from 0 to 2pi and r = rad is constant.
            r = self.dangerD/2
            theta1 = np.linspace(0, np.pi, 100)
            theta2 = np.linspace(np.pi, 2*np.pi, 100)
            # Transform from polar to cartesian coords
            # Move the circle centre from (0, 0) to its real centre
            x = (r*np.cos(theta1)) + x
            y1 = (r*np.sin(theta1)) + y
            y2 = (r*np.sin(theta2)) + y
            ax.fill_between(x, y1, y2, color = 'red', alpha=0.1)
            
        # ax.legend(bbox_to_anchor=(1, 1))
        ax.set_axis_off()
        plt.savefig(r'auto_assembly_plots\attempt_' + str(self.count))
        
        
        
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
            
        time.sleep(3)
        print('Completed slow current setpoint move - Current SetPoint = ' + str(self.Current_GetSetPoint())+' A')

                    


                    
assembly = AutoAssembly('InSb(110)', gridName = '6_atoms')    
        