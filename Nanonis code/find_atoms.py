# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 13:34:14 2024

@author: Sofia Alonso Perez
"""

import numpy as np
from skimage.feature import canny
from skimage import filters
from skimage.transform import hough_circle, hough_circle_peaks
import lmfit
import matplotlib.pyplot as plt


class AtomFinder():
    
    def __init__(self, sxm, xCentre, yCentre, width, 
                 angle, atomDiameterGuess=1e-9):
        
        if sxm.shape[0] != sxm.shape[1]: 
            raise ValueError('Atom Finder requires square sxm')
            
        sxm = (sxm - sxm.min()) / (sxm.max() - sxm.min()) # normalised im
        sxm = self.FlattenImTilt(sxm) # flatten image tilt (as done by nanonis)
        
        self.im = np.array(sxm, dtype=np.float32)
        self.xCentre = xCentre
        self.yCentre = yCentre
        self.widthInM = width
        self.widthInPx = sxm.shape[0]
        self.angle = angle
        self.radGuess = self.M2Px(atomDiameterGuess / 2)

        # empty mask and lists
        self.mask = np.zeros((self.widthInPx, self.widthInPx),
                             dtype=np.float32) 
        self.xCentresInPx = []
        self.yCentresInPx = []
        self.radiiInPx = []
        
        self.xCentresInM = []
        self.yCentresInM = []
        self.radiiInM = []
        
        
                  
    def Run(self, sigmaForGaussianFilter=0, confidenceThreshold=0.5):
        # guess atom sites using a circle Hough Transform
        xCentresInPx, yCentresInPx, radiiInPx = self.HoughTransform(sigmaForGaussianFilter, confidenceThreshold)
        
        nAtoms = len(xCentresInPx)
        print(nAtoms, ' atoms found')
        
        xCentresInM = np.zeros(nAtoms)
        yCentresInM = np.zeros(nAtoms)
        radiiInM = np.zeros(nAtoms)
        
        # imporove guess by performing a 2D Gaussian fit around the guessed sites
        # px to m coord tranf
        for i in range(len(radiiInPx)):
            xCentresInPx[i], yCentresInPx[i], radiiInPx[i] = self.CentreNRadOptimiser(xCentresInPx[i], yCentresInPx[i], self.radGuess)
            xCentresInM[i], yCentresInM[i] = self.Px2NanoCoord(xCentresInPx[i], yCentresInPx[i])
            radiiInM[i] = self.Px2M(radiiInPx[i])
        
        # populate mask 
        self.mask = self.UpdateMask(xCentresInPx, yCentresInPx, radiiInPx)
        
        fig, ax = plt.subplots()
        plt.imshow(self.im, origin='lower')
        ax.scatter(xCentresInPx, yCentresInPx)
        
        # populate lists
        self.xCentresInPx = xCentresInPx
        self.yCentresInPx = yCentresInPx
        self.radiiInPx = radiiInPx
        
        self.xCentresInM = xCentresInM
        self.yCentresInM = yCentresInM
        self.radiiInM = radiiInM
        
        fig, ax = plt.subplots()
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
        
        x = np.linspace(self.xCentre - self.widthInM/2, self.xCentre + self.widthInM/2, num = self.widthInPx)
        y = np.linspace(self.yCentre - self.widthInM/2, self.yCentre + self.widthInM/2, num = self.widthInPx)

        
        xx, yy = np.meshgrid(x,y) 
        
        if self.angle != 0:
            xx, yy = rotate(xx, yy, xPivot=self.xCentre, yPivot=self.yCentre, rot=self.angle)
        
        ax.pcolor(xx,yy,self.im)
        ax.scatter(xCentresInM, yCentresInM)
        
        
    
    def M2Px(self, m):

        return int(np.round( (self.widthInPx / self.widthInM) * m))
        


    def Px2M(self, px):

        return  (self.widthInM / self.widthInPx) * px 
        
    
    
    def Px2NanoCoord(self, xPx, yPx):  
        
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
        xPx = xPx - (self.widthInPx / 2)
        yPx = yPx - (self.widthInPx / 2)
        
        # px -> metre lengths
        xM = self.Px2M(xPx)
        yM = self.Px2M(yPx)
        
        # origin trasformation: scan centre -> nanonis origin
        xM = xM + self.xCentre
        yM = yM + self.yCentre
        
        # angle trasformation
        xM, yM = rotate(xM, yM, xPivot=self.xCentre, yPivot=self.yCentre,
                        rot=self.angle)
        return xM, yM
    
    
    
    def HoughTransform(self, sigmaForGaussianFilter=0, confidenceThreshold=0.5):
        
        def EdgeDetection(im):
            """
            edge detection using "Canny" algorithm.
            """
            # If possible, define an optimal upper threshold using "Otsu's" method.
            high_thresh = filters.threshold_otsu(im)
            if high_thresh > 0:
                low_thresh = high_thresh / 2
                im = canny(im, sigma=1.5, high_threshold=high_thresh,
                           low_threshold=low_thresh)  # edge detection
            else:
                im = canny(im, sigma=1.5)  # edge detection
            return im
        
        def RemoveHorizontalEdges(im):
            """
            *** THIS FUNC IS CURRENTLY NOT USED ***
            remove horizontal edges, as they tend to be mistaken for circle edges
            (horizontal edges appear whith tip changes or in unfinished ims)
            remove by discarding all edges detected in a row, where the number of
            edges detected is abnormally large.
            "abnormally large" is defined as:
            3*interquartile range of the number of edges detected per row, among 
            the subset of rows where 1 or more edges were detected
            (its important to only consider the subset where edges were detected 
             to avoid errors when dealing with space images)
            """
            s = np.sum(im.astype(int), axis=1)  # Number of detected edges per row.
            s_non0 = s[s != 0]  # remove zeros
            q1, q3 = np.percentile(s_non0, 25), np.percentile(s_non0, 75)
            iqr = q3 - q1
            outlier_threshold = 3 * iqr  # define a high outlier threshold
            outlier_idx = np.where(s > q3 + outlier_threshold)
            for i in range(len(outlier_idx)):
                # remove all edges within the outlier row
                im[outlier_idx[i], :] = False
            return im
                         
            
        im = np.copy(self.im)
                
        # preprocessing
        im = filters.gaussian(im, sigma=int(sigmaForGaussianFilter))
        im = EdgeDetection(im)
        
        # Hough Transform 
        expectedRad = self.radGuess
        expectedErr = self.radGuess
        possibleRadii = np.arange(1, expectedRad + expectedErr + 1)
        
        hough_res = hough_circle(im, possibleRadii)
        accums, xCentres, yCentres, radii = hough_circle_peaks(hough_res,
                                                   possibleRadii,
                                                   min_xdistance=round(
                                                       expectedRad * 1.5),
                                                   min_ydistance=round(
                                                       expectedRad * 1.5),
                                                   threshold=confidenceThreshold)
        
        return xCentres.tolist(), yCentres.tolist(), radii.tolist()
    
    
  
        
    def CentreNRadOptimiser(self, xCentre, yCentre, rad, fixedRad = False,
                            plotFit = False, 
                            reportFitInfo = False):
        """
        2D gaussian fit (using lmfit library) to datapoints around the estimated 
        atom position. The result of the fit is used to optimise the atom's
        centre and radius guess. 

        """
        
        im = np.copy(self.im)
 
        cx = xCentre
        cy = yCentre
        
        # define area to be fitted to a 2D Gaussian
        
        xmin, xmax = cx - 2 * self.radGuess, cx + 2 * self.radGuess
        ymin, ymax = cy - 2 * self.radGuess, cy + 2 * self.radGuess
    
        if xmin < 0: xmin = 0
        if ymin < 0: ymin = 0
        if xmax > self.widthInPx: xmax = self.widthInPx
        if ymax > self.widthInPx: ymax = self.widthInPx
        
        imAtom = im[ymin:ymax, xmin:xmax]
        
        x = np.arange(start = xmin, stop = xmax, step = 1)
        y = np.arange(start = ymin, stop = ymax, step = 1)
        
        # fit done as shown here: https://lmfit.github.io/lmfit-py/examples/example_two_dimensional_peak.html 
        # Documentation here: https://lmfit.github.io/lmfit-py/builtin_models.html
        x2D, y2D = np.meshgrid(x,y)
        x1D, y1D, z1D = x2D.ravel(), y2D.ravel(), imAtom.ravel()
        
        model = lmfit.models.Gaussian2dModel()
        
        # inbuilt method of guessing the starting values for the fitting params
        params = model.guess(z1D, x1D, y1D) 
        
        # set the fitting params' limits
        if fixedRad == True:
            params['sigmax'].set(value=rad, vary=False)
            params['sigmay'].set(value=rad, vary=False)
        elif fixedRad == False:
            params['sigmax'].set(max=self.M2Px(5e-9))
            params['sigmay'].set(max=self.M2Px(5e-9))
            
        params['centerx'].set(min=(xCentre - self.M2Px(0.5e-9)),
                              max=(xCentre + self.M2Px(0.5e-9)))
        params['centery'].set(min=(yCentre - self.M2Px(0.5e-9)),
                              max=(yCentre + self.M2Px(0.5e-9)))
        
        # make the fit
        result = model.fit(z1D, x=x1D, y=y1D, params=params)
        
        # redefine the atom centre and rad based on the fit's result
        newXCentre = result.params["centerx"].value
        newYCentre = result.params["centery"].value
        # defining the rad as  1 * sigma,
        xSigma = result.params["sigmax"].value
        ySigma = result.params["sigmay"].value
        newRad = (xSigma + ySigma) / 2
        
        
        if reportFitInfo == True:
            lmfit.report_fit(result)
            
        if plotFit == True:
            fig, axs = plt.subplots(2, 2, figsize=(10, 10))
            vmax = np.max(imAtom)
    
            ax = axs[0, 0]
            art = ax.pcolor(x2D, y2D, imAtom, vmin=0, vmax=vmax, shading='auto')
            plt.colorbar(art, ax=ax)
            ax.set_title('Data')
    
            ax = axs[0, 1]
            fit = model.func(x2D, y2D, **result.best_values)
            art = ax.pcolor(x2D, y2D, fit, vmin=0, vmax=vmax, shading='auto')
            plt.colorbar(art, ax=ax)
            ax.set_title('Fit')
    
            ax = axs[1, 0]            
            art = ax.pcolor(x2D, y2D, imAtom - fit, shading='auto')
            plt.colorbar(art, ax=ax)
            ax.set_title('Data - Fit')
    
            ax = axs[1, 1]
            ax.imshow(im, cmap='gray')
            self.DrawAtom(ax, xCentre, yCentre, rad=rad, label='Initial guess')
            self.DrawAtom(ax, newXCentre, newYCentre,
                          rad=newRad, label='Optimised guess')
            plt.show()
    
        return newXCentre, newYCentre, newRad
    
    
    
    
    def UpdateMask(self, xCentresInPx, yCentresInPx, radiiInPx):
        # empty mask
        mask = np.zeros((self.widthInPx, self.widthInPx),
                             dtype=np.float32) 
        
        # populate mask
        for atom in range(len(radiiInPx)):
            for i in range(np.shape(self.mask)[0]):
              for j in range(np.shape(self.mask)[1]):
                if (i - xCentresInPx[atom])**2 + (j - yCentresInPx[atom])**2 <= radiiInPx[atom]**2:
                  mask[j,i] = 1
        
        return mask
            
    

                  
                  
    
    def FlattenImTilt(self, im, kx = 1, ky = 1):
        """
        Image processing. Flattenng tilt as done by nanonis.

        """
        
        #Function allowing multiple polynomial order fits on 2D arrays 
        def polyfit2d(x, y, z, kx, ky, order=None):
            '''
            Two dimensional polynomial fitting by least squares.
            Fits the functional form f(x,y) = z.

            Notes
            -----
            Resultant fit can be plotted with:
            np.polynomial.polynomial.polygrid2d(x, y, soln.reshape((kx+1, ky+1)))

            Parameters
            ----------
            x, y: array-like, 1d
                x and y coordinates.
            z: np.ndarray, 2d
                Surface to fit.
            kx, ky: int, default is 3
                Polynomial order in x and y, respectively.
            order: int or None, default is None
                If None, all coefficients up to maxiumum kx, ky, ie. up to and including x^kx*y^ky, are considered.
                If int, coefficients up to a maximum of kx+ky <= order are considered.

            Returns
            -------
            Return paramters from np.linalg.lstsq.

            soln: np.ndarray
                Array of polynomial coefficients.
            residuals: np.ndarray
            rank: int
            s: np.ndarray

            '''

            # grid coords
            x, y = np.meshgrid(x, y)
            # coefficient array, up to x^kx, y^ky
            coeffs = np.ones((kx+1, ky+1))

            # solve array
            a = np.zeros((coeffs.size, x.size))

            # for each coefficient produce array x^i, y^j
            for index, (j, i) in enumerate(np.ndindex(coeffs.shape)):
                # do not include powers greater than order
                if order is not None and i + j > order:
                    arr = np.zeros_like(x)
                else:
                    arr = coeffs[i, j] * x**i * y**j
                a[index] = arr.ravel()

            # do leastsq fitting and return leastsq result
            return np.linalg.lstsq(a.T, np.ravel(z), rcond=None)

        
        x_pixels = np.arange(np.size(im,1))
        y_pixels = np.arange(np.size(im,0))

        #Find the fit!
        fit_data = polyfit2d(x_pixels, y_pixels, im, kx, ky)
        soln = fit_data[0]
        
        # Create the background
        fitted_surf = np.polynomial.polynomial.polygrid2d(x_pixels, y_pixels, soln.reshape((kx+1,ky+1)))
        
        #flatten the data using the background
        flattened_data = im - fitted_surf

        return flattened_data
    
    
