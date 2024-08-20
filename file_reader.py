import os
import numpy as np
from read_sxm import output_data_from_sxm
from read_spectra import output_data_spectra_dat
import warnings
import sys
import matplotlib.pyplot as plt

#%%
class MyFiles():
   
    
    def DirFilter(self, path, extension=None, baseName=None, keyStr=None,
                  fileNums=None, fileSize=(None, None)):
        """
        Parameters
        ----------
        path : str
        extension : str, optional
            file type/extension 'InSb(110)_001.XXX'
        baseName : str, optional
            start of file name eg. XXXXXXXXX_001.dat. The default is None.
        keyStr : str, optional
            any string within the file name, eg. InSb(110)_XXX.dat .
            The default is None.
        numRange : tuple, optional
            files named with numbers within the range. Only the two following 
            formats are considered XXX_[number].XXX or XXX_XX0[number].XXX to 
            avoid errors.
            (num range start, num range end). 
            The default is (None, None).
        fileSize : tuple, optional
            (approximate file size, file size tolerance). 
            The default is (None, None).

        Returns
        -------
        filteredFiles : list
            names of files in directory that fulfill the specified conditions.

        """
        
        allFiles = [f.name for f in os.scandir(path) if f.is_file()]
        filteredFiles = []
        
        
        for f in allFiles:
            if extension is not None and not f.endswith(extension):
                continue           
            
            if baseName is not None and not f.startswith(baseName):
                continue
            
            if keyStr is not None and keyStr not in f:
                continue
            
            
            if fileNums is not None:
                
                # format: XXX_[number].XXX 
                numFormat1 = ['_{}.'.format(i) for i in fileNums]
                # format: XXX_XX0[number].XXX
                numFormat2 = ['0{}.'.format(i) for i in fileNums]
                
                if not any(num in f for num in numFormat1 + numFormat2):
                    continue
            
            
            if fileSize != (None, None):
                requiredFileSize = fileSize[0]
                fileSizeTolerance = fileSize[1]
                
                size = os.stat(path + r'\\' + f).st_size
                
                if not np.isclose(size, requiredFileSize,
                                  atol = int(fileSizeTolerance)):
                    continue
            
            filteredFiles.append(f)
         
            
        # test ================================================================
        if fileNums is not None: 
            if len(filteredFiles) != len(fileNums): 
                warnings.warn('Dir filter may be finding more/less files than intended.')
        # test will fail if no file was for a specified number, or, more
        # than one file was found. This might be interntional.
        # =====================================================================
        
        print('Dir filter found ', len(filteredFiles), ' files:')
        print(filteredFiles)
        
        return filteredFiles
    
    
    
    def LatestFile(self, path, extension=None, baseName=None,
                           keyStr=None, fileSize=(None, None)):
        """
        Parameters
        ----------
        path : str
        extension : str, optional
            file type/extension 'InSb(110)_001.XXX'
        baseName : str, optional
            start of file name eg. XXXXXXXXX_001.dat. The default is None.
        keyStr : str, optional
            any string within the file name, eg. InSb(110)_XXX.dat .
            The default is None.
        fileSize : tuple, optional
            (approximate file size, file size tolerance). 
            The default is (None, None).

        Returns
        -------
        latestFile : str
            file name of the latest file in the dir, that fulfills the 
            specified conditions.

        """
        
        filteredFileNames = self.FilterDir(path,  extension, baseName,
                               keyStr, fileSize)
        
        latestTime = 0
        latestFile = None
        
        for f in os.scandir(path): # iterate over all file names in the dir
            if f.name in filteredFileNames:
                time = f.stat().st_mtime_ns # file modification time
                if time > latestTime:
                    latestFile = f.name
                    latestTime = time
                            
        print('File found: ', latestFile)
        
        return latestFile
        
    
    
#%%
class Sxm(output_data_from_sxm):
    
    def __init__(self):
        super().__init__() 
        
        
        
    def ReadSxm(self, path, fileName):
            
        self.get_file(path, fileName)
        
        xCentre, yCentre, xWidth, yWidth, angle = self.position
        
        channels = self.get_channel_names()
        zFwdIdx = channels.index('Z_Fwd')
        zBwdIdx = channels.index('Z_Bwd')
        
        imFwd, _ = self.get_select_image(zFwdIdx)
        imBwd, _ = self.get_select_image(zBwdIdx)
        
        """
        the y-axis origin is flipped 
        when loading an image from a saved sxm file vs. from grabbing the 
        current image displayed on the scan control. To avoid confusion, I
        prefer flipping sxm file images, when loading. 
        """
        imFwd = np.flipud(imFwd)
        imBwd = np.flipud(imBwd)
    
        self.imFwd = imFwd
        self.imBwd = imBwd
        self.xCentre = xCentre
        self.yCentre = yCentre
        self.xWidth = xWidth
        self.yWidth = yWidth
        self.angle = angle
        
        return imFwd
        
        
    def Plot(self, ax=None, cmap='gray', scanDirection = 'Fwd'):
        
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
        
        if scanDirection == 'Fwd': im = self.imFwd
        elif scanDirection == 'Bwd': im = self.imBwd
        else: print('scanDirection takes either Fwd or Bwd as input.')
        
        pxPerLine = np.shape(im)[0]
        
        x = np.linspace(self.xCentre - self.xWidth/2, self.xCentre + self.xWidth/2, num = pxPerLine)
        y = np.linspace(self.yCentre - self.yWidth/2, self.yCentre + self.yWidth/2, num = pxPerLine)

        
        xx, yy = np.meshgrid(x,y) 
        
        if self.angle != 0:
            xx, yy = rotate(xx, yy, xPivot=self.xCentre, yPivot=self.yCentre, rot=self.angle)
        
        if ax is None: fig, ax = plt.subplots()
        ax.pcolor(xx,yy,im, cmap=cmap)
        return ax
    
    
    def FilteredReadSxm(self, path, fileName, squarePx=True, squareWidth=True,
                        NaNVals=False, zeroAngle=False):
        
        try:
            self.ReadSxm(path, fileName)
            
            if self.imFwd.shape[0] != self.imFwd.shape[1]: 
                raise ValueError('sxm px are not square')
            
            if self.xWidth != self.yWidth:
                raise ValueError('sxm is not square')
                
            if np.any(np.isnan(self.imFwd))==True:
                raise ValueError('sxm contains NaN')
                
            if zeroAngle == True and self.angle != 0:
                raise ValueError('scan window angle is non-zero')
            
        except Exception as e: 
            print(e)
            print(fileName, ' is not valid')
            return Exception
        
        else:
            return self.imFwd
            
        

#%%
class Spectrum(output_data_spectra_dat):
    
    def __init__(self):
        super().__init__() 
        
        
        
    def ReadSpectrum(self, path, fileName, channel):
        self.get_file(path, fileName)  
        
        # if channel not in file, print list of possible channels
        if channel not in list(self.df): 
            print('Choice of channel not found in ' + fileName)
            self.show_method_fun()
            sys.exit()
        
        x = self.give_data(0)[0] 
        y = self.give_data(channel)[0]
        
        self.x = x
        self.y = y
        
        """
        note that the spectra metadata was also loaded, eg. self.x_pos,
        self.y_pos... (See output_data_spectra_dat for more info)
        """
        
        return x, y 
    
#%%
import os

def FindLatestFile(path):
    latestTime = 0
    latestFile = None
    # iterate over the files in the directory using os.scandir
    for f in os.scandir(path):
        if f.is_file():
            # get the modification time of the file using entry.stat().st_mtime_ns
            time = f.stat().st_mtime_ns
            if time > latestTime:
                # update the most recent file and its modification time
                latestFile = f.name
                latestTime = time
    return latestFile

sxm= Sxm()
sxm.ReadSxm(r'D:\Results\2024\III-Vs', 'InSb(110)_2802.sxm')
    
