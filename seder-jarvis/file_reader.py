import os
import numpy as np
from read_sxm import output_data_from_sxm
from read_spectra import output_data_spectra_dat
import warnings
import sys
import matplotlib.pyplot as plt
import warnings

#%%
class MyFiles():

    def DirFilter(self, path, extension=None, baseName=None, keyStr=None,
                  fileNums=None, fileSize=(None, None), fileNameLength=None):
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


            if fileNameLength is not None and len(f) != fileNameLength:
                continue


            filteredFiles.append(f)

            
        # test ================================================================
        if fileNums is not None: 
            if len(filteredFiles) != len(fileNums): 
                warnings.warn('Dir filter may be finding more/less files than intended.')
        # test will fail if no file was for a specified number, or, more
        # than one file was found. This might be intentional.
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
        """
        self.xCentre = xCentre
        self.yCentre = yCentre
        self.xWidth = xWidth
        self.yWidth = yWidth
        self.angle = angle
        """
        
        return imFwd
        
        
    def Plot(self, ax=None, cmap='gray', scanDirection = 'Fwd', alpha=1):
        
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
        ax.pcolor(xx,yy,im, cmap=cmap, alpha=alpha)
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

    def ReadSpectrum(self, path, fileName, yChannel, xChannel=0):

        def CheckChannelExists(channel):
            if channel not in list(self.df):
                print('Choice of channel not found in ' + fileName)
                self.show_method_fun()
                print('Index')
                sys.exit()


        self.get_file(path, fileName)

        if type(yChannel) == str: CheckChannelExists(yChannel)
        y = self.give_data(yChannel)[0]

        if xChannel == 'Index':
            x = list(range(len(y)))
            """
            Can be later converted to time, if we made note of sampling freq. 
            TCP receiver sampling is limited to 20 kHz. So, if measurement made using data logger 
            through Matt's python_interface_nanonis.py, default is 20 kHz. 
            Note: we may be able to play with TCP receiver to lower the 20kHz limit, if needed.
            """
        else:
            if type(xChannel) == str: CheckChannelExists(xChannel)
            x = self.give_data(xChannel)[0]

        self.x = x
        self.y = y
        
        """
        note that the spectra metadata was also loaded, eg. self.x_pos,
        self.y_pos... (See output_data_spectra_dat for more info)
        """

        return x, y

"""
# ===============================================================================================
# Example Use
# ===============================================================================================

path = r'data/manipulation_force_data'

def BeforeAfterEventPlot(path, fileNameBefore, fileNameAfter):
    plt.style.use("dark_background")
    fig, ax = plt.subplots(1, 2, sharey=True, sharex=True, figsize=(15, 7))

    before = Sxm()
    before.ReadSxm(path, fileNameBefore)
    before.Plot(ax=ax[0])

    after = Sxm()
    after.ReadSxm(path, fileNameAfter)
    after.Plot(ax=ax[1])

    #ax[0].scatter(after.xCentre, after.yCentre, color='red')
    #ax[1].scatter(after.xCentre, after.yCentre, color='red', label='atom tracked apex, before')

    a = 0.648e-9
    b = 0.458e-9

    ymin, ymax = ax[1].get_ylim()
    ysitespos = np.arange(after.yCentre + b, ymax, b)
    ysitesneg = np.arange(after.yCentre, ymin, -b)
    ysites = np.concatenate((ysitespos, ysitesneg))
    ax[1].set_yticks(ysites)

    xmin, xmax = ax[1].get_xlim()
    xsitespos = np.arange(after.xCentre + a, xmax, a)
    xsitesneg = np.arange(after.xCentre, xmin, -a)
    xsites = np.concatenate((xsitespos, xsitesneg))
    ax[1].set_xticks(xsites)

    fig.legend()

    ax[0].grid(alpha=0.3)
    ax[1].grid(alpha=0.3)

    ax[0].set_aspect('equal')
    ax[1].set_aspect('equal')
    ax[0].set_title('before')
    ax[1].set_title('after')
    ax[0].set_xticklabels([])
    ax[0].set_yticklabels([])
    ax[1].set_xticklabels([])
    ax[0].set_yticklabels([])
    plt.subplots_adjust(wspace=0, hspace=0)


BeforeAfterEventPlot(path, 'InSb_0080.sxm', 'InSb_0082.sxm')
#BeforeAfterEventPlot(path, 'InSb_0967.sxm', 'InSb_0968.sxm')


plt.show()

"""