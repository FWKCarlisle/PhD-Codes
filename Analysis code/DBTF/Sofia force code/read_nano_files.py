import os
import numpy as np
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
import warnings
import re
from typing import List, Optional, Tuple

from read_sxm import output_data_from_sxm
from read_spectra import output_data_spectra_dat
from coord_trans import NanoCoordTrans
from itertools import compress

mpl.use('Qt5Agg')  # Returns interactive plots which I find useful when testing


#%%
class MyFiles():

    def __init__(self, path: str):
        # Ensure the path exists and is a directory
        if not os.path.isdir(path):
            raise NotADirectoryError(f"The provided path '{path}' is not a valid directory.")

        self.path = path



    def dir_filter(self,
                  extension: Optional[str] = None,
                  base_name: Optional[str] = None,
                  key_str: Optional[str] = None,
                  file_nums: Optional[str] = None,
                  file_size: Optional[Tuple[int, int]] = (None, None),
                  file_name_length: Optional[int] = None) -> List[str]:
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
        all_files = [f.name for f in os.scandir(self.path) if f.is_file()]
        filtered_files = []

        for f in all_files:

            name, ext = os.path.splitext(f)

            if extension is not None and not ext:
                continue           
            
            if base_name is not None and not f.startswith(base_name):
                continue
            
            if key_str is not None and key_str.lower() not in name.lower():
                continue
            
            
            if file_nums is not None:
                matches = re.findall(r'\d+', f)
                if not matches:
                    continue
                digit = matches[-1]
                if str(int(digit)) not in [str(int(num)) for num in file_nums]:
                    continue
            
            
            if file_size != (None, None):
                requiredFileSize = file_size[0]
                fileSizeTolerance = file_size[1]
                
                size = os.stat(os.path.join(self.path, f)).st_size
                
                if not np.isclose(size, requiredFileSize,
                                  atol = int(fileSizeTolerance)):
                    continue


            if file_name_length is not None and len(f) != file_name_length:
                continue


            filtered_files.append(f)


        print('dir_filter found ', len(filtered_files), ' files: \n', filtered_files)
        
        return filtered_files
    
    
    
    def latest_file(self, **kwargs):
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
        filtered_files = self.dir_filter(**kwargs)
        
        latest_time = 0
        latest_file = None
        
        for f in os.scandir(self.path): # iterate over all file names in the dir
            if f.name in filtered_files:
                time = f.stat().st_mtime_ns # file modification time
                if time > latest_time:
                    latest_file = f.name
                    latest_time = time
                            
        print('latest_file found: ', latest_file)
        
        return latest_file



    def are_file_names_ordered(self, file_names):
        digits = []
        for f in file_names:
            digit_groups =  re.findall(r'\d+', f) # scan through string looking for digit groups
            if not digit_groups:
                raise ValueError(f"Cannot check if are_file_names_ordered bc file '{f}' does not contain any digits.")
            # the file base name may have numerous digit groups eg. XXX_[other non-important digits]_00001.dat
            digits.append(int(digit_groups[-1])) # we only care about the *last* digit group

        # Check that digits are in ascending order *and* no digits are missing
        order_bool = [earlier + 1 == later for earlier, later in zip(digits[:-1], digits[1:])]
        if not all(order_bool):
            print('non ordered file names: ', list(compress(file_names[1:], np.logical_not(order_bool))), list(compress(file_names[:-1],  np.logical_not(order_bool))))
        return all(earlier + 1 == later for earlier, later in zip(digits[:-1], digits[1:]))



    def find_one_file(self, **kwargs) -> str:
        filtered_files = self.dir_filter(**kwargs)

        if not filtered_files:
            raise FileNotFoundError("No file found matching the specified conditions.")

        elif len(filtered_files) > 1:
            raise FileNotFoundError("Multiple files found to match the specified conditions.")

        found_file = filtered_files[0]

        print("find_one_file found:", found_file)

        return found_file



#%%
class Sxm(output_data_from_sxm):
    
    def __init__(self, path, file_name):
        super().__init__()
        self.get_file(path, file_name)


        
    def read_channel(self, channel='Z_Fwd'):

        if not self.does_channel_exist(channel):
            sys.exit()

        if type(channel) == str:
            channels = self.get_channel_names()
            channel_idx = channels.index(channel)

        elif type(channel) == int:
            channel_idx = channel

        else:
            raise ValueError('channel must be str or int.')
        
        im, _ = self.get_select_image(channel_idx)
        
        """
        the y-axis origin is flipped 
        when loading an image from a saved sxm file vs. from grabbing the 
        current image displayed on the scan control. To avoid confusion, I
        prefer flipping sxm file images, when loading. This matches how 
        the image is displayed on nanonis.
        """
        im = np.flipud(im)
    
        self.im = im
        
        return im



    def does_channel_exist(self, channel):
        exist = False

        channel_names = self.get_channel_names()

        if type(channel) == str:
            if channel in channel_names:
                exist = True

        if type(channel) == int:
            if channel < len(channel_names):
                warnings.warn('define sxm channels as str (not int) to avoid loading an unwanted channel when the recorded channels change.')
                exist = True

        if exist == False:
            print('Choice of channel not found')
            print('Available channels: ', channel_names)

        return exist

        
    def plot_sxm(self, fig= None, ax=None, cmap='gray', **kwargs):

        x_px_per_line = np.shape(self.im)[1]
        y_px_per_line = np.shape(self.im)[0]

        x_px_width = self.x_width / x_px_per_line
        y_px_width = self.y_width / y_px_per_line
        
        x = np.linspace(self.x_centre - (self.x_width/2 - x_px_width/2),
                        self.x_centre + (self.x_width/2 - x_px_width/2), num = x_px_per_line)
        y = np.linspace(self.y_centre - (self.y_width/2 - y_px_width/2),
                        self.y_centre + (self.y_width/2 - y_px_width/2), num = y_px_per_line)

        xx, yy = np.meshgrid(x,y)

        if self.angle != 0:
            xx, yy = NanoCoordTrans(xx, yy).rot(x_pivot=self.x_centre, y_pivot=self.y_centre, rot=self.angle)

        # plot
        if ax is None or fig is None:
            fig, ax = plt.subplots()

        ax.pcolor(xx, yy, self.im, cmap=cmap, **kwargs)

        ax.set_aspect(self.y_width / self.x_width)

        return fig, ax
    
    
    def filtered_read_sxm(self, path, fileName, squarePx=True, squareWidth=True,
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






# ===============================================================================================
# Example Use
# ===============================================================================================
"""
path = r'data_simulator'
fileName = 'unnamed0006.sxm'
im = Sxm(path=path, file_name=fileName)
print(im.does_channel_exist(0))
im.read_channel(channel=0)
fig, ax = im.plot_sxm()

# if the x and y channels were saved, you can check how the image px are formed compared to the tip measuring pos
for c in ['X_Fwd', 'Y_Fwd', 'X_Bwd', 'Y_Bwd']:
    if im.does_channel_exist(c):
    xx_Fwd = im.read_sxm(path=path, file_name=fileName, channel_name='X_wd')
    yy_Fwd = im.read_sxm(path=path, file_name=fileName, channel_name='Y_Fwd')

    xx_Bwd = im.read_sxm(path=path, file_name=fileName, channel_name='X_Bwd')
    yy_Bwd = im.read_sxm(path=path, file_name=fileName, channel_name='Y_Bwd')

    ax.scatter(xx_Fwd, yy_Fwd, marker='.', color='tab:blue')
    ax.scatter(xx_Bwd, yy_Bwd, marker='.', color='tab:red')

plt.show()






def BeforeAfterEventPlot(path, fileNameBefore, fileNameAfter):
    plt.style.use("dark_background")
    fig, ax = plt.subplots(1, 2, sharey=True, sharex=True, figsize=(15, 7))

    before = Sxm()
    im = before.ReadSxm(path, fileNameBefore) # outputs image as 2d numpy array 
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

plt.show()

"""