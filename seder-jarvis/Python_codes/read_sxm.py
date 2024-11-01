# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 16:03:33 2023

@author: physicsuser
"""

import numpy as np
from numpy.polynomial import polynomial as P
import re
import io
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import struct
import datetime
import os
import glob

class output_data_from_sxm():
    
    def print_value(self):
        print(self.filename_path)
        
    def convert_time_to_epoch(self, time):
        dt = datetime.datetime.strptime(str(time[0]), '%d.%m.%Y %H:%M:%S')
        return dt.timestamp()
    
    def get_channel_names(self):
        
        #Get the correct name
        direction = []
        channel = []
        count = 0
        for i in range(len(self.df.Direction)):
            if self.df.Direction.iloc[i] == 'both':
                direction.append('Fwd')
                direction.append('Bwd')
                channel.append(self.df.Name.iloc[i])
                channel.append(self.df.Name.iloc[i])
            else: 
                TypeError('Not yet coded this - don\'t know what one direction only looks like')
        names = []
        for i in range(len(channel)):
            ###### Need to add for only one direction!!   
            name = channel[i] +'_'+direction[i]
            names.append(name)
        
        return names
    
    def get_select_image(self,selection):
        #Once images have been got, how can we select the image required?
        image_data_temp = self.image[int(0+self.ypixels*selection):int(self.ypixels+self.ypixels*selection),:]

        #Get the correct name
        direction = []
        channel = []
        count = 0
        for i in range(len(self.df.Direction)):
            if self.df.Direction.iloc[i] == 'both':
                direction.append('Fwd')
                direction.append('Bwd')
                channel.append(self.df.Name.iloc[i])
                channel.append(self.df.Name.iloc[i])
            else: 
                TypeError('Not yet coded this - don\'t know what one direction only looks like')
                
        ###### Need to add for only one direction!!   
        name = channel[selection] +'_'+direction[selection]
     
        #We must ensure it is the correct way around! 
        #print('This direction part needs to be fixed')
        if self.scan_direction == 'up':
            if channel[selection] == 'Z':
                image_data_temp = image_data_temp
            
            if direction[selection] == 'Fwd':
                image_data_temp = image_data_temp[::-1]
                
            if direction[selection] == 'Bwd':
                image_data_temp = np.rot90(np.transpose(image_data_temp[::-1]),-1)
        
        
        if self.scan_direction == 'down':
            if channel[selection] == 'Z':
                image_data_temp = image_data_temp
            
            if direction[selection] == 'Fwd':
                image_data_temp = image_data_temp
                
            if direction[selection] == 'Bwd':
                image_data_temp = np.rot90(image_data_temp[::-1],2)
        
        
                
                
        return image_data_temp, name
    
    def get_image(self):
        
        #if image_number = 0, then it is the first image in the file
        #if Fwd_or_Bwd = 0 then forward
        #if Fwd_or_Bwd = 0 then backwards
        
        #We have get the image data from self.read_sxm_image(), let's extract here
        total_pixels_all_images = int(self.xpixels*self.xpixels*self.total_image_num)
    
        self.image = np.zeros((total_pixels_all_images))
        
        for i in range(total_pixels_all_images):
            self.image[i] = struct.unpack('>f',self.all_non_sorted_image_data[0+(i*4) : 4+(i*4) ])[0]
        
        #image = image.reshape((int(self.xpixels), int(self.ypixels)))
        self.image = self.image.reshape((int(self.xpixels*len(self.df)*2), int(self.ypixels)))
        
        
        
        
    def read_sxm_image(self):
        
        f = open(self.filename_path, 'rb')
        f.seek(0)
        data = f.read()
        f.close()
        
        #Split into after and before the data part
        data_split = data.split(b':SCANIT_END:')
        
        #For the image data
        image_data = data_split[1]
        
        #For the meta data - split one final time to get the data on channels saved=
        data_split_meta = data_split[0].split(b':DATA_INFO:')
        
        #Channels
        data_channels = data_split_meta[1][2:-1].decode("utf-8") 
        
        #Meta data of the file
        data_meta_data = data_split_meta[0][0:-1].decode("utf-8").split(':')
        
        #Extract info... stored in a list with the value on the next line. 
        self.xpixels = float(data_meta_data[data_meta_data.index('Scan>pixels/line')+1].strip())
        self.ypixels = float(data_meta_data[data_meta_data.index('Scan>lines')+1].strip())
        
        # =====================================================================
        
        
        widths = data_meta_data[data_meta_data.index('SCAN_RANGE')+1].strip().split('           ')

        self.xWidth = float(widths[0])
        self.yWidth = float(widths[1])
        
        centre = data_meta_data[data_meta_data.index('SCAN_OFFSET')+1].strip().split('         ')
        self.xCentre = float(centre[0])
        self.yCentre = float(centre[1])
        print(self.xCentre, self.yCentre)
        
        self.angle = float(data_meta_data[data_meta_data.index('SCAN_ANGLE')+1].strip())
        
       # =====================================================================
        self.scan_name = data_meta_data[data_meta_data.index('SCAN_FILE')+2].strip().split('\\')[-1]
        self.acq_time = float(data_meta_data[data_meta_data.index('ACQ_TIME')+1].strip())
        self.time_rec = [data_meta_data[data_meta_data.index('REC_DATE')+1].strip() \
                         + ' ' + data_meta_data[data_meta_data.index('REC_TIME')+1].strip() \
                         + ':' + data_meta_data[data_meta_data.index('REC_TIME')+2].strip() \
                         + ':' + data_meta_data[data_meta_data.index('REC_TIME')+3].strip()]
        self.time_rec_epoch = self.convert_time_to_epoch(self.time_rec)
        
        self.scan_direction = data_meta_data[data_meta_data.index('SCAN_DIR')+1].strip()[:]
        
        self.bias = float(data_meta_data[data_meta_data.index('Bias>Bias (V)')+1].strip())
        
        self.position = data_meta_data[data_meta_data.index('Scan>Scanfield')+1].strip()[:].split(';')
        self.position = [float(self.position[0]),float(self.position[1]),float(self.position[2]),float(self.position[3]),float(self.position[4])]
        # If I find a better way, rewrite this:
        #Channel data - all channel data in a pandas df
        data_channels = data_split_meta[1][2:-1].decode("utf-8") 
        self.df = pd.DataFrame([x.split() for x in data_channels.splitlines()])
        self.df.rename(columns=self.df.iloc[0], inplace = True)
        self.df.drop(self.df.index[0], inplace = True)
        
        
        # Note, this is since "The binary data begins after the header and is introduced by the (hex) code \1A\04."
        self.all_non_sorted_image_data = data_split[1].split(b'\x1a\x04',1)[1]
        
        # Let's now find out how many channels or images there are... 
        # div by 4 as this in # of bytes per float
        # div by how many pixels in each image
        # Other ways of doing this (looking at the meta data, but this is a nice check)
        self.total_image_num = (len(self.all_non_sorted_image_data)/4)/(self.xpixels*self.ypixels)
        
        #Now also reading the sxm image into variables! 
        self.get_image()
        
        print('Image loaded')

        
    def get_file(self, pathname, filename):
        
        #Find get the file name
        self.filename_path = pathname + "\\" + filename
        
        #Now we want to get the data and read it. 
        self.read_sxm_image()
        
    
    def __init__(self):
        
        print('Class for reading SXM is initalised')


