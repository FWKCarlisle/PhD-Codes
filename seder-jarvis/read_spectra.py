# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 14:26:52 2023

@author: ppzme
"""

#%% Import the modules
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


class output_data_spectra_dat():
    
    def get_file(self, pathname, filename, show_methods=0):

        #Find get the file name
        self.filename_path = pathname + "\\" + filename
        
        #Now we want to get the data and read it. 
        self.load_file(show_methods)
        
        if show_methods == 1:
            self.show_method_fun()
    
    def convert_time_to_epoch(self, time):
        dt = datetime.datetime.strptime(str(time), '%d.%m.%Y %H:%M:%S')
        return dt.timestamp()

    def load_file(self, show_methods):

        def read_metadata_info(meta_data):
            """
            Note: some files will not have this info e.g. data logger files
            Can be extended/edited to access more/different metadata info.
            """

            # Extract position information:
            try: self.x_pos = float(meta_data.split('X (m)\t')[1].split('\n')[0][0:-1])
            except: pass
            try: self.y_pos = float(meta_data.split('Y (m)\t')[1].split('\n')[0][0:-1])
            except: pass
            try: self.z_pos = float(meta_data.split('Z (m)\t')[1].split('\n')[0][0:-1])
            except: pass

            # Extract the time information:
            try: self.start_time = self.convert_time_to_epoch(
                meta_data[meta_data.index('Start time') + 11:meta_data.index('Start time') + 11 + 19])
            except: pass

            # Extract comment
            try: self.comment = meta_data.split('Comment01\t')[1].split('\n')[0][0:-1]
            except: pass

        #Open the file. The data and headings start after the '[DATA]' section.
        f = open(self.filename_path, "r")
        f.seek(0)
        file_all_data = f.read()
        f.close()
        
        #Split at the '[DATA]' section. The second in the list is our data with the headers.
        file_data = re.split(r"\[DATA\]\n",file_all_data)[1]
        meta_data = re.split(r"\[DATA\]\n", file_all_data)[0]

        read_metadata_info(meta_data)

        #Load as df to make life easy
        self.df = pd.read_csv(io.StringIO(file_data),delimiter = '\t')
        
        self.list_of_methods = list(self.df)
     
    def show_method_fun(self):
        
        print('Possible methods to use are:')
        
        for count, i in enumerate(self.list_of_methods):
            print(str(count) +') '+ i)
        
    def give_data(self, method_number):
        
        #if type(method_number) is int:
            #Do nothing, as this is the expected format
        if type(method_number) is str:
            #Convert to number but use the lookup table
            for count, i in enumerate(self.list_of_methods):
                if method_number in i:
                    method_number = count
                    break
            if method_number is not count:
                #We did not convert to an int value
                raise Exception("This input is not found in the list, check spelling is correct!")
                
        elif type(method_number) is not int: 
            raise Exception("Must be either and int or str variable")
        
        name_to_export = self.list_of_methods[method_number]
        
        exported_data = (self.df[name_to_export],self.list_of_methods[method_number])
        
        return exported_data

