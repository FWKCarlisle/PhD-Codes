import numpy as np
import re
import io
import scipy.signal as spy
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import time
import os
from pathlib import Path
import pptx
import pptx.util

class output_data_spectra_dat():
    
    def get_file(self, pathname, filename, show_methods=0):
        #Find get the file name
        self.filename_path = pathname + "\\" + filename
        
        #Now we want to get the data and read it. 
        self.meta = self.load_file(show_methods)
        
        if show_methods == 1:
            self.show_method_fun()
    
    def convert_time_to_epoch(self, time):
        dt = datetime.datetime.strptime(str(time), '%d.%m.%Y %H:%M:%S')
        return dt.timestamp()

    def load_file(self, show_methods):
        #Open the file. The data and headings start after the '[DATA]' section.
        f = open(self.filename_path, "r")
        f.seek(0)
        file_all_data = f.read()
        f.close()
        
        #Split at the '[DATA]' section. The second in the list is our data with the headers.
        
       
        meta_data = re.split(r"\[DATA\]\n",file_all_data)[0]
        file_data = re.split(r"\[DATA\]\n",file_all_data)[1]
        # print(meta_data)
        #Extract the time infomation:
        # self.start_time = self.convert_time_to_epoch(meta_data[meta_data.index('Start time')+11:meta_data.index('Start time')+11+19])
        self.start_time = self.convert_time_to_epoch(meta_data[meta_data.index('Saved Date')+11:meta_data.index('Saved Date')+11+19])
        

        meta_split = meta_data.split('\n') # Extracting the Z relative value 
        #Extract position infomation: 
        # self.x_pos = float(file_data.split('X (m)\t')[1].split('\n')[0][0:-1])
        # self.y_pos = float(file_data.split('Y (m)\t')[1].split('\n')[0][0:-1])
        
        
        #Load as df to make life easy
        self.df = pd.read_csv(io.StringIO(file_data),delimiter = '\t')
        self.meta_split = meta_split
        
        self.list_of_methods = list(self.df)
        return meta_data
     
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
    

class Spectrum(output_data_spectra_dat):
    
    def __init__(self, path, fileName, channel):
        """
        Parameters
        ----------
        path : str
            path to spectra file.
        fileName : str
            spectra file name (dat file).
        channel : str
            Channel to read.
            For KPFM it'll be 'OC M1 Freq. Shift [AVG] (Hz)'.
            If the given channel is not found, the available channels will be 
            returned. So, if you're not sure of the exact channel name, just 
            type in non sense.

        Returns
        -------
        Class instance for the specified spectra file, with 2 attributes: the x
        and y data for the specified channel. 
        
        """
        super().__init__()

        self.path = path
        self.fileName = fileName
        self.channel = channel
        
        self.x, self.y = self._OpenSpectra(path=self.path, fileName=self.fileName, channel = self.channel)
        
    
    
    def _OpenSpectra(self, path, fileName, channel):
        """
        Parameters
        ----------
        path : str
            path to spectra file.
        filename : str
            spectra file (dat file).
        channel : str
            Channel to read.
            

        Returns
        -------
        x : 1D array
            spectra x data. For KPFM it'll be bias in Volts.
        y : 1D array
            spectra y data. For KPFM it'll be freq shift in Hz.

        """
        self.get_file(path, fileName)  
        
        # if channel not in file, print list of possible channels
        if channel not in list(self.df): 
            print('Choice of channel not found in ' + fileName)
            self.show_method_fun()
         
        
        x = self.give_data(0)[0] 
        y = self.give_data(channel)[0]

        return x, y 
    
def setup_presentation():
    prs = pptx.Presentation()
    return prs

def save_to_presentation(prs, save_path,image_path,  filename = "presentation", save= "900mV"):
    slide_layout = prs.slide_layouts[5]
    slide = prs.slides.add_slide(slide_layout)
    
    # Step 4: Add image to the slide
    left = pptx.util.Inches(0.25)   # Adjust positioning
    top = pptx.util.Inches(1.5)
    width = pptx.util.Inches(8)  # Set width (optional)
    height = pptx.util.Inches(6) # Set height (optional)
    # print("Image path: ", image_path)
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    
    slide.shapes.add_picture(image_path, left, top, width, height)

    # Step 5: Save the PowerPoint file
    pptx_filename = save_path + fr"\{filename}.pptx"
    prs.save(pptx_filename)
    
def main():
    temp = "300mK"
    temp = "4.8K"

    if temp == "300mK":
        dir_path = r"D:\DBTF\up down 330"
    elif temp == "4K":
        dir_path = r"D:\DBTF\Up_down_18th"
    elif temp == "4.8K":
        dir_path = r"E:\DBTF\Up_down_12_11_500"

    files = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]

    files = np.sort(files)

    Ten_array = []
    Hundred_array = []
    SevenFifty_array = []
    FiveHundred_array = []
    if temp != "4.8K":
        for file in files:
            
            if file.split("_")[4] == "100":
                Hundred_array.append(file)
            elif file.split("_")[4] == "10":
                Ten_array.append(file)
            elif file.split("_")[4] == "0.75": 
                SevenFifty_array.append(file)
            elif file.split("_")[4] == "1.0": 
                FiveHundred_array.append(file)
    else:
        for file in files:
            FiveHundred_array.append(file)

        



    powerpoint_path = rf"C:\Users\ppxfc1\OneDrive - The University of Nottingham\Desktop\PhD\Code\PhD-Codes\Analysis code\DBTF\Up_down_powerpoints"
    

    prs = setup_presentation()

    #Change between 10 and 100 here
    number = 1 # or 0.75 for the 750mV ones. 
    background_file = 0  #23 for 0.75, 35 for ten_array 
    files = FiveHundred_array
   
    #delete old powerpoint of same name
    pptx_filename = powerpoint_path + fr"\{number}.pptx"
    if os.path.exists(pptx_filename):
        os.remove(pptx_filename)
    current_array = []
    length_array = []
    z_array = []
    # files = files[399:]

    times = []

    files.insert(0, files.pop(background_file))

    # print(files[:25])
    for i, file in enumerate(files):
        start_time = time.perf_counter()
        # Remove the existing file if it exists
        
        file_spec = Spectrum(path=dir_path, fileName=file, channel= 'Bias (V)')

        current = file_spec.x
        freq_shift = file_spec.y
        meta_split = file_spec.meta_split
        z = file_spec.give_data("Z (m)")[0] ## Use this as its a standard height in the middle of measurement
        # x = file_spec.give_data("X (m)")[0]
        # y = file_spec.give_data("Y (m)")[0]

        sample_period = float(meta_split[3].split(")")[1])* 0.001 # in s
        
        if i == 0:
            background = z


        current_array.append(current) 
        z_array.append(z)
        
        z_true = z
        length_t = np.arange(0, len(z_true))
        
        # z = z-background
        z_sg = spy.savgol_filter(z, 200, 3)  # window size 51, polynomial order 3
        length = np.arange(0, len(z))
        dZ = np.diff(z)
        dZ_t = np.diff(z_true)
        dZ_sg = np.diff(z_sg)

        length_dz = np.arange(0,len(dZ))
        length_dzt = np.arange(0,len(dZ_t))

        approach = np.argmin(dZ_t)
            
        mean_threshold = np.mean(z[approach+500:])
        # attraction = 0
        # #Find the first index after the data passes the mean threshold
        # for idx in range(approach+50, len(z)):
        #     if z[idx] > mean_threshold:
        #         attraction = idx 
        #         # print(f"Attraction point found at index: {attraction} and approach at {approach} for file {file}")
        #         break

        attraction = np.argmax(z[approach + 50:]) + (approach + 50)
                

        # print(approach, len(dZ), len(dZ[approach:]), attraction, attraction + approach )

        diff_time = (attraction - approach) * sample_period
        if diff_time > -1:
            times.append(diff_time)

        fig, [Iax, ZTax, Zax ] = plt.subplots(3, 1, figsize=(10, 6), sharex=True)

        # Iax.plot(length,current)
        # Iax.plot(length_dz[approach:],dZ[approach:], "g", label="dZ") #[plotting_1:plotting_2]
        # Iax.plot(length_dzt[approach:],dZ_t[approach:], "b", label="dZ_true")
        Iax.plot(np.arange(0,len(dZ_sg))[approach:], dZ_sg[approach:], "r", label="dZ_sg")
        Iax.axvline(attraction)
        Iax.legend()
        Iax.set_ylabel("Current")

        ZTax.plot(length_t[approach:],z_true[approach:], "g", label = "Z_true")
        ZTax.plot(np.arange(0,len(background))[approach:], background[approach:], "r", label="Z_0")
        ZTax.axvline(attraction)
        ZTax.legend()
        ZTax.set_ylabel("Z")

        # Zax.plot(length, z)
        Zax.plot(length[approach: ],z[approach:],"g")
        Zax.plot(length[approach: ],z_sg[approach:],"b", label="Z - Z_0 (smoothed)")
        Zax.axhline(mean_threshold, color='r', linestyle='--', label='Mean Threshold')
        Zax.axvline(attraction, label=f'Attraction Point - {diff_time}s')
        
        Zax.legend()
        
    
    
        Zax.set_ylabel("Z-Z_0")

        plt.title(file)
        plt.xlabel("Index")
        # plt.show()
        plt.savefig(f"{powerpoint_path}\\temp_img.png")
        plt.close()
        
        save_to_presentation(prs, powerpoint_path, f"{powerpoint_path}\\temp_img.png", filename=f"{number}_{temp}")
        finish_time = time.perf_counter()
        if i % 10 == 0:
            # print(f"{plotting_1}, {plotting_2} ")
            print(f"Saving file {i} / {len(files)}, time: {(finish_time - start_time)*10}")
    
    counts, edges, bars = plt.hist(times, bins=100, color='blue', alpha=0.7, edgecolor='black')
    plt.bar_label(bars)
    
    plt.title(f"Histogram of approach to attraction times for 1V at {temp}")
    plt.xlabel("Time (s)")
    plt.ylabel("Counts")
    plt.show()
    plt.savefig(f"{powerpoint_path}\\temp_img.png")
    plt.close()
    


    save_to_presentation(prs, powerpoint_path, f"{powerpoint_path}\\temp_img.png", filename=f"{number}_{temp}")
        

    
if __name__ == "__main__":


    main()