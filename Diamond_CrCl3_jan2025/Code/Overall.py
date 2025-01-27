#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 20:47:35 2024

@author: briankiraly
"""


from nexusformat.nexus import nxload
import numpy as np
import lmfit as lm
import matplotlib.pyplot as plt 
import lmfitxps

class Nexus_handling:
    def get_nexus_data_I06(self,file):
        """Function that loads the data from a nexus file and returns it as a list of numpy arrays"""
        entry_string = "entry"
        data_region_list = []
        metadata_region_list = None
        x_array = file[entry_string]["instrument"]["fastEnergy"]["value"].nxvalue
        y_array = file[entry_string]["instrument"]["fesData"]["C1"].nxvalue 
        data_region_list.append({"x": x_array, "y": y_array})
        y_array = file[entry_string]["instrument"]["fesData"]["C5"].nxvalue 
        data_region_list.append({"x": x_array, "y": y_array})
        y_array = file[entry_string]["instrument"]["fesData"]["idio"].nxvalue 
        data_region_list.append({"x": x_array, "y": y_array})
        y_array = file[entry_string]["instrument"]["fesData"]["ifiofb"].nxvalue 
        data_region_list.append({"x": x_array, "y": y_array})
        magnet_field = file[entry_string]["instrument"]["scm"]["field_z"].nxvalue 
        metadata_region_list = {"magnet_field": magnet_field}
        polar = file[entry_string]["instrument"]["id"]["polarisation"].nxvalue 
        metadata_region_list["polarisation"] = polar
        return data_region_list, metadata_region_list
    
    def open_single_spectra(self,file_number,directory_path,file_prefix,sensor): 
        file_name = directory_path + file_prefix + str(file_number) + ".nxs" 
        spectra_file = nxload(file_name)
        data,meta_data = self.get_nexus_data_I06(spectra_file)
        if sensor == "TEY":
            spectra = {"x" : data[2]["x"],"y":data[2]["y"],"meta" : meta_data}
        elif sensor == "TFY":
            spectra = {"x" : data[3]["x"],"y":data[3]["y"],"meta" : meta_data}
            return spectra
    
    def group_spectra(self,XMCD_spectra_id):
        grouped_data = []
        for i in range(XMCD_spectra_id, XMCD_spectra_id+self.group_size):
            x,y = self.open_single_spectra(i,self.directory_path,self.file_prefix) 
            data_set = np.array([x,y])
            grouped_data.append(data_set)
            return grouped_data
    
class normalisation_procedures:
    # def normalise_spectra_vy_oI_edge_point(spectra_set,oI_edge = 542): 
    #     """Norm procedure by an of single point from the oI edge""" 
    #     normalised_set = []
    #     for i in range(len(spectra_set)):
    #         x = spectra_set[i][0]
    #         y = spectra_set[i][1]
    #         #get y value at index closest to where 
    #         x = oI_edge y_oI_edge = y[np.argmin(np.abs(x-oI_edge))]
    #         y = y/y_oI_edge
    #         normalised_set.append(np.array([x,y]))
    #         return normalised_set
        
    def normalise_spectra_by_range(self,spectra_set,oI_edge_range = [570,573]): 
        """Norm procedure by an of range along the oI edge"""
        x = spectra_set["x"]
        y = spectra_set["y"]
        #get y value at index closest to where x = oI_edge
        y_oI_edge = np.mean(y[np.where((x > oI_edge_range[0]) & (x < oI_edge_range[1]))]) 
        return y - y_oI_edge
    
    def normalise_spectra_by_linear_fit(spectra_set,oI_edge_range = [570,573]): 
        """Norm procedure by fitting a line to the oI edfe"""
        normalised_set = []
        for i in range(len(spectra_set)):
            x = spectra_set[i][0]
            y = spectra_set[i][1]
            #get y value at index closest to where x = oI_edge
            y_oI_edge = y[np.where((x > oI_edge_range[0]) & (x < oI_edge_range[1]))] 
            x_oI_edge = x[np.where((x > oI_edge_range[0]) & (x < oI_edge_range[1]))] #fit a line to the oI edge
            def line(x,m,c):
                return m*x + c
            model = lm.Model(line)
            params = model.make_params(m=0,c=0)
            result = model.fit(y_oI_edge,params,x=x_oI_edge) 
            y = y - result.best_fit 
            normalised_set.append(np.array([x_oI_edge,y]))
            return normalised_set

    def _calculate_shirley_background_full_range(xps: np.ndarray, eps=1e-7, max_iters=50, n_samples=5) -> np.ndarray:
        """Core routine for calculating a Shirley background on np.ndarray data.""" 
        background = np.copy(xps)
        cumulative_xps = np.cumsum(xps, axis=0)
        total_xps = np.sum(xps, axis=0)
        rel_error = np.inf
        i_left = np.mean(xps[:n_samples], axis=0) 
        i_right = np.mean(xps[-n_samples:], axis=0)
        iter_count = 0
        k = i_left - i_right
        for iter_count in range(max_iters):
            cumulative_background = np.cumsum(background, axis=0) 
            total_background = np.sum(background, axis=0)
            new_bkg = np.copy(background)
            for i in range(len(new_bkg)): 
                new_bkg[i] = i_right + k * ((total_xps - cumulative_xps[i] - (total_background - cumulative_background[i]))/ (total_xps - total_background + 1e-5) )
                rel_error = np.abs(np.sum(new_bkg, axis=0) - total_background) / (total_background)
                background = new_bkg
                if np.any(rel_error < eps): 
                    break
                if (iter_count + 1) == max_iters: 
                    # warnings.warn("Shirley background calculation did not converge "+ "after {} steps with relative error {}!".format(max_iters, rel_error) )
                    return background
                

class XMCD_data_analysis(Nexus_handling): 
    
    def __init__(self):
        self.on_oI_edge_points = [571, 576.9] 
        self.directory_path = None 
        self.file_prefix = None
        self.energy = None
        self.spectra = [] 
        self.hyst = [] 
        self.XMCD = [] 
        self.magz = []
    
    def set_directory_path(self, directory_path):
        self.directory_path = directory_path
        
    def set_file_prefix(self, file_prefix): 
        self.file_prefix = file_prefix
    
    def load_spectra(self, directory_path, file_prefix, XMCD_spectra_Set,sensor = "TEY"): 
        self.directory_path = directory_path
        self.file_prefix = file_prefix
        for file_number in XMCD_spectra_Set:
            #print(file_number)
            data_set = self.open_single_spectra(file_number, directory_path,file_prefix,sensor) 
            self.spectra.append(data_set)
            self.energy = self.spectra[0]["x"]
            
    def normalise_all_spectra(self, procedure="normalise_spectra_by_range", oI_edge= [570, 573]):
        for i in range(len(self.spectra)):
            y = getattr(normalisation_procedures, procedure)(self, self.spectra[i], oI_edge)
            y = y - normalisation_procedures._calculate_shirley_background_full_range(y) 
            self.spectra[i]["y"] = y
    
    def calculate_XMCD_w_hysterisis(self):
        pc_set = []
        nc_set = []
        print("number of spectra = ", len(self.spectra))
        for i in range(len(self.spectra)):
            if self.spectra[i]["meta"]["polarisation"] == "pc": 
                pc_set.append(self.spectra[i])
            else: nc_set.append(self.spectra[i])
            if i == len(self.spectra) - 1:
                hyst_point,XMCD_arr = self.calculate_XMCD_w_hyst_point(pc_set, nc_set) 
                self.hyst.append(hyst_point)
                self.XMCD.append(XMCD_arr) 
                self.magz.append(self.spectra[i]["meta"]["magnet_field"]*1)
            elif (np.round(self.spectra[i]["meta"]["magnet_field"]*1, 1) != np.round(self.spectra[i + 1]["meta"]["magnet_field"]*1, 1)):
                print(np.round(self.spectra[i]["meta"]["magnet_field"]*1,2)) 
                hyst_point,XMCD_arr = self.calculate_XMCD_w_hyst_point(pc_set, nc_set)
                self.hyst.append(hyst_point)
                self.XMCD.append(XMCD_arr) 
                self.magz.append(self.spectra[i]["meta"]["magnet_field"]*1) 
                pc_set = []
                nc_set = []
                return
#fig,ax = plt.subplots()
#ax.plot(self.energy,XMCD_arr)
#ax.set_ylim(-1,1) #print(self.spectra[i]["meta"]["magnet_field"])
#plt.show()

    def calculate_XMCD_w_hysterisis_by_group(self,group_size = 5): 
        pc_set = []
        nc_set = []
        print("number of spectra = ", len(self.spectra))
        print("number of points = ",len(self.spectra)/5) 
        for i in range(0,len(self.spectra),group_size):
            for j in range(1,group_size):
                if self.spectra[i+j]["meta"]["polarisation"] == "pc":
                    pc_set.append(self.spectra[i+j]) 
                else:
                    nc_set.append(self.spectra[i+j]) 
                    self.magz.append(self.spectra[i]["meta"]["magnet_field"]) 
                    hyst_point,XMCD_arr = self.calculate_XMCD_w_hyst_point(pc_set, nc_set) 
                    self.hyst.append(hyst_point)
                    self.XMCD.append(XMCD_arr)
                    pc_set = []
                    nc_set = []
                    return
                
    def check_spectra(self,start = 0,finish = 20, group_size = 5): 
        for i in range(start,finish,group_size):
            for j in range(group_size):
                x = self.spectra[i+j]["x"]
                y = self.spectra[i+j]["y"]
                back = normalisation_procedures._calculate_shirley_background_full_range(y) 
                fig,ax = plt.subplots()
                plt.plot(x,y)
                plt.plot(x,back) 
                ax.annotate(str(self.spectra[i+j]["meta"]["magnet_field"]),xy=(0.9,0.9),xytext =(0.95,0.95),xycoords = "axes fraction") 
                print("")
                
    def calculate_XMCD_w_hyst_point(self, pc_set, nc_set): 
        ncy=0
        ncx = nc_set[0]["x"] 
        pcx = pc_set[0]["x"] 
        for spec in nc_set:
            ncy += spec["y"] 
            ncy /= len(nc_set) 
            pcy=0
            for spec in pc_set:
                pcy += spec["y"] 
                pcy /= len(pc_set)
                XMCD=pcy-ncy
                #XMCD_oI_edge = XMCD[np.argmin(np.abs(ncx - self.on_oI_edge_points[0]))] 
                #XMCD_on_edge = XMCD[np.argmin(np.abs(pcx - self.on_oI_edge_points[1]))] 
                A=pcy[np.argmin(np.abs(pcx - self.on_oI_edge_points[1]))] - pcy[np.argmin(np.abs(ncx - self.on_oI_edge_points[0]))] 
                B=ncy[np.argmin(np.abs(pcx - self.on_oI_edge_points[1]))] - ncy[np.argmin(np.abs(ncx - self.on_oI_edge_points[0]))] 
                #print(XMCD_oI_edge - XMCD_on_edge,XMCD_oI_edge,XMCD_on_edge)
                ##################HYST POINT CALCULATION######################## 
                #hystp = abs(XMCD_oI_edge - XMCD_on_edge)
                hystp = (A-B)/(A+B)
                return hystp,XMCD

    def plot_XMCD_curves(self,tag = "all",tag_value = None,ylim = 1): 
        for i,XMCD_set in enumerate(self.XMCD):
            if tag == "by field" and tag_value is not None: 
                XMCD_plots = self.XMCD[np.where(np.round(np.array(self.magz),self.decimal_length(tag_value)))] 
            elif tag == "by range" and tag_value is not None:
                for i in range(tag_value): 
                    ax.plot(self.energy,XMCD_set)
                    break 
                else:
                    fig,ax = plt.subplots()
                    ax.plot(self.energy,XMCD_set)
                    #ax.set_ylim(-ylim,ylim)
                    ax.annotate(str(self.magz[i]),xy=(0.1,0.1),xytext = (0.15,0.15),xycoords = "axesfraction") 
                    plt.show()
                    return
                
    def decimal_length(number):
        # Convert the number to a string 
        number_str = str(number)
        # Find the position of the decimal point 
        decimal_position = number_str.find('.')
        # If there's no decimal point, return 0
        if decimal_position == -1:
            return 0 
        # Return the length of the string, excluding the decimal point 
        return len(number_str) - 1

    
# %%  Data Analysis

directory_path = '/Users/briankiraly/Library/CloudStorage/OneDrive-TheUniversityofNottingham/Shared/Kiraly Group/Projects/2D Magnets/Data/Diamond_Feb2024/February_Beamtime_data_2024/' 
file_prefix = "i06-1-"
 

XMCD_file_Set = list(range(330645, 330730))
#XMCD_file_Set = list(range(329206, 329291))
#XMCD_file_Set = list(range(329594, 329679))
#XMCD_file_Set = list(range(329960, 330115)) 
#XMCD_file_Set = list(range(330306, 330391))
sensor = ["TEY"]
XMCD = XMCD_data_analysis()
XMCD.load_spectra(directory_path, file_prefix, XMCD_file_Set,sensor[0]) 
XMCD.normalise_all_spectra()
#XMCD.check_spectra(start = 0,finish = 5)
#XMCD.check_spectra(start = 85,finish = 90) XMCD.calculate_XMCD_w_hysterisis_by_group() #XMCD.plot_XMCD_curves()
magz1,hyst1 = XMCD.magz,XMCD.hyst


XMCD_file_Set = list(range(330546, 330636))
#XMCD_file_Set = list(range(330391, 330476))
#XMCD_file_Set = list(range(329292, 329377))
#XMCD_file_Set = list(range(329516, 329591)) 
# XMCD_file_Set = list(range(330788, 330873))
XMCD = XMCD_data_analysis() 
XMCD.load_spectra(directory_path, file_prefix, XMCD_file_Set) #MCD.check_spectra()
XMCD.normalise_all_spectra() 
XMCD.calculate_XMCD_w_hysterisis_by_group() #XMCD.plot_XMCD_curves(ylim = 0.8)
magz2,hyst2 = XMCD.magz,XMCD.hyst
#print(magz2,hyst2)
#split = 1
#l = int(len(magz1)/split)
# fig, ax = plt.subplots()
# ax.plot(magz[:l], hyst[:l], 'o--') # Plot the first 10 elements of magz and hyst
# ax.plot(magz[l:], hyst[l:], 'o--') # Plot from the 11th element to the end of magz hyst

# plt.show()
plt.figure()
plt.plot(np.asarray(magz1), np.asarray(hyst1), 'o--') # Plot the first 10 elements of magz and hyst
plt.plot(np.asarray(magz2), np.asarray(hyst2), 'o--') # Plot from the 11th element to the end of magz and hyst
plt.show()
