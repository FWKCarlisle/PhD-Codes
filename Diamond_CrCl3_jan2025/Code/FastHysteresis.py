#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 01:00:17 2024

@author: briankiraly
"""


from nexusformat.nexus import * 
import matplotlib.pyplot as plt
import numpy as np

plt.close('all')
folder = '/Users/briankiraly/Library/CloudStorage/OneDrive-TheUniversityofNottingham/Shared/Kiraly Group/Projects/2D Magnets/Data/Diamond_Feb2024/February_Beamtime_data_2024/' 
files=list(range(329930, 329936))
fl=[]
for i in files:
    filename=folder+'i06-1-'+str(i)+'.nxs' 
    a=nxload(filename)
    T1=a.entry.instrument.ca65sr 
    T2=a.entry.instrument.magz 
    B=np.array(T2.value) 
    F=np.array(T1.value) 
    # f=np.polyfit(B, F, 1) 
    # fit=np.polyval(f,B)
    # fl=np.c_[fl, np.transpose(F)]

plt.figure(1);plt.plot(B,F)