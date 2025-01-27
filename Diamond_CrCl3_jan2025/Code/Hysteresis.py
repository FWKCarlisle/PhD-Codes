#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 20:26:39 2024

@author: briankiraly
"""

from nexusformat.nexus import * 
import matplotlib.pyplot as plt 
import numpy as np 

plt.close('all')
folder = '/Users/briankiraly/Library/CloudStorage/OneDrive-TheUniversityofNottingham/Shared/Kiraly Group/Projects/2D Magnets/Data/Diamond_Feb2024/February_Beamtime_data_2024/' 
files=[328560, 328561, 328562, 328563]

filename=folder+'i06-1-'+str(files[0])+'.nxs' 
a=nxload(filename)
#E=a.entry.instrument.fastEnergy.value 
#idi0=a.entry.instrument.fesData.idio

B1=a.entry.instrument.hyst2.value.nxvalue 
D1A=a.entry.instrument.hyst2.detector1_A.nxvalue 
D1B=a.entry.instrument.hyst2.detector1_B.nxvalue 
D2A=a.entry.instrument.hyst2.detector2_A.nxvalue 
D2B=a.entry.instrument.hyst2.detector2_B.nxvalue 
D3A=a.entry.instrument.hyst2.detector3_A.nxvalue 
D3B=a.entry.instrument.hyst2.detector3_B.nxvalue

#plt.figure(1);plt.plot(B1,D1A,'b.-') 
#plt.figure(1);plt.plot(B1,D1B,'r.-') 
#plt.figure(2);plt.plot(B1,D2A,'b.-') 
#plt.figure(2);plt.plot(B1,D2B,'r.-') 
#plt.figure(3);plt.plot(B1,D3A,'b.-') 
#plt.figure(3);plt.plot(B1,D3B,'r.-')
#plt.figure(1);plt.plot(B1,(D3B/D2B)/(D3A/D2A),'b.-') 
plt.figure(1);plt.plot(B1,((D3B/D2B)-(D3A/D2A))/((D3B/D2B)+(D3A/D2A)),'b.-') 
asym1=((D3B/D2B)-(D3A/D2A))/((D3B/D2B)+(D3A/D2A)) 
minasym1=np.min(asym1)
maxasym1=np.max(asym1) 
normasym1=2*(asym1-(maxasym1+minasym1)/2)/(maxasym1-minasym1) 
plt.figure(2);plt.plot(B1,normasym1,'b.-')

#--------------------

filename=folder+'i06-1-'+str(files[1])+'.nxs' 
b=nxload(filename)
#E=a.entry.instrument.fastEnergy.value 
#idi0=a.entry.instrument.fesData.idio

B2=b.entry.instrument.hyst2.value.nxvalue 
D1Ab=b.entry.instrument.hyst2.detector1_A.nxvalue 
D1Bb=b.entry.instrument.hyst2.detector1_B.nxvalue 
D2Ab=b.entry.instrument.hyst2.detector2_A.nxvalue 
D2Bb=b.entry.instrument.hyst2.detector2_B.nxvalue 
D3Ab=b.entry.instrument.hyst2.detector3_A.nxvalue 
D3Bb=b.entry.instrument.hyst2.detector3_B.nxvalue

#plt.figure(1);plt.plot(B1,D1A,'b.-') 
#plt.figure(1);plt.plot(B1,D1B,'r.-') 
#plt.figure(2);plt.plot(B1,D2A,'b.-') 
#plt.figure(2);plt.plot(B1,D2B,'r.-') 
#plt.figure(3);plt.plot(B1,D3A,'b.-') 
#plt.figure(3);plt.plot(B1,D3B,'r.-')
plt.figure(1);plt.plot(B2,((D3Bb/D2Bb)-(D3Ab/D2Ab))/((D3Bb/D2Bb)+(D3Ab/D2Ab)),'r.-') 
asym2=((D3Bb/D2Bb)-(D3Ab/D2Ab))/((D3Bb/D2Bb)+(D3Ab/D2Ab))
minasym2=np.min(asym2)
maxasym2=np.max(asym2) 
normasym2=2*(asym2-(maxasym2+minasym2)/2)/(maxasym2-minasym2) 
plt.figure(2);plt.plot(B2,normasym2,'r.-')

#--------------------- 

filename=folder+'i06-1-'+str(files[1])+'.nxs' 
a3=nxload(filename)
#E=a.entry.instrument.fastEnergy.value 
#idi0=a.entry.instrument.fesData.idio

B3=a3.entry.instrument.hyst2.value.nxvalue 
D1Ac=a3.entry.instrument.hyst2.detector1_A.nxvalue 
D1Bc=a3.entry.instrument.hyst2.detector1_B.nxvalue 
D2Ac=a3.entry.instrument.hyst2.detector2_A.nxvalue 
D2Bc=a3.entry.instrument.hyst2.detector2_B.nxvalue 
D3Ac=a3.entry.instrument.hyst2.detector3_A.nxvalue 
D3Bc=a3.entry.instrument.hyst2.detector3_B.nxvalue

#plt.figure(1);plt.plot(B1,D1A,'b.-') 
#plt.figure(1);plt.plot(B1,D1B,'r.-') 
#plt.figure(2);plt.plot(B1,D2A,'b.-') 
#plt.figure(2);plt.plot(B1,D2B,'r.-') 
#plt.figure(3);plt.plot(B1,D3A,'b.-') 
#plt.figure(3);plt.plot(B1,D3B,'r.-')
plt.figure(1);
plt.plot(B3,((D3Bc/D2Bc)-(D3Ac/D2Ac))/((D3Bc/D2Bc)+(D3Ac/D2Ac)),'k.-') 
asym3=((D3Bc/D2Bc)-(D3Ac/D2Ac))/((D3Bc/D2Bc)+(D3Ac/D2Ac))

#-------------------- 

filename=folder+'i06-1-'+str(files[2])+'.nxs' 
a4=nxload(filename)
#E=a.entry.instrument.fastEnergy.value
#idi0=a.entry.instrument.fesData.idio

B4=a4.entry.instrument.hyst2.value.nxvalue 
D1Ad=a4.entry.instrument.hyst2.detector1_A.nxvalue
D1Bd=a4.entry.instrument.hyst2.detector1_B.nxvalue 
D2Ad=a4.entry.instrument.hyst2.detector2_A.nxvalue 
D2Bd=a4.entry.instrument.hyst2.detector2_B.nxvalue 
D3Ad=a4.entry.instrument.hyst2.detector3_A.nxvalue 
D3Bd=a4.entry.instrument.hyst2.detector3_B.nxvalue
 
#plt.figure(1);plt.plot(B1,D1A,'b.-') 
#plt.figure(1);plt.plot(B1,D1B,'r.-') 
#plt.figure(2);plt.plot(B1,D2A,'b.-') 
#plt.figure(2);plt.plot(B1,D2B,'r.-') 
#plt.figure(3);plt.plot(B1,D3A,'b.-') 
#plt.figure(3);plt.plot(B1,D3B,'r.-')
plt.figure(1);
plt.plot(B4,((D3Bd/D2Bd)-(D3Ad/D2Ad))/((D3Bd/D2Bd)+(D3Ad/D2Ad)),'c.-') 
asym4=((D3Bd/D2Bd)-(D3Ad/D2Ad))/((D3Bd/D2Bd)+(D3Ad/D2Ad))
M1=(asym1-asym3)/(asym1+asym3) 
M2=(asym2-asym4)/(asym2+asym4) 
plt.figure(5);
plt.plot(B1,M1,'b.-') 
plt.plot(B2,M2,'r.-')