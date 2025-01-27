#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 21:47:58 2024

@author: briankiraly
"""

from nexusformat.nexus import * 
import matplotlib.pyplot as plt
import numpy as np
import lmfitxps 

plt.close('all')

folder = '/Users/briankiraly/Library/CloudStorage/OneDrive-TheUniversityofNottingham/Shared/Kiraly Group/Projects/2D Magnets/Data/Diamond_Feb2024/February_Beamtime_data_2024/' 
# files=[328379, 328384] #BP 0T
# files=[328392, 328393] #BP 2T
# files=[328394, 328395] #BP -2T
# files=[328396, 328397] #BP 4T 
# files=[328398, 328400] #BP 6T
# files=[329060, 329061] #HOPG 0T
# files=[329064, 329065] #HOPG 6T
# files=[329842, 329843] #Capped SiC 6T
# files=[329845, 329849] #Capped SiC 0T
# files=[329873, 329875] #Capped HOPG 6T
# files=[329862, 329866] #Capped HOPG 0T
# files=[329891, 329885] #Capped BP 6T
# files=[329902, 329905] #Capped BP 0T
files=[330542, 330543] #BP 6T


filename=folder+'i06-1-'+str(files[0])+'.nxs' 
a=nxload(filename)
En=a.entry.instrument.fastEnergy.value 
idi0=a.entry.instrument.fesData.idio
E=np.array(En)
I=np.array(idi0)
In=I-np.average(I[0:20])

filename=folder+'i06-1-'+str(files[1])+'.nxs' 
a=nxload(filename)
En=a.entry.instrument.fastEnergy.value 
idi0=a.entry.instrument.fesData.idio
E1=np.array(En)
I1=np.array(idi0)
In1=I1-np.average(I1[0:20])






XMCD=-In+In1
XAS=In+In1
# bkgnd=lmfitxps.backgrounds.shirley_calculate(E, XAS, tol=1e-05, maxit=10)



# A=np.trapz(XMCD[39:119])
# B=np.trapz(XMCD[129:199])
# C=np.trapz(In+In1)

p=np.trapz(XMCD[0:119])
q=np.trapz(XMCD)
r=np.trapz(In+In1)

# ms=(-A+2*B)/(C)
# ml=-2*(A+B)/(3*C)

# plt.figure(1);
# plt.title('CrCl3 on HOPG')
# plt.plot(E,In)
# # plt.plot(E1,In1)
# plt.xlabel('Energy (eV)')
# plt.ylabel('Normalised Intensity (a.u.)')
# plt.savefig("HOPGXAS.pdf", format="pdf", bbox_inches="tight")

# plt.figure(2);
# plt.plot(E,XMCD,'k')
# plt.xlabel('Energy (eV)')
# plt.ylabel('XMCD')

# fig, (ax1, ax2) = plt.subplots(2)
# ax1.plot(E, In, E, In1)
# ax1.set_ylabel('XAS')
# ax2.plot(E, XMCD, 'k')
# plt.xlabel('Energy (eV)')
# plt.ylabel('XMCD')
# fig.suptitle('CrCl3/HOPGc 6T')
# plt.savefig("HOPGc6T.pdf", format="pdf", bbox_inches="tight")

# plt.figure(3);
# plt.plot(E,XMCD,'k')
# plt.plot(E[39:119],XMCD[39:119])
# plt.plot(E[129:199],XMCD[129:199])
# plt.xlabel('Energy (eV)')
# plt.ylabel('XMCD')