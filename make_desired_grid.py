# -*- coding: utf-8 -*-
"""
Created on Wed May  1 15:40:43 2024

@author: physicsuser
"""
import numpy as np 
import matplotlib.pyplot as plt

def RealAtomPos(xAtomsVec, yAtomsVec, a, b, theta):
    
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
    
    x = np.array(xAtomsVec, dtype=float)
    y = np.array(yAtomsVec, dtype=float)
    
    x = a * x 
    y = b * y
    
    x, y = rotate(x, y, xPivot=x[0], yPivot=y[0], rot=theta)

    # x = x + xt
    # y = y + yt
    
    return x, y

# xAtomsVec = np.array([0, 4, -2, 6, 0, 4], dtype=float)
# yAtomsVec = np.array([0, 0, -5, -5, -10, -10], dtype=float)
# xAtomsVec = np.array([0, -3, 3, -4, 4, -3, 3, 0], dtype=float)
# yAtomsVec = np.array([0, -1, -1, -5, -5, -9, -9, -10], dtype=float)
# xAtomsVec = np.array([0, -2, -4, -5, -4, -2, 0, 2, 3, 2], dtype=float)
# yAtomsVec = np.array([0, -0, -2, -5, -8, -10, -10, -8, -5, -2], dtype=float)
xAtomsVec = np.array([0, 5, 10, 15, 20], dtype=float)
yAtomsVec = np.array([0, 5, 10, 15, 20], dtype=float)
xAtomsVec, yAtomsVec = np.meshgrid(xAtomsVec, yAtomsVec)

a = 0.648e-9
b = 0.458e-9
theta = 2.4
xDesiredGrid, yDesiredGrid = RealAtomPos(xAtomsVec, yAtomsVec, a, b, theta)
fig, ax = plt.subplots()
ax.scatter(xDesiredGrid, yDesiredGrid)
substrate = 'InSb(110)'
gridName = 'square'

path = r'desired_grids'

np.save(path + r'\xdg_' + substrate + '_' + gridName, xDesiredGrid)
np.save(path + r'\ydg_' + substrate + '_' + gridName, yDesiredGrid)
