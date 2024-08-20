import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# from read_sxm import output_data_from_sxm
from file_reader import Sxm
matplotlib.rc('image', cmap='gray')

class SamplingGrid():
    
    
    def SquareGrid(self, sideLength, PointsPerLine, xCentre, yCentre):
        """
        Parameters
        ----------
        sideLength : float
            side length of the square grid in metres
        PointsPerLine : int
            number of sampling points within the side length 
        xCentre : float
            x centre coordinate in metres
        yCentre : float
            y centre coordinate in metres
            
        Returns
        -------
        x : 1D numpy array
            x coordinates of sampling points in metres
        y : 1D numpy array
            y coordinates of sampling points in metres
        
        Note - x and y are ordered snaking upwards to minimise drift. (See example below).
        """
        
        # Define a 2D grid of xy coordinates 
        xMin, xMax = xCentre - 0.5 * sideLength, xCentre + 0.5 * sideLength 
        yMin, yMax = yCentre - 0.5 * sideLength, yCentre + 0.5 * sideLength
        
        x2D, y2D = np.meshgrid(np.linspace(xMin, xMax, num=PointsPerLine),
                           np.linspace(yMin, yMax, num=PointsPerLine))
        x2D, y2D = x2D.T, y2D.T
        # Use a numpy left right flip function on every other row for a snaking sampling order. 
        # Every other row is sliced as [start = 0 : end = -1 : step = 2]
        x2D[0:-1:2], y2D[0:-1:2] = np.fliplr(x2D[0:-1:2]), np.fliplr(y2D[0:-1:2]) 
        
        # flatten the 2D arrays, to 1D
        x = x2D.ravel()
        y = y2D.ravel()
        y = np.flip(y)
        
        
        
        return x, y
        
        
    def CircleGrid(self, diameterLength, pointsInDiameter, xCentre, yCentre):
        """
        Parameters
        ----------
        diameterLength : float
            diamter of the sampling grid in metres
        pointsInDiameter : int
            number of sampling points within the length of the circle's diameter. 
            note - this will be only an approximation bc we want sampling points aranged 
            in a square manner (so we get sqare pixels in our eg. KPFM spectra map) and a 
            round border. 
        xCentre : float
            x centre coordinate in metres
        yCentre : float
            y centre coordinate in metres

        Returns
        -------
        x : 1D numpy array
            x coordinates of sampling points in metres
        y : 1D numpy array
            y coordinates of sampling points in metres
        
        Note - x and y are ordered snaking upwards to minimise drift. (See example below).
        """
        
        # make a square grid of size diameterLength x diameterLength
        x, y = self.SquareGrid(diameterLength, pointsInDiameter, xCentre, yCentre)
        
        # remove unwanted sampling points
        # coordinate trasformation to polar, with origin at (xCentre, yCentre)
        xDash = x - xCentre 
        yDash = y - yCentre
        r = np.sqrt(xDash**2 + yDash**2)
        
        # In this new coordinate system, sampling points with r>rmax are unwanted
        rmax = 0.5 * diameterLength 
        x = x[r <= rmax]
        y = y[r <= rmax]
        
        return x, y
        
    
    def RemoveSamplingAboveAtoms(self, xSampling, ySampling, 
                                 xAtoms, yAtoms, forbiddenDiameter):
        """
        Parameters
        ----------
        xSampling : 1D array like
            x coordinates of sampling points in metres.
        ySampling : 1D array like
            y coordinates of sampling points in metres.
        xAtoms : 1D array like
            x coordinates of atoms positions in metres.
        yAtoms : 1D array like
            y coordinates of atoms positions in metres.
        forbiddenDiameter : float
            diameter, in metres, around atom centre to avoid sampling over.

        Returns
        -------
        xGrid : 1D array
            x coordinates of sampling points, with points above atoms removed.
        yGrid : 1D array
            x coordinates of sampling points, with points above atoms removed.

        """
        # need to use polar coords to define forbidden regions
        for atomNumber in range(len(xAtoms)):
            # move the origin to the atom centre
            xDash = xSampling - xAtoms[atomNumber]
            yDash = ySampling - yAtoms[atomNumber]
            
            # now the forbidden region is simply anything with r <= forbidden radius
            rDash = np.sqrt(xDash**2 + yDash**2)
            forbiddenRad = forbiddenDiameter / 2
            xSampling = xSampling[rDash >= forbiddenRad]
            ySampling = ySampling[rDash >= forbiddenRad]

        return xSampling, ySampling
    
    
    
    def Rotate(self, xSampling, ySampling, xPivot=0, yPivot=0, rot=0):
        rot = -np.deg2rad(rot)
        
        x = xSampling - xPivot
        y = ySampling - yPivot
        
        # carteesian -> polar
        r = np.sqrt(x**2 + y**2)
        theta = np.arctan2(y, x) # element-wise arctan
        
        # rotation
        theta = theta + rot
        
        # polar -> carteesian
        x = (r*np.cos(theta)) + xPivot
        y = (r*np.sin(theta)) + yPivot

        return x, y


# class Sxm(output_data_from_sxm):
    
#     def __init__(self):
#         super().__init__() 
        

        
#     def ReadSxm(self, path, fileName):
            
#         self.get_file(path, fileName)
        
#         xCentre, yCentre, _, _, angle = self.position
    
#         channels = self.get_channel_names()
#         zFwdIdx = channels.index('Z_Fwd')
#         zBwdIdx = channels.index('Z_Bwd')
        
#         imFwd, _ = self.get_select_image(zFwdIdx)
#         imBwd, _ = self.get_select_image(zBwdIdx)
        
#         """
#         the y-axis origin is flipped 
#         when loading an image from a saved sxm file vs. from grabbing the 
#         current image displayed on the scan control. To avoid confusion, I
#         prefer flipping sxm file images, when loading. 
#         """
#         imFwd = np.flipud(imFwd)
#         imBwd = np.flipud(imBwd)
    
#         self.imFwd = imFwd
#         self.imBwd = imBwd
#         self.xCentre = xCentre
#         self.yCentre = yCentre
#         self.angle = angle
        
#         return imFwd
    
    
#     def PlotAtAngle(self, theta):
        
#         def rotate(x, y, xPivot=0, yPivot=0, rot=0):
#             rot = -np.deg2rad(rot)
            
#             x = x - xPivot
#             y = y - yPivot
            
#             # carteesian -> polar
#             r = np.sqrt(x**2 + y**2)
#             theta = np.arctan2(y, x) # element-wise arctan 
            
#             # rotation
#             theta = theta + rot
            
#             # polar -> carteesian
#             x = (r*np.cos(theta)) + xPivot
#             y = (r*np.sin(theta)) + yPivot

#             return x, y
        
#         pxPerLine = np.shape(im)[0]
        
#         x = np.linspace(self.xCentre - self.xWidth/2, self.xCentre + self.xWidth/2, num = pxPerLine)
#         y = np.linspace(self.yCentre - self.yWidth/2, self.yCentre + self.yWidth/2, num = pxPerLine)

        
#         xx, yy = np.meshgrid(x,y) 
#         xx, yy = rotate(xx, yy, xPivot=self.xCentre, yPivot=self.yCentre, rot=theta)
        
#         fig, ax = plt.subplots()
#         ax.pcolor(xx,yy,im)
#         return ax
        
    
class CheckSamplingGrid():
    
    def __init__(self, xAtomsVec, yAtomsVec, a, b, theta, gridDiameter, 
    gridPointsPerDiameter, gridAtomDiameterToAvoid, xTrackingAtom, yTrakingAtom):
        
        self.xAtomsVec = xAtomsVec
        self.yAtomsVec = yAtomsVec
        self.a = a
        self.b = b
        self.theta = theta
        self.gridDiameter = gridDiameter
        self.gridPointsPerDiameter = gridPointsPerDiameter
        self.gridAtomDiameterToAvoid = gridAtomDiameterToAvoid
        self.xt = xTrackingAtom
        self.yt = yTrakingAtom
        

    def RealAtomPos(self):
        
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
        
        x = np.array(self.xAtomsVec, dtype=float)
        y = np.array(self.yAtomsVec, dtype=float)
        
        x = self.a * x 
        y = self.b * y
        
        x, y = rotate(x, y, xPivot=x[0], yPivot=y[0], rot=self.theta)

        x = x + self.xt
        y = y + self.yt
        
        return x, y
    
    
    
    def Grid(self, xAtoms, yAtoms):
        
        xCentre = np.average(xAtoms)
        yCentre = np.average(yAtoms)
        
        sg = SamplingGrid()
    
        xGrid, yGrid = sg.CircleGrid(self.gridDiameter,
                                          self.gridPointsPerDiameter,
                                           xCentre, yCentre)
        
        xGrid, yGrid = sg.Rotate(xGrid, yGrid, xPivot=xCentre, yPivot=yCentre,
                                 rot=self.theta)
    
        xGrid, yGrid = sg.RemoveSamplingAboveAtoms(xGrid, yGrid, xAtoms, yAtoms,
                                                   self.gridAtomDiameterToAvoid)
        
        return xGrid, yGrid


path = r'D:\Results\2024\III-Vs\May 2024'
fileName = 'InSb(110)_0626.sxm'
sxm = Sxm()
im = sxm.ReadSxm(path, fileName)

xAtomsVec = np.array([0, 4, -2, 6, 0, 4], dtype=float)
yAtomsVec = np.array([0, 0, -5, -5, -10, -10], dtype=float)
# xAtomsVec = np.array([0, -3, 3, -4, 4, -3, 3, 0], dtype=float)
# yAtomsVec = np.array([0, -1, -1, -5, -5, -9, -9, -10], dtype=float)
# xAtomsVec = np.array([0, -2, -4, -5, -4, -2, 0, 2, 3, 2], dtype=float)
# yAtomsVec = np.array([0, -0, -2, -5, -8, -10, -10, -8, -5, -2], dtype=float)
a = 0.648e-9
b = 0.458e-9
theta = 2.4
gridDiameter = 16e-9
gridPointsPerDiameter = 40
gridAtomDiameterToAvoid = 2.6e-9

xTrackingAtom = -154.35e-9
yTrakingAtom = 33.453e-9

# sg = CheckSamplingGrid(xAtomsVec, yAtomsVec, a, b, theta, gridDiameter,
#                         gridPointsPerDiameter, gridAtomDiameterToAvoid,
#                         xTrackingAtom, yTrakingAtom)

# xAtoms, yAtoms = sg.RealAtomPos()
# xGrid, yGrid = sg.Grid(xAtoms, yAtoms)



# ax = sxm.PlotAtAngle(theta)

# ax.scatter(xGrid, yGrid)
# for i in range(len(xGrid)):
#     ax.annotate(i, (xGrid[i], yGrid[i]))

#%%
csg = CheckSamplingGrid(xAtomsVec, yAtomsVec, a, b, theta, gridDiameter, gridPointsPerDiameter, gridAtomDiameterToAvoid, xTrackingAtom, yTrakingAtom)

xAtoms, yAtoms = csg.RealAtomPos()

xCentre = np.average(xAtoms)
yCentre = np.average(yAtoms)

sg = SamplingGrid()
x0, y0 = sg.SquareGrid(gridDiameter, gridPointsPerDiameter, xCentre, yCentre)

x0, y0 = sg.Rotate(x0, y0, xPivot=xCentre, yPivot=yCentre,
                         rot=theta)


x, y = csg.Grid(xAtoms, yAtoms)

fig, ax = plt.subplots(figsize=(7,7))
ax.scatter(x0,y0,marker='.', color='red', alpha=0.3)
ax.scatter(xAtoms[0],yAtoms[0],marker='o', color='red', lw=7)
ax.scatter(x,y,marker='.', color='black')
ax.set_aspect('equal')
ax.set_axis_off()


path = r'D:\Results\2024\III-Vs\May 2024'
fileName = 'InSb(110)_0626.sxm'
sxm = Sxm()
im = sxm.ReadSxm(path, fileName)

fig, ax = plt.subplots(figsize=(7,7))
ax.imshow(im, extent=(sxm.xCentre - sxm.xWidth/2,
                      sxm.xCentre + sxm.xWidth/2,
                      sxm.yCentre - sxm.yWidth/2,
                      sxm.yCentre + sxm.yWidth/2) )
ax.set_aspect('equal')
# fig, ax = plt.subplots(figsize=(7,7))
ax = sxm.Plot(ax=ax)
# ax.scatter(xAtoms[0],yAtoms[0],marker='.', color='red', lw=7)
ax.set_aspect('equal')
# ax.set_axis_off()