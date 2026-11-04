import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# mpl.use('Qt5Agg')  # for testing purposes only

class NanoCoordTrans():

    def __init__(self, x, y):
        self.x = np.array(x)
        self.y = np.array(y)



    def rot(self, rot, x_pivot = None, y_pivot = None):
        if x_pivot is None: x_pivot = np.mean(self.x)
        if y_pivot is None: y_pivot = np.mean(self.y)

        rot = -np.deg2rad(rot)

        x = self.x - x_pivot
        y = self.y - y_pivot

        # carteesian -> polar
        r = np.sqrt(x ** 2 + y ** 2)
        theta = np.arctan2(y, x)  # element-wise arctan

        # rotation
        theta = theta + rot

        # polar -> carteesian
        x = (r * np.cos(theta)) + x_pivot
        y = (r * np.sin(theta)) + y_pivot

        self.x = x
        self.y = y

        return self.x, self.y



    def scale(self, x_scale_factor, y_scale_factor):
        self.x = x_scale_factor * self.x
        self.y = y_scale_factor * self.y

        return self.x, self.y



    def translate(self, x_delta, y_delta):
        self.x = self.x + x_delta
        self.y = self.y + y_delta

        return self.x, self.y



    def clip_rot(self, x_ref, y_ref, x_target, y_target, x_pivot=None, y_pivot=None, plot=False):
        x_ref_mean, y_ref_mean = np.mean(x_ref), np.mean(y_ref)

        x_r = np.array(x_ref) - x_ref_mean
        y_r = np.array(y_ref) - y_ref_mean
        x_t = np.array(x_target) - np.mean(x_target)
        y_t = np.array(y_target) - np.mean(y_target)

        theta = np.arctan2(x_t * y_r - y_t * x_r,
                           x_t * x_r + y_t * y_r)

        theta = np.rad2deg(theta)
        theta = np.mean(theta)

        self.rot(theta, x_pivot= x_pivot, y_pivot= y_pivot)

        if plot == True:
            fig, ax = plt.subplots(1,2, sharex=True, sharey=True)

            # before plot
            self.rot(-theta, x_pivot=x_pivot, y_pivot=y_pivot) # undo rot
            ax[0].set_title('before')
            ax[0].plot(self.x, self.y, c='gold', marker='.', label='xy', alpha=1)
            ax[0].plot(x_ref, y_ref, c='tab:red', marker='.', label='Ref points')
            ax[0].plot(x_target, y_target, c='black', marker='o', label='Target for ref points')

            # after plot
            ax[1].set_title('after')
            self.rot(theta, x_pivot=x_pivot, y_pivot=y_pivot)  # redo rot
            ax[1].plot(self.x, self.y, c='gold', marker='.', alpha=1)
            clip = NanoCoordTrans(x_ref, y_ref)
            clip.rot(theta) # apply same rot to ref points
            ax[1].plot(clip.x, clip.y, c='tab:red', marker='.')
            ax[1].plot(x_target, y_target, c='black', marker='o')

            ax[0].set_aspect('equal')
            ax[1].set_aspect('equal')
            plt.subplots_adjust(wspace=0)
            fig.legend()
            fig.suptitle('ClipRot')

            plt.show()



    def calc_polar(self, x_centre, y_centre):
        # transformation: origin @ nanonis origin -> origin @ adatom centre
        x_dash = self.x - x_centre
        y_dash = self.y - y_centre

        # evaluate r and theta
        r_sq = (x_dash ** 2) + (y_dash ** 2)
        r = np.sqrt(r_sq)
        theta = np.arctan2(y_dash, x_dash)  # element-wise arctan

        theta = np.rad2deg(theta)

        return r, theta



    def _get_angle(self, x, y, xPivot, yPivot, xRef=None, yRef=None):
        if xRef is None: xRef = xPivot
        if yRef is None: yRef = yPivot

        x = x - xPivot
        y = y - yPivot

        xRef -= xPivot
        yRef -= yPivot

        thetaRef = np.arctan2(yRef, xRef)
        theta = np.arctan2(y, x)

        return np.rad2deg(theta - thetaRef)





"""

sg = SamplingGrid(10, 0.5, 0.5)
mask = Mask(sg.x, sg.y).Circle(1)


sg.SnakeOrder(orientation='horizontal', startLR='left', startUD='down')


tcp = python_Nanonis_TCP()

def GetApex():
    # initialise atom traking
    tcp.AtomTrack_CtrlSet('modulation', 'on')  # Turn on Atom Tracking Modulation
    tcp.AtomTrack_CtrlSet('controller', 'on')  # Turn on Atom Tracking Controller
    print("check atom tracking is on")

    time.sleep(10)  # wait

    # end atom tracking
    tcp.AtomTrack_CtrlSet('controller', 'off')
    tcp.AtomTrack_CtrlSet('modulation', 'off')
    print("check atom tracking is off")
    time.sleep(0.5)  # wait

    return tcp.XY_tip_pos()  # update xt and yt to calibrated vals


yes = ['', 'yes', 'y', 'YES', 'Y']




x0 = [-2.5e-9, 2.5e-9]
y0 = [0, 0]

x1, y1 = [], []

if input('place tip near atom 0') in yes:
    x, y = GetApex()
    x1.append(x)
    y1.append(y)

if input('place tip near atom 1') in yes:
    x, y = GetApex()
    x1.append(x)
    y1.append(y)


tcp.close_socket()

path = r'data_simulator'
fileName = 'unnamed0014.sxm'
im = Sxm()
im.ReadSxm(path=path, fileName=fileName, channelName='Z_Fwd')

trans = NanoCoodTrans(sg.x, sg.y)

trans.Scale(1e-8, 1e-8)


fig, ax = trans.ClipRot(x0, y0, x1, y1, plot=True)

im.Plot(ax=ax[0], alpha=0.25, zorder=-1, cmap='gray')
im.Plot(ax=ax[1], alpha=0.25, zorder=-1, cmap='gray')
#ax.scatter(trans.x, trans.y)

plt.show()





x = [-1, 0, -1, -1]
y = [0, 0, 1, 0]

t = NanoCoodTrans(x, y)
t.Translate(2,1)
t.Rot(30, xPivot=np.mean(t.x), yPivot=np.mean(t.y))


fig, ax = plt.subplots()
ax.scatter(x, y)
ax.plot(x, y)
ax.scatter(np.mean(x), np.mean(y), c='black')

ax.scatter(t.x, t.y)
ax.plot(t.x, t.y)
ax.scatter(np.mean(t.x), np.mean(t.y), c='black')
plt.show()

xGoal, yGoal = t.x, t.y

error = 0.01*np.max(yGoal)
xGoalDelta = np.random.normal(scale=error, size=len(xGoal))
yGoalDelta = np.random.normal(scale=error, size=len(xGoal))
xGoal, yGoal = xGoal + xGoalDelta, yGoal + yGoalDelta

test = NanoCoodTrans(x, y)

test.Translate(np.mean(xGoal)-np.mean(test.x), np.mean(yGoal)-np.mean(test.y))

theta = test.ClipRot(xGoal, yGoal, test.x, test.y, plot=True)


print(theta)


xGoal0, yGoal0 = xGoal - np.mean(xGoal), yGoal - np.mean(yGoal)
x0, y0 = test.x - np.mean(test.x), test.y - np.mean(test.y)

print(xGoal0, yGoal0, x0, y0)
theta = np.arctan2(xGoal0*y0-yGoal0*x0, xGoal0*x0 + yGoal0*y0)
theta = np.rad2deg(theta)
print(theta)

"""