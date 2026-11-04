import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.colors import LinearSegmentedColormap
import math
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import lmfit
import traceback
from matplotlib.widgets import RangeSlider
from mpl_toolkits.axes_grid1 import make_axes_locatable
from typing import List, Optional
import os
import math
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
import io
from pptx.util import Inches
import sys

from analysis_spectrum import Spectrum
from coord_trans import NanoCoordTrans
from read_nano_files import MyFiles

#mpl.use('Qt5Agg')  # Returns interactive plots which I find useful when testing
    
class SpectraSet():
    
    def __init__(self, path: str, file_names: List[str], chrono_order: bool = True):
        """
        A list of Spectrum instances.
        Encapsulates a dataset, to be analysed / visualised together.
        @param path: path to spectra files
        @type path: str
        @param fileNames: spectra file names
        @type fileNames: list of str

        The list of file names must be ordered

        Note: The list's order is set by the order of fileNames provided.
        Every object in this class will follow this order.
        """
        
        self.file_names = file_names
        self.spectra = [Spectrum(path, f) for f in file_names]
        self.spectra_path = path
        self.set_spectra_attribute('outlier', np.zeros(len(self.spectra), dtype=bool))

        if chrono_order:
            if not MyFiles(path).are_file_names_ordered(file_names):
                raise ValueError('The file_names digits (XXXXX[digits].dat) are not in order, or some are missing.')
        


    def slice_spectra_attribute(self, attr_name: str) -> List:
        """
        List of the sliced attribute out of each Spectrum instance.
        @param attribute: attribute to be sliced
        @type attribute: str
        @return: sliced attribute, in order of self.spectra
        @rtype:ndarray
        """
        return np.array([getattr(s, attr_name, np.nan) for s in self.spectra])


    def set_spectra_attribute(self, attribute_name, attribute_vals):
        for ds, val in zip(self.spectra, attribute_vals):
            setattr(ds, attribute_name, val)


    def strip_outliers(self, test_attr_name, attr_min=None, attr_max=None):
        attr = self.slice_spectra_attribute(test_attr_name)
        outlier_mask = np.zeros(len(attr), dtype=bool)
        if attr_min is not None:
            outlier_mask[attr <= attr_min] = True
        if attr_max is not None:
            outlier_mask[attr >= attr_max] = True

        for s, is_out in zip(self.spectra, outlier_mask):
            if is_out:
                x, y, up = s.x_adj, s.y_adj, s.upsampling
                s.__dict__.clear() # Remove all attributes
                s.x_adj = x # Restore only x_pos, y_pos, upsampling for plotting purposes
                s.y_adj = y
                s.upsampling = up


    def read_channel(self, channel: str) -> List[List]:
        """
        A list of the channel's data from each Spectrum instance.
        Note: The channel's data is, itself, a list. So we form a list of lists.
        Note: The list's order is set by the order of fileNames provided.
        @param channel: channel name
        @type channel: str
        If the channel is not found, the available channels are
        printed. So, if you're not sure of the exact channel name, just
        type in nonsense.
        Note: channel = 'Index' is an option. May seem redundant, but may
        be useful in future to convert to a time channel, if we make
        note of sampling freq. TCP receiver sampling is limited to
        20 kHz. So, if measurement made using data logger through
        Matt's python_interface_nanonis.py, default is 20 kHz.
        Note: we may be able to play with TCP receiver to lower the 20kHz limit.
        @return: the channel's data, of each Spectrum instance.
        @rtype: list of lists
        """
        return [s.read_channel(channel) for s in self.spectra]



    def load_extra_info(self, path: str, extra_info: dict, chrono_order: bool = True):
        # if the info to be loaded is in chrono order, check the spectra is also chrono ordered, for correct appending
        if chrono_order:
            file_names = self.slice_spectra_attribute('file_name')
            if not MyFiles(self.spectra_path).are_file_names_ordered(file_names):
                raise ValueError('The file_names digits (XXXXX[digits].dat) are not in order, or some are missing.')

        for attribute_name, file_name in extra_info.items():
            info = np.load(os.path.join(path, file_name))

            if len(info) == 1: # assume it is an attribute of the whole data set, ie. of self
                setattr(self, attribute_name, info[0])

            elif len(info) == len(self.spectra): # assume each element in info, is an attribute of each individual spectrum
                for s, inf in zip(self.spectra, info):
                    setattr(s, attribute_name, inf)

            elif len(info) > len(self.spectra): # assume measurement is unfinished
                for i in range(len(self.spectra)):
                    setattr(self.spectra[i], attribute_name, info[i])
                    setattr(self, 'full_' + attribute_name, info) # ensure the info from the unfinished set of spec is not lost
            else:
                raise ValueError('shape of extra info is irregular, unsure of how to append')




    def load_drift_adj_info(self, path: str,
                            x_adj_file_name: Optional[str] = None,
                            y_adj_file_name: Optional[str] = None,
                            up_file_name: Optional[str] = None):

        if x_adj_file_name is None:
            # if x_adj file name not provided, assume it contains the substring 'x'
            x_adj_file_name = MyFiles(path).find_one_file(key_str='x')

        if y_adj_file_name is None:
            # if y_adj file name not provided, assume it contains the substring 'y'
            y_adj_file_name = MyFiles(path).find_one_file(key_str='y')

        if up_file_name is None:
            # if up file name not provided, assume it contains the substring 'up' or 'mask'
            # (sorry for my naming inconsistency here!)
            try:
                up_file_name = MyFiles(path).find_one_file(key_str='up')
            except FileNotFoundError:
                try:
                    up_file_name = MyFiles(path).find_one_file(key_str='mask')
                except FileNotFoundError:
                    pass # if no file found, assume there is no up-sampling

        extra_info = {'x_adj': x_adj_file_name,
                      'y_adj': y_adj_file_name}

        if up_file_name is not None:
            extra_info['upsampling'] = up_file_name
        else:
            for s in self.spectra:
                setattr(s, 'upsampling', None)

        self.load_extra_info(path, extra_info)



    def straighten_xy_adj(self):

        x = self.slice_spectra_attribute('x_adj')
        y = self.slice_spectra_attribute('y_adj')

        trans = NanoCoordTrans(x, y)
        # Calculate angles between adjacent points
        angles = trans._get_angle(x[:-1], y[:-1], x[1:], y[1:])
        angles = np.abs(angles)
        unique_angles, counts = np.unique(angles, return_counts=True)
        angle_adj = unique_angles[np.argmax(counts)]
        x, y = trans.rot(-angle_adj)

        for spec, xi, yi, in zip(self.spectra, x, y):
            spec.x_adj = xi
            spec.y_adj = yi



    def redefine_attribute_units(self, attribute: str, scale: float):
        for s in self.spectra:
            try:
                data = getattr(s, attribute)
                data = np.array(data) * scale
                setattr(s, attribute, data)
            except AttributeError:
                pass



    # ==================================================================================================================
    # Individual Analysis
    # ==================================================================================================================

    def kpfm_analysis(self, **kwargs):
        """
        Runs the KPFM analysis on each Spectrum instance.
        See Spectrum.KPFMAnalysis in spectrum_analysis.py for details.
        @param xAdatomCentre: x coordinate of the artificial atom's centre.
        @type xAdatomCentre: float, optional
        @param yAdatomCentre: y coordinate of the artificial atom's centre.
        @type yAdatomCentre: float, optional
        If xAdatomCentre and yAdatomCentre are specified, each spectrum's distance
        from the artificial atom's centre will be calculated and added as Spectrum.r (float).
        @param yChannel: Use if you want to analyse only one of the repeat sweeps e.g.
        'OC M1 Freq. Shift [00002] (Hz)'. Otherwise, the default is 'OC M1 Freq. Shift [AVG] (Hz)' or, in its absence,
        'OC M1 Freq. Shift (Hz)'.
        @type yChannel: str
        @return: The useful info from the analysis is added as attributes to each Spectrum instance.
        See Spectrum.KPFMAnalysis in spectrum_analysis.py for details.
        (Remember self is a list of Spectrum instances)
        """
        for s in self.spectra:
            s.kpfm_analysis(**kwargs)


    def iz_analysis(self, **kwargs):
        for s in self.spectra:
            s.iz_analysis(**kwargs)



    def didv_analysis(self, **kwargs):
        """
        Runs the dI/dV analysis on each Spectrum instance.
        See Spectrum.DIDVAnalysis in spectrum_analysis.py for details.
        @param peakType: accepts 'Gaussian' or 'Lorentzian'. Default is 'Gaussian'
        @type peakType: str, optional
        @param amplitudeGuess: peak amplitude guess
        @type amplitudeGuess: float, optional
        @param centerGuess: peak center guess
        @type centerGuess: float, optional
        @param sigmaGuess: peak sigma guess
        @type sigmaGuess: float, optional
        @param amplitudeMaxLim: restrict peak amplitude maximum limit
        @type amplitudeMaxLim: float, optional
        @param centerMaxLim: restrict peak center maximum limit
        @type centerMaxLim: float, optional
        @param sigmaMaxLim: restrict peak sigma maximum limit
        @type sigmaMaxLim: float, optional
        @param amplitudeMinLim: restrict peak amplitude minimum limit
        @type amplitudeMinLim: float, optional
        @param centerMinLim: restrict peak center minimum limit
        @type centerMinLim: float, optional
        @param sigmaMinLim: restrict peak sigma minimum limit
        @type sigmaMinLim: float, optional
        @param yChannel: specify the y channel, Otherwise, the default is 'LI Demod 1 Y [AVG] (A)' or, in its absence,
        'LI Demod 1 Y (A)'.
        @return: The useful info from the analysis is added as attributes to each Spectrum instance.
        See Spectrum.DIDVAnalysis in spectrum_analysis.py for details.
        (Remember self is a list of Spectrum instances)
        """
        for s in self.spectra:
            s.didv_analysis(**kwargs)



    def force_analysis(self, threshold, plot_calculation=False):
        """
        work in progress
        """

        for s in self.spectra:
            s.ForceAnalysis(threshold, plotCalculation=plot_calculation)
        self.redefine_attribute_units('z_av', 1e12)
        self.redefine_attribute_units('v_av', 1e3)
        z_0 = self.spectra[0].z_av
        for s in self.spectra:
            s.z_rel_av = s.z_av - z_0


    
     
    # ==================================================================================================================
    # Group Analysis
    # ==================================================================================================================

    def calc_polar_adj(self, x_centre: Optional[float] = None,
                             y_centre: Optional[float] = None):
        try:
            x = self.slice_spectra_attribute('x_adj')
            y = self.slice_spectra_attribute('y_adj')
        except Exception:
            raise ValueError('the drift adjusted info was not found. Load it using the load_drift_adj_info method.')

        if x_centre is None:
            x_centre = np.average(x)
        if y_centre is None:
            y_centre = np.average(y)

        r_adj, theta_adj = NanoCoordTrans(x, y).calc_polar(x_centre, y_centre)

        theta_adj = np.array(theta_adj)

        # I prefer theta to be 0 to 360, (rather than the default -180 to 180)
        theta_adj = np.where(theta_adj < 0, theta_adj + 360, theta_adj)

        for spec, r, th, in zip(self.spectra, r_adj, theta_adj):
            spec.r_adj = r
            spec.theta_adj = th

        return r_adj, theta_adj


    def calc_angle_xy_adj(self):
        try:
            x = self.slice_spectra_attribute('x_adj')
            y = self.slice_spectra_attribute('y_adj')
        except Exception:
            raise ValueError('the drift adjusted info was not found. Load it using the load_drift_adj_info method.')

        # Calculate angles between adjacent points
        angles = NanoCoordTrans(x,y)._get_angle(x[:-1], y[:-1], x[1:], y[1:])
        angles = np.abs(angles)
        angles = angles % 90

        # assume the modal angle is the sampling grid's angles
        unique_angles, counts = np.unique(angles, return_counts=True)
        angle_adj = unique_angles[np.argmax(counts)]

        if angle_adj > 45:
            angle_adj = 90 - angle_adj

        self.angle_xy_adj = angle_adj



    def idx_ascending_r_order(self, r = 'r_adj'):
        """Returns the indices that would sort the Spectrum instance list in ascending r order."""
        if type(r) is str:
            r = self.slice_spectra_attribute(r)
        return np.argsort(r)



    def subregion_mask(self, r_min = 0.0, r_max = np.inf, theta_min = 0.0, theta_max = 360.0):
        if 0 > theta_min or theta_min > 360: raise ValueError('0 > theta_min > 360')
        if 0 > theta_max or theta_max > 360: raise ValueError('0 > theta_max > 360')
        if r_min < 0 or r_max < 0: raise ValueError('r < 0')

        r = self.slice_spectra_attribute('r_adj')
        th = self.slice_spectra_attribute('theta_adj')

        return (r >= r_min) & (r <= r_max) & (th >= theta_min) & (th <= theta_max)


    def calc_cpd_norm(self):
        in_well_mask = self.subregion_mask(r_max=2.5)
        cpd = self.slice_spectra_attribute('vContact')

        cpd_in_well = cpd[in_well_mask]

        if not all(np.isnan(cpd_in_well)):
            cpd_0 = np.nanmean(cpd_in_well)
            cpd_norm = np.array(cpd) - cpd_0

        else:
            cpd_norm = cpd
            cpd_norm[:] = np.nan

        self.set_spectra_attribute('cpd_norm', cpd_norm)



    def calc_well_depth(self):
        in_well_mask = self.subregion_mask(r_max=2.5)
        inf_mask = self.subregion_mask(r_min=12)

        if any(in_well_mask) and any(inf_mask):
            cpd = self.slice_spectra_attribute('vContact')

            self.well_depth = np.nanmean(cpd[in_well_mask]) - np.nanmean(cpd[inf_mask])
        else:
            self.well_depth = np.nan



    def kpfm_point_charge_fit(self, r, cpd, D_guess=0.1, C_guess=-0.2, r0_guess=0.5):
        def our_model(r, D, C, r0):
            return D + (C) / (r - r0)
        
        params = lmfit.Parameters() # define a dict to store the fitting parameters
        params.add('D', value=D_guess) # define the fitting parameters and an initial guess for their value
        params.add('C', value=C_guess)
        params.add('r0', value=r0_guess)
        
        # specify which variables are independent (r) and which are fitting parameters (D, C, r0) in the model function (self._Vcontact_model)
        model = lmfit.Model(our_model, independent_vars='r', param_names=['D','C','r0'])
        
        # perform the fit
        fit_info =model.fit(data=cpd, params=params, r=r)
        fit = fit_info.best_fit
        
        # calculate the fit's confidence bound to 2 sigma, ie ~95%
        # fitConfBound = fitInfo.eval_uncertainty(params=fitInfo.params, sigma=2)

        self.kpfm_fit = fit
        self.kpfm_fit_param_D = fit_info.params['D']
        self.kpfm_fit_param_C = fit_info.params['C']
        self.kpfm_fit_param_r0 = fit_info.params['r0']




    # ==================================================================================================================
    # General Visualisation Methods
    # ==================================================================================================================

    def plot_spectra_pos(self, x='x_pos', y='y_pos', fig=None, ax=None, **kwargs):
        x = self.slice_spectra_attribute(x)
        y = self.slice_spectra_attribute(y)

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))

        ax.scatter(x, y, **kwargs)

        for i in range(len(x)):
            ax.annotate(str(i), (x[i], y[i]), fontsize='xx-small')

        ax.set_aspect('equal')
        ax.set_ylabel('y / m')
        ax.set_xlabel('x / m')
        
      
    def scatter_map(self, z, x='x_pos', y='y_pos', angle=0, x_pivot=None, y_pivot=None, ax=None, fig=None,
                    cmap='seismic', cmap_scale = 'linear', colourbar_label = None):
        
        if type(z) is str:
            if colourbar_label is None: colourbar_label = z
            z = self.slice_spectra_attribute(z)
        
        if type(x) is str: 
            x = self.slice_spectra_attribute(x)

        if type(y) is str: 
            y = self.slice_spectra_attribute(y)

        if angle != 0:
            if x_pivot is None: x_pivot = np.average(x)
            if y_pivot is None: y_pivot = np.average(y)
            x, y = NanoCoordTrans(x, y).rot(rot=-angle, x_pivot=x_pivot, y_pivot=y_pivot)

        if ax is None: fig, ax = plt.subplots()

        if cmap_scale == 'log': cmap_scale = colors.LogNorm()
        elif cmap_scale == 'linear': cmap_scale =colors.Normalize()
        else: raise ValueError('cmap_scale must be either "linear" or "log"')

        plot = ax.scatter(x, y, c=np.array(z), cmap=cmap, norm=cmap_scale,
                          marker="s", linewidths=3)

        fig.colorbar(plot, label=colourbar_label)
        ax.set_ylabel('y / m')
        ax.set_xlabel('x / m')



    def map_adj(self, z, ax=None, fig=None,
            cmap='spring', colourbarLabel=None, cmapScale='linear', cmapMin=None, cmapMax=None, scalebar=3e-9,
                cbar_max_text_color='black', cbar_min_text_color='white'):

        # slice data, if needed
        if type(z) is str:
            z = self.slice_spectra_attribute(z)


        if fig is None or ax is None:
            fig, ax = plt.subplots()

        try:
            x = self.full_x_adj
            y = self.full_y_adj
            upsampling = self.full_upsampling
            z_new = np.empty(len(x))
            z_new[:] = np.nan
            z_new[:len(z)] = z
            z = z_new
        except:
            x = self.slice_spectra_attribute('x_adj')
            y = self.slice_spectra_attribute('y_adj')
            upsampling = self.slice_spectra_attribute('upsampling')




        #if np.isnan(upsampling).all():
            #upsampling=None

        map = DriftCorrectedMap(z, x, y, angle=self.angle_xy_adj, xPivot=None, yPivot=None, mask=upsampling)

        map.Plot(fig=fig, ax=ax, cmapScale=cmapScale, cmap=cmap, colourbarLabel=colourbarLabel,
                 scalebar=scalebar, cmapMin=cmapMin, cmapMax=cmapMax,  cbar_max_text_color=cbar_max_text_color, cbar_min_text_color=cbar_min_text_color)


        outlier_cmap = LinearSegmentedColormap.from_list(
            'transparent2red',
            [(1, 0, 0, 0), (1, 0, 0, 1)])

        map = DriftCorrectedMap(np.isnan(z).astype(int), x, y, angle=self.angle_xy_adj, xPivot=None, yPivot=None, mask=upsampling)


        map.Plot(fig=fig, ax=ax, cmapScale=cmapScale, cmap=outlier_cmap, colourbarLabel=None,
                 scalebar=scalebar, cmapMin=0, cmapMax=1)





    # ==================================================================================================================
    # Specific Visualisation Methods
    # ==================================================================================================================

    def radial_plot(self, r='r_adj', y='vContact', fig=None, ax=None, fmt='k.', legend=False, x_label='', y_label='', **kwargs):

        if type(y) is str:
            y = self.slice_spectra_attribute(y)

        if type(r) is str:
            r = self.slice_spectra_attribute(r)

        if ax is None:
            fig, ax = plt.subplots()

        ax.plot(r, y, fmt, **kwargs)

        if legend:
            ax.legend(fontsize=7, loc='lower right')

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)


    def stacked_plot(self, x='x', y='y', ncols=1, fig=None, ax=None, alpha=1, xLabel='',
                    yLabel='', color='black', cmap=None):

        # form / format axes
        if ax is None:
            spectraPerAx = math.ceil(len(self.spectra) / ncols)
            nrows = spectraPerAx
            fig, ax = plt.subplots(nrows, ncols, sharex=True, sharey=True,
                                   frameon=False, figsize=(24, 9))
            for col in range(ncols):
                for row in range(nrows):
                    ax[row, col].axis('off')
        else:
            if type(ax) is np.ndarray:
                nrows, ncols = np.shape(ax)
            else:
                ncols = 1

        if ncols == 1: ax = [[ax]]

        # form colour range
        if cmap is not None:
            cmap = self._TuneCmap(ax[0, 0], cmap=cmap, Ndivisions=len(self.spectra),
                                  colourbarLabel='spectrum number',
                                  plotCBar=True, axForCBarPlot=ax[:, ncols - 1])

        # plot
        spectraCount = 0

        for col in range(ncols):
            for row in range(nrows):
                if spectraCount < len(self.spectra):

                    if cmap is not None: color = cmap(spectraCount)
                    try:
                        x_data = getattr(self.spectra[spectraCount], x)
                        y_data = getattr(self.spectra[spectraCount], y)
                        ax[row, col].plot(x_data, y_data,
                                          color=color,
                                          alpha=alpha)
                    except AttributeError:
                        pass


                    spectraCount = spectraCount + 1
                    print(spectraCount)

        plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, hspace=0)
        ax[0, 0].set_ylabel(yLabel)
        ax[0, 0].set_xlabel(xLabel)

        return fig, ax



    def my_force_plot(self, fontsize=6):
        dfFail = np.zeros(len(self.spectra)-1)
        dfWin = np.zeros(1)
        avDfFail = np.zeros(len(self.spectra)-1)
        avDfWin = np.zeros(1)

        sLast = self.spectra[-1]
        if any(sLast.eventMask) == True:
            win = True
            idxWin = np.argwhere(sLast.eventMask == True)[0][0] # index of first win
            dfWin[0] = sLast.y[int(idxWin)]
            avDfWin[0] = np.average(sLast.y[:idxWin+1])

        # check
        else:
            win = False
            print('no win found in last spectrum')
            idxWin = len(self.spectra[0].df)-5
            dfWin[0] = sLast.y[int(idxWin)]
            avDfWin[0] = np.average(sLast.y[:idxWin + 1])

        for i in range(len(dfFail)):
            s = self.spectra[i]
            dfFail[i] = s.y[idxWin]
            avDfFail[i] = np.average(s.y[:idxWin+1])

            # check
            if any(s.eventMask) == True:
                print('win found somewhere on a spectrum that is not the last')

        plt.style.use("dark_background")

        fig1, ax = plt.subplots(nrows=3, ncols=len(self.spectra), sharex=True, sharey='row', figsize=(16,4.5))
        for i in range(len(self.spectra)):
            if win == True:
                win_color='lime'
            else:
                win_color='darkslategray'

            ax[0, i].plot(self.spectra[i].y, '.', color='darkslategray', markersize=3)
            dfMask = np.copy(self.spectra[i].I)
            dfMask[~self.spectra[i].eventMask] = np.nan
            ax[1, i].plot(dfMask, '.', color=win_color, markersize=3, label='right b4 move')
            ax[1, i].plot(self.spectra[i].I, '.', color='darkslategray', markersize=3)
            IMask = np.copy(self.spectra[i].I)
            IMask[~self.spectra[i].eventMask] = np.nan
            ax[1, i].plot(IMask, '.', color=win_color, markersize=3, label='right b4 move')
            ax[2, i].plot(self.spectra[i].absDI, '.', color='darkslategray', markersize=3)
            ax[2, i].plot([0, len(IMask)],
                          [self.spectra[i].threshold, self.spectra[i].threshold], '--', color='white', lw=1)
            absDIMask = np.copy(self.spectra[i].absDI)
            absDIMask[~self.spectra[i].eventMask] = np.nan
            ax[2, i].plot(absDIMask, '.', color=win_color, markersize=3, label='right b4 move')
            ax[0, i].tick_params(axis='both', which='major', labelsize=fontsize)
            ax[1, i].tick_params(axis='both', which='major', labelsize=fontsize)
            ax[2, i].tick_params(axis='both', which='major', labelsize=fontsize)


        for i in range(3):
            yMin, yMax = ax[i, -1].get_ylim()
            ax[i, -1].plot([idxWin + 1, idxWin + 1],
                          [yMin, yMax], '--', color=win_color, lw=1)


        for i in range(len(self.spectra)):
            if i == 0:
                ax[0, i].set_title(
                    'z (pm) = ' + '{:.0f}'.format(self.spectra[i].z_rel_av) + '\n v (mV) = ' + '{:.0f}'.format(
                        self.spectra[i].v_av),
                    fontsize=fontsize, loc='right')

            else:
                ax[0, i].set_title(
                    '{:.0f}'.format(self.spectra[i].z_rel_av) + '\n' + '{:.0f}'.format(
                        self.spectra[i].v_av),
                    fontsize=fontsize)


        ax[2, -1].set_xticks([idxWin + 1])
        ax[0, 0].set_ylabel('df (Hz)', fontsize=fontsize)
        ax[1, 0].set_ylabel('I (A)', fontsize=fontsize)
        ax[2, 0].set_ylabel('abs(dI) (A)', fontsize=fontsize)
        ax[2, 1].set_xlabel('Index, up to move (+1 to see dI jump)', fontsize=fontsize)

        plt.tight_layout()
        plt.subplots_adjust(wspace=0, hspace=0)



        fig2, ax = plt.subplots(1,2, figsize=(12,4.5))

        ax[0].plot(np.append(dfFail, dfWin), marker='.', color='darkslategray', markersize=5, label='unsuccessful')
        ax[0].plot([len(dfFail)], dfWin, '.', color=win_color, markersize=5, label='successful')
        ax[0].set_xlabel('attempt')
        ax[0].set_ylabel('df (Hz) at position of first move')
        ax2 = ax[0].twinx()  # instantiate a second Axes that shares the same x-axis
        ax2.plot(np.append(dfFail, dfWin)-dfFail[0], marker='.', color='darkslategray', markersize=5)
        ax2.plot([len(dfFail)], dfWin-dfFail[0], '.', color=win_color, markersize=5)
        ax2.set_ylabel(r'$\Delta$ df (Hz) at position of first move')
        ax[0].grid(alpha=0.2, axis='y')
        ax2.grid(alpha=0.2, axis='y')
        #fig.subplots_adjust(bottom=0.2, right=0.8)
        #fig.legend(loc='lower center', bbox_transform=fig.transFigure, ncol=2)


        #fig, ax = plt.subplots()
        ax[1].plot(np.append(avDfFail, avDfWin), marker='.', color='darkslategray', markersize=5, label='unsuccessful')
        ax[1].plot(len(avDfFail), avDfWin, '.', color=win_color, markersize=5, label='successful')
        ax[1].set_xlabel('attempt')
        ax[1].set_ylabel('average df (Hz) up to the position of first move')
        ax2 = ax[1].twinx()  # instantiate a second Axes that shares the same x-axis
        ax2.plot(np.append(avDfFail, avDfWin)-avDfFail[0], marker='.', color='darkslategray', markersize=5)
        ax2.plot(len(avDfFail), avDfWin-avDfFail[0], '.', color=win_color, markersize=5)
        ax2.set_ylabel(r'$\Delta$ average df (Hz) up to the position of first move')
        ax[1].grid(alpha=0.2, axis='y')
        ax2.grid(alpha=0.2, axis='y')
        #fig.subplots_adjust(bottom=0.2, right=0.8)
        #fig.legend(loc='lower center', bbox_transform=fig.transFigure, ncol=2)
        plt.tight_layout()
        return fig1, fig2




class PlotStacked():

    def __init__(self, x='x', y='y', ncols=1, ax=None, alpha=1, xLabel='',
                    yLabel='', color='black', cmap=None):
        # form / format axes
        if ax is None:
            spectraPerAx = math.ceil(len(self.spectraData) / ncols)
            nrows = spectraPerAx
            fig, ax = plt.subplots(nrows, ncols, sharex=True, sharey=True,
                                   frameon=False, figsize=(24, 9))
            for col in range(ncols):
                for row in range(nrows):
                    ax[row, col].axis('off')
        else:
            if type(ax) is np.ndarray:
                nrows, ncols = np.shape(ax)
            else:
                ncols = 1

        if ncols == 1: ax = [[ax]]

        # form colour range
        if cmap is not None:
            cmap = self._TuneCmap(ax[0, 0], cmap=cmap, Ndivisions=len(self.spectraData),
                                  colourbarLabel='spectrum number',
                                  plotCBar=True, axForCBarPlot=ax[:, ncols - 1])

        # plot
        spectraCount = 0

        for col in range(ncols):
            for row in range(nrows):
                if spectraCount < len(self.spectraData):

                    if cmap is not None: color = cmap(spectraCount)
                    ax[row, col].plot(x[spectraCount], y[spectraCount],
                                      color=color,
                                      alpha=alpha)

                    spectraCount = spectraCount + 1
                    print(spectraCount)

        plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, hspace=0)
        ax[0, 0].set_ylabel(yLabel)
        ax[0, 0].set_xlabel(xLabel)


    def _TuneCmap(self, ax, cmap='gist_rainbow', minVal=None, maxVal=None,
                 nDivisions=None, cmapScale='linear',
                 colourbarLabel=None, plotCBar=False, location='right',
                 shrink=0, axForCBarPlot=None):

        if nDivisions is None: Ndivisions = len(self.spectraData)
        if minVal is None: minVal = 0
        if maxVal is None: maxVal = Ndivisions

        if cmapScale == 'linear':
            scale = np.linspace(minVal, maxVal, nDivisions)
        if cmapScale == 'log':
            scale = np.logspace(minVal, maxVal, nDivisions)

        dummy_xyz = [minVal, maxVal]
        dummie_cax = ax.scatter(dummy_xyz, dummy_xyz, dummy_xyz,
                                cmap=cmap, norm=cmapScale)
        ax.cla()
        cmap = plt.get_cmap(cmap)

        if plotCBar == True:
            plt.colorbar(dummie_cax, label=colourbarLabel, location=location,
                         shrink=shrink, ax=axForCBarPlot)

        return cmap, scale





class DriftCorrectedMap():


    def __init__(self, z, x, y, angle=0, xPivot=None, yPivot=None, mask=None):
        if angle != 0:
            if xPivot is None: xPivot = np.average(x)
            if yPivot is None: yPivot = np.average(y)
            x, y = NanoCoordTrans(x, y).rot(-angle, xPivot, yPivot) # note: - angle bc. we want to rotate back to 0 deg.

        self.z = np.array(z)
        self.x = np.array(x)
        self.y = np.array(y)


        if mask is not None:
            mask = np.array(mask)
        self.mask = mask


    def CoordTransform(self, z, x, y):
        def _sig_fig_round(x, p):
            x = np.asarray(x)
            x_positive = np.where(np.isfinite(x) & (x != 0), np.abs(x), 10**(p-1))
            mags = 10 ** (p - 1 - np.floor(np.log10(x_positive)))
            return np.round(x * mags) / mags

        x, y = _sig_fig_round(x, 10), _sig_fig_round(y, 10)
        # remove duplicates
        xDash = list(set(x))
        yDash = list(set(y))

        xDash.sort()
        yDash.sort()

        zDash = np.empty((len(xDash), len(yDash)))
        zDash.fill(np.nan)

        for s in range(len(z)):
            i = xDash.index(x[s])
            j = yDash.index(y[s])
            zDash[i, j] = z[s]

        zDash = np.array(zDash).T
        xDash, yDash = np.array(xDash), np.array(yDash)

        xDash = xDash - np.average(xDash)
        yDash = yDash - np.average(yDash)

        return zDash, xDash, yDash



    def DrawOutline(self, ax, z, x, y):

        z = np.isnan(z)  # bool array, if True at nan, false otherwise

        # upsample using Kronecker product so outline follows square px edges
        z = np.kron(z, np.ones((100, 100), dtype=bool))

        # edge border
        z[0, :], z[-1, :], z[:, 0], z[:, -1] = True, True, True, True

        # upsample x
        xStep = x[1] - x[0]
        x = np.linspace(min(x) - xStep / 2, max(x) + xStep / 2, num=len(x) * 100)
        # upsample y
        yStep = y[1] - y[0]
        y = np.linspace(min(y) - yStep / 2, max(y) + yStep / 2, num=len(y) * 100)

        ax.contour(x, y, z, linewidths=3, colors='black', zorder=-1) # zorder = -1 to send to the very back


    def CMapSlider(self, fig, plot, z, cmapScale, ax):

        # Create the RangeSlider
        #divider = make_axes_locatable(ax)
        #slider_ax = divider.append_axes("right", size="7%", pad=0.2)  # size controls width

        slider_ax = plt.axes([0.95, 0.4, 0.03, 0.2])

        slider = RangeSlider(ax=slider_ax, label="", valmin=np.nanmin(z), valmax=np.nanmax(z), valinit=(np.nanmin(z), np.nanmax(z)),
                             orientation="vertical", facecolor='darkgray',
                                track_color='lightgray')

        for handle in slider._handles:
            handle.set_color('black')
            handle.set_marker('_')
            handle.set_markersize(15)

        slider.valtext.set_visible(False)

        def update(val):
            # Update the image's colormap
            plot.norm.vmin = val[0]
            plot.norm.vmax = val[1]


            self.cbar_min_txt.set_text(f"{val[0]:.2f}")
            self.cbar_max_txt.set_text(f"{val[1]:.2f}")

            self.cbar.update_normal(plot)  # Force colorbar to update
            # Redraw the figure to ensure it updates
            fig.canvas.draw_idle()


        slider.on_changed(update)
        plt.show()


    def CBar(self, plot, ax, colourbarLabel, cmapMax, cmapMin, cmapScale, cax=None, max_color='black', min_color='white'):
        if cax is None:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="10.5%", pad=0.05)  # size controls width

        cbar = plt.colorbar(plot, cax=cax)
        self.cbar=cbar

        cbar.ax.text(2, 0.5, colourbarLabel, rotation='vertical', horizontalalignment='center',
                     verticalalignment='center', fontsize=15,
                     color='black', transform=cbar.ax.transAxes)

        """
        self.cbar_max_txt = cbar.ax.text(0.55, 0.95, f'{cmapMax:.2f}', rotation='vertical', horizontalalignment='center',
                     verticalalignment='top', fontsize=10,
                     color=max_color,
                     transform=cbar.ax.transAxes)
        self.cbar_min_txt = cbar.ax.text(0.55, 0.05, f'{cmapMin:.2f}', rotation='vertical', horizontalalignment='center',
                     verticalalignment='bottom', fontsize=10,
                     color=min_color,
                     transform=cbar.ax.transAxes)
        """

        self.cbar_max_txt = cbar.ax.text(0.55, 0.9, 'High', rotation='vertical', horizontalalignment='center',
                         verticalalignment='center', fontsize=12,
                         color='white', transform=cbar.ax.transAxes)
        self.cbar_min_txt = cbar.ax.text(0.55, 0.1, 'Low', rotation='vertical', horizontalalignment='center',
                         verticalalignment='center', fontsize=12,
                         color='black', transform=cbar.ax.transAxes)


        cbar.ax.set_ylabel(colourbarLabel)

        if cmapScale == 'log':
            cbar.ax.text(0.55, 0.5, 'log', rotation='vertical', horizontalalignment='center',
                         verticalalignment='center', fontsize=10,
                         color='white', weight='bold', transform=cbar.ax.transAxes)
        cbar.ax.set_axis_off()


    def Plot(self, fig=None, ax=None, cmapScale='linear', cmap='spring', colourbarLabel=None,
             scalebar=None, cmapMin=None, cmapMax=None, cbar_max_text_color='black', cbar_min_text_color='white',):
        if ax is None: fig, ax = plt.subplots(figsize=(4,5))

        cmap = mpl.cm.get_cmap(cmap).copy()
        cmap.set_under('red')
        cmap.set_over('red')

        if cmapMin is None: cmapMin = np.nanmin(self.z)
        if cmapMax is None: cmapMax = np.nanmax(self.z)



        if cmapScale == 'log': cmapNorm = colors.LogNorm(vmin=cmapMin,
                                                          vmax=cmapMax, clip=False)
        if cmapScale == 'linear': cmapNorm = colors.Normalize(vmin=cmapMin,
                                                               vmax=cmapMax, clip=False)

        if self.mask is None:
            z, x, y = self.CoordTransform(self.z, self.x, self.y)
            plot = ax.pcolormesh(x, y, z, norm=cmapNorm, cmap=cmap)

            self.DrawOutline(ax, z, x, y)


        else:
            for i in set(self.mask):
                m = np.isin(self.mask, i)
                zm, xm, ym = self.z[m], self.x[m], self.y[m]

                zm, xm, ym = self.CoordTransform(zm, xm, ym)

                plot = ax.pcolormesh(xm, ym, zm, norm=cmapNorm, cmap=cmap)

                self.DrawOutline(ax, zm, xm, ym)


        if scalebar is not None:
            scalebar = AnchoredSizeBar(ax.transData,
                                       size=scalebar, label=str(scalebar)+' nm', loc='lower right',
                                       frameon=False, color='black',
                                       pad=1)
            ax.add_artist(scalebar)


        ax.set_axis_off()
        ax.use_sticky_edges = False
        ax.margins(0.05)
        ax.set_aspect('equal')
        
        if colourbarLabel is not None:
            print(cbar_max_text_color)
            self.CBar(plot, ax, colourbarLabel, cmapMax, cmapMin, cmapScale, max_color=cbar_max_text_color,
                      min_color=cbar_min_text_color)


        self.CMapSlider(fig, plot, self.z, cmapScale, ax)







#%%

class GroupKPFMAnalysis():
    def __init__(self, all_paths_to_data, all_paths_to_info, all_file_names_data):
        self.datasets = []
        self.all_paths_to_data, self.all_paths_to_info, self.all_file_names_data = all_paths_to_data, all_paths_to_info, all_file_names_data


    def didv(self):
        for path_to_data, path_to_info, file_names_data in zip(all_paths_to_data, all_paths_to_info,
                                                               all_file_names_data):
            ds = SpectraSet(path=path_to_data, file_names=file_names_data, chrono_order=False)
            self.datasets.append(ds)
            print('data set loaded')
            ds.didv_analysis(yChannel='Current [AVG] (A)')
            ds.load_drift_adj_info(path_to_info)
            ds.redefine_attribute_units('x_adj', 1e9)
            ds.redefine_attribute_units('y_adj', 1e9)
            ds.calc_polar_adj()  # assumes centre is at grid's mean xy
            ds.calc_angle_xy_adj()

    def kpfm(self):
        for path_to_data, path_to_info, file_names_data in zip(self.all_paths_to_data, self.all_paths_to_info,
                                                               self.all_file_names_data):
            ds = SpectraSet(path=path_to_data, file_names=file_names_data, chrono_order=True)
            self.datasets.append(ds)
            print('data set loaded')

            ds.kpfm_analysis()  # loads the 'OC M1 Freq. Shift [AVG] (Hz)' channel
            ds.load_drift_adj_info(path_to_info)
            ds.redefine_attribute_units('x_adj', 1e9)
            ds.redefine_attribute_units('y_adj', 1e9)
            ds.calc_polar_adj()  # assumes centre is at grid's mean xy
            ds.calc_angle_xy_adj()

            print('data set base analysis done')

    def set_dataset_attribute(self, attribute_name, attribute_vals):
        for ds, val in zip(self.datasets, attribute_vals):
            setattr(ds, attribute_name, val)

    def strip_outliers(self, test_attr_name, attr_min=None, attr_max=None):
        for ds in self.datasets:
            ds.strip_outliers(test_attr_name, attr_min, attr_max)


    def calc_cpd_norm(self):
        for ds in self.datasets:
            ds.calc_cpd_norm()

    def calc_well_depth(self):
        for ds in self.datasets:
            ds.calc_well_depth()

    def my_avg_plot_format(self, x_attribute_name, y_attribute_name, fig=None, ax=None,
                       xlabel=None, ylabel=None, fmt='.', **kwargs):

        if fig is None or ax is None:
            fig, ax = plt.subplot_mosaic([['6', '6n8'], ['8', '6n8']], sharey=True, sharex=True, figsize=(10, 7))

        for ds in self.datasets:

            color = 'k'
            if hasattr(ds, 'well_type') and hasattr(ds, 'era'):
                if ds.well_type == 6:
                    if ds.era == 0:
                        color = '#165580'

                    elif ds.era == 1:
                        color = '#1F77B4'

                    elif ds.era == 2:
                        color = '#48A2E0'

                if ds.well_type == 8:
                    if ds.era == 0:
                        color = '#B05400'

                    elif ds.era == 1:
                        color = '#FF810E'

                    elif ds.era == 2:
                        color = '#FFAA5D'

            try:
                x = getattr(ds, x_attribute_name)
            except:
                try:
                    x = ds.slice_spectra_attribute(x_attribute_name)
                except:
                    raise AttributeError(x_attribute_name + ' not found')

            try:
                y = getattr(ds, y_attribute_name)  # attributes of 1 whole Spectra Set
            except:
                try:
                    y = ds.slice_spectra_attribute(y_attribute_name)  # attributes of each Spectrum instance
                except:
                    raise AttributeError(y_attribute_name + ' not found')

            ax['6n8'].plot(x, y, fmt, color=color, **kwargs)
            print(ds.spectra_path)
            if ds.well_type == 6:
                ax['6'].plot(x, y, fmt, color=color, **kwargs)
            elif ds.well_type == 8:
                ax['8'].plot(x, y, fmt, color=color, **kwargs)
            else:
                raise ValueError('Well type not supported')

        for axi in fig.axes:
            axi.yaxis.minorticks_on()
            axi.grid(which='major', color='k', alpha=0.3)
            axi.grid(which='minor', color='k', alpha=0.1)
            axi.yaxis.set_tick_params(labelbottom=True)

        if xlabel is not None:
            fig.supxlabel(xlabel)
        if ylabel is not None:
            fig.supylabel(ylabel)

        plt.subplots_adjust(hspace=0)

        return fig, ax

    def my_hist_format(self, attribute_name, bins=25, xlabel=None, ylabel='Count', **kwargs):
        fig, ax = plt.subplots()

        data = [getattr(ds, attribute_name) for ds in self.datasets]
        try:
            well_type = [ds.well_type for ds in self.datasets]
            era = [ds.era for ds in self.datasets]

            data, well_type, era = np.array(data), np.array(well_type), np.array(era)

            ax.hist([data[np.logical_and(well_type == 6, era == 0)],
                     data[np.logical_and(well_type == 6, era == 1)],
                     data[np.logical_and(well_type == 6, era == 2)],
                     data[np.logical_and(well_type == 8, era == 0)],
                     data[np.logical_and(well_type == 8, era == 1)],
                     data[np.logical_and(well_type == 8, era == 2)]],
                    bins=bins,
                    color=['#165580', '#1F77B4', '#48A2E0', '#B05400', '#FF810E', '#FFAA5D'],
                    edgecolor='black', stacked=True, alpha=1, **kwargs)

        except:
            ax.hist(data, bins=bins, color='gray', edgecolor='black', alpha=1, **kwargs)

        if xlabel is not None:
            ax.set_xlabel(xlabel)
        if ylabel is not None:
            ax.set_ylabel(ylabel)

    def my_subplot_format(self, x=None, y=None, z=None, fig=None, ax=None, xlabel=None, ylabel=None,
                          grid=True, ticks=True, xlim=None, ylim=None, fmt='k.', ms=3, shared_cmap=False,
                          cmap_min=None, cmap_max=None, **kwargs):
        num = len(self.datasets)

        if z is not None and shared_cmap:
            foo=[]
            for d in num:
                foo.extend(d.slice_spectra_attribute('z'))
            cmap_min = min(foo)
            cmap_max = max(foo)
        elif z is not None and not shared_cmap:
            cmap_min = None
            cmap_max = None


        if fig is None or ax is None:
            num_subplots = math.ceil(math.sqrt(num)) ** 2
            num_rows = int(math.sqrt(num_subplots))
            num_cols = num_rows

            fig, ax = plt.subplots(num_rows, num_cols, figsize=(9, 9), sharex=True, sharey=True)
            print(num, num_subplots, num_rows, num_cols)
        else:
            num_rows, num_cols = np.shape(ax)
            num_subplots = num_rows * num_cols

        for i in range(num_subplots):
            row = int(math.floor(i / num_cols))
            col = int(i - row * num_cols)
            print(row, col)

            if i < num:
                if z is None and x is not None and y is not None:
                    ax[row, col].plot(x[i], y[i], fmt, ms=ms, **kwargs)
                if z is not None and x is None and y is None:
                    self.datasets[i].map_adj(z, ax=ax[row, col], fig=fig, **kwargs)
                else:
                    raise ValueError(
                        'input either x and y (not z) for a 2D plot, or just z (str) for a 3D color map.')

                spine_color = 'k'
                if hasattr(self.datasets[i], 'well_type') and hasattr(self.datasets[i], 'era'):
                    if self.datasets[i].well_type == 6:
                        if self.datasets[i].era == 0:
                            spine_color = '#165580'
                        elif self.datasets[i].era == 1:
                            spine_color = '#1F77B4'
                        elif self.datasets[i].era == 2:
                            spine_color = '#48A2E0'

                    if self.datasets[i].well_type == 8:
                        if self.datasets[i].era == 0:
                            spine_color = '#B05400'
                        elif self.datasets[i].era == 1:
                            spine_color = '#FF810E'
                        elif self.datasets[i].era == 2:
                            spine_color = '#FFAA5D'

                ax[row, col].spines[['left', 'right', 'bottom', 'top']].set_color(spine_color)
                ax[row, col].spines[['left', 'right', 'bottom', 'top']].set_linewidth(2)

                if grid:
                    ax[row, col].yaxis.minorticks_on()
                    ax[row, col].grid(which='major', color=spine_color, alpha=0.3)
                    ax[row, col].grid(which='minor', color=spine_color, alpha=0.1)
                    ax[row, col].tick_params(which='both', colors=spine_color)

                if not ticks:
                    ax[row, col].set_xticks([])
                    ax[row, col].set_yticks([])

                if xlim is not None:
                    ax[row, col].set_xlim(xlim[0], xlim[1])
                if ylim is not None:
                    ax[row, col].set_ylim(ylim[0], ylim[1])

            else:
                ax[row, col].axis('off')

        if xlabel is not None:
            ax[-1, 0].set_xlabel(xlabel)
        if ylabel is not None:
            ax[-1, 0].set_ylabel(ylabel)

        plt.subplots_adjust(wspace=0.1, hspace=0.1)

        return fig, ax

    def my_plot_format(self, x_attribute_name, y_attribute_name, fig=None, ax=None,
                       xlabel=None, ylabel=None, fmt='.', **kwargs):

        if fig is None or ax is None:
            fig, ax = plt.subplot_mosaic([['6','6n8'],['8','6n8']], sharey=True, sharex=True, figsize=(10,7))

        for ds in self.datasets:

            color = 'k'
            if hasattr(ds, 'well_type') and hasattr(ds, 'era'):
                if ds.well_type == 6:
                    if ds.era == 0:
                        color = '#165580'

                    elif ds.era == 1:
                        color = '#1F77B4'

                    elif ds.era == 2:
                        color = '#48A2E0'
                    else:
                        color = '#1F77B4'


                if ds.well_type == 8:
                    if ds.era == 0:
                      color = '#B05400'

                    elif ds.era == 1:
                        color = '#FF810E'

                    elif ds.era == 2:
                        color = '#FFAA5D'
                    else:
                        color = '#FF810E'

            try:
                x = getattr(ds, x_attribute_name)
            except:
                try:
                    x = ds.slice_spectra_attribute(x_attribute_name)
                except:
                    raise AttributeError(x_attribute_name + ' not found')

            try:
                y = getattr(ds, y_attribute_name) # attributes of 1 whole Spectra Set
            except:
                try:
                    y = ds.slice_spectra_attribute(y_attribute_name) # attributes of each Spectrum instance
                except:
                    raise AttributeError(y_attribute_name + ' not found')


            ax['6n8'].plot(x, y, fmt, color=color, **kwargs)
            print(ds.spectra_path)
            if ds.well_type == 6:
                ax['6'].plot(x, y, fmt, color=color, **kwargs)
            elif ds.well_type == 8:
                ax['8'].plot(x, y, fmt, color=color, **kwargs)
            else:
                raise ValueError('Well type not supported')

        for axi in fig.axes:
            axi.yaxis.minorticks_on()
            axi.grid(which='major', color='k', alpha=0.3)
            axi.grid(which='minor', color='k', alpha=0.1)
            axi.yaxis.set_tick_params(labelbottom=True)

        if xlabel is not None:
            fig.supxlabel(xlabel)
        if ylabel is not None:
            fig.supylabel(ylabel)

        plt.subplots_adjust(hspace=0)

        return fig, ax



class AllData():
    def __init__(self, all_paths_to_data, all_paths_to_info, all_file_names_data):
        self.datasets = []

        for path_to_data, path_to_info, file_names_data in zip(all_paths_to_data, all_paths_to_info, all_file_names_data):
            ds = self._load_dataset(path_to_data, file_names_data)
            self.datasets.append(ds)
            self.base_analysis(ds, path_to_info)



    def _load_dataset(self, path_to_data, file_names_data):
        ds = SpectraSet(path=path_to_data, file_names=file_names_data, chrono_order=True)

        print('data set loaded')

        return ds



    def base_analysis(self, dataset, path_to_info):
        dataset.kpfm_analysis()  # loads the 'OC M1 Freq. Shift [AVG] (Hz)' channel

        dataset.load_drift_adj_info(path_to_info)
        dataset.redefine_attribute_units('x_adj', 1e9)
        dataset.redefine_attribute_units('y_adj', 1e9)
        dataset.calc_polar_adj()  # assumes centre is at grid's mean xy
        dataset.calc_angle_xy_adj()

        print('data set base analysis done')



    def bin_cpd_norm(self, rmin=0, rmax=12, rstep=1):
        r_binned = np.arange(rmin+rstep/2, rmax, rstep)
        cpd_norm_binned_6 = []
        cpd_norm_binned_8 = []

        for ds in self.datasets:
            ds.r_binned = r_binned
            ds.cpd_norm_binned = []

            for r in r_binned:
                bin_mask = ds.subregion_mask(r_min=r-rstep/2, r_max=r+rstep/2)

                if any(bin_mask):
                    cpd_norm_in_bin = ds.cpd_norm[bin_mask]
                    ds.cpd_norm_binned.append(np.average(cpd_norm_in_bin))

                else:
                    ds.cpd_norm_binned.append(np.nan)
            if ds.well_type == 6:
                cpd_norm_binned_6.append(ds.cpd_norm_binned)
            elif ds.well_type == 8:
                cpd_norm_binned_8.append(ds.cpd_norm_binned)
            else:
                raise ValueError('Well type not recognized')

        self.cpd_norm_binned_6_avg = np.nanmean(cpd_norm_binned_6, axis=0)
        self.cpd_norm_binned_8_avg = np.nanmean(cpd_norm_binned_8, axis=0)
        self.cpd_norm_binned_6_weight = np.sum(~np.isnan(cpd_norm_binned_6), axis=0)
        self.cpd_norm_binned_8_weight = np.sum(~np.isnan(cpd_norm_binned_8), axis=0)
        self.r_binned_avg = r_binned


    def set_dataset_attribute(self, attribute_name, attribute_vals):
        for ds, val in zip(self.datasets, attribute_vals):
            setattr(ds, attribute_name, val)

    def calc_cpd_norm(self):
        for ds in self.datasets:
            ds.calc_cpd_norm()

    def calc_well_depth(self):
        for ds in self.datasets:
            ds.calc_well_depth()


    def my_plot_format(self, x_attribute_name, y_attribute_name, fig=None, ax=None,
                       xlabel=None, ylabel=None, fmt='.', **kwargs):

        if fig is None or ax is None:
            fig, ax = plt.subplot_mosaic([['6','6n8'],['8','6n8']], sharey=True, sharex=True, figsize=(10,7))

        for ds in self.datasets:

            color = 'k'
            if hasattr(ds, 'well_type'):
                if ds.well_type == 6:
                    color = '#1F77B4'
                    if hasattr(ds, 'era'):
                        if ds.era == 0:
                            color = '#165580'

                        elif ds.era == 1:
                            color = '#1F77B4'

                        elif ds.era == 2:
                            color = '#48A2E0'




                if ds.well_type == 8:
                    color = '#FF810E'
                    if hasattr(ds, 'era'):
                        if ds.era == 0:
                          color = '#B05400'

                        elif ds.era == 1:
                            color = '#FF810E'

                        elif ds.era == 2:
                            color = '#FFAA5D'


            try:
                x = getattr(ds, x_attribute_name)
            except:
                try:
                    x = ds.slice_spectra_attribute(x_attribute_name)
                except:
                    raise AttributeError(x_attribute_name + ' not found')

            try:
                y = getattr(ds, y_attribute_name) # attributes of 1 whole Spectra Set
            except:
                try:
                    y = ds.slice_spectra_attribute(y_attribute_name) # attributes of each Spectrum instance
                except:
                    raise AttributeError(y_attribute_name + ' not found')


            ax['6n8'].plot(x, y, fmt, color=color, **kwargs)
            print(ds.spectra_path)
            if ds.well_type == 6:
                ax['6'].plot(x, y, fmt, color=color, **kwargs)
            elif ds.well_type == 8:
                ax['8'].plot(x, y, fmt, color=color, **kwargs)
            else:
                raise ValueError('Well type not supported')

        for axi in fig.axes:
            axi.yaxis.minorticks_on()
            axi.grid(which='major', color='k', alpha=0.3)
            axi.grid(which='minor', color='k', alpha=0.1)
            axi.yaxis.set_tick_params(labelbottom=True)

        if xlabel is not None:
            fig.supxlabel(xlabel)
        if ylabel is not None:
            fig.supylabel(ylabel)

        plt.subplots_adjust(hspace=0)

        return fig, ax




    def my_hist_format(self, attribute_name, bins=25, xlabel=None, ylabel='Count', **kwargs):
        fig, ax = plt.subplots()

        data = [getattr(ds, attribute_name) for ds in self.datasets]

        if hasattr(self.datasets[0], 'well_type') and hasattr(self.datasets[0], 'era'):
            well_type = [ds.well_type for ds in self.datasets]
            era = [ds.era for ds in self.datasets]

            data, well_type, era = np.array(data), np.array(well_type), np.array(era)

            ax.hist([data[np.logical_and(well_type == 6, era == 0)],
                     data[np.logical_and(well_type == 6, era == 1)],
                     data[np.logical_and(well_type == 6, era == 2)],
                     data[np.logical_and(well_type == 8, era == 0)],
                     data[np.logical_and(well_type == 8, era == 1)],
                     data[np.logical_and(well_type == 8, era == 2)]],
                    bins=bins,
                    color=['#165580', '#1F77B4', '#48A2E0', '#B05400', '#FF810E', '#FFAA5D'],
                    edgecolor='black', stacked=True, alpha=1, **kwargs)
        elif hasattr(self.datasets[0], 'well_type'):
            well_type = [ds.well_type for ds in self.datasets]

            data, well_type = np.array(data), np.array(well_type)

            ax.hist([data[well_type == 6],
                     data[well_type == 8]],
                    bins=bins,
                    color=['#1F77B4', '#FF810E'],
                    edgecolor='black', stacked=True, alpha=1, **kwargs)
        else:
            ax.hist(data, bins=bins, color='gray', edgecolor='black', alpha=1, **kwargs)

        if xlabel is not None:
            ax.set_xlabel(xlabel)
        if ylabel is not None:
            ax.set_ylabel(ylabel)


    def my_subplot_format(self, x=None, y=None, z=None, fig=None, ax=None, xlabel=None, ylabel=None,
                          grid=True, ticks=True, xlim=None, ylim=None, fmt='k.', ms=3, **kwargs):
        num = len(self.datasets)

        if fig is None or ax is None:
            num_subplots = math.ceil(math.sqrt(num))**2
            num_rows = int(math.sqrt(num_subplots))
            num_cols = num_rows
            fig, ax = plt.subplots(num_rows, num_cols, figsize=(9,9), sharex=True, sharey=True)
            print(num, num_subplots, num_rows, num_cols)
        else:
            num_rows, num_cols = np.shape(ax)
            num_subplots = num_rows * num_cols

        for i in range(num_subplots):
            row = int(math.floor(i / num_cols))
            col = int(i - row * num_cols)
            #print(row, col)

            if i < num:
                if z is None and x is not None and y is not None:
                    ax[row, col].plot(x[i], y[i], fmt, ms=ms, **kwargs)
                    ax[row, col].axhline(-1*self.datasets[i].well_depth, color='gray')
                    ax[row, col].title.set_text(self.datasets[i].file_names[0][:4])
                elif z is not None and x is None and y is None:
                    self.datasets[i].map_adj(z, ax=ax[row, col], fig=fig, **kwargs)
                else:
                    raise ValueError('input either x and y (not z) for a 2D plot, or just z (str) for a 3D color map.')


                spine_color = 'k'
                if hasattr(self.datasets[i], 'well_type') and hasattr(self.datasets[i], 'era'):
                    if self.datasets[i].well_type == 6:
                        if self.datasets[i].era == 0:
                            spine_color = '#165580'
                        elif self.datasets[i].era == 1:
                            spine_color = '#1F77B4'
                        elif self.datasets[i].era == 2:
                            spine_color = '#48A2E0'

                    if self.datasets[i].well_type == 8:
                        if self.datasets[i].era == 0:
                            spine_color = '#B05400'
                        elif self.datasets[i].era == 1:
                            spine_color = '#FF810E'
                        elif self.datasets[i].era == 2:
                            spine_color = '#FFAA5D'

                ax[row, col].spines[['left', 'right', 'bottom', 'top']].set_color(spine_color)
                ax[row, col].spines[['left', 'right', 'bottom', 'top']].set_linewidth(2)

                if grid:
                    ax[row, col].yaxis.minorticks_on()
                    ax[row, col].grid(which='major', color=spine_color, alpha=0.3)
                    ax[row, col].grid(which='minor', color=spine_color, alpha=0.1)
                    ax[row, col].tick_params(which='both', colors=spine_color)

                if not ticks:
                    ax[row, col].set_xticks([])
                    ax[row, col].set_yticks([])

                if xlim is not None:
                    ax[row, col].set_xlim(xlim[0], xlim[1])
                if ylim is not None:
                    ax[row, col].set_ylim(ylim[0], ylim[1])

            else:
                ax[row, col].axis('off')

        if xlabel is not None:
            ax[-1,0].set_xlabel(xlabel)
        if ylabel is not None:
            ax[-1,0].set_ylabel(ylabel)

        plt.subplots_adjust(wspace=0.1, hspace=0.1)

        return fig, ax


#%% define all paths and file names




if __name__ == "__main__":

 
    path = r"D:\DBTF\Manip 3"

    ppt_path = r"force_in_manip_3.pptx"

    prs = Presentation()
    prs.slide_width = Inches(16)  # standard 16:9 width
    prs.slide_height = Inches(9)  # standard 16:9 height

    j = 1
    try:
        while True:
            key_str = f'manip_'
            # key_str = None
            print(key_str)

            #path_to_info = path + r'\info'
            file_names = MyFiles(path).dir_filter(extension='dat', key_str=key_str)
            
            ds = SpectraSet(path, file_names, chrono_order=False)
            ds.force_analysis(threshold=0.1e-11)
            fig1, fig2 = ds.my_force_plot()


            slide = prs.slides.add_slide(prs.slide_layouts[6])
            # Add a transparent text box
            left = Inches(0.3)
            top = Inches(0.3)
            width = Inches(3.5)
            height = Inches(3)

            textbox = slide.shapes.add_textbox(left, top, width, height)
            text_frame = textbox.text_frame

            # Add your metadata text
            p = text_frame.paragraphs[0]
            s = ds.spectra[0]
            matadata_items = ['speed', 'x_path', 'y_path', 'i_calibration', 'v_calibration', 'i_idle', 'v_idle', 'z_idle', 'q_factor', 'amplitude_stp']
            pres_txt = key_str
            for i in matadata_items:
                pres_txt = pres_txt + '\n' + i + ' ' + getattr(s, i)
            p.text = pres_txt  # metadata is a string with newlines

            # Set font style
            text_frame.clear()  # clears default empty paragraph
            for i in range(len(pres_txt.split('\n'))):
                line = pres_txt.split('\n')[i]
                p = text_frame.add_paragraph()
                run = p.add_run()
                run.text = line
                if i == 0:
                    run.font.size = Pt(50)
                    run.font.color.rgb = RGBColor(255, 255, 255)
                else:
                    run.font.size = Pt(16)
                    run.font.color.rgb = RGBColor(100, 100, 100)


            buf1 = io.BytesIO()
            fig1.savefig(buf1, format='png', bbox_inches='tight')
            buf1.seek(0)
            buf2 = io.BytesIO()
            fig2.savefig(buf2, format='png', bbox_inches='tight')
            buf2.seek(0)


            # fig1: top-right (next to the metadata box)
            slide.shapes.add_picture(
                buf2,
                left=Inches(4),  # starts after your left-side text box
                top=Inches(0),
                width=Inches(12),  # use the rest of the width (13.33 - 4.0 - 0.3)
                height=Inches(4.5)
            )

            # fig2: bottom full-width image
            slide.shapes.add_picture(
                buf1,
                left=Inches(0),                    # small left margin
                top=Inches(4.5),                     # adjust vertical as needed
                width=Inches(16),                 # full width minus 0.3" margin on each side
                height=Inches(4.5)                  # proportional height
            )
            j = j + 1
    except Exception as e:
        print(e)

    prs.save(ppt_path)
    sys.exit()
    

#     """

    
#     file_names = MyFiles(path).dir_filter(extension='dat', key_str='iz')

#     ds = SpectraSet(path, file_names, chrono_order=True)



#     ds.iz_analysis()
#     ds.load_drift_adj_info(path_to_info)
#     ds.redefine_attribute_units('x_adj', 1e9)
#     ds.redefine_attribute_units('y_adj', 1e9)
#     ds.calc_polar_adj()  # assumes centre is at grid's mean xy
#     ds.calc_angle_xy_adj()

#     fig, ax = plt.subplots()
#     ax.plot(ds.slice_spectra_attribute('z_pos'))

#     fig, ax = ds.stacked_plot(y='log_curr', x='x', ncols=15, color='black')
#     ds.stacked_plot(y='fit', x='x', ncols=15, color='red', fig=fig, ax=ax)
#     plt.show()
#     ds.redefine_attribute_units('decay_length', 1e-9)
#     ds.map_adj('decay_length', cmap='coolwarm', colourbarLabel=r'$\kappa (nm^{-1})$ ', scalebar=2,
#                cbar_max_text_color='white', cbar_min_text_color='black')

#     ds.redefine_attribute_units('maxAbsCurrent', 1e12)
#     ds.map_adj('maxAbsCurrent', cmap='bone_r', cmapScale='log', colourbarLabel='max abs current (pA)', scalebar=2,
#                cbar_max_text_color='white', cbar_min_text_color='black')

#     ds.redefine_attribute_units('meanAbsRes', 1e3)
#     ds.map_adj('meanAbsRes', cmap='bone_r', colourbarLabel='mean abs res (mlnA)', scalebar=2,
#                cbar_max_text_color='white', cbar_min_text_color='black')

#     fig, ax = plt.subplots(figsize=(2.5, 5))
#     for s in ds.spectra:
#         try:
#             ax.plot(s.x, s.log_curr, color='black', alpha=0.05)
#         except:
#             pass
#     ax.set_title('All spectra')
#     ax.set_xlabel(r'z (m)')
#     ax.set_ylabel(r'ln[I] (lnA)')
#     ax.grid()
#     plt.tight_layout()

#     fig, ax = plt.subplots(figsize=(3, 3))
#     ax.plot(ds.slice_spectra_attribute('r_adj'), ds.slice_spectra_attribute('decay_length'), '.', c='black')
#     ax.grid()
#     ax.set_xlabel('r (nm)')
#     ax.set_ylabel(r"$\kappa (nm^{-1})$")
#     plt.tight_layout()



#     k = ds.slice_spectra_attribute('decay_length')
#     fig, axFit, axResiduals = ds.spectra[np.argmin(k)].iz_analysis(plotCalculation=True)

#     axFit.set_title(r'$\bf S$mallest $\kappa$')
#     plt.tight_layout()
#     fig, axFit, axResiduals = ds.spectra[np.argmax(k)].iz_analysis(plotCalculation=True)

#     axFit.set_title(r'$\bf L$argest $\kappa$')
#     plt.tight_layout()

#     fig, ax = plt.subplots()
#     x = ds.slice_spectra_attribute('x_adj')
#     y = ds.slice_spectra_attribute('y_adj')
#     ax.scatter(x, y, color='black', alpha=0.05)
#     ax.scatter(x[np.nanargmax(k)], y[np.argmax(k)], color='black', alpha=1, marker='$L$')
#     ax.scatter(x[np.nanargmin(k)], y[np.nanargmin(k)], color='black', alpha=1, marker='$S$')
#     ax.set_axis_off()
#     ax.set_aspect('equal')
#     plt.show()

#     ds.plot_spectra_pos(color='white')
#     plt.show()

#     """


#     r"""

#     file_names = MyFiles(path).dir_filter(extension='dat', key_str='didv')

#     ds = SpectraSet(path, file_names, chrono_order=True)


#     ds.didv_analysis(yChannel='Current [AVG] (A)', peakType='arctan', centerGuess=-0.1, centerMinLim=-0.105, centerMaxLim=-0.1, amplitudeMinLim=0, sigmaMaxLim=0.02, sigmaMinLim=0.002, plotCalculation=False)

#     ds.load_drift_adj_info(path_to_info)
#     ds.redefine_attribute_units('x_adj', 1e9)
#     ds.redefine_attribute_units('y_adj', 1e9)
#     ds.calc_polar_adj()  # assumes centre is at grid's mean xy
#     ds.calc_angle_xy_adj()

    
    

#     fig, ax = ds.stacked_plot(y='curr',ncols=15, color='black')
#     ds.stacked_plot(y='iv_fit', ncols=15, color='red', fig=fig, ax=ax)


#     h = ds.slice_spectra_attribute('fitHeight')

#     fig, axDIDV, axIV, axResiduals = ds.spectra[np.nanargmin(h)].didv_analysis(yChannel='Current [AVG] (A)', peakType='arctan',  amplitudeMinLim=0, sigmaMaxLim=0.01, sigmaMinLim=0.002, plotCalculation=True)

#     axDIDV.set_title(r'$\bf S$hortest fitted peak')
#     plt.tight_layout()
#     fig, axDIDV, axIV, axResiduals = ds.spectra[np.nanargmax(h)].didv_analysis(yChannel='Current [AVG] (A)', peakType='arctan',  amplitudeMinLim=0, sigmaMaxLim=0.01, sigmaMinLim=0.002, plotCalculation=True)

#     axDIDV.set_title(r'$\bf T$allest fitted peak')
#     plt.tight_layout()
    
#     fig, ax = plt.subplots(1, 6, figsize=(12, 6))
#     x = ds.slice_spectra_attribute('r_adj')
#     for j in range(6):
#         params = ['fitArea', 'fitCentre', 'fitSigma', 'fitSlope', 'fitIntercept', 'fitK']
#         y = ds.slice_spectra_attribute(params[j])
#         ax[j].plot(x, y, '.', c='black')
#         ax[j].set_ylabel('r')
#         ax[j].set_ylabel(params[j])
#         plt.tight_layout()
    
#     fig, ax = plt.subplots(figsize=(3,3))
#     ax.plot(x, h, '.', c='black')
#     ax.grid()
#     ax.set_xlabel('r (nm)')
#     ax.set_ylabel("peak's height (a.u.)")
#     plt.tight_layout()




#     fig, ax = plt.subplots(figsize=(2.5, 5))
#     for s in ds.spectra:
#         try:
#             ax.plot(s.x, s.curr, color='black', alpha=0.05)
#         except:
#             pass
#     ax.set_title('All spectra')
#     ax.set_xlabel(r'bias (V)')
#     ax.set_ylabel(r'curr (A)')
#     ax.grid()
#     plt.tight_layout()

#     fig, ax = plt.subplots()
#     x = ds.slice_spectra_attribute('x_adj')
#     y = ds.slice_spectra_attribute('y_adj')
#     ax.scatter(x, y, color='black', alpha=0.05)
#     ax.scatter(x[np.nanargmax(h)], y[np.nanargmax(h)], color='black', alpha=1, marker='$T$')
#     ax.scatter(x[np.nanargmin(h)], y[np.nanargmin(h)], color='black', alpha=1, marker='$S$')
#     ax.set_axis_off()
#     ax.set_aspect('equal')

#     ds.redefine_attribute_units('fitHeight', 1e12)
#     ds.map_adj('fitHeight', cmap='BuPu', colourbarLabel="peak height (1e-12 a.u.)", scalebar=2,
#                cbar_max_text_color='white', cbar_min_text_color='black', cmapScale='log')

#     ds.redefine_attribute_units('fitCentre', 1e3)
#     ds.map_adj('fitCentre', cmap='bone_r', colourbarLabel="peak's centre (mV)", scalebar=2,
#                cbar_max_text_color='white', cbar_min_text_color='black')

#     ds.slice_spectra_attribute('fitCentre')
    

#     ds.redefine_attribute_units('fitSigma', 1e3)
#     ds.map_adj('fitSigma', cmap='bone_r', colourbarLabel="peak's sigma (mV)", scalebar=2,
#                cbar_max_text_color='white', cbar_min_text_color='black')

#     ds.redefine_attribute_units('fitArea', 1e12)
#     ds.map_adj('fitArea', cmap='bone_r', colourbarLabel="peak's area (1e-12 a.u.)", cmapScale='log', scalebar=2,
#                cbar_max_text_color='white', cbar_min_text_color='black')

#     ds.redefine_attribute_units('fitSlope', 1e12)
#     ds.map_adj('fitSlope', cmap='bone_r', colourbarLabel='background slope (1e-12 a.u.)', scalebar=2,
#                cbar_max_text_color='white', cbar_min_text_color='black')

#     ds.redefine_attribute_units('fitIntercept', 1e12)
#     ds.map_adj('fitIntercept', cmap='bone_r', colourbarLabel='background intercept (1e-12 a.u.)', scalebar=2,
#                cbar_max_text_color='white', cbar_min_text_color='black')

#     ds.redefine_attribute_units('maxAbsCurrent', 1e12)
#     ds.map_adj('maxAbsCurrent', cmap='bone_r', colourbarLabel='max abs current (pA)', scalebar=2,
#                cbar_max_text_color='white', cbar_min_text_color='black', cmapScale='log')

#     ds.redefine_attribute_units('meanAbsRes', 1e12)
#     ds.map_adj('meanAbsRes', cmap='bone_r', colourbarLabel='mean abs res (1e-12 a.u.)', scalebar=2,
#                cbar_max_text_color='white', cbar_min_text_color='black')

#     ds.map_adj('fitDW', cmap='bone_r', colourbarLabel='DW fit', scalebar=2,
#                cbar_max_text_color='white', cbar_min_text_color='black')

#     """





#     r"""
#     file_names = MyFiles(path).dir_filter(extension='dat', key_str='dfv')

#     ds = SpectraSet(path, file_names, chrono_order=True)

#     ds.kpfm_analysis()  # loads the 'OC M1 Freq. Shift [AVG] (Hz)' channel

#     ds.load_drift_adj_info(path_to_info)
#     ds.redefine_attribute_units('x_adj', 1e9)
#     ds.redefine_attribute_units('y_adj', 1e9)
#     ds.calc_polar_adj()  # assumes centre is at grid's mean xy
#     ds.calc_angle_xy_adj()

    
#     fig, ax = ds.stacked_plot(ncols=15, color='black')
#     ds.stacked_plot(x='x_filt', y='y_filt', ncols=15, color='red', fig=fig, ax=ax)
#     plt.show()

#     ds.redefine_attribute_units('vContact', 1e3)


#     ds.map_adj('vContact', cmap='inferno', colourbarLabel='CPD (mV)', scalebar=2,
#                cbar_max_text_color='black', cbar_min_text_color='white')

#     ds.redefine_attribute_units('biasAtMaxDf', 1e3)
#     ds.map_adj('biasAtMaxDf', cmap='inferno', colourbarLabel='bias @ max df after 4 $\sigma$ Gauss smooth (mV)', scalebar=2,
#                cbar_max_text_color='black', cbar_min_text_color='white')

    
#     cpd1 = ds.slice_spectra_attribute('vContact')
    

#     path_base = r'C:\Users\Usuario\OneDrive - The University of Nottingham\PhD\data'
#     path = path_base + r'\2025-07-13-2'
#     path_to_info = path + r'\info'
#     """


#     """
#     file_names = MyFiles(path).dir_filter(extension='dat', key_str='dfv')

#     ds = SpectraSet(path, file_names, chrono_order=True)

#     ds.kpfm_analysis()  # loads the 'OC M1 Freq. Shift [AVG] (Hz)' channel

#     ds.load_drift_adj_info(path_to_info)
#     ds.redefine_attribute_units('x_adj', 1e9)
#     ds.redefine_attribute_units('y_adj', 1e9)
#     ds.calc_polar_adj()  # assumes centre is at grid's mean xy
#     ds.calc_angle_xy_adj()

    
    
#     fig, ax = ds.stacked_plot(ncols=15, color='black')
#     ds.stacked_plot(y='fit', ncols=15, color='red', fig=fig, ax=ax)
#     plt.show()

#     ds.redefine_attribute_units('vContact', 1e3)

#     ds.map_adj('vContact', cmap='BuPu', colourbarLabel='z', scalebar=2,
#                cbar_max_text_color='black', cbar_min_text_color='white')







#     """
#     # cpd2 = ds.slice_spectra_attribute('vContact')

#     # cpd_diff = cpd1-cpd2

#     # ds.set_spectra_attribute('cpd_diff', cpd_diff)

#     # ds.map_adj('cpd_diff', cmap='inferno', colourbarLabel='CPD @ -100pm - CPD @ 1nm (mV)', scalebar=2,
#     #            cbar_max_text_color='black', cbar_min_text_color='white')

#     # fig, ax = plt.subplots(figsize=(3, 3))
#     # ax.plot(ds.slice_spectra_attribute('r_adj'), ds.slice_spectra_attribute('cpd_diff'), '.', c='black')
#     # ax.grid()
#     # ax.set_xlabel('r (nm)')
#     # ax.set_ylabel("CPD @ -100pm - CPD @ 1nm (mV)")
#     # plt.tight_layout()
    

#     """







#     a = ds.slice_spectra_attribute('fitA')
#     b = ds.slice_spectra_attribute('fitB')
#     c = ds.slice_spectra_attribute('fitC')

#     const2 = c - ((b**2) / (4*a))

#     ds.map_adj('fitA', cmap='bone_r', colourbarLabel=r'fit $a = \frac{\partial ^2C}{\partial z^2}$', scalebar=2,
#                cbar_max_text_color='white', cbar_min_text_color='black')

#     ds.map_adj(const2, cmap='bone_r', colourbarLabel=r'fit $c - \frac{b^2}{4*a} = \frac{\partial F_{other}}{\partial z}$ (Hz)', scalebar=2,
#                cbar_max_text_color='white', cbar_min_text_color='black')

#     ds.map_adj('dfAtVContact', cmap='bone_r', colourbarLabel=r'$\Delta f$ @ CPD (Hz)', scalebar=2,
#                cbar_max_text_color='white', cbar_min_text_color='black')


#     ds.redefine_attribute_units('maxAbsCurrent', 1e12)
#     ds.map_adj('maxAbsCurrent', cmap='bone_r', colourbarLabel='max abs current (pA)', scalebar=2,
#                cbar_max_text_color='white', cbar_min_text_color='black', cmapScale='log')

#     ds.redefine_attribute_units('meanAbsRes', 1e3)
#     ds.map_adj('meanAbsRes', cmap='bone_r', colourbarLabel='mean abs res (mHz)', scalebar=2,
#                cbar_max_text_color='white', cbar_min_text_color='black')

#     ds.map_adj('dw_parabola', cmap='bone_r', colourbarLabel='DW parabola', scalebar=2,
#                cbar_max_text_color='white', cbar_min_text_color='black')

#     ds.redefine_attribute_units('meanExcitation', 1e3)
#     ds.map_adj('meanExcitation', cmap='bone_r', colourbarLabel=r'mean excitation (mV)', scalebar=2,
#                cbar_max_text_color='white', cbar_min_text_color='black')

#     ds.map_adj('dw_excitation', cmap='bone_r', colourbarLabel=r'DW flat excitation', scalebar=2,
#                cbar_max_text_color='white', cbar_min_text_color='black')

#     ds.redefine_attribute_units('meanAmplitude', 1e12)
#     ds.map_adj('meanAmplitude', cmap='bone_r', colourbarLabel=r'mean amplitude (pm)', scalebar=2,
#                cbar_max_text_color='white', cbar_min_text_color='black')

#     ds.map_adj('dw_amplitude', cmap='bone_r', colourbarLabel=r'DW flat amplitude', scalebar=2,
#                cbar_max_text_color='white', cbar_min_text_color='black')

#     fig, ax = plt.subplots(figsize=(2.5, 5))
#     for s in ds.spectra:
#         try:
#             ax.plot(s.x, s.y, color='black', alpha=0.05)
#         except:
#             pass
#     ax.set_title('All spectra')
#     ax.set_xlabel(r'bias (V)')
#     ax.set_ylabel(r'$\Delta f$')
#     ax.grid()
#     plt.tight_layout()

#     fig, ax = plt.subplots(figsize=(3, 3))
#     ax.plot(ds.slice_spectra_attribute('r_adj'), ds.slice_spectra_attribute('vContact'), '.', c='black')
#     ax.grid()
#     ax.set_xlabel('r (nm)')
#     ax.set_ylabel("CPD (mV)")
#     plt.tight_layout()

#     cpd = ds.slice_spectra_attribute('vContact')
#     fig, axFit, axResiduals = ds.spectra[np.argmin(cpd)].kpfm_analysis(plotCalculation=True)

#     axFit.set_title(r'$\bf S$mallest CPD')
#     plt.tight_layout()
#     fig, axFit, axResiduals = ds.spectra[np.argmax(cpd)].kpfm_analysis(plotCalculation=True)

#     axFit.set_title(r'$\bf L$argest CPD')
#     plt.tight_layout()

#     fig, ax = plt.subplots()
#     x = ds.slice_spectra_attribute('x_adj')
#     y = ds.slice_spectra_attribute('y_adj')
#     ax.scatter(x, y, color='black', alpha=0.05)
#     ax.scatter(x[np.nanargmax(cpd)], y[np.argmax(cpd)], color='black', alpha=1, marker='$L$')
#     ax.scatter(x[np.nanargmin(cpd)], y[np.nanargmin(cpd)], color='black', alpha=1, marker='$S$')
#     ax.set_axis_off()
#     ax.set_aspect('equal')
#     plt.show()








#     r"""




# #     dfv_path_endings = [r'\2024-04-19-1', r'\2024-04-21-2',
# #                     r'\2024-04-22', r'\2024-04-26-3',
# #                     r'\2024-06-07', r'\2024-07-13-1',
# #                     r'\2024-12-10', r'\2024-12-16-1',
# #                     r'\2024-12-17-5', r'\2024-12-18', r'\2024-12-18-2', r'\2024-12-19',
# #                     r'\2024-12-21']
# #     didv_path_endings = [r'\2024-04-19-2', r'\2024-04-21-1',
# #                         r'\2024-04-23', r'\2024-04-26-2',
# #                         r'\2024-06-08', r'\2024-07-13-2',
# #                         r'\2024-12-10', r'\2024-12-16-1',
# #                         r'\2024-12-17-5', r'\2024-12-18', r'\2024-12-18-2', r'\2024-12-19',
# #                         r'\2024-12-21']

    
    
    
    
# #     path_endings = dfv_path_endings

# #     all_paths_to_data = [path_base + ending for ending in path_endings]
# #     all_paths_to_info = [path + r'\info' for path in all_paths_to_data]
# #     all_file_names_data = []

# #     for path_to_data in all_paths_to_data:
# #         file_names = MyFiles(path_to_data).dir_filter(extension='dat', key_str='dfv')
# #         all_file_names_data.append(file_names)


# #     d = GroupKPFMAnalysis(all_paths_to_data, all_paths_to_info, all_file_names_data)
# #     d.kpfm()
# #     d.set_dataset_attribute('era',
# #                             [0,0,0,0,0,1,2,2,2,2,2,2,2])
# #     d.set_dataset_attribute('well_type',
# #                             [8,8,8,8,8,8,8,6,6,6,6,8,8])
# #     d.my_subplot_format(z='vContact', cmap='inferno', scalebar=None,
# #                         cmapScale='linear', xlim=(-12, 12), ylim=(-12, 12),
# #                         ticks=False, cmapMin=None, cmapMax=None, colourbarLabel='CPD (V)')

# #     plt.show()
    



# #     dfv_all_paths_to_data = [path_base + ending for ending in dfv_path_endings]
# #     dfv_all_paths_to_info = [path + r'\info' for path in dfv_all_paths_to_data]

# #     didv_all_paths_to_data = [path_base + ending for ending in didv_path_endings]
# #     didv_all_paths_to_info = [path + r'\info' for path in didv_all_paths_to_data]


# #     for i in range(len(didv_all_paths_to_data)):
# #         i=6
# #         path_to_data = didv_all_paths_to_data[i]
# #         path_to_info = didv_all_paths_to_info[i]

# #         if path_to_data in [path_base + ending for ending in [r'\2024-12-10', r'\2024-12-16-1',
# #                         r'\2024-12-17-5', r'\2024-12-18', r'\2024-12-18-2', r'\2024-12-19',
# #                         r'\2024-12-21']]:
# #             file_names = MyFiles(path_to_data).dir_filter(extension='dat', key_str='didv')
# #         else:
# #             file_names = MyFiles(path_to_data).dir_filter(extension='dat')


# #         ds = SpectraSet(path_to_data, file_names, chrono_order=True)

# #         ds.load_drift_adj_info(path_to_info)
# #         ds.redefine_attribute_units('x_adj', 1e9)
# #         ds.redefine_attribute_units('y_adj', 1e9)
# #         ds.calc_polar_adj()  # assumes centre is at grid's mean xy
# #         ds.calc_angle_xy_adj()



# #         if r'\2024-04-23' in path_to_data:
# #             ds.spectra = ds.spectra[:942]

# #         ds.didv_analysis(peakType='Lorentzian', yChannel='Current [AVG] (A)',
# #                          centerGuess=-0.0885, sigmaGuess=0.008, amplitudeMaxLim=1e-10, amplitudeMinLim=0.2e-15, sigmaMinLim=0.001, sigmaMaxLim=0.02)
# #         h = ds.slice_spectra_attribute('fitHeight')
# #         fig, axFit, axResiduals = ds.spectra[np.argmin(h)].didv_analysis(peakType='Lorentzian', yChannel='Current [AVG] (A)',
# #                                                                          plotCalculation=True,
# #                          centerGuess=-0.0885, sigmaGuess=0.008, amplitudeMaxLim=1e-10, amplitudeMinLim=0.2e-15, sigmaMinLim=0.001, sigmaMaxLim=0.02)


# #         axFit.set_title(r'$\bf S$hortest fitted peak')
# #         plt.tight_layout()
# #         fig, axFit, axResiduals = ds.spectra[np.argmax(h)].didv_analysis(peakType='Lorentzian', yChannel='Current [AVG] (A)',
# #                                                                          plotCalculation=True,
# #                          centerGuess=-0.0885, sigmaGuess=0.008, amplitudeMaxLim=1e-10, amplitudeMinLim=0.2e-15, sigmaMinLim=0.001, sigmaMaxLim=0.02)

# #         axFit.set_title(r'$\bf T$allest fitted peak')
# #         plt.tight_layout()

# #         fig, ax = plt.subplots(1,5, figsize=(12,6))
# #         x = ds.slice_spectra_attribute('r_adj')
# #         for j in range(5):
# #             params = ['fitArea', 'fitCentre', 'fitSigma', 'fitSlope', 'fitIntercept']
# #             y = ds.slice_spectra_attribute(params[j])
# #             ax[j].plot(x, y, '.', c='black')
# #             ax[j].set_ylabel('r')
# #             ax[j].set_ylabel(params[j])
# #             plt.tight_layout()

# #         fig, ax = plt.subplots()
# #         ds.redefine_attribute_units('fitHeight', 1e12)
# #         ds.map_adj('fitHeight', cmap='BuPu', cmapScale='log', colourbarLabel="peak's height (1e-12 a.u.)", scalebar=2,
# #                    cbar_max_text_color='white', cbar_min_text_color='black', fig=fig, ax=ax)

# #         xmin, xmax = ax.get_xlim()
# #         ymin, ymax = ax.get_ylim()

    
    
    
    
# #         fig, ax = plt.subplots(figsize=(2.5,5))
# #         for s in ds.spectra:
# #             ax.plot(s.x, s.y, color='black', alpha=0.05)
# #         ax.set_title('All spectra')
# #         ax.set_xlabel(r'bias (V)')
# #         ax.set_ylabel(r'dI/dV')
# #         ax.grid()
# #         plt.tight_layout()

# #         fig, ax = plt.subplots()
# #         x = ds.slice_spectra_attribute('x_adj')
# #         y = ds.slice_spectra_attribute('y_adj')
# #         ax.scatter(x, y, color='black', alpha=0.05)
# #         ax.scatter(x[np.argmax(h)], y[np.argmax(h)], color='black', alpha=1, marker='$T$')
# #         ax.scatter(x[np.argmin(h)], y[np.argmin(h)], color='black', alpha=1, marker='$S$')
# #         ax.set_axis_off()
# #         ax.set_aspect('equal')



# #         ds.redefine_attribute_units('fitCentre', 1e3)
# #         ds.map_adj('fitCentre', cmap='bone_r', colourbarLabel="peak's centre (mV)", scalebar=2,
# #                    cbar_max_text_color='white', cbar_min_text_color='black')

# #         ds.redefine_attribute_units('fitSigma', 1e3)
# #         ds.map_adj('fitSigma', cmap='bone_r', colourbarLabel="peak's sigma (mV)", scalebar=2,
# #                       cbar_max_text_color='white', cbar_min_text_color='black')

# #         ds.redefine_attribute_units('fitArea', 1e12)
# #         ds.map_adj('fitArea', cmap='bone_r', colourbarLabel="peak's area (1e-12 a.u.)", cmapScale='log', scalebar=2,
# #                    cbar_max_text_color='white', cbar_min_text_color='black')

# #         ds.redefine_attribute_units('fitSlope', 1e12)
# #         ds.map_adj('fitSlope', cmap='bone_r', colourbarLabel='background slope (1e-12 a.u.)', scalebar=2,
# #                    cbar_max_text_color='white', cbar_min_text_color='black')

# #         ds.redefine_attribute_units('fitIntercept', 1e12)
# #         ds.map_adj('fitIntercept', cmap='bone_r', colourbarLabel='background intercept (1e-12 a.u.)', scalebar=2,
# #                    cbar_max_text_color='white', cbar_min_text_color='black')

# #         ds.redefine_attribute_units('meanAbsRes', 1e12)
# #         ds.map_adj('meanAbsRes', cmap='bone_r', colourbarLabel='mean abs res (1e-12 a.u.)', scalebar=2,
# #                    cbar_max_text_color='white', cbar_min_text_color='black')

# #         ds.map_adj('fitDW', cmap='bone_r', colourbarLabel='DW fit', scalebar=2,
# #                    cbar_max_text_color='white', cbar_min_text_color='black')
        
    
    
    
    
    
# #         path_to_data = dfv_all_paths_to_data[i]
# #         path_to_info = dfv_all_paths_to_info[i]
# #         file_names = MyFiles(path_to_data).dir_filter(extension='dat', key_str='dfv')
# #         ds = SpectraSet(path=path_to_data, file_names=file_names, chrono_order=True)

# #         ds.kpfm_analysis()  # loads the 'OC M1 Freq. Shift [AVG] (Hz)' channel

# #         ds.load_drift_adj_info(path_to_info)
# #         ds.redefine_attribute_units('x_adj', 1e9)
# #         ds.redefine_attribute_units('y_adj', 1e9)
# #         ds.calc_polar_adj()  # assumes centre is at grid's mean xy
# #         ds.calc_angle_xy_adj()

# #         fig, ax = plt.subplots()
# #         ax.set_xlim(xmin, xmax)
# #         ax.set_ylim(ymin, ymax)
# #         #ds.redefine_attribute_units('vContact', 1e3)

# #         ds.map_adj('vContact', cmap='inferno', colourbarLabel='intercept (Hz)', scalebar=2,
# #                    cbar_max_text_color='black', cbar_min_text_color='white', fig=fig, ax=ax)




    
# #     d = GroupKPFMAnalysis(all_paths_to_data, all_paths_to_info, all_file_names_data)
# #     d.didv()
# #     d.set_dataset_attribute('era',
# #                             [0, 0, 0, 0, 0, 1, 2, 2, 2, 2, 2, 2, 2])
# #     d.set_dataset_attribute('well_type',
# #                             [8, 8, 8, 8, 8, 8, 8, 6, 6, 6, 6, 8, 8])
# #     #d.my_subplot_format(z='fitSigma', cmap='inferno', scalebar=None,
# #     #                    cmapScale='linear', xlim=(-12, 12), ylim=(-12, 12),
# #     #                    ticks=False, cmapMin=None, cmapMax=None, colourbarLabel='CPD (V)')

# #     for s in d.datasets:
        
# #         s.spectra[int(len(s.spectra)/2)].didv_analysis(plotCalculation=True)

# #         s.redefine_attribute_units('fitSigma', 1e3)
# #         s.map_adj('fitSigma', cmap='bone_r', colourbarLabel='fit sigma (mV)', scalebar=None,
# #                       cbar_max_text_color='white', cbar_min_text_color='black')
# #     """



# #     r"""                       
# #     path = r'C:\Users\Usuario\OneDrive - The University of Nottingham\PhD\data\2025-04-16-1'
# #     file_names = MyFiles(path).dir_filter(extension='dat')
# #     s = SpectraSet(path, file_names)
# #     s.kpfm_analysis(yChannel='OC M1 Freq. Shift (Hz)')  # loads the 'OC M1 Freq. Shift [AVG] (Hz)' channel
# #     x = s.slice_spectra_attribute('x_pos')
# #     s.set_spectra_attribute('x_adj', x)
# #     y = s.slice_spectra_attribute('y_pos')
# #     s.set_spectra_attribute('y_adj', y)
# #     s.angle_xy_adj = 0.0
# #     s.redefine_attribute_units('x_adj', 1e9)
# #     s.redefine_attribute_units('y_adj', 1e9)
# #     s.redefine_attribute_units('z_pos', 1e9)

# #     fig, ax = plt.subplots(3,4, figsize=(12,9))

# #     vars = ['vContact', 'dfAtVContact', 'fitA', 'fitB', 'fitC', 'z_pos', 'maxAbsCurrent', 'meanAbsRes',
# #             'dw_parabola', 'dw_excitation', 'dw_amplitude', 'meanAmplitude']

# #     vars_name = ['CPD (V)', 'df at CPD (Hz)', 'parabolic fit param a', 'parabolic fit param b', 'parabolic fit param c (Hz)',
# #                  'z (nm)', 'max abs current (pA)', 'mean abs res (Hz)', 'DW parabolic df(V)', 'DW flat excitation',
# #                  'DW flat amplitude', 'mean amplitude (pm)']

# #     for i in range(3*4):
# #         row = int(math.floor(i / 4))
# #         col = int(i - row * 4)

# #         var = vars[i]

# #         var_name = vars_name[i]


# #         if var_name == 'max abs current (pA)':
# #             s.redefine_attribute_units(var, 1e12)
# #         if var_name == 'mean amplitude (pm)':
# #             s.redefine_attribute_units(var, 1e12)



# #         #ax[row, col].hist(s.slice_spectra_attribute(var), 20, edgecolor='black', color='gray')

        
# #         if i==0:
# #             s.map_adj(var, cmap='inferno', colourbarLabel=var_name, scalebar=3, fig=fig, ax=ax[row, col])
# #         else:
# #             s.map_adj(var, cmap='bone_r', colourbarLabel=var_name, scalebar=None, fig=fig, ax=ax[row, col],
# #                       cbar_max_text_color='white', cbar_min_text_color='black')
       
# #         ax[row, col].set_ylabel('Count')
# #         ax[row, col].set_xlabel(var_name)

# #     plt.tight_layout()
# #     plt.show()
    
# #     """

    
# #     path_base = r'C:\Users\Usuario\OneDrive - The University of Nottingham\PhD\data'

# #     path_endings = [ r'\2025-05-22',
# #                     r'\2025-05-25-1', r'\2025-05-29-1', r'\2025-05-29-3', r'\2025-06-02-1',
# #                      r'\2025-06-03-1', r'\2025-06-03-2', r'\2025-06-04-1', r'\2025-06-04-2',
# #                     r'\2025-06-06', r'\2025-06-10-2',r'\2025-06-10-4',r'\2025-06-11-2',
# #                     r'\2025-06-12',r'\2025-06-14',]

# #     well_type = [6, 4, 8, 6,
# #                  6, 8, 6, 4,
# #                  6, 8, 8, 6,
# #                  6, 6, 8, 8,
# #                  6, 8]

# #     path_endings = [  r'\2025-05-23', r'\2025-05-29-1', r'\2025-05-29-3',
# #                     r'\2025-06-03-1', r'\2025-06-03-2', r'\2025-06-04-1', r'\2025-06-04-2',
# #                     r'\2025-06-06', r'\2025-06-10-2', r'\2025-06-10-4', r'\2025-06-11-2',
# #                     r'\2025-06-12', r'\2025-06-14-1', ]

# #     well_type = [8, 8, 6,
# #                  6, 8, 8, 6,
# #                  6, 6, 8, 8,
# #                  6, 8]

# #     path_endings = [ r'\2025-06-11-2', r'\2025-06-12',
# #                     ]

# #     well_type = [8, 6]

# #     all_paths_to_data = [path_base + ending for ending in path_endings]
# #     all_paths_to_info = [path + r'\info' for path in all_paths_to_data]
# #     all_file_names_data = []

# #     for path_to_data in all_paths_to_data:
# #         file_names = MyFiles(path_to_data).dir_filter(extension='dat', key_str='dfv')
# #         all_file_names_data.append(file_names)

# #     d = AllData(all_paths_to_data, all_paths_to_info, all_file_names_data)

# #     d.set_dataset_attribute('well_type',
# #                             well_type)
# #     d.set_dataset_attribute('era',
# #                             [1,
# #                              1])

# #     for ds in d.datasets:
# #         ds.strip_outliers('dw_parabola', attr_min=1)

# #     d.calc_cpd_norm()
# #     d.calc_well_depth()

# #     #d.my_subplot_format(z='vContact', cmap='inferno', scalebar=None,
# #     #                    cmapScale='linear',
# #     #                    ticks=False, cmapMin=None, cmapMax=None, colourbarLabel='CPD (V)')

# #     r_adj = [ds.slice_spectra_attribute('r_adj') for ds in d.datasets]
# #     cpd = [ds.slice_spectra_attribute('vContact') for ds in d.datasets]
# #     d.my_subplot_format(x=r_adj, y=cpd, xlabel='drift adjusted r (nm)',
# #                                ylabel='CPD (V) ')

# #     d.my_subplot_format(z='cpd_norm', cmap='inferno', scalebar=None,
# #                                   cmapScale='linear',
# #                                   ticks=False,  colourbarLabel=None)

# #     fig, ax = d.my_plot_format('r_adj', 'vContact', xlabel='r (nm)',
# #                                ylabel='CPD (V)', alpha=0.5)

# #     d.my_hist_format('well_depth', xlabel='well depth (V)')
# #     plt.show()











# #     path_endings = [r'\2024-04-19-1', r'\2024-04-21-2', r'\2024-04-22',
# #                     r'\2024-04-25', r'\2024-04-26-3', r'\2024-06-07',
# #                     r'\2024-06-14-1', r'\2024-06-14-2', r'\2024-06-14-3', r'\2024-06-15-1',
# #                     r'\2024-06-15-2', r'\2024-06-15-3', r'\2024-06-15-4', r'\2024-06-15-5', r'\2024-06-20-2',
# #                     r'\2024-06-20-3', r'\2024-06-20-4', r'\2024-06-21-1', r'\2024-06-21-2',
# #                     r'\2024-06-21-3', r'\2024-06-22-1', r'\2024-06-22-2', r'\2024-06-22-3',
# #                     r'\2024-06-23-1', r'\2024-06-25', r'\2024-07-12-1', r'\2024-07-12-2',
# #                     r'\2024-07-13-1', r'\2024-07-16', r'\2024-12-10', r'\2024-12-16-1',
# #                     r'\2024-12-17-5', r'\2024-12-18', r'\2024-12-18-2', r'\2024-12-19',
# #                     r'\2024-12-21']


# #     all_paths_to_data = [path_base + ending for ending in path_endings]
# #     all_paths_to_info = [path + r'\info' for path in all_paths_to_data]
# #     all_file_names_data = []

# #     for path_to_data in all_paths_to_data:
# #         file_names = MyFiles(path_to_data).dir_filter(extension='dat', key_str='dfv')
# #         all_file_names_data.append(file_names)


# #     d = GroupKPFMAnalysis(all_paths_to_data, all_paths_to_info, all_file_names_data)
# #     d.set_dataset_attribute('era',
# #                             [0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2])
# #     d.set_dataset_attribute('well_type',
# #                             [8,8,8,6,8,8,8,8,8,8,8,8,8,8,6,6,6,6,6,6,6,6,6,6,6,6,8,8,8,8,6,6,6,6,8,8])

# #     #d.strip_outliers("meanAbsRes", attr_max=0.1)
# #     #d.strip_outliers("meanAbsCurrent", attr_max=2e-11)
# #     #d.strip_outliers("dw_excitation", attr_min=1.0)
# #     #d.strip_outliers("dw_parabola", attr_min=0.5)

# #     d.calc_cpd_norm()
# #     d.calc_well_depth()

# #     d.my_subplot_format(z='vContact', cmap='inferno', scalebar=None,
# #                                   cmapScale='linear', xlim=(-12, 12), ylim=(-12, 12),
# #                                   ticks=False,  cmapMin=None, cmapMax=None, colourbarLabel='CPD (V)')

# #     #d.my_subplot_format(z='cpd_norm', cmap='inferno', scalebar=None,
# #     #                              cmapScale='linear', xlim=(-12, 12), ylim=(-12, 12),
# #     #                              ticks=False, cmapMin=-0.05, cmapMax=0.2, colourbarLabel='CPD')




# #     #d.my_hist_format('well_depth')

# #     #d.datasets[17].map_adj('vContact', cmap='inferno', colourbarLabel='CPD')

# #     plt.show()
    

    
# #     d.dataset_attr_to_nan(25, 'vContact')
# #     d.dataset_attr_to_nan(26, 'vContact')
# #     d.dataset_attr_to_nan(27, 'vContact')

# #     d.dataset_attr_to_nan(23, 'vContact')

# #     d.dataset_attr_to_nan(30, 'vContact')
# #     d.dataset_attr_to_nan(31, 'vContact')

# #     d.outlier_to_nan(test_attr_name='dw_excitation',exec_attr_name=['dw_excitation', 'vContact'], test_attr_min=1)

# #     d.outlier_to_nan(test_attr_name='dw_parabola',exec_attr_name=['dw_parabola', 'vContact'], test_attr_min=1)
    
    
# #     d.calc_cpd_norm()
# #     d.calc_well_depth()

    


# #     fig, ax = d.my_subplot_format(z='fitA', cmap='bwr_r', scalebar=None,
# #                    cmapScale='linear', xlim=(-12, 12), ylim=(-12,12), cmapMin=-np.inf, cmapMax=-np.inf, ticks=False)

# #     d.my_subplot_format(z='cpd_norm', cmap='inferno', scalebar=None,
# #                    cmapScale='linear', xlim=(-12, 12), ylim=(-12,12), cmapMin=-0.05, cmapMax=0.2, fig=fig, ax=ax, ticks=False)


# #     fig, ax = d.my_subplot_format(z='fitA', cmap='bwr_r', scalebar=None,
# #                    cmapScale='linear', xlim=(-12, 12), ylim=(-12,12), cmapMin=-np.inf, cmapMax=-np.inf, ticks=False)

# #     d.my_subplot_format(z='cpd_norm', cmap='inferno', scalebar=None,
# #                    cmapScale='linear', xlim=(-12, 12), ylim=(-12,12), fig=fig, ax=ax, ticks=False)

# #     fig, ax = d.my_plot_format('r_adj','cpd_norm', xlabel='drift adjusted r (nm)',
# #                      ylabel='CPD (V) (CPD=0 @ in well avg)', alpha=0.1)

    
# #     d.bin_cpd_norm(rmax=15)
# #     #d.my_plot_format('r_binned','cpd_norm_binned', fmt='o', lw=4, fig=fig, ax=ax)

# #     ax['6n8'].plot(d.r_binned_avg, d.cpd_norm_binned_6_avg, 'o', c='k', lw=4)
# #     ax['6n8'].plot(d.r_binned_avg, d.cpd_norm_binned_6_avg, '.', c='tab:blue', lw=4)
# #     ax['6n8'].plot(d.r_binned_avg, d.cpd_norm_binned_8_avg, 'o', c='k', lw=4)
# #     ax['6n8'].plot(d.r_binned_avg, d.cpd_norm_binned_8_avg, '.', c='tab:orange', lw=4)
# #     for i in range(len(d.cpd_norm_binned_6_avg)):
# #         ax['6n8'].annotate(str(d.cpd_norm_binned_6_weight[i]), (d.r_binned_avg[i], d.cpd_norm_binned_6_avg[i]), size='small', ha='center', va='bottom')
# #     for i in range(len(d.cpd_norm_binned_8_avg)):
# #         ax['6n8'].annotate(str(d.cpd_norm_binned_8_weight[i]), (d.r_binned_avg[i], d.cpd_norm_binned_8_avg[i]), size='small', ha='center', va='bottom')

# #     ax['6'].plot(d.r_binned_avg, d.cpd_norm_binned_6_avg, 'o', c='k', lw=4)
# #     ax['6'].plot(d.r_binned_avg, d.cpd_norm_binned_6_avg, '.', c='tab:blue')
# #     for i in range(len(d.cpd_norm_binned_6_avg)):
# #         ax['6'].annotate(str(d.cpd_norm_binned_6_weight[i]), (d.r_binned_avg[i], d.cpd_norm_binned_6_avg[i]), size='small', ha='center', va='bottom')

# #     ax['8'].plot(d.r_binned_avg, d.cpd_norm_binned_8_avg, 'o', c='k', lw=4)
# #     ax['8'].plot(d.r_binned_avg, d.cpd_norm_binned_8_avg, '.', c='tab:orange')
# #     for i in range(len(d.cpd_norm_binned_8_avg)):
# #         ax['8'].annotate(str(d.cpd_norm_binned_8_weight[i]), (d.r_binned_avg[i], d.cpd_norm_binned_8_avg[i]), size='small', ha='center', va='bottom')


# #     #d.my_hist_format('cpd_norm_shift', xlabel='map avg cpd @ r < 2.5 nm (V)')
# #     d.my_hist_format('well_depth', xlabel='well depth (V)')
# #     plt.show()


# # # %%
