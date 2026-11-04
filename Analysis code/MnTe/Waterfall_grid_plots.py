import sys
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
import io
import scipy.signal as spy

import datetime

from pathlib import Path
import pptx
import pptx.util


from PyQt5 import QtWidgets, QtCore
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

# reuse Spectrum class from your existing script
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
        file_data = re.split(r"\[DATA\]\n",file_all_data)[1]
        meta_data = re.split(r"\[DATA\]\n",file_all_data)[0]
        # print(meta_data)
        #Extract the time infomation:
        # self.start_time = self.convert_time_to_epoch(meta_data[meta_data.index('Start time')+11:meta_data.index('Start time')+11+19])
        self.start_time = self.convert_time_to_epoch(meta_data[meta_data.index('Saved Date')+11:meta_data.index('Saved Date')+11+19])
        

        meta_split = meta_data.split('\n') # Extracting the Z relative value 
        #Extract position infomation: 
        self.x_pos = float(meta_data.split('X (m)\t')[1].split('\n')[0][0:-1])
        self.y_pos = float(meta_data.split('Y (m)\t')[1].split('\n')[0][0:-1])
        self.z_pos = float(meta_data.split('Z (m)\t')[1].split('\n')[0][0:-1])
        
        
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
                    # print("Found match: ", i)
                    method_number = count
                    break
            if method_number is not count:
                #We did not convert to an int value
                raise Exception(f"{method_number} This input is not found in the list, check spelling is correct!")
                
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

        return x, y  # fallback will raise if not present

class WaterfallCanvas(FigureCanvas):
    def __init__(self, parent=None):
        self.fig, self.ax = plt.subplots(figsize=(6,6))
        super().__init__(self.fig)
        self.setParent(parent)
        plt.tight_layout()
        self.overlay_line = None  # reference to current overlay line

    def plot_mesh(self, X, Y, Z):
        self.ax.clear()
        self.mesh = self.ax.pcolormesh(X, Y, Z, cmap="plasma")
        self.fig.colorbar(self.mesh, ax=self.ax)
        self.ax.set_title("Z Position Map")
        self.ax.set_xlabel("X (m)")
        self.ax.set_ylabel("Y (m)")
        self.draw()

    def draw_overlay(self, orientation, coord, **style):
        """
        orientation: 'row' -> vertical line at x=coord (fixed X)
                     'col' -> horizontal line at y=coord (fixed Y)
        coord: numeric coordinate in same units as axes
        style: optional matplotlib.Line2D kwargs
        """
        # remove previous overlay
        try:
            if self.overlay_line is not None:
                self.overlay_line.remove()
        except Exception:
            pass

        line_style = dict(color='k', linewidth=1.5, linestyle='--', zorder=10)
        line_style.update(style)

        if orientation == 'row':
            self.overlay_line = self.ax.axvline(coord, **line_style)
        else:
            self.overlay_line = self.ax.axhline(coord, **line_style)

        self.draw()



class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, start_dir=None):
        super().__init__()
        self.setWindowTitle("Waterfall Grid Viewer")
        self.resize(1000,700)

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        vbox = QtWidgets.QVBoxLayout(central)

        # Figure
        self.canvas = WaterfallCanvas(self)
        toolbar = NavigationToolbar(self.canvas, self)
        vbox.addWidget(toolbar)
        vbox.addWidget(self.canvas)

        # Controls
        hbox = QtWidgets.QHBoxLayout()
        hbox_settings = QtWidgets.QHBoxLayout()
        self.dir_edit = QtWidgets.QLineEdit(start_dir or "")


        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(100)

        self.slider.valueChanged.connect(self.value_changed)
        
        self.index_label = QtWidgets.QLabel("Index: 0")
        

        self.r_or_c = QtWidgets.QComboBox()
        self.r_or_c.addItems(["Rows", "Columns", "Line"])
        self.rows_or_cols = 0

        self.r_or_c.currentIndexChanged.connect(self.index_changed)

        btn_browse = QtWidgets.QPushButton("Browse...")
        btn_load = QtWidgets.QPushButton("Load & Show Mesh")
        btn_process = QtWidgets.QPushButton("Process rows & cols")

        self.Avg_btn = QtWidgets.QCheckBox("Avg")



        hbox_settings.addWidget(self.r_or_c)
        hbox_settings.addWidget(self.index_label)
        hbox_settings.addWidget(self.slider)
        hbox_settings.addWidget(self.Avg_btn)
        hbox.addWidget(self.dir_edit)
        hbox.addWidget(btn_browse)
        hbox.addWidget(btn_load)
        hbox.addWidget(btn_process)

        vbox.addLayout(hbox)
        vbox.addLayout(hbox_settings)


        # Output area
        self.log = QtWidgets.QTextEdit()
        self.log.setReadOnly(True)
        vbox.addWidget(self.log, 1)

        # Connections
        btn_browse.clicked.connect(self.browse)
        btn_load.clicked.connect(self.load_mesh)
        btn_process.clicked.connect(self.process_grid)

        # data holders
        self.X = self.Y = self.Z = None
        self.position_pivot = None

    def browse(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Select data directory", self.dir_edit.text() or os.getcwd())
        if d:
            self.dir_edit.setText(d)
    def index_changed(self, i):
        self.rows_or_cols = i  # 0=rows, 1=cols, 2=line

        if getattr(self, "spectra_matrix", None) is not None:
            self.slider.setMaximum(self.spectra_matrix.shape[0]-1 if i == 0 else self.spectra_matrix.shape[1]-1)

        # update overlay immediately (use current selected_index if set)
        if getattr(self, "position_pivot", None) is not None and hasattr(self, "selected_index"):
            if i == 0:
                # row: vertical line at x coordinate
                x_val = self.position_pivot.index[self.selected_index]
                self.canvas.draw_overlay('row', float(x_val))
            else:
                # column: horizontal line at y coordinate
                y_val = self.position_pivot.columns[self.selected_index]
                self.canvas.draw_overlay('col', float(y_val))

    def value_changed(self, v):
        self.selected_index = v
        self.index_label.setText(f"Index: {v}")

        # update overlay on slider move
        if getattr(self, "position_pivot", None) is not None:
            if self.rows_or_cols == 0:
                x_val = self.position_pivot.index[v]
                self.canvas.draw_overlay('row', float(x_val))
            else:
                y_val = self.position_pivot.columns[v]
                self.canvas.draw_overlay('col', float(y_val))


    def log_msg(self, s):
        ts = time.strftime("%H:%M:%S")
        self.log.append(f"[{ts}] {s}")

    def build_grid_from_dir(self, dir_path):
        if Spectrum is None:
            raise RuntimeError("Cannot import Spectrum from Waterfall_plot.py. Ensure file is on PYTHONPATH.")
        files = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]
        files = np.sort(files)

        x_list = []
        y_list = []
        z_list = []

        demod_y_list = []
        current_list = []
        bias_list = []

        spectra_map = {}

        for i, file in enumerate(files):
            if i % 500 == 0:
                self.log_msg(f"Processed {i}/{len(files)}")
                QtWidgets.QApplication.processEvents()
            try:
                if self.Avg_btn.isChecked():
                    spec = Spectrum(path=dir_path, fileName=file, channel='LI Demod 1 Y [AVG] (A)') #[AVG]
                else:
                    spec = Spectrum(path=dir_path, fileName=file, channel='LI Demod 1 Y (A)') #[AVG]
            except Exception as e:
                self.log_msg(f"Skipping {file}: {e}")
                continue
            

            demod_y = spec.y
            if self.Avg_btn.isChecked():
                current = spec.give_data('Current [AVG] (A)')[0] #[AVG]
            else:
                current = spec.give_data('Current (A)')[0] #[AVG]
            bias = spec.x  

            x_val = spec.x_pos
            y_val = spec.y_pos
            z_val = spec.z_pos

            x_list.append(x_val)
            y_list.append(y_val)
            z_list.append(z_val)
            demod_y_list.append(demod_y)
            current_list.append(current)
            bias_list.append(bias)

            spectra_map[(x_val, y_val)] = {
                'bias': bias,
                'demod_y': demod_y,
                'current': current,
                'filename': file
            }

        position_grid = pd.DataFrame({'X': x_list, 'Y': y_list, 'Z': z_list})
        position_pivot = position_grid.pivot(index='X', columns='Y', values='Z')
        X, Y = np.meshgrid(position_pivot.index.values, position_pivot.columns.values, indexing='ij')
        Z = position_pivot.values

        nx = len(position_pivot.index)
        ny = len(position_pivot.columns)
        spectra_matrix = np.empty((nx, ny), dtype=object)

        for ix, x_val in enumerate(position_pivot.index):
            for iy, y_val in enumerate(position_pivot.columns):
                spectra_matrix[ix, iy] = spectra_map.get((x_val, y_val), None)

        self.position_pivot = position_pivot
        self.spectra_map = spectra_map        # dict keyed by (x,y)
        self.spectra_matrix = spectra_matrix
        self.slider.setMaximum(spectra_matrix.shape[0]-1 if self.rows_or_cols == 0 else spectra_matrix.shape[1]-1)
        return X, Y, Z, position_pivot

    def load_mesh(self):
        dir_path = self.dir_edit.text().strip()
        if not dir_path or not os.path.isdir(dir_path):
            QtWidgets.QMessageBox.warning(self, "No directory", "Please choose a valid directory containing data files.")
            return
        self.log.clear()
        self.log_msg("Building grid...")
        try:
            X, Y, Z, pivot = self.build_grid_from_dir(dir_path)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", str(e))
            return
        self.X, self.Y, self.Z = X, Y, Z
        self.position_pivot = pivot
        self.canvas.plot_mesh(X, Y, Z)

        # draw initial overlay at index 0
        self.selected_index = getattr(self, "selected_index", 0)
        if self.rows_or_cols == 0:
            x0 = self.position_pivot.index[self.selected_index]
            self.canvas.draw_overlay('row', float(x0))
        else:
            y0 = self.position_pivot.columns[self.selected_index]
            self.canvas.draw_overlay('col', float(y0))

        self.log_msg("Mesh displayed.")

    def process_grid(self):
        if self.Z is None:
            QtWidgets.QMessageBox.warning(self, "No data", "Load a mesh first.")
            return
        self.log_msg("Processing every row and column...")
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        QtWidgets.QApplication.processEvents()

        # example function: integrate (trapz) along each column and each row
        # rows correspond to X (axis 0), cols correspond to Y (axis 1)
        try:
            # For each row (fixed X), integrate Z[row, :] over Y (use pivot.columns)

            Spectra_fig, ax = plt.subplots(figsize=(8,6))

            cmap = plt.get_cmap("plasma")
            if self.rows_or_cols == 0:  # Rows
                # process selected row
                col_idx = self.selected_index
                spec_data = self.spectra_matrix[col_idx, :]
                n_cols = spec_data.size
                for iy, spectra in enumerate(spec_data):
                    if spectra is None:
                        self.log_msg(f"No spectra at column {col_idx}, col {iy}")
                        continue
                    bias = np.array(spectra['bias'])
                    demod_y = np.array(spectra['demod_y'])
                    color = cmap(iy / max(1, n_cols - 1))
                    ax.plot(bias, (demod_y + (iy*0.08e-12)), color=color, label=f'col {iy}')
                ax.set_title(f'Spectra for Row {col_idx}')
            elif self.rows_or_cols == 1:  # Columns
                row_idx = self.selected_index
                spec_data = self.spectra_matrix[:, row_idx]
                n_rows = spec_data.size
                for ix, spectra in enumerate(spec_data):
                    if spectra is None:
                        self.log_msg(f"No spectra at row {ix}, col {row_idx}")
                        continue
                    bias = np.array(spectra['bias'])
                    demod_y = np.array(spectra['demod_y'])
                    color = cmap(ix / max(1, n_rows - 1))
                    ax.plot(bias, (demod_y + (ix*0.08e-12)), color=color, label=f'Row {ix}')
                ax.set_title(f'Spectra for row {row_idx}')
            elif self.rows_or_cols == 2:  # Line (diagonal)
                ## process diagonal line
                print("Line option selected - not implemented.")
            
                rows = list(self.position_pivot.index)
                cols = list(self.position_pivot.columns)
                all_specs = []
                for ix, x_val in enumerate(rows):
                    for iy, y_val in enumerate(cols):
                        spec = self.spectra_matrix[ix, iy]
                        if spec is not None:
                            all_specs.append(spec)

                n_specs = len(all_specs)
                if n_specs == 0:
                    raise RuntimeError("No spectra found to display in Line mode.")

                # compute a sensible vertical spacing: median peak-to-peak of spectra
                p2ps = []
                for spec in all_specs:
                    y = np.array(spec['demod_y'])
                    if y.size > 0:
                        p2ps.append(np.nanmax(y) - np.nanmin(y))
                median_p2p = np.median(p2ps) if len(p2ps) else 1.0
                spacing = median_p2p * 1.01 if median_p2p > 0 else 1.0
                
                for idx, spec in enumerate(all_specs):
                    bias = np.array(spec['bias'])
                    demod_y = np.array(spec['demod_y'])
                    offset = idx * spacing
                    color = cmap(idx / max(1, n_specs - 1))
                    ax.plot(bias, demod_y + offset, color=color, alpha=0.9)
                ax.set_title(f'Waterfall (all spectra) — {n_specs} traces')
                # label shows offset units; you may adjust formatting
                ax.set_ylabel(f"Demod Y + offset ({spacing:.3e})")

            
            ax.set_xlabel('Bias (V)')
            ax.set_ylabel('Demod Y (A)')
            Spectra_fig.tight_layout()

            # embed in a Qt dialog (non-blocking for the main window)
            self._show_figure_dialog(Spectra_fig, title="Spectra Viewer")
            plt.close(Spectra_fig)  # close the standalone figure
            self.log_msg("Spectra displayed.")
                    
        except Exception as e:
            self.log_msg(f"Processing failed: {e}")
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
    
    def _show_figure_dialog(self, fig, title="Spectra"):
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle(title)
        layout = QtWidgets.QVBoxLayout(dlg)
        canvas = FigureCanvas(fig)
        layout.addWidget(canvas)
        btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
        btns.rejected.connect(dlg.reject)
        layout.addWidget(btns)
        dlg.resize(900, 650)
        canvas.draw()
        dlg.exec_()

def run_app(default_dir=None):
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow(start_dir=default_dir)
    w.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    run_app()