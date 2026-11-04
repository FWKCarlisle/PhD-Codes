import sys
import os
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QPushButton, QWidget, QFileDialog, QHBoxLayout, QLabel, QCheckBox, QScrollArea, QListWidget
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from force_analysis_oli import Spectrum  # Import your Spectrum class
from matplotlib.offsetbox import OffsetImage, AnnotationBbox



from PyQt5.QtWidgets import QListWidget

# class PlotWindow(QMainWindow):
#     def __init__(self):
#         super().__init__()
#         self.setWindowTitle("Spectrum Plot Generator")
#         self.setGeometry(100, 100, 1000, 600)

#         # Main widget and layout
#         self.main_widget = QWidget(self)
#         self.setCentralWidget(self.main_widget)
#         self.layout = QVBoxLayout(self.main_widget)

#         # Plot area
#         self.figure = plt.figure(figsize=(6, 4))
#         self.canvas = FigureCanvas(self.figure)
#         self.layout.addWidget(self.canvas, stretch = 3)

#         # Button area
#         self.button_layout = QHBoxLayout()
#         self.layout.addLayout(self.button_layout, stretch =1)

#         # Generate Plot Button
#         self.generate_button = QPushButton("Generate Data and Plot")
#         self.generate_button.clicked.connect(self.generate_data_and_update_plot)
#         self.button_layout.addWidget(self.generate_button)

#         # File Selection Button
#         self.file_button = QPushButton("Select Directory")
#         self.file_button.clicked.connect(self.select_directory)
#         self.button_layout.addWidget(self.file_button)

#         self.toggle_button = QPushButton("Check/Uncheck All")
#         self.toggle_button.clicked.connect(self.toggle_checkboxes)
#         self.button_layout.addWidget(self.toggle_button)


#         # Label to show selected directory
#         self.directory_label = QLabel("No directory selected")
#         self.layout.addWidget(self.directory_label)

#         self.file_data_layout = QHBoxLayout()

#         # Scroll area for checkboxes
#         self.scroll_area = QScrollArea()
#         self.scroll_area_widget = QWidget()
#         self.scroll_area_layout = QVBoxLayout(self.scroll_area_widget)
#         self.scroll_area.setWidget(self.scroll_area_widget)
#         self.scroll_area.setWidgetResizable(True)
#         self.file_data_layout.addWidget(self.scroll_area)

#         # List widget for Z_rel values
#         self.zrel_list = QListWidget()
#         self.file_data_layout.addWidget(self.zrel_list)

#         self.layout.addLayout(self.file_data_layout)
#         # Initialize variables
#         self.dir_path = None
#         self.checkboxes = {}  # Store checkboxes for each plot
#         self.all_checked = True

#     def select_directory(self):
#         # Open a file dialog to select a directory
#         self.dir_path = QFileDialog.getExistingDirectory(self, "Select Directory")
#         if self.dir_path:
#             self.directory_label.setText(f"Selected Directory: {self.dir_path}")
#             self.populate_checkboxes()
#         else:
#             self.directory_label.setText("No directory selected")

#     def populate_checkboxes(self, sort_by="number"):
#         """
#         Populate checkboxes based on files in the directory.
#         sort_by: "number" or "time"
#         """
#         # Clear existing checkboxes
#         for checkbox in self.checkboxes.values():
#             checkbox.deleteLater()
#         self.checkboxes.clear()

#         # Clear Z_rel list
#         self.zrel_list.clear()

#         # Get files in the directory
#         files = [f for f in os.listdir(self.dir_path) if os.path.isfile(os.path.join(self.dir_path, f))]

#         # Sort files
#         if sort_by == "number":
#             # Sort by numeric values in filenames
#             files.sort(key=lambda x: int(''.join(filter(str.isdigit, x))) if any(c.isdigit() for c in x) else 0)
#         elif sort_by == "time":
#             # Sort by time modified
#             files.sort(key=lambda x: os.path.getmtime(os.path.join(self.dir_path, x)))

#         # Add checkboxes for sorted files
#         for file in files:
#             checkbox = QCheckBox(file)
#             checkbox.setChecked(True)  # Default to checked
#             self.scroll_area_layout.addWidget(checkbox)
#             self.checkboxes[file] = checkbox

#             # Add Z_rel value to the list
#     def toggle_checkboxes(self):
#         """
#         Toggle the state of all checkboxes (check/uncheck all).
#         """
#         self.all_checked = not self.all_checked  # Toggle the state
#         for checkbox in self.checkboxes.values():
#             checkbox.setChecked(self.all_checked)      

#     def generate_data_and_update_plot(self):
#         if not self.dir_path:
#             self.directory_label.setText("Please select a directory first!")
#             return

#         # Clear the plot
#         self.figure.clear()
#         ax = self.figure.add_subplot(111)

#         self.zrel_list.clear()

#         # Generate data using the Spectrum class for selected files
#         for file, checkbox in self.checkboxes.items():
#             if checkbox.isChecked():  # Only plot if the checkbox is checked
#                 file_spec = Spectrum(path=self.dir_path, fileName=file, channel='OC M1 Freq. Shift (Hz)')
#                 current = file_spec.x
#                 z = file_spec.give_data('Z (m)')[0][0]
#                 zrel = file_spec.zrel

#                 # Plot the data
#                 ax.plot(np.arange(0, len(current)), current, "o-", label=f'{file}')
#                 self.zrel_list.addItem(f"{file}: Z_rel = {zrel*1e12:.2f}pm")

#         # Update the plot
#         ax.set_xlabel('Indices')
#         ax.set_ylabel('Current (A)')
#         ax.legend()
#         self.canvas.draw()




from PyQt5.QtWidgets import QComboBox
from PyQt5.QtWidgets import QCheckBox

class PlotWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Spectrum Plot Generator")
        self.setGeometry(100, 100, 1000, 600)

        # Main widget and layout
        self.main_widget = QWidget(self)
        self.setCentralWidget(self.main_widget)
        self.layout = QVBoxLayout(self.main_widget)

        # Plot area
        self.figure = plt.figure(figsize=(8, 6))
        self.canvas = FigureCanvas(self.figure)
        self.layout.addWidget(self.canvas, stretch=3)

        # Button area
        self.button_layout = QHBoxLayout()
        self.layout.addLayout(self.button_layout, stretch=1)

        # Generate Plot Button
        self.generate_button = QPushButton("Generate Data and Plot")
        self.generate_button.clicked.connect(self.generate_data_and_update_plot)
        self.button_layout.addWidget(self.generate_button)

        # File Selection Button
        self.file_button = QPushButton("Select Directory")
        self.file_button.clicked.connect(self.select_directory)
        self.button_layout.addWidget(self.file_button)

        # Check/Uncheck All Button
        self.toggle_button = QPushButton("Check/Uncheck All")
        self.toggle_button.clicked.connect(self.toggle_checkboxes)
        self.button_layout.addWidget(self.toggle_button)

        # Dropdown menu for selecting what to plot
        self.plot_dropdown = QComboBox()
        self.plot_dropdown.addItems(["Current", "Frequency Shift", "X", "Y", "Z"])
        self.button_layout.addWidget(self.plot_dropdown)

        # Checkbox for including Z_rel list
        self.include_zrel_checkbox = QCheckBox("Include Z_rel List")
        self.include_zrel_checkbox.setChecked(False)  # Default to checked
        self.button_layout.addWidget(self.include_zrel_checkbox)

        self.sub_layout = QHBoxLayout()

        # Label to show selected directory
        self.directory_label = QLabel("No directory selected")
        self.sub_layout.addWidget(self.directory_label, stretch=1)

        # Scroll area for checkboxes
        self.scroll_area = QScrollArea()
        self.scroll_area_widget = QWidget()
        self.scroll_area_layout = QVBoxLayout(self.scroll_area_widget)
        self.scroll_area.setWidget(self.scroll_area_widget)
        self.scroll_area.setWidgetResizable(True)
        self.sub_layout.addWidget(self.scroll_area, stretch=1)

        # List widget for Z_rel values
        self.zrel_list = QListWidget()
        self.sub_layout.addWidget(self.zrel_list, stretch=1)
        self.layout.addLayout(self.sub_layout)

        # Initialize variables
        self.dir_path = None
        self.checkboxes = {}  # Store checkboxes for each plot
        self.all_checked = True  # Track the state of the check/uncheck all button

    def select_directory(self):
        # Open a file dialog to select a directory
        self.dir_path = QFileDialog.getExistingDirectory(self, "Select Directory")
        if self.dir_path:
            self.directory_label.setText(f"Selected Directory: {self.dir_path}")
            self.populate_checkboxes()
        else:
            self.directory_label.setText("No directory selected")

    def populate_checkboxes(self, sort_by="number"):
        """
        Populate checkboxes based on files in the directory.
        sort_by: "number" or "time"
        """
        # Clear existing checkboxes
        for checkbox in self.checkboxes.values():
            checkbox.deleteLater()
        self.checkboxes.clear()

        # Clear Z_rel list
        self.zrel_list.clear()

        # Get files in the directory
        files = [f for f in os.listdir(self.dir_path) if os.path.isfile(os.path.join(self.dir_path, f))]

        # Sort files
        if sort_by == "number":
            files.sort(key=lambda x: int(''.join(filter(str.isdigit, x))) if any(c.isdigit() for c in x) else 0)
        elif sort_by == "time":
            files.sort(key=lambda x: os.path.getmtime(os.path.join(self.dir_path, x)))

        # Add checkboxes for sorted files
        for file in files:
            checkbox = QCheckBox(file)
            checkbox.setChecked(True)  # Default to checked
            self.scroll_area_layout.addWidget(checkbox)
            self.checkboxes[file] = checkbox

    def toggle_checkboxes(self):
        """
        Toggle the state of all checkboxes (check/uncheck all).
        """
        self.all_checked = not self.all_checked  # Toggle the state
        for checkbox in self.checkboxes.values():
            checkbox.setChecked(self.all_checked)

    def generate_data_and_update_plot(self):
        if not self.dir_path:
            self.directory_label.setText("Please select a directory first!")
            return

        # Clear the plot
        self.figure.clear()
        ax = self.figure.add_subplot(111)

        # Clear Z_rel list if the checkbox is checked
        if self.include_zrel_checkbox.isChecked():
            self.zrel_list.clear()

        # Get the selected plot type from the dropdown menu
        plot_type = self.plot_dropdown.currentText()

        # Generate data using the Spectrum class for selected files
        indices_addition = 0
        count = 0
        for file, checkbox in self.checkboxes.items():
            if checkbox.isChecked():  # Only process files with checked checkboxes
                file_spec = Spectrum(path=self.dir_path, fileName=file, channel='OC M1 Freq. Shift (Hz)')
                indicies = np.arange(indices_addition, indices_addition+len(file_spec.x))
                time_step = 0.001 #s
                time_seconds = indicies * time_step
                indices_addition = indicies[-1] + 1  # Update the starting index for the next file
                # Select the data to plot based on the dropdown menu
                if plot_type == "Current":
                    data_to_plot = file_spec.x
                elif plot_type == "Frequency Shift":
                    data_to_plot = file_spec.y
                elif plot_type == "X":
                    data_to_plot = file_spec.give_data('X (m)')[0]
                elif plot_type == "Y":
                    data_to_plot = file_spec.give_data('Y (m)')[0]
                elif plot_type == "Z":
                    data_to_plot = file_spec.give_data('Z (m)')[0]

                # Plot the data
                ax.plot(time_seconds, data_to_plot*1e9, "o-", label=f'Atom track file {count + 1}')
                
                # Add Z_rel value to the list if the checkbox is checked
                if self.include_zrel_checkbox.isChecked():
                    zrel_value = file_spec.zrel
                    self.zrel_list.addItem(f"{file}: Z_rel = {zrel_value*1e12:.2f}pm")
                count +=1
        # Update the plot
        arr_img = plt.imread(r"C:\Users\ppxfc1\Downloads\STM_data.png")
        im = OffsetImage(arr_img, zoom=0.35)
        ab = AnnotationBbox(im, (0.75, 0.35), xycoords='axes fraction')
        ax.add_artist(ab)
        ax.set_xlabel('Time (s)', fontsize=16)
        ax.set_ylabel("X position (nm)", fontsize=16)
        ax.legend()
        self.canvas.draw()
        plt.savefig(rf"C:\Users\ppxfc1\Downloads\STM_data_{count}.png", dpi=300, bbox_inches='tight')
        
        
app = QApplication(sys.argv)
window = PlotWindow()
window.show()
sys.exit(app.exec_())