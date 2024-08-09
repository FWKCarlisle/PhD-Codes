# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 10:49:30 2022

@author: Matthew Edmondson

v1 of the pressure monitor program to display system pressures on the PC 
screen. Using the .ini file, the program sets up the initial window for you. 
Update the .ini file to include anything you want. 
"""


#%% Import required packages

import configparser
import serial
import re
import pathlib




#%% Import the config file

config_path = pathlib.Path(__file__).parent.absolute() / 'pressure_monitor_parameter.ini'

# instantiate
config = configparser.ConfigParser()

# parse existing file
config.read(config_path)


#%% Read the file's data
num_of_input = int(config["Setup"]["Number_of_inputs"])
email = config["Setup"]["email_to_send"]

input1 = config["inputname"]["Input1_str"]
input2 = config["inputname"]["Input2_str"]
input3 = config["inputname"]["Input3_str"]
input4 = config["inputname"]["Input4_str"]
input5 = config["inputname"]["Input5_str"]

input1_warn = config["Pressure_warn_value"]["Input1_warn"]
input2_warn = config["Pressure_warn_value"]["Input2_warn"]
input3_warn = config["Pressure_warn_value"]["Input3_warn"]
input4_warn = config["Pressure_warn_value"]["Input4_warn"]
input5_warn = config["Pressure_warn_value"]["Input5_warn"]

time_spacing = int(config["Setup"]["time_update_milli_seconds"])
warn_spacing = float(config["Setup"]["warn_timer_minutes"])

#COM port details!
COM_port = config["COM_port_details"]["port"]
COM_baudrate = config["COM_port_details"]["baudrate"]
COM_timeout = config["COM_port_details"]["timeout"]

#%% Setup the connection here...

#Setup the connection to the controller
ser = serial.Serial()
ser.baudrate = int(COM_baudrate)
ser.port = "/dev/"+COM_port
ser.timeout = int(COM_timeout)
ser
ser.open()



#%%
import sys
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QMainWindow, QLabel, QGridLayout, QWidget, QCheckBox, QPushButton
from PyQt5.QtCore import QSize, QTimer
from PyQt5.QtGui import QFont


class HelloWindow(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        
        self.setMinimumSize(QSize(900, 500))    
        self.setWindowTitle("Pressure Monitor") 
        
        centralWidget = QWidget(self)          
        self.setCentralWidget(centralWidget)   
 
        gridLayout = QGridLayout(self)     
        centralWidget.setLayout(gridLayout)  
 
        title = QLabel("Pressure location", self) 
        title.setAlignment(QtCore.Qt.AlignLeft)
        gridLayout.addWidget(title, 0, 0)
        title.setFont(QFont('Sans Serif 10',50))
        
        #self.warn_text = QLabel("Warn status: ", self) 
        #self.warn_text.setAlignment(QtCore.Qt.AlignLeft)
        #gridLayout.addWidget(self.warn_text, 0, 1)
        
        #self.reset_warn = QPushButton('Reset Warn Timer', self)
        #self.reset_warn.clicked.connect(self.on_clicked_reset_warn)
        #gridLayout.addWidget(self.reset_warn, 0, 2)
        
        #self.warn_timer = warn_spacing + 1
        
        
        if num_of_input >= 1:
            #Setup the label
            self.pressure1 = QLabel(input1, self) 
            
            self.pressure1.setAlignment(QtCore.Qt.AlignLeft)
            gridLayout.addWidget(self.pressure1, 1, 0)
            self.pressure1.setFont(QFont('Sans Serif 10',50))
            
            #Now setup the checkbox for warning... 
            #self.cb_pressure1 = QCheckBox('Warn?', self)
            #gridLayout.addWidget(self.cb_pressure1, 1, 1)
            
            self.input1_warn = float(input1_warn)
            
            
        
        if num_of_input >= 2:
            self.pressure2 = QLabel(input2, self) 
            self.pressure2.setAlignment(QtCore.Qt.AlignLeft)
            gridLayout.addWidget(self.pressure2, 2, 0)
            self.pressure2.setFont(QFont('Sans Serif 10',50))
            
            #Now setup the checkbox for warning... 
            #self.cb_pressure2 = QCheckBox('Warn?', self)
            #gridLayout.addWidget(self.cb_pressure2, 2, 1)
            
            self.input2_warn = float(input2_warn)
            
            
        
        if num_of_input >= 3:
            self.pressure3 = QLabel(input3, self) 
            self.pressure3.setAlignment(QtCore.Qt.AlignLeft)
            gridLayout.addWidget(self.pressure3, 3, 0)
            
            #Now setup the checkbox for warning... 
            #self.cb_pressure3 = QCheckBox('Warn?', self)
            #ridLayout.addWidget(self.cb_pressure3, 3, 1)
            
            self.input3_warn = float(input3_warn)
            
            
        
        
        
        # Every (time_spacing)ms the system will run self.onTimeout!
        timer = QTimer(self)
        timer.timeout.connect(self.onTimeout)
        timer.start(time_spacing)
        
        self.x = 0
    
    def closeEvent(self,*args):
        # When the window is closed, the serial port is closed! 
        super(QMainWindow,self).closeEvent(*args)
        ser.close()
        #print('Test, closed')
        
    def send_email_note(self):
        outlook = win32.Dispatch('outlook.application')
        mail = outlook.CreateItem(0)
        mail.To = email
        
        if self.pressure_errors == 0:
            mail.Subject = 'Pressure change detected in '+ self.name_of_sensor
            mail.Body = 'Hi there, \
            The pressure in the POLAR has increased. Please check this is OK. \
            Presure = '+str(self.pressure_value)+' mbar\
            Thanks, POLAR :)'
    
        elif self.pressure_errors == 1:
            mail.Subject = 'Pressure error detected in '+ self.name_of_sensor
            mail.Body = 'Hi there, \
            The pressure in the POLAR cannot be read. Please check this is OK. \
            Thanks, POLAR :)'
    
        mail.Send()
        
    
        
    def on_clicked_reset_warn(self):
            self.warn_timer = warn_spacing + 1 
     
    def update_warning(self):
        if self.warn_timer > warn_spacing:
            self.warn_text.setText("Warn status: Able to Warn")
        else:
            self.warn_text.setText("Warn status: Cannot Warn")
            
    def warning_tests(self):
        
        if self.input_selection == 1:
            self.pressure_value = self.all_pressure_values[0]
            test_pressure = self.input1_warn
            checked_val = self.cb_pressure1.isChecked()
            
        elif self.input_selection == 2:
            self.pressure_value = self.all_pressure_values[1]
            test_pressure = self.input2_warn
            checked_val = self.cb_pressure2.isChecked()
            
        elif self.input_selection == 3:
            self.pressure_value = self.all_pressure_values[2]
            test_pressure = self.input3_warn
            checked_val = self.cb_pressure3.isChecked()
            
        #Is the pressure high enough to warn?
        if self.pressure_value>test_pressure:
            # Only warn if the button is ticked!
            if checked_val:
                # now check if enough time has past?
                if self.warn_timer > warn_spacing:
                    #able to warn
                    print('Warning that pressure too high')
                    
                    if self.input_selection == 1:
                        self.name_of_sensor = input1
                    elif self.input_selection == 2:
                        self.name_of_sensor = input2
                    elif self.input_selection == 3:
                        self.name_of_sensor = input3
                    
                    #Will send email
                    self.send_email_note()
                    
                    #also reset timer for 
                    self.warn_timer = 0
                else:
                    print('not enough time')
        
    
    
    def get_pressure_from_device(self):
        #Asks the controller for all pressures. 
        ser.write('#000F\r'.encode())
        read_pressure_data = ser.readline()[1:-1]
        ser.flush()
        test = read_pressure_data.decode("utf-8")
        #Splits into two pressures here:
        self.all_pressure_values = re.split(',',test)
        
        count = 0
        self.pressure_errors = 0
        for i in self.all_pressure_values[0:num_of_input]:
        
            try:
                self.all_pressure_values[count] = float(i)
            except: 
                self.all_pressure_values[count] = 1000
                self.pressure_errors = 1
                
            count = count + 1

    
    
    
    
    
        
    def onTimeout(self):
        
        #Update able to warn text:
        #self.update_warning()
        
        #now check each pressure!
        self.get_pressure_from_device()
        
        if num_of_input >= 1:
            self.input_selection = 1
            self.x = self.x + 1
            self.pressure1.setText(input1+' = '+ "%1.2e" %self.all_pressure_values[0] +' mbar')
            
            #if pressure too high set to red
            if self.all_pressure_values[0]>self.input1_warn:
                self.pressure1.setStyleSheet("background-color: red")
            else:
                self.pressure1.setStyleSheet("background-color: lightgreen") 
                
            #Test if can send!    
            #self.warning_tests()
                
            
        if num_of_input >= 2:
            self.input_selection = 2
            self.x = self.x + 1
            self.pressure2.setText(input2+' = '+"%1.2e" %self.all_pressure_values[1] +' mbar')
            
            #if pressure too high set to red
            if self.all_pressure_values[1]>self.input2_warn:
                self.pressure2.setStyleSheet("background-color: red")
            else:
                self.pressure2.setStyleSheet("background-color: lightgreen") 
            
            #Test if can send!    
            #self.warning_tests()
            
            
            
        if num_of_input >= 3:
            self.input_selection = 3
            self.x = self.x + 1
            self.pressure3.setText(input3+' = '+"%1.2e" %self.all_pressure_values[2] +' mbar')
            
            #if pressure too high set to red
            if self.all_pressure_values[2]>self.input3_warn:
                self.pressure3.setStyleSheet("background-color: red")
            else:
                self.pressure3.setStyleSheet("background-color: lightgreen") 
            
            #Test if can send!    
            #self.warning_tests()
                
                
            
        
        #Add on time_spacing in minutes for after the last warning! 
        #self.warn_timer = self.warn_timer + (time_spacing/1000)/60
        #print(self.warn_timer)
        
        
        
        
        
 
if __name__ == "__main__":
    def run_app():
        app = QtWidgets.QApplication(sys.argv)
        app.setQuitOnLastWindowClosed(True)
        mainWin = HelloWindow()
        mainWin.show()
        sys.exit(app.exec_())
        app.exec_()
        
        
    run_app()
    









