# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 10:14:47 2023

@author: ppzme

Aim of this code is to interface with Nanonis v5e to control the scan using 
TCP interface. 

We must convert the text to hexadecimal code. This code provides a wrapper to 
interface with the TCP port in a easier way. You should not need to worry 
about the TCP interfacing, and only consider your experiment! 

From the Doc: 
The Nanonis software works as a TCP Server, and a remote application works as a TCP Client. The TCP Server 
listens at the ports specified in the Options window (under the System menu). The Client can open one, two, three or 
four different connections to the Server at these ports.
Each individual connection handles commands serially (one command after another) guaranteeing synchronized 
execution. These connections can be found and configured in the Options window (under the System menu).
Every message sent from the client to the server (request message), and viceversa (response message) when Send 
response back is set to True in the request message (see Request message>Header section), consists of header and 
body.
All numeric values are sent in binary form (e.g. a 32 bit integer is encoded in 4 bytes). The storage method of binary 
encoded numbers is big-endian, that is, the most significant byte is stored at the lowest address.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
v6 Change Log:
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

1) Change behaviour of the scan detector hold function, allow it to detect if the scan isn't running due to either
    ZSpec or BiasSpec.
    This means that spectra is not counted as 'stopped'. So that you can run a spectra and it not 
    count as completed the scan. 
    ### Need to deal with always needing to open them to check... 
    ### (Could check if the windows are closed by the error message?)

2) Add function: Z_SpectraStatusGet, Z_SpectraStart, Z_SpectraWinOpen
    
3) Added Continuous scan check in the scan action, if it is on, the code will turn it off. 
    
4) Added Bias pulse
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
v7 Change log: 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    1) Move the user defined functions to end of this class (hopefully this makes it more easy to distinush between
    'user' defined functions and Nanonis functions that should be touched unless needed! 
    
    2) Made the 'nano' class setup have the socket info within it!
    Allows for multiple classes to be made: 
    x = python_Nanonis_TCP(6501)
    x1 = python_Nanonis_TCP(6502)
    
    or just use the standard:
    x = python_Nanonis_TCP() # to open using 6501 port
    
    3) Added the following Nanonis functions: 
    LockIn_ModSignalSet [not working on demo mode]
    
    4) Added the following user defined functions:
    lock_in_setup() [Allows for one function setup for dI/dV measurements]
                                                         
    
    

"""


import matplotlib.pyplot as plt
import numpy as np
import struct
import socket
import time


class python_Nanonis_TCP():
    
    '''
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Class Setup
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    Note: These functions should not need to be touched! 
    '''
    
    def __init__(self, socket_number=6501):
        
        """This function runs when the class is created. It creates the \'types of data\' to be referenced in the code.
        
        To see the sent and recieved full message, set self.debug_mode = 1
        To see the error messages every time, set self.error_status_every_fun = 1
        To see the values the function returns in the console, use self.supress_return_value_print = 1
        
        The types_of_data contain the name of the type in string format in python dict, with some addtional infomation
        that corresponds to the number of bytes, or if we need it to 'escape' the normal execution path as it is a 
        special file format (e.g 1D string array that has vairbale byte length and that must be determined)."""
        
        #Setup the dictionary for types of values used.
        #Note the system uses big-endian => '>'
        self.types_of_data = {'string' : ['','string',''], 'int': ['>i',4,''], 'uint16': ['>H',2,''] \
                         ,'uint32': ['>I',4,''], 'float32': ['>f',4,''], 'float64': ['>d',8,''] \
                             , '1Dfloat32': ['>f', 4,'1D'],'1Dint': ['>i', 4,'1D'], '1Dstringarray':['>I',4,'1Dstringarray'] \
                                 , '2Dfloat32': ['>f', 4,'2D']}
        
            
        #Set debug mode to be 0... will not be in debug mode
        #Use to output all response!
        self.debug_mode = 0
        self.error_status_every_fun = 0 #Make sure that we don't say it was OK every run
        self.supress_return_value_print = 0 #Prevent values printing if set to 0
        
        #Note the system uses big-endian => '>'
        self.set_up_socket(socket_number)
        
        #Set the lenght of the response
        self.response_length = 50000
        
        
    def set_up_socket(self, socket_number):
        """Sets up the connection between python and Nanonis via the TCP protocal on the 6501 port. 
        
        Other ports may be chosen by calling 'self.set_up_socket(6502)' to setup the 6502 port.
        The timeout on the response is 30 seconds. This may be extended for the case of spectra where
        the reponse is required and is 'locking' (preventing the excution of the rest of the python script). """
        
        #Sets up the connection
        host = socket.gethostname()
        #port = 6501                   # The same port as used by the server
        port = socket_number
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.settimeout(30) #30 seconds timeout
        self.s.connect((host, port))
    
    def close_socket(self):
        """Closes the port to the TCP protocal."""
        self.s.close()
    
    
    '''
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Command functions to allow request to be setup/recieved correctly
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    Note: These functions should not need to be touched! 
    '''
    
        
    def command_convert_to_hex(self,command):
        """Converts the command to a 32 byte length as required. 
        
        This will set up the command to be :
        Command name (string) (32 bytes) is the name of the executed command. 
        It matches one of the function names described in the Functions section
        of this document (i.e. BiasSpectr.Open). Maximum number of characters 
        is 32.
        """
        #Need to check that it will not be longer than needed! 
        length_command = len(command)
        if length_command > 32:
            raise ValueError('You cannot have a string that gives a command larger than 32 Bytes!')
            
        #Now we must now convert to byte string and add the padding! 
        self.command_name = command.encode('utf-8') + (0).to_bytes(32-length_command,byteorder='big')
        

    def body_size(self):
        """Works out the length of the body of the request.
        
        In the header of the request, the length of the body must be provided. This is worked out 
        from the length of bytes (as determined by the bytes for each data type).
        
        This is done in the != 'string' and != '1D'. 
        In the case of strings == 'string' and != '1D'
        In the case of 1D arrays of values == '1D'    
        
        This value is then packed into an 'int' number.
        """
        
        #Create body size of the request. 
        
        if len(self.sending_infomation_types) == 0:
            #Not sending anything... therefore, we can set this to 4, as this is the minimum body size
            self.body_size_value = 4
            #self.body_size_command = (self.body_size_value).to_bytes(4,byteorder='big')
            self.body_size_command = struct.pack( self.types_of_data['int'][0], self.body_size_value)
            
        else: 
            #Add up the numbers of bytes required.
            self.body_size_value = 4
            for index, i in enumerate(self.sending_infomation_types):
                
                #If string, we need to make it the length of the string
                if self.types_of_data[i][1] != 'string' and self.types_of_data[i][2] != '1D':
                    self.body_size_value += self.types_of_data[i][1]
                elif self.types_of_data[i][1] == 'string' and self.types_of_data[i][2] != '1D':
                    list_return = list(self.sending_infomation.keys())
                    self.body_size_value += len(self.sending_infomation[list_return[index]])
                
                elif self.types_of_data[i][2] == '1D':
                    list_return = list(self.sending_infomation.keys())
                    self.body_size_value += self.types_of_data[i][1]*self.sending_infomation[list_return[self.sending_special_length_loc[index]]]
                    
                
                
            #self.body_size_command = (self.body_size_value).to_bytes(4,byteorder='big')
            self.body_size_command = struct.pack( self.types_of_data['int'][0], self.body_size_value)
        
        
    def response_required(self):
        """Sets the self.response_command to be 1 for a response required, or 0 for no response in 2 bytes."""
        
        #Do we require a response?
        if self.response_required_value == 'Yes':
            self.response_command = (1).to_bytes(2,byteorder='big')
        elif self.response_required_value == 'No':
            self.response_command = (0).to_bytes(2,byteorder='big')
        else:
            raise TypeError('You need to ensure the response is required (\'Yes\') or (\'No\')')
            
        if self.debug_mode != 0:
            #Need a response regardless!! 
            self.response_command = (1).to_bytes(2,byteorder='big')
            
    
    
    def command_creater(self):
        
        """Uses other fucntions to create the header to the command, and forms the command byte string to be sent.
        
        This function can deal with the 0D values, 0D strings, and 1D arrays of values. This is done in the body of the
        message section. I recomend if you want to add addtional functionalilty, consider the body size too. """
        
        #Creates the header to the command (40 bytes in length)
        self.headercommand_to_send = self.command_name + self.body_size_command + self.response_command + b'\x00\x00'
        
        #This is creating the end of the command
        self.wait_for_data = (1).to_bytes(4,byteorder='big')
        
        #This creates the body of the message.
        self.body_data_to_send = b''
        count = 0
        
        if self.body_size_value > 4:
            for i in self.sending_infomation:
                
                if isinstance(self.sending_infomation[i],str)==0:
                    #This is for value types
                    try:
                        #if only one length
                        self.body_data_to_send = self.body_data_to_send + struct.pack(self.types_of_data[self.sending_infomation_types[count]][0],self.sending_infomation[i])
                    except:
                        #in the case where the values are a 1D array, the above will fail and this will do what is required. 
                        for i1 in self.sending_infomation[i]:
                            self.body_data_to_send = self.body_data_to_send + struct.pack(self.types_of_data[self.sending_infomation_types[count]][0],i1)
                          
                elif isinstance(self.sending_infomation[i],str):
                    #This is for string types
                    self.body_data_to_send = self.body_data_to_send + self.sending_infomation[i].encode('utf-8')
                count += 1
            
            
        #Sets the full command together
        command = self.headercommand_to_send + self.body_data_to_send + self.wait_for_data
        
        return command
    
    
    
    def send_recv_command(self, command):
        
        """This send/recieves the data from the port.
        
        The response is not always collected, as this is something the user can chose. Note: this must be also set in 
        the header to the command. The length of the recv is set at 5000, as it may be this long if you ask for a 
        list of all of the availble channels.
        
        Note: this command may fail, if you did not close the connection (using self.close_socket()) and you
        restarted the connection by calling the class again. """
        
        if self.debug_mode == 1:
            print('Header:')
            print(command[:40])
            print('Body:')
            print(command[40:])
        
        self.s.sendall(command)
        #print('Sent')
        #print(command)
        
        if self.response_required_value == 'Yes' or self.debug_mode != 0:     
            try: 
                self.data = self.s.recv(self.response_length)
                if self.debug_mode != 0:
                    print('Output:')
                    print(self.data)
                    
            except: 
                raise TypeError('You closed the python script without using self.close_socket(). \n The socket is still open. \n Restart the kernal.')
        
        
        #print('Recieved')
        #print(self.data)
        
        
    def convert_output_to_readable(self, given_infomation, infomation_types):
        """ Converting the recieved data to something that is usable.
        
        The given_infomation is a dictionary of what we expect to reviece, with infomation_types being the 
        expected byte length and types to correctly unpack this data. 
        
        The first 40 bytes are the header to the response. The data is given by the body.  
        
        We have three data types:
        1 - Normal values (+ 0D string) (== '')
        These are dealt with by finding the position in the body of the message and 
        splicing the data from the body. then upacking it and seting it equal to the 
        dictionary key. The 0D string is and extenstion of this, and relies on the 
        self.special_length_loc vairable telling us how long the string is in bytes,
        we should write this in the function if required... ususually the length is 
        the value before, but not always... 
        
        2 - 1D arrays of values (== '1D')
        Similar to the normal values, but relies on the self.special_length_loc
        variable telling us how long the 1D array is. 
        
        3 - 1D array of strings
        Similar to #2, but the number of string vairables is given in the pervious 
        value (given by self.special_length_loc), with the length of the string given by 
        and int directly before the string. We loop over the number of strings, and 
        extreact them as a list in the dict. 
        """
        
        header = self.data[:40]
        body = self.data[40:]
        
        #Not required, but we can extract info from the header
        #print(header)
        #print(body)
        
        base_bytes = 0 
        list_return = list(given_infomation.keys())
        
        for index, i in enumerate(given_infomation):
            
            if self.types_of_data[infomation_types[index]][2] == '':
                #This is a normal only number type
                type_of_data = self.types_of_data[infomation_types[index]][0]
                bytes_count = self.types_of_data[infomation_types[index]][1]
                
                if bytes_count == 'string':
                    bytes_count = given_infomation[list_return[self.special_length_loc[index]]]
                    given_infomation[i] = body[base_bytes: base_bytes+bytes_count].decode("utf-8") 
                else:
                    given_infomation[i] = struct.unpack(type_of_data, body[base_bytes: base_bytes+bytes_count])[0]
                
                base_bytes += bytes_count
                
                
            elif self.types_of_data[infomation_types[index]][2] == '1D':
                # There are given_infomation[list_return[self.special_length_loc[index]]] spaces to go through to form the 1D array
                # Number in array given by the self.special_length_loc[index] array...
                temp_variable = []
                for i1 in range(given_infomation[list_return[self.special_length_loc[index]]]):
                    type_of_data = self.types_of_data[infomation_types[index]][0]
                    bytes_count = self.types_of_data[infomation_types[index]][1]
                    temp_variable.append(struct.unpack(type_of_data, body[base_bytes: base_bytes+bytes_count])[0])
                    base_bytes += bytes_count
                
                given_infomation[i] = temp_variable
                
                
            elif self.types_of_data[infomation_types[index]][2] == '1Dstringarray':
                #When a strong array need to find out how long it will be....
                #This part is in the self.special_length_loc
                length_of_1D_string_array = given_infomation[list_return[self.special_length_loc[index]]]
                
                all_1Darray = body[base_bytes:base_bytes+length_of_1D_string_array]
                bytes_count_of_value = self.types_of_data[infomation_types[index]][1]
                inital_base_bytes = base_bytes
                
                temp_variable = []
                
                while base_bytes < length_of_1D_string_array+inital_base_bytes:
                    length_of_string = struct.unpack('>i', body[base_bytes:base_bytes+bytes_count_of_value] )[0]
                    base_bytes += bytes_count_of_value
                    temp_variable.append(body[base_bytes:base_bytes+length_of_string])
                    base_bytes += length_of_string
                    
                given_infomation[i] = temp_variable
            
            elif self.types_of_data[infomation_types[index]][2] == '2D':
                #for 2D arrays, it has a row and col value. Need to find the full length. Can reshape later. 
                temp_variable = []
                size_of_2D_array = self.special_length_loc[index]
                
                for index1, i2 in enumerate(size_of_2D_array):
                    if index1 == 0:
                        #When it is the first in the loop
                        size_of_2d_array_value = given_infomation[list_return[i2]]
                    else:
                        #second in the loop
                        size_of_2d_array_value = size_of_2d_array_value * given_infomation[list_return[i2]]
                
                for i1 in range(size_of_2d_array_value):
                    type_of_data = self.types_of_data[infomation_types[index]][0]
                    bytes_count = self.types_of_data[infomation_types[index]][1]
                    temp_variable.append(struct.unpack(type_of_data, body[base_bytes: base_bytes+bytes_count])[0])
                    base_bytes += bytes_count
                
                given_infomation[i] = temp_variable
                
        
        return given_infomation
    
    
    
    def deal_with_error(self):
        """Determines if there is an error, looking at the specific bytes in the body. It will print the error. 
        
        The first 4 bytes in the body are False if no error, or true if there is an error. """
        
        #Now we deal with the error. 
        #The next to the last four bytes of the header tell us how long the body is in bytes... 
        count_bytes_to_error = 0
        
        if len(self.recieved_infomation_types) > 0:
            #If not empty
            for i in self.recieved_infomation_types:
                count_bytes_to_error += self.types_of_data[i][1]
        else:
            #If empty...
            count_bytes_to_error = 0
            
        #Now what is the status of the error? (4 bytes)
        header_body_defined_length = self.data[40-4-4:40-4]
        body = self.data[40:]
        error_status = body[count_bytes_to_error:]
        
        if struct.unpack('>I',error_status[0:4])[0] == 0:
            #Now this is showing no error
            if self.error_status_every_fun == 1:
                print('Ran command with no error')
            
        else: 
            #There is an error
            print(error_status[8:])
            
            
        if self.debug_mode == 1:  
            print('Full debug mode output:' + str(self.data))
            
            
        
        #We can use the 'self.recieved_infomation_types' to give info about where to expect the data
        
        
    
    
    
    def run_command_fun(self, command_name, convert_required):
        """
        Function to bring together all command creating functions together. From the command and whether you want the response converted to a readable value.  
        """
        
        #Makes, and then sends the command
        self.command_convert_to_hex(command_name)
        self.body_size()
        self.response_required()
        command = self.command_creater()
        self.send_recv_command(command)
        
        
        # Runs the converter of the response if required
        self.return_output_data = []
        if convert_required == 'Yes':
            given_infomation = self.convert_output_to_readable(self.recieved_infomation, self.recieved_infomation_types)
            
            if self.supress_return_value_print != 0:
                print(given_infomation)
                
            self.return_output_data = given_infomation
        
        if self.print_error == 'Yes':
            self.deal_with_error()
            #print('here')
            
            
            
    '''
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Nanonis Functions - Commands 
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    '''
        
            
    
    def XY_tip_pos(self):
        """
        ()
        Finds the current position of the tip in X and Y. Returns in (m). 
        
        Returns
        -------
        X-Position (float)
            The X position of the tip.
        Y-Position (float)
            The Y position of the tip.

        """
        
        #Command setup:
        command_name = 'FolMe.XYPosGet'
        self.sending_infomation = {'Wait?':0}
        self.sending_infomation_types = ['uint32']
        
        
        self.response_required_value = 'Yes'
        
        #Do you want to see the error?
        self.print_error = 'Yes'
        
        #How to deal with returned data:
        self.recieved_infomation = {'X(m)':'','Y(m)':''}
        self.recieved_infomation_types = ['float64','float64']
        list_return = list(self.recieved_infomation.keys())
        
        #Do we need it to convert for us?  
        convert_required='Yes'
        
        #Run the function to generate command        
        self.run_command_fun(command_name, convert_required)
        
        #Uncomment if you want it to return values
        return self.return_output_data[list_return[0]], self.return_output_data[list_return[1]]
        
        
    def XY_tip_set_pos(self,X,Y):
        """ 
        (X, Y)
        Set the X & Y position of the tip. Input using (m) units. e.g: 1e-9 = 1 nm   

        Parameters
        ----------
        X : X-Position (float)
            The X position to move the tip. 
        Y : Y-Position (float)
            The Y position to move the tip.

        Returns
        -------
        None.

        """
        
        #ste to move immediately! 
        
        #Command setup:
        command_name = 'FolMe.XYPosSet'
        self.sending_infomation = {'X(m)': X,'Y(m)': Y,'Wait?':0}
        self.sending_infomation_types = ['float64','float64','uint32']
        
        
        self.response_required_value = 'No'
        
        
        #Do you want to see the error?
        self.print_error = 'No'
        
        #How to deal with returned data:
        self.recieved_infomation = {}
        self.recieved_infomation_types = []
        list_return = list(self.recieved_infomation.keys())
        
        #Do we need it to convert for us?  
        convert_required= 'No'
        
        #Run the function to generate command        
        self.run_command_fun(command_name, convert_required)
        
        #Uncomment if you want it to return values
        #return self.return_output_data[list_return[0]]
        
        
    def XY_tip_move_set_speed(self, speed, custom_or_scan_speed):
        """
        (speed, custom_or_scan_speed)
        Set a 'custom' slower XY follow me tip movement speed. Or set the same as that given by the scan speed. 
        

        Parameters
        ----------
        speed : Tip speed  (float)
            Only for 'custom' type. Set speed of the tip in (m/s). 1e-9 = 1 nm/s. If 'scan', this parameter is ignored. 
        custom_or_scan_speed : type of scan (string or int)
            Set to 'custom' (or 1) for custom tip move speed. Set to 'scan' (or 0) for scan tip move speed.

        Returns
        -------
        None.

        """
        
         #### Use this section to convert from user inputs to TCP inputs ####
        if custom_or_scan_speed == 0 or custom_or_scan_speed == 1:
            custom_or_scan_speed_value = custom_or_scan_speed
        elif custom_or_scan_speed.lower() == 'custom':
            custom_or_scan_speed_value = 1
        elif custom_or_scan_speed.lower() == 'scan':
            custom_or_scan_speed_value = 0

        #Set the XY_tip movement using follow me to:
        # custom value (custom_or_scan_speed=1) or scan speed (custom_or_scan_speed=0)
        # speed of the tip in m/s
        
        #Command setup:
        command_name = 'FolMe.SpeedSet'
        self.sending_infomation = {'Speed (m/s)': speed,'Custom Speed?': custom_or_scan_speed_value}
        self.sending_infomation_types = ['float32', 'uint32']

        self.response_required_value = 'Yes'
        
        
        #Do you want to see the error?
        self.print_error = 'Yes'
        
        #How to deal with returned data:
        self.recieved_infomation = {}
        self.recieved_infomation_types = []
        list_return = list(self.recieved_infomation.keys())
        
        #Do we need it to convert for us?  
        convert_required= 'No'
        
        #Run the function to generate command        
        self.run_command_fun(command_name, convert_required)
        
        #Uncomment if you want it to return values
        #return self.return_output_data[list_return[0]]        
        
        
        
    def Bias_set(self,bias):
        """
        (bias)
        Set the bias of the sample to be given by bias. 
        

        Parameters
        ----------
        bias : set bias applied to sample (float)
            Applies this value to the bias controller.

        Returns
        -------
        None.

        """
        
        
        #Command setup:
        command_name = 'Bias.Set'
        self.sending_infomation = {'V(V)': bias}
        self.sending_infomation_types = ['float32']
        
        
        self.response_required_value = 'Yes'
        
        #Do you want to see the error?
        self.print_error = 'Yes'
        
        #How to deal with returned data:
        self.recieved_infomation = {}
        self.recieved_infomation_types = []
        list_return = list(self.recieved_infomation.keys())
        
        #Do we need it to convert for us?  
        convert_required= 'No'
        
        #Run the function to generate command        
        self.run_command_fun(command_name, convert_required)
        
        #Uncomment if you want it to return values
        #return self.return_output_data[list_return[0]]
        
    def Bias_get(self):
        """
        ()
        Gets the bias currently set from the system.
        

        Returns
        -------
        bias : bias applied to sample (float)
            bias applied to the system in (V)

        """
        
        #Command setup:
        command_name = 'Bias.Get'
        self.sending_infomation = {}
        self.sending_infomation_types = []
        
        
        self.response_required_value = 'Yes'
        
        #Do you want to see the error?
        self.print_error = 'Yes'
        
        #How to deal with returned data:
        self.recieved_infomation = {'V(V)':''}
        self.recieved_infomation_types = ['float32']
        list_return = list(self.recieved_infomation.keys())
        
        #Do we need it to convert for us?  
        convert_required= 'Yes'
        
        #Run the function to generate command        
        self.run_command_fun(command_name, convert_required)
        
        #Uncomment if you want it to return values
        return self.return_output_data[list_return[0]]
    

    
    def Bias_Pulse(self, seconds, pulse_bias_value):
        """
        (seconds, pulse_bias_value)
        The command runs the Bias Pulse button on the bias window follwoing changing the parameters to those set.

        Parameters
        ----------
        seconds : float or int
            Indicates the length of time the bias is changed.
        pulse_bias_value : float or int
            The bias value that is applied to the system. 

        Returns
        -------
        None.

        """
       
        #### Use this section to convert from user inputs to TCP inputs ####
        wait_value = 1 #will wait
        z_ctrl_hold = 1 #will hold
        pulse_abs_rel = 2 #absolute
        
        seconds = float(seconds)
        pulse_bias_value = float(pulse_bias_value)


        
        #### Standard Command Setup ####
        

        #Command setup:
        command_name = 'Bias.Pulse'
        self.sending_infomation = {'wait until done':wait_value, 'Bias pulse width': seconds, \
                                   'Bias Pulse (V)': pulse_bias_value, 'Z-Crtl hold?':z_ctrl_hold, \
                                   'Pulse abs/rel':pulse_abs_rel}
            
        self.sending_infomation_types = ['uint32', 'float32', 'float32','uint16','uint16']
        self.sending_special_length_loc = []

        self.response_required_value = 'Yes'

        #Do you want to see the error?
        self.print_error = 'No'

        #How to deal with returned data:
        self.recieved_infomation = {}
        self.recieved_infomation_types = []
        self.special_length_loc = []
        list_return = list(self.recieved_infomation.keys())

        #Do we need it to convert for us?
        convert_required = 'No'

        #Run the function to generate command
        self.run_command_fun(command_name, convert_required)

        #### Use this section to return values if required ####
        #if self.return_output_data[list_return[0]] == 0:
        #    return 'off'
        #elif self.return_output_data[list_return[0]] == 1:
        #   return 'on'

        #Uncomment if you want it to return values
        #return self.return_output_data[list_return[0]]
    

    ################################
    ###### Current/feedback ########
    ################################

    def Current_Get(self):
        """
        ()
        Returns the current current value. This is an intaneous value and therefore is likely to be noisy. Given in A. 

        Returns
        -------
        current_value : float
            This value is given in A. So 1e-9 = 1 nA.

        """
        
        #Command setup:
        command_name = 'Current.Get'
        self.sending_infomation = {}
        self.sending_infomation_types = []
        
        self.response_required_value = 'Yes'
        
        #Do you want to see the error?
        self.print_error = 'Yes'
        
        #How to deal with returned data:
        self.recieved_infomation = {'I(A)':''}
        self.recieved_infomation_types = ['float32']
        list_return = list(self.recieved_infomation.keys())
        
        #Do we need it to convert for us?  
        convert_required= 'Yes'
        
        #Run the function to generate command        
        self.run_command_fun(command_name, convert_required)
        
        current_value = self.return_output_data[list_return[0]]
        
        #Uncomment if you want it to return values
        return current_value
    
    
    def Current_SetPoint(self, current_setpoint):
        """
        (current_setpoint)
        Sets the ZCtrl setpoint. Called current_setpoint, although may be used for anything... df etc. in SI units
        

        Parameters
        ----------
        current_setpoint : float
            Set the setpoint for the ZCtrl. Units in SI. So 1e-9 = 1 nA setpoint. 

        Returns
        -------
        None.

        """
        
        
        
        #Command setup:
        command_name = 'ZCtrl.SetpntSet'
        self.sending_infomation = {'I(A) setpoint':current_setpoint}
        self.sending_infomation_types = ['float32']
        
        
        self.response_required_value = 'Yes'
        
        #Do you want to see the error?
        self.print_error = 'Yes'
        
        #How to deal with returned data:
        self.recieved_infomation = {}
        self.recieved_infomation_types = []
        list_return = list(self.recieved_infomation.keys())
        
        #Do we need it to convert for us?  
        convert_required= 'No'
        
        #Run the function to generate command        
        self.run_command_fun(command_name, convert_required)
        
        #Uncomment if you want it to return values
        #return self.return_output_data[list_return[0]]

    def Current_GetSetPoint(self):
        """
        ()
        Returns what the current current setpoint is. Can be used for current and ZCrtl feedback parameter. 
        

        Returns
        -------
        current_setpoint : float
            Get the setpoint for the ZCtrl. Units in SI. So 1e-9 = 1 nA setpoint. 

        """
        
        
        
        #Command setup:
        command_name = 'ZCtrl.SetpntGet'
        self.sending_infomation = {}
        self.sending_infomation_types = []
        
        
        self.response_required_value = 'Yes'
        
        #Do you want to see the error?
        self.print_error = 'Yes'
        
        #How to deal with returned data:
        self.recieved_infomation = {'Current setpoint (A)':''}
        self.recieved_infomation_types = ['float32']
        list_return = list(self.recieved_infomation.keys())
        
        #Do we need it to convert for us?  
        convert_required= 'Yes'
        
        #Run the function to generate command        
        self.run_command_fun(command_name, convert_required)
        
        #Uncomment if you want it to return values
        return self.return_output_data[list_return[0]]

    
    
    ################################
    ####### Scan  functions ########
    ################################
        
    
    def Scan_Action(self, action, direction, turn_off_contin_scan = 'yes'):
        """
        (action, direction, turn_off_contin_scan = 'yes')
        
        Can perform the usual actions of a scan.
        Non-latching action (once actioned will allow continued excution of the python script). Use self.Scan_Check_Hold() to hold exc. 
        NOTE: Default action is to ensure contin scan is off. 

        Parameters
        ----------
        action : int or string
            Can 'start' (or 0) a scan. Can 'stop' (or 1) a scan. Can 'pause' (or 2) and scan. Can 'resume' (or 3) a scan. 
        direction : int or string
            The scan can be started 'up' (or 1). The scan can also be started 'down' (or 0).
        turn_off_contin_scan : string, optional
            Setting to default ensures that contin scan is off. The default is 'yes'. Set to anything else to leave scan as what you want!

        Returns
        -------
        None.

        """
        
        
        if turn_off_contin_scan == 'yes':
            #Note: Turning off the Continuous scan if it is still on by acident!
            Continuous_scan, _, _, Series_name_string, Comment = self.Scan_PropsGet()
        
            if Continuous_scan == 'on':
                self.Scan_PropsSet(Series_name_string, Comment,'off')
                print('Turned off continuous_scan!')
        
        
        #Convert string to numbers - so you can use more intiuative string (or you can use the numbers...)
        if action == 3 or action == 2 or action == 1 or action == 0:
            action_value = action
        elif action.lower() == 'start':
            action_value = 0
        elif action.lower() == 'stop':
            action_value = 1
        elif action.lower() == 'pause':
            action_value = 2
        elif action.lower() == 'resume':
            action_value = 3
        
        if direction == 1 or direction == 0:
            direction_value = direction
        elif direction.lower() == 'up':
            direction_value = 1
        elif direction.lower() == 'down':
            direction_value = 0
         
        
        # action: 0 means Start, 1 is Stop, 2 is Pause, and 3 is Resume
        # Direction: if 1, scan direction is set to up. If 0, direction is down
        
        #Command setup:
        command_name = 'Scan.Action'
        self.sending_infomation = {'Scan Action': action_value, 'Scan Direction': direction_value}
        self.sending_infomation_types = ['uint16', 'uint32']
        
        
        self.response_required_value = 'Yes'
        
        #Do you want to see the error?
        self.print_error = 'Yes'
        
        #How to deal with returned data:
        self.recieved_infomation = {}
        self.recieved_infomation_types = []
        list_return = list(self.recieved_infomation.keys())
        
        #Do we need it to convert for us?  
        convert_required= 'No'
        
        #Run the function to generate command        
        self.run_command_fun(command_name, convert_required)
        
        #Uncomment if you want it to return values
        #return self.return_output_data[list_return[0]]
        
        
        
    def Scan_StatusGet(self):
        """
        ()
        Returns the status of a scan. Is the scan 'on' or 'off'.
        

        Returns
        -------
        scan_status : string
            Returns 'off' if the scan is no longer scanning. Returns 'on' if the scan is still scanning.

        """
        
        #Command setup:
        command_name = 'Scan.StatusGet'
        self.sending_infomation = {}
        self.sending_infomation_types = []
        
        self.response_required_value = 'Yes'
        
        #Do you want to see the error?
        self.print_error = 'Yes'
        
        #How to deal with returned data:
        self.recieved_infomation = {'Scan Status':''}
        self.recieved_infomation_types = ['uint32']
        list_return = list(self.recieved_infomation.keys())
        
        #Do we need it to convert for us?  
        convert_required= 'Yes'
        
        #Run the function to generate command        
        self.run_command_fun(command_name, convert_required)
        
        #Uncomment if you want it to return values
        # Return value creator 0 = not scanning, 1 = if scanning.
        # To 'off' = not scanning, 'on' = if scanning
        if self.return_output_data[list_return[0]] == 0:
            scan_status = 'off'
        elif self.return_output_data[list_return[0]] == 1:
            scan_status = 'on'
        
        return scan_status
    
        
    
    def Scan_FrameSet(self, centreX, centreY, widthX,  widthY, angle):
        """
        (centreX, centreY, widthX,  widthY, angle)
        Sets the frame of the scan window. 
        

        Parameters
        ----------
        centreX : float
            X Position of the centre scan window. In SI units, so 1e-9 is 1 nm. 
        centreY : float
            Y Position of the centre scan window. In SI units, so 1e-9 is 1 nm. 
        widthX : float
            X width of the scan window. In SI units, so 1e-9 is 1 nm. 
        widthY : float
            Y width of the scan window. In SI units, so 1e-9 is 1 nm. 
        angle : float
            Angle the scan window. From 0 -> 360 degrees.

        Returns
        -------
        None.

        """
        
        # Use to set the frame position of the scan
        
        #Command setup:
        command_name = 'Scan.FrameSet'
        self.sending_infomation = {'Centre X': centreX ,'Centre Y': centreY, 'Width X': widthX, 'Width Y': widthY, 'Angle':angle}
        self.sending_infomation_types = ['float32','float32','float32','float32','float32']
        
        self.response_required_value = 'Yes'
        
        #Do you want to see the error?
        self.print_error = 'Yes'
        
        #How to deal with returned data:
        self.recieved_infomation = {}
        self.recieved_infomation_types = []
        list_return = list(self.recieved_infomation.keys())
        
        #Do we need it to convert for us?  
        convert_required= 'Yes'
        
        #Run the function to generate command        
        self.run_command_fun(command_name, convert_required)
        
        #Uncomment if you want it to return values
        # Return value creator 0 = not scanning, 1 = if scanning.
        # To 'off' = not scanning, 'on' = if scanning
    
        
    def Scan_FrameGet(self):
        """
        ()
        Returns the centre position, size and angle of the current scan window. 
        

        Returns
        -------
        centreX : float
            X Position of the centre scan window. In SI units, so 1e-9 is 1 nm. 
        centreY : float
            Y Position of the centre scan window. In SI units, so 1e-9 is 1 nm. 
        widthX : float
            X width of the scan window. In SI units, so 1e-9 is 1 nm. 
        widthY : float
            Y width of the scan window. In SI units, so 1e-9 is 1 nm. 
        angle : float
            Angle the scan window. From 0 -> 360 degrees.

        """
        
        # Returns if the scan is running or not. 
        
        #Command setup:
        command_name = 'Scan.FrameGet'
        self.sending_infomation = {}
        self.sending_infomation_types = []
        
        self.response_required_value = 'Yes'
        
        #Do you want to see the error?
        self.print_error = 'Yes'
        
        #How to deal with returned data:
        self.recieved_infomation = {'Centre X': '' ,'Centre Y': '', 'Width X': '', 'Width Y': '', 'Angle': ''}
        self.recieved_infomation_types = ['float32','float32','float32','float32','float32']
        list_return = list(self.recieved_infomation.keys())
        
        #Do we need it to convert for us?  
        convert_required= 'Yes'
        
        #Run the function to generate command        
        self.run_command_fun(command_name, convert_required)
        
        #Return value section
    
        return self.return_output_data[list_return[0]], self.return_output_data[list_return[1]], self.return_output_data[list_return[2]], self.return_output_data[list_return[3]], self.return_output_data[list_return[4]] 
        
    def Scan_PropsGet(self):
        """
        ()
        Returns the variables/data for the status of:

        Returns
        -------
        Continuous_scan : string
            Returns if continuous scan is 'on' or 'off'.
        Bouncy_scan : string
            Returns if bouncy scan is 'on' or 'off'.
        AutoSave : string
            Returns if autosave is 'off', on the 'next' image or on for 'all' images. 
        Series_name_string : string
            Returns the basename of the scan name. 
        Comment : string
            Returns any comment string. 

        """
        #### Use this section to convert from user inputs to TCP inputs ####
   
        #### Standard Command Setup ####
        # Opens the pattern exp window
   
        #Command setup:
        command_name = 'Scan.PropsGet'
        self.sending_infomation = {}
        self.sending_infomation_types = []
        
        self.response_required_value = 'Yes'
        
        #Do you want to see the error?
        self.print_error = 'No'
        
        #How to deal with returned data:
        self.recieved_infomation = {'Continuous_scan':'', 'Bouncy_scan':'', 'AutoSave':'', \
                                    'Series_name_size':'', 'Series_name_string':'', 'Comment_size':'', \
                                    'Comment':''}
        self.recieved_infomation_types = ['uint32', 'uint32', 'uint32', 'int', 'string', 'int', 'string']
        self.special_length_loc = ['','','','',3,'',5]
        list_return = list(self.recieved_infomation.keys())
        
        #Do we need it to convert for us?  
        convert_required= 'Yes'
        
        #Run the function to generate command        
        self.run_command_fun(command_name, convert_required)
        
        #### Use this section to return values if required ####
        
        if self.return_output_data[list_return[0]] == 0:
            Continuous_scan = 'off' #Is not on
        elif self.return_output_data[list_return[0]] == 1:
            Continuous_scan = 'on' #Is on
            
        if self.return_output_data[list_return[1]] == 0:
            Bouncy_scan = 'off' #Is not on
        elif self.return_output_data[list_return[1]] == 1:
            Bouncy_scan = 'on' #Is on
        
        if self.return_output_data[list_return[2]] == 0:
            AutoSave = 'all' #Is all saved
        elif self.return_output_data[list_return[2]] == 1:
            AutoSave = 'next' #Is next saved
        elif self.return_output_data[list_return[2]] == 2:
            AutoSave = 'off' #Is off
        
        Series_name_string = self.return_output_data[list_return[4]]  
        Comment = self.return_output_data[list_return[6]]  
        
        #Uncomment if you want it to return values
        return Continuous_scan, Bouncy_scan, AutoSave, Series_name_string, Comment
        
    
    def Scan_PropsSet(self, Scan_Name_String, Comment_String, Continuous_scan = 'off', autosave = 'same', bouncy_scan = 'same'):
        """
        (Scan_Name_String, Comment_String, Continuous_scan = 'off', autosave = 'same', bouncy_scan = 'same')
        Set the name of base sting of the scan. 

        Parameters
        ----------
        Scan_Name_String : string
            Sets the base name of the scan.
        Comment_String : string
            Sets the comment of the image. 
        Continuous_scan : string, optional
            Can set continuous scan 'on', 'off' or keep the 'same'. The default is 'off'.
        autosave : string, optional
            Can set autosave 'off', on for the 'next' or 'all' images. The default is 'same' as is currently set.
        bouncy_scan : string, optional
            Can set bouncy_scan 'on', 'off' or keep the 'same'. The default is 'off'.

        Returns
        -------
        None.

        """
       
        #### Use this section to convert from user inputs to TCP inputs ####
        
        #Set Continuous_scan to be off... = 2
        if Continuous_scan.lower() == 'off':
            Continuous_scan = 2
        elif Continuous_scan.lower() == 'on':
            Continuous_scan = 1
        elif Continuous_scan.lower() == 'same':
            Continuous_scan = 0 #No change! 
            
        #Set autosave to be a value
        if autosave.lower() == 'next':
            autosave = 2
        elif autosave.lower() == 'all':
            autosave = 1
        elif autosave.lower() == 'same':
            autosave = 0 #No change! 
            
        #Set bouncy_scan to be a value
        if bouncy_scan.lower() == 'off':
            bouncy_scan = 2
        elif bouncy_scan.lower() == 'on':
            bouncy_scan = 1
        elif bouncy_scan.lower() == 'same':
            bouncy_scan = 0 #No change! 
            
            
            
        
        #Need the size of the comments
        Scan_Name_String_len = len(Scan_Name_String)
        Comment_String_len = len(Comment_String)
        
        #### Standard Command Setup ####

        #Command setup:
        command_name = 'Scan.PropsSet'
        self.sending_infomation = {'Continuous_scan': Continuous_scan, 'Bouncy_scan':bouncy_scan, 'AutoSave':autosave, \
                                    'Series_name_size':Scan_Name_String_len, 'Series_name_string':Scan_Name_String, 'Comment_size':Comment_String_len, \
                                    'Comment':Comment_String}
            
        self.sending_infomation_types = ['uint32', 'uint32', 'uint32', 'int', 'string', 'int', 'string']
        self.sending_special_length_loc = ['','','','',3,'',5]
        
        self.response_required_value = 'No'
        
        #Do you want to see the error?
        self.print_error = 'No'
        
        #How to deal with returned data:
        self.recieved_infomation = {}
        self.recieved_infomation_types = []
        self.special_length_loc = []
        list_return = list(self.recieved_infomation.keys())
        
        #Do we need it to convert for us?  
        convert_required= 'No'
        
        #Run the function to generate command        
        self.run_command_fun(command_name, convert_required)
        

        #Uncomment if you want it to return values
        #return self.return_output_data[list_return[0]]
        
        
        
    def Scan_SpeedGet(self):
        """
        ()
        Returns the scan speed of the scan as it is currently setup. 

        Returns
        -------
        Forward_Linear_Speed : float
            Forward scan sweep speed of the tip. 
        Backwards_Linear_Speed : float
            Backwards scan sweep speed of the tip. 
        Forward_time_per_line : float
            Forward scan time to complete one line of the tip. 
        Backwards_time_per_line : float
            Backwards scan time to complete one line of the tip. 
        Keep_para_constant : string
            Either 'time', 'string'. Selects where the lock is applied. 
        Speed_ratio : float
            TCP manual describes as "defines the backward tip speed related to the forward speed". If set to 1, implies that the forward and backwards speed are the same. 

        """
       
        #### Use this section to convert from user inputs to TCP inputs ####

        #### Standard Command Setup ####

        #Command setup:
        command_name = 'Scan.SpeedGet'
        self.sending_infomation = {}
            
        self.sending_infomation_types = []
        self.sending_special_length_loc = []
        
        self.response_required_value = 'Yes'
        
        #Do you want to see the error?
        self.print_error = 'No'
        
        #How to deal with returned data:
        self.recieved_infomation = {'Forward Linear Speed (m/s)':'', 'Backwards Linear Speed (m/s)':'', \
                                    'Forward time per line (s)':'', 'Backwards time per line (s)':'', \
                                    'Keep para constant': '', 'Speed ratio': ''}
        self.recieved_infomation_types = ['float32','float32','float32','float32','uint16','float32']
        self.special_length_loc = []
        list_return = list(self.recieved_infomation.keys())
        
        #Do we need it to convert for us?  
        convert_required= 'Yes'
        
        #Run the function to generate command        
        self.run_command_fun(command_name, convert_required)
        
        Forward_Linear_Speed = self.return_output_data[list_return[0]]
        Backwards_Linear_Speed = self.return_output_data[list_return[1]]
        Forward_time_per_line = self.return_output_data[list_return[2]]
        Backwards_time_per_line = self.return_output_data[list_return[3]]
        Keep_para_constant = self.return_output_data[list_return[4]]
        Speed_ratio = self.return_output_data[list_return[5]]
        
        if Keep_para_constant == 0:
            Keep_para_constant = 'speed'
        elif Keep_para_constant == 1:
            Keep_para_constant = 'time'

        #Uncomment if you want it to return values
        return Forward_Linear_Speed, Backwards_Linear_Speed, Forward_time_per_line, Backwards_time_per_line, Keep_para_constant, Speed_ratio
    
    
    def Scan_SpeedSet(self, Forward_Linear_Speed, Backwards_Linear_Speed, Forward_time_per_line, Backwards_time_per_line, Keep_para_constant, Speed_ratio):
        """
        (Forward_Linear_Speed, Backwards_Linear_Speed, Forward_time_per_line, Backwards_time_per_line, Keep_para_constant, Speed_ratio)
        Allows for the setting of the scan speed, aswell as additional parameters hidden by the Nanonis GUI. 
        
        Parameters
        ----------
        Forward_Linear_Speed : float
            Forward scan sweep speed of the tip. 
        Backwards_Linear_Speed : float
            Backwards scan sweep speed of the tip. 
        Forward_time_per_line : float
            Forward scan time to complete one line of the tip. 
        Backwards_time_per_line : float
            Backwards scan time to complete one line of the tip. 
        Keep_para_constant : string
            Either 'time', 'string' or 'same'. Selects where the lock is applied. Note: 'same' means no lock change from the setting already being used. 
        Speed_ratio : float
            Set to 1 normally. TCP manual describes as "defines the backward tip speed related to the forward speed".

        Returns
        -------
        None.

        """
       
        #### Use this section to convert from user inputs to TCP inputs ####
        if Keep_para_constant.lower() == 'same':
            Keep_para_constant = 0
        elif Keep_para_constant.lower() == 'time':
            Keep_para_constant = 2
        elif Keep_para_constant.lower() == 'speed':
            Keep_para_constant = 1
        
        #### Standard Command Setup ####

        #Command setup:
        command_name = 'Scan.SpeedSet'
        self.sending_infomation = {'Forward Linear Speed (m/s)':Forward_Linear_Speed, 'Backwards Linear Speed (m/s)':Backwards_Linear_Speed, \
                                    'Forward time per line (s)':Forward_time_per_line, 'Backwards time per line (s)':Backwards_time_per_line, \
                                    'Keep para constant': Keep_para_constant, 'Speed ratio': Speed_ratio}
        self.sending_infomation_types = ['float32','float32','float32','float32','uint16','float32']
        self.sending_special_length_loc = []
        
        self.response_required_value = 'No'
        
        #Do you want to see the error?
        self.print_error = 'No'
        
        #How to deal with returned data:
        self.recieved_infomation = {}
        self.recieved_infomation_types = []
        self.special_length_loc = []
        list_return = list(self.recieved_infomation.keys())
        
        #Do we need it to convert for us?  
        convert_required= 'No'
        
        #Run the function to generate command        
        self.run_command_fun(command_name, convert_required)
    

        #Uncomment if you want it to return values
        #return self.return_output_data[list_return[0]]
    
    
    def Scan_BufferGet(self):
        """
        ()
        Provides the number of scan channels being recorded (count of channels), a list of the channel's number, as well as the number of lines and pixels in the image.

        Returns
        -------
        Num_of_channels : int
            Count of the total number of channels being recorded using the scan control window. 
        Channel_indexes : list of int
            Lists the channel number, of each of the channels being recorded in the scan control window. 
        Pixels : int
            Number of pixels along the fast scan axis of the image.
        Num_of_lines : int
            Number of pixels along the slow scan axis of the image. 

        """
       
        #### Use this section to convert from user inputs to TCP inputs ####

        
        #### Standard Command Setup ####

        #Command setup:
        command_name = 'Scan.BufferGet'
        self.sending_infomation = {}
        self.sending_infomation_types = []
        self.sending_special_length_loc = []
        
        self.response_required_value = 'Yes'
        
        #Do you want to see the error?
        self.print_error = 'No'
        
        #How to deal with returned data:
        self.recieved_infomation = {'Num_of_channels':'','Channel indexes':'','Pixels':'','Num_of_lines':''}
        self.recieved_infomation_types = ['int', '1Dint', 'int', 'int']
        self.special_length_loc = ['',0]
        list_return = list(self.recieved_infomation.keys())
        
        #Do we need it to convert for us?  
        convert_required= 'Yes'
        
        #Run the function to generate command        
        self.run_command_fun(command_name, convert_required)
        
        Num_of_channels = self.return_output_data[list_return[0]]
        Channel_indexes = self.return_output_data[list_return[1]]
        Pixels = self.return_output_data[list_return[2]]
        Num_of_lines = self.return_output_data[list_return[3]]

        #Uncomment if you want it to return values
        return Num_of_channels, Channel_indexes, Pixels, Num_of_lines
    
    def Scan_BufferSet(self, channel_list, pixel_num, num_of_lines='same'):
        """
        (channel_list, pixel_num, num_of_lines='same')
        Set the channels to record, the number of pixels in a line and the number of lines in an image. 

        Parameters
        ----------
        channel_list : int, or list of int
            Select channel numbers to be recorded in the scan. Can provide single int, or list of int values. 
        pixel_num : int
            Sets number of pixels across the image in the fast scan axis. 
        num_of_lines : int or 'same', optional
            If you want the same number of pixels as num of lines, leave as 'same'. The default is 'same'. Change to int for a difference between pixels and num of lines. 

        Returns
        -------
        None.

        """
       
        #### Use this section to convert from user inputs to TCP inputs ####
        num_of_channels = len(channel_list)
        
        check_after = 0
        if num_of_lines == 'same':
            #This means that you want pixel_num to be the same an num_of_lines
            num_of_lines = pixel_num
            #NOTE: May be problematic...  since the lines may not default to the same as pixel (in bit resentation)
            check_after = 1 # use this check to ensure we are the same! 
            
        #### Standard Command Setup ####

        #Command setup:
        command_name = 'Scan.BufferSet'
        self.sending_infomation = {'Num_of_channels':num_of_channels,'Channel indexes':channel_list,\
                                   'Pixels':pixel_num,'Num_of_lines':num_of_lines}
            
        self.sending_infomation_types = ['int','1Dint','int','int']
        self.sending_special_length_loc = ['',0]
        
        self.response_required_value = 'Yes'
        
        #Do you want to see the error?
        self.print_error = 'No'
        
        #How to deal with returned data:
        self.recieved_infomation = {}
        self.recieved_infomation_types = []
        self.special_length_loc = []
        list_return = list(self.recieved_infomation.keys())
        
        #Do we need it to convert for us?  
        convert_required= 'No'
        
        #Run the function to generate command        
        self.run_command_fun(command_name, convert_required)
        
        if check_after == 1:
            _,_,newPixel,newNumLines = self.Scan_BufferGet()
            if newPixel != newNumLines:
                #if they do not match... set them to be correct! 
                self.Scan_BufferSet(channel_list, newPixel, newPixel)

        
    
    def Scan_FrameDataGrab(self, channel_num, direction='forward'):
        """
        (channel_num, direction='forward')
        Provides the Scan Control window data for the channel number supplied, as well as the name of the channel. Default is the 'forward' direction. Not tested on if you are not recording one of the directions. 
        

        Parameters
        ----------
        channel_num : int
            Select the channel number to get the scan winow data from. 
        direction : string, optional
            Either select the 'forward' or 'backward' scan direction. The default is 'forward'.

        Returns
        -------
        Channel_Name : string
            Returns the channel name, useful to check you are getting what you want. 
        Scan_data_reshaped : float 2D array (numpy)
            Reshaped 2D numpy float array of the channel values. 

        """
       
        #### Use this section to convert from user inputs to TCP inputs ####
        if direction.lower() == 'forward':
            direction_value = 1
        elif direction.lower() == 'backward':
            direction_value = 0
        
        #### Standard Command Setup ####
        self.response_length = 50000000 # I expect these to be MUCH larger than normal responses! 
        

        #Command setup:
        command_name = 'Scan.FrameDataGrab'
        self.sending_infomation = {'Channel Index':channel_num, 'Direction': direction_value}
        self.sending_infomation_types = ['uint32','uint32']
        self.sending_special_length_loc = []
        
        self.response_required_value = 'Yes'
        
        #Do you want to see the error?
        self.print_error = 'No'
        
        #How to deal with returned data:
        self.recieved_infomation = {'Channel Name size':'', 'Channel Name':'', \
                                    'Scan rows':'', 'Scan columns':'', 'Scan data':'', \
                                    'Scan direction':''}
        self.recieved_infomation_types = ['int','string','int','int','2Dfloat32','uint32']
        self.special_length_loc = ['',0,'','',[2,3],'']
        list_return = list(self.recieved_infomation.keys())
        
        #Do we need it to convert for us?  
        convert_required= 'Yes'
        
        #Run the function to generate command
        self.run_command_fun(command_name, convert_required)
        
        Channel_Name = self.return_output_data[list_return[1]]
        #Scan_data = self.return_output_data[list_return[4]]
        
        self.response_length = 50000 #Return to a normal response length, just in case of garbage! 
        
        # reshape here to image shape! 
        Scan_data_reshaped = np.reshape(self.return_output_data[list_return[4]], (self.return_output_data[list_return[2]], self.return_output_data[list_return[3]]))
        
        return Channel_Name, Scan_data_reshaped

            
    
    ################################
    ######Z Related functions#######
    ################################
        
        
    def Z_pos_get(self): 
        """
        ()
        Gets the Z position of the tip. Note: The range of the tip can go from +ve values to -ve values. 
        

        Returns
        -------
        Z_tip_height: float
            Height of the tip. Note: in SI units, so 1e-9 = 1 nm. 
        """
        
        # Gets the Z position 
        
        #Command setup:
        command_name = 'ZCtrl.ZPosGet'
        self.sending_infomation = {}
        self.sending_infomation_types = []
        
        
        self.response_required_value = 'Yes'
        
        #Do you want to see the error?
        self.print_error = 'Yes'
        
        #How to deal with returned data:
        self.recieved_infomation = {'Z position (m)': ''}
        self.recieved_infomation_types = ['float32']
        list_return = list(self.recieved_infomation.keys())
        
        #Do we need it to convert for us?  
        convert_required= 'Yes'
        
        #Run the function to generate command        
        self.run_command_fun(command_name, convert_required)
        
        #Uncomment if you want it to return values
        return self.return_output_data[list_return[0]]
    
    
    def Z_pos_set(self, height):
        """
        (height)
        Sets the Z position of the tip. Note: The range of the tip can go from +ve values to -ve values. 
        

        Parameters
        -------
        Z_tip_height: float
            Height of the tip. Note: in SI units, so 1e-9 = 1 nm. 
            
        Returns
        -------
        None.
        
        """
        
        # Sets the Z position 
        
        #Command setup:
        command_name = 'ZCtrl.ZPosSet'
        self.sending_infomation = {'Z position (m)': height}
        self.sending_infomation_types = ['float32']
        
        
        self.response_required_value = 'Yes'
        
        #Do you want to see the error?
        self.print_error = 'Yes'
        
        #How to deal with returned data:
        self.recieved_infomation = {}
        self.recieved_infomation_types = []
        list_return = list(self.recieved_infomation.keys())
        
        #Do we need it to convert for us?  
        convert_required= 'No'
        
        #Run the function to generate command        
        self.run_command_fun(command_name, convert_required)
        
        #Uncomment if you want it to return values
        #return self.return_output_data[list_return[0]]
        
        
    
    def Z_feedback_set(self, on_off):
        """
        (on_off)
        Sets the feedback of the tip to be 'on' or 'off'. 
        

        Parameters
        ----------
        on_off : string or int
            Turn the feedback 'on' (or 1). Turn the feedback 'off' (or 0).

        Returns
        -------
        None.

        """
        
        #### Use this section to convert from user inputs to TCP inputs ####
        
        if on_off == 0 or on_off == 1:
            on_off_value = on_off
        elif on_off.lower() == 'on':
            on_off_value = 1
        elif on_off.lower() == 'off':
            on_off_value = 0
        
        
        
        # Sets the Z feedback status
        # 0 turns off the controller
        # 1 turns on the controller
        
        #Command setup:
        command_name = 'ZCtrl.OnOffSet'
        self.sending_infomation = {'Z-Controller status': on_off_value }
        self.sending_infomation_types = ['uint32']
        
        
        self.response_required_value = 'Yes'
        
        #Do you want to see the error?
        self.print_error = 'Yes'
        
        #How to deal with returned data:
        self.recieved_infomation = {}
        self.recieved_infomation_types = []
        list_return = list(self.recieved_infomation.keys())
        
        #Do we need it to convert for us?  
        convert_required= 'No'
        
        #Run the function to generate command        
        self.run_command_fun(command_name, convert_required)
        
        #Uncomment if you want it to return values
        #return self.return_output_data[list_return[0]]
        
        
    def Z_feedback_get(self):
        """
        ()
        Gets the feedback of the tip, described as 'on' or 'off'. 
        

        Returns
        -------
        on_off : string
            Feedback is 'on' or 'off'.

        """
        
        
        # Gets the Z feedback status
        # Returns 0 if 'off' and 1 if 'on'.
        
        #Command setup:
        command_name = 'ZCtrl.OnOffGet'
        self.sending_infomation = {}
        self.sending_infomation_types = []
        
        
        self.response_required_value = 'Yes'
        
        #Do you want to see the error?
        self.print_error = 'Yes'
        
        #How to deal with returned data:
        self.recieved_infomation = {'Z-Controller status':''}
        self.recieved_infomation_types = ['uint32']
        list_return = list(self.recieved_infomation.keys())
        
        #Do we need it to convert for us?  
        convert_required= 'Yes'
        
        #Run the function to generate command        
        self.run_command_fun(command_name, convert_required)
        
        return_value = self.return_output_data[list_return[0]]
        
        if return_value == 0:
            #The Auto approach if off
            return 'off'
        elif return_value == 1:
            #The Auto approach if off
            return 'on'
        
        #Uncomment if you want it to return values
        #return self.return_output_data[list_return[0]]
        
        
        
    def Z_fine_withdraw(self, wait_value = 1, timeout_value = 9999):
        """ 
        (wait_value = 1, timeout_value = 9999)
        Withdraws the tip in fine ZCtrl. Wait_value and timeout_value are aplicable if you have a very slow Z movement speed. 
        

        Parameters
        ----------
        wait_value : int, optional
            If you want Nanonis to wait until until the tip is retracted before allowing the continued exc of code set to 1, if not 0. The default is 1.
        timeout_value : int, optional
            Time in (ms) that Nanonis will wait for the tip to retract before timeout. The default is 9999.

        Returns
        -------
        None.

        """
        
        
        # Withdraws the tip to a fine motor retracted state
        # Wait value will hold exc of the script until tip retracted
        # waits until the tip is fully withdrawn (=1) or 
        # it does not wait (=0) 
        # timeout in ms for how long to wait... -1 = indefinitely
        
        #Command setup:
        command_name = 'ZCtrl.Withdraw'
        self.sending_infomation = {'Wait_Value': wait_value, 'Timeout (ms)':timeout_value}
        self.sending_infomation_types = ['uint32','int']
        
        self.response_required_value = 'Yes'
        
        #Do you want to see the error?
        self.print_error = 'Yes'
        
        #How to deal with returned data:
        self.recieved_infomation = {}
        self.recieved_infomation_types = []
        list_return = list(self.recieved_infomation.keys())
        
        #Do we need it to convert for us?  
        convert_required= 'No'
        
        #Run the function to generate command        
        self.run_command_fun(command_name, convert_required)
        
        #Uncomment if you want it to return values
        #return self.return_output_data[list_return[0]]
        
    def ZCtrl_GainGet(self):
        """
        ()
        Returns the gain settings of the Z controller. 

        Returns
        -------
        P_gain : float
            Proportional gain as given by the ZCtrl window. 
        Time_Constant : float
            Time constant as given by the ZCtrl window. 
        I_Gain : float
            Time constant as given by the ZCtrl window. 

        """
        
        #Command setup:
        command_name = 'ZCtrl.GainGet'
        self.sending_infomation = {}
        self.sending_infomation_types = []
        
        self.response_required_value = 'Yes'
        
        #Do you want to see the error?
        self.print_error = 'No'
        
        #How to deal with returned data:
        self.recieved_infomation = {'P-gain':'', 'Time Constant':'', 'I-Gain':''}
        self.recieved_infomation_types = ['float32','float32','float32']
        list_return = list(self.recieved_infomation.keys())
        
        #Do we need it to convert for us?  
        convert_required= 'Yes'
        
        #Run the function to generate command        
        self.run_command_fun(command_name, convert_required)
        
        P_gain = self.return_output_data[list_return[0]]  
        Time_Constant = self.return_output_data[list_return[1]]  
        I_Gain = self.return_output_data[list_return[2]]  
        
        #Uncomment if you want it to return values
        return P_gain, Time_Constant, I_Gain
        
    def ZCtrl_GainSet(self, P_gain, I_Gain):
        """
        (P_gain, I_Gain)
        Sets the proportional and intergral gain settings. Currently setup for P and I only, time constant is calulated from P and I.

        Parameters
        ----------
        P_gain : float
            Sets the proportional gain in the ZCtrl window.
        I_Gain : float
            Sets the intergral gain in the ZCtrl window. 

        Returns
        -------
        None.

        """
        Time_Constant = P_gain/I_Gain
        
        #Command setup:
        command_name = 'ZCtrl.GainSet'
        self.sending_infomation = {'P-gain':P_gain, 'Time Constant':Time_Constant, 'I-Gain':I_Gain}
        self.sending_infomation_types = ['float32','float32','float32']
        
        self.response_required_value = 'No'
        
        #Do you want to see the error?
        self.print_error = 'No'
        
        #How to deal with returned data:
        self.recieved_infomation = {}
        self.recieved_infomation_types = []
        list_return = list(self.recieved_infomation.keys())
        
        #Do we need it to convert for us?  
        convert_required= 'No'
        
        #Run the function to generate command        
        self.run_command_fun(command_name, convert_required)
        
        #Uncomment if you want it to return values
        
        
        
        
        
    def ZCtrl_LimitsEnabledSet(self, on_off):
        """
        (on_off)
        Turn the ZCtrl limits 'on' or 'off'. Useful for preventing tip going too far into the surface when attempting tasks. 

        Parameters
        ----------
        on_off : string or int
            Turn the limits 'on' (or 1), or 'off' (or 0).

        Returns
        -------
        None.

        """
        
        #### Use this section to convert from user inputs to TCP inputs ####

        #Convert to number if required!
        if on_off == 1 or on_off == 0:
            on_off_value = on_off
        elif on_off.lower() == 'on':
            on_off_value = 1
        elif on_off.lower() == 'off':
            on_off_value = 0
        
                
        #### Standard Command Setup ####
        # Returns the number of channels, and their names! 
        
        #Command setup:
        command_name = 'ZCtrl.LimitsEnabledSet'
        self.sending_infomation = {'Enable or disable':on_off_value}
        self.sending_infomation_types = ['uint32']
        self.sending_special_length_loc = []
        
        self.response_required_value = 'Yes'
        
        #Do you want to see the error?
        self.print_error = 'Yes'
        
        #How to deal with returned data:
        self.recieved_infomation = {}
        self.recieved_infomation_types = []
        self.special_length_loc = []
        list_return = list(self.recieved_infomation.keys())
        
        #Do we need it to convert for us?  
        convert_required= 'No'
        
        #Run the function to generate command        
        self.run_command_fun(command_name, convert_required)
        
        #### Use this section to return values if required ####
        
        #Uncomment if you want it to return values
        
        
    def ZCtrl_LimitsEnabledGet(self):
        """
        ()
        Returns if the limits on the ZCtrl are 'on' or 'off'. 

        Returns
        -------
        on_off_value : string
            Returns if the limits on the ZCtrl are 'on' or 'off'.

        """
        
        #### Use this section to convert from user inputs to TCP inputs ####

                       
        #### Standard Command Setup ####
        # Returns the number of channels, and their names! 
        
        #Command setup:
        command_name = 'ZCtrl.LimitsEnabledGet'
        self.sending_infomation = {}
        self.sending_infomation_types = []
        self.sending_special_length_loc = []
        
        self.response_required_value = 'Yes'
        
        #Do you want to see the error?
        self.print_error = 'Yes'
        
        #How to deal with returned data:
        self.recieved_infomation = {'Enabled or disabled':''}
        self.recieved_infomation_types = ['uint32']
        self.special_length_loc = []
        list_return = list(self.recieved_infomation.keys())
        
        #Do we need it to convert for us?  
        convert_required= 'Yes'
        
        #Run the function to generate command        
        self.run_command_fun(command_name, convert_required)
        
        #### Use this section to return values if required ####
        on_off = self.return_output_data[list_return[0]]
        
        
        if on_off == 1:
            on_off_value = 'on'
        elif on_off == 0:
            on_off_value = 'off'
        
        #Uncomment if you want it to return values
        return on_off_value
        
        
    def ZCtrl_LimitsSet(self, Zhigh, Zlow):
        """
        (Zhigh, Zlow)
        Set the ZCtrl limits on the Z Controller! NOTE: Limits must be ON, prior to using this function. 

        Parameters
        ----------
        Zhigh : float
            Set the high Z limit (furthest from the surface)
        Zlow : float
            Set the low Z limit (closest to the surface)

        Returns
        -------
        None.

        """        
        #### Use this section to convert from user inputs to TCP inputs ####
        
                
        #### Standard Command Setup ####
        # Returns the number of channels, and their names! 
        
        #Command setup:
        command_name = 'ZCtrl.LimitsSet'
        self.sending_infomation = {'Zhigh value':Zhigh, 'Zlow value':Zlow}
        self.sending_infomation_types = ['float32', 'float32']
        self.sending_special_length_loc = []
        
        self.response_required_value = 'Yes'
        
        #Do you want to see the error?
        self.print_error = 'Yes'
        
        #How to deal with returned data:
        self.recieved_infomation = {}
        self.recieved_infomation_types = []
        self.special_length_loc = []
        list_return = list(self.recieved_infomation.keys())
        
        #Do we need it to convert for us?  
        convert_required= 'No'
        
        #Run the function to generate command        
        self.run_command_fun(command_name, convert_required)
        
        #### Use this section to return values if required ####
        
        #Uncomment if you want it to return values
        #return self.return_output_data[list_return[1]], self.return_output_data[list_return[2]]
        
    def ZCtrl_LimitsGet(self):
        """
        ()
        Return the ZCtrl limits on the Z Controller.

        Returns
        -------
        Zhigh : float
            Get the high Z limit (furthest from the surface)
        Zlow : float
            Get the low Z limit (closest to the surface)

        """
        
        #### Use this section to convert from user inputs to TCP inputs ####
        
                
        #### Standard Command Setup ####
        # Returns the number of channels, and their names! 
        
        #Command setup:
        command_name = 'ZCtrl.LimitsGet'
        self.sending_infomation = {}
        self.sending_infomation_types = []
        self.sending_special_length_loc = []
        
        self.response_required_value = 'Yes'
        
        #Do you want to see the error?
        self.print_error = 'Yes'
        
        #How to deal with returned data:
        self.recieved_infomation = {'Zhigh value':'', 'Zlow value':''}
        self.recieved_infomation_types = ['float32', 'float32']
        self.special_length_loc = []
        list_return = list(self.recieved_infomation.keys())
        
        #Do we need it to convert for us?  
        convert_required= 'Yes'
        
        #Run the function to generate command        
        self.run_command_fun(command_name, convert_required)
        
        #### Use this section to return values if required ####
        Zhigh = self.return_output_data[list_return[0]]
        Zlow = self.return_output_data[list_return[1]]
        
        return Zhigh, Zlow
        
        #Uncomment if you want it to return values
        #return self.return_output_data[list_return[1]], self.return_output_data[list_return[2]]
        
        
    ################################
    ########### XYZ coarse #########
    ############    &    ###########
    ######### Auto approach ########
    ################################
    
    def XYZ_coarse(self, direction, number_of_steps):
        """
        (direction, number_of_steps)
        Move the tip or sample using the Coarse motor. This can be achived by giving a direction, and then the number of steps required. 
        

        Parameters
        ----------
        direction : string
            Give the direction of movement. Z direction: '-z' or '+z'. X direction: '-x' or '+x'. Y direction '-y' or '+y'
        number_of_steps : int
            Number of steps to take in this direction. 
            

        Returns
        -------
        None.

        """
        #### Use this section to convert from user inputs to TCP inputs ####
        direction_dict = {'z-':5, 'z+':4, 'y-':3, 'y+':2, 'x-':1, 'x+':0, \
                          '-z':5, '+z':4, '-y':3, '+y':2, '-x':1, '+x':0} 
        
        if direction == 0 or direction == 1 or direction == 2 or direction == 3 or direction == 4 or direction == 5: 
            direction_value = direction #assuming you give this a number...
        
        # or you now are using letters
        else:
            direction_value = direction_dict[direction.lower()]
        
        
        # Moves the Coarse motors
        # Valid values are 0=X+, 1=X-, 2=Y+, 3=Y-, 4=Z+, 5=Z-
        # Number of steps to take
        
        #Command setup:
        command_name = 'Motor.StartMove'
        self.sending_infomation = {'Direction': direction_value, 'Steps to take':number_of_steps, 'Group': 0, 'Wait until finished': 1 }
        self.sending_infomation_types = ['uint32','uint16','uint32','uint32']
        
        self.response_required_value = 'Yes'
        
        #Do you want to see the error?
        self.print_error = 'Yes'
        
        #How to deal with returned data:
        self.recieved_infomation = {}
        self.recieved_infomation_types = []
        list_return = list(self.recieved_infomation.keys())
        
        #Do we need it to convert for us?  
        convert_required= 'No'
        
        #Run the function to generate command        
        self.run_command_fun(command_name, convert_required)
        
        #Uncomment if you want it to return values
        #return self.return_output_data[list_return[0]]
    
    
    def AutoApproach_OnOffSet(self, on_off):
        """
        (on_off) 
        Turn 'on' the auto approach, or turn 'off' the auto approach. To prevent crashes, this function also opens the AP module first.
        Non-latching action (once actioned will allow continued excution of the python script). Use self.AutoApproach_Check_Hold() to hold exc. 
        

        Parameters
        ----------
        on_off : string or int
            Turn 'on' (or 1) the auto approach. Or turn 'off' (or 0) the auto approach. 

        Returns
        -------
        None.

        """
        
        #Note this requires the Auto approach window to be open - opening here:
        self.AutoApproach_OpenModule()
        
        #### Use this section to convert from user inputs to TCP inputs ####
        
        #Convert to number if required!
        if on_off == 1 or on_off == 0:
            on_off_value = on_off
        elif on_off.lower() == 'on':
            on_off_value = 1
        elif on_off.lower() == 'off':
            on_off_value = 0
        
        
        #### Standard Command Setup ####
        #Turns on or off the auto approach
        #=1 turn on
        #=0 turn off
        
        #Command setup:
        command_name = 'AutoApproach.OnOffSet'
        self.sending_infomation = {'On/Off': on_off_value }
        self.sending_infomation_types = ['uint16']
        
        self.response_required_value = 'Yes'
        
        #Do you want to see the error?
        self.print_error = 'Yes'
        
        #How to deal with returned data:
        self.recieved_infomation = {}
        self.recieved_infomation_types = []
        list_return = list(self.recieved_infomation.keys())
        
        #Do we need it to convert for us?  
        convert_required= 'No'
        
        #Run the function to generate command        
        self.run_command_fun(command_name, convert_required)
        
        #### Use this section to return values if required ####
        
        #Uncomment if you want it to return values
        #return self.return_output_data[list_return[0]]
    
    def AutoApproach_OnOffGet(self):
        """
        ()
        Gets the auto approach status. This is returned as a string ('on' or 'off')
        

        Returns
        -------
        on_off : string
            Auto approach is 'on', or 'off'. 

        """
        
        #Note this requires the Auto approach window to be open - opening here:
        self.AutoApproach_OpenModule()
        
        
        #### Use this section to convert from user inputs to TCP inputs ####
        
        # Not required here
        
        
        #### Standard Command Setup ####
        #Determines if the AutoApproach is on or off
        
        #Command setup:
        command_name = 'AutoApproach.OnOffGet'
        self.sending_infomation = {}
        self.sending_infomation_types = []
        
        self.response_required_value = 'Yes'
        
        #Do you want to see the error?
        self.print_error = 'Yes'
        
        #How to deal with returned data:
        self.recieved_infomation = {'Status': ''}
        self.recieved_infomation_types = ['uint16']
        list_return = list(self.recieved_infomation.keys())
        
        #Do we need it to convert for us?  
        convert_required= 'Yes'
        
        #Run the function to generate command        
        self.run_command_fun(command_name, convert_required)
        
        #### Use this section to return values if required ####
        
        return_value = self.return_output_data[list_return[0]]
        
        if return_value == 0:
            #The Auto approach if off
            return 'off'
        elif return_value == 1:
            #The Auto approach if off
            return 'on'
        
        #Uncomment if you want it to return values
        #return self.return_output_data[list_return[0]]
        
        
    
    def AutoApproach_OpenModule(self):
        """
        ()
        In order to use the Auto Approach, the module window must be opened. This function does this. 
        

        Returns
        -------
        None.

        """
        
        
        #### Use this section to convert from user inputs to TCP inputs ####
        
        # Not required here
        
        
        #### Standard Command Setup ####
        #The other AutoApproach commands require the module to be open. This
        #function opens the module for you. 
        
        #Command setup:
        command_name = 'AutoApproach.Open'
        self.sending_infomation = {}
        self.sending_infomation_types = []
        
        self.response_required_value = 'Yes'
        
        #Do you want to see the error?
        self.print_error = 'Yes'
        
        #How to deal with returned data:
        self.recieved_infomation = {'Status': ''}
        self.recieved_infomation_types = ['uint16']
        list_return = list(self.recieved_infomation.keys())
        
        #Do we need it to convert for us?  
        convert_required= 'No'
        
        #Run the function to generate command        
        self.run_command_fun(command_name, convert_required)
        
        #### Use this section to return values if required ####
        
        #Uncomment if you want it to return values
        #return self.return_output_data[list_return[0]]
        
        
        
        ################################
        ########### Tip Shaper #########
        ################################
    
    def TipShaper_Start(self, wait_until_finished = 1, timeout = -1):
        """
        (wait_until_finished = 1, timeout = -1)
        NOTE: Tip Shaper window MUST be open. Will run the Tip Shaper where the tip is currently situated. Will wait until finished (=1), and will not timeout (=-1).  

        Parameters
        ----------
        wait_until_finished : int (0 or 1), optional
            The default is 1, which means wait until complete. Setting to 0, will allow more code to run whilst the Tip shaper is run. 
        timeout : int, optional
            Length of time to time out. The default is -1, which means never timeout.

        Returns
        -------
        None.

        """
        
        #### Use this section to convert from user inputs to TCP inputs ####
        self.s.settimeout(2000) #To ensure that if you have a really long tip shaper even it does not impact the code.
                
        #### Standard Command Setup ####
        # Returns the number of channels, and their names! 
        
        #Command setup:
        command_name = 'TipShaper.Start'
        self.sending_infomation = {'Wait until finished': wait_until_finished, 'Timeout': timeout}
        self.sending_infomation_types = ['uint32', 'int']
        self.sending_special_length_loc = []
        
        self.response_required_value = 'Yes'
        
        #Do you want to see the error?
        self.print_error = 'Yes'
        
        #How to deal with returned data:
        self.recieved_infomation = {}
        self.recieved_infomation_types = []
        self.special_length_loc = []
        list_return = list(self.recieved_infomation.keys())
        
        #Do we need it to convert for us?  
        convert_required= 'No'
        
        #Run the function to generate command        
        self.run_command_fun(command_name, convert_required)
        
        #### Use this section to return values if required ####
        self.s.settimeout(30) # Returning to our default value of 30s.
        
        #Uncomment if you want it to return values
        #return self.return_output_data[list_return[1]], self.return_output_data[list_return[2]]
        
    
    def TipShaper_PropsGet(self):
        """
        ()
        Returns all relavant changeable values and checkboxes in the Tip Shaper window. NOTE: Tip Shaper window MUST be open. 

        Returns
        -------
        Switch_Off_Delay : float
            "Time during which the Z position is averaged right before switching the Z-Controller off"*
        Change_Bias : string
            Is the bias changed prior to first Z ramp? 'yes' or 'no'.
        Bias : float
            Value the bias is changed to if Change_Bias = 'yes'
        Tip_Lift : float
            "Defines the relative height the tip is going to ramp for the first time (from the current Z position)."*
        Lift_Time : float
            "Defines the time to ramp Z from the current Z position by the Tip Lift amount."*
        Bias_Lift : float
            "Bias voltage applied just after the first Z ramping."*
        Bias_Settling_time : float
            "Time to wait after applying the Bias Lift value, and it is also the time to wait after applying Bias (V) before ramping Z for the first time."*
        Lift_Height : float
            "Defines the height the tip is going to ramp for the second time"*
        Lift_Time_2 : float
            "Given time to ramp Z in the second ramping"*
        End_Wait_Time : float
            "Time to wait after restoring the initial Bias voltage (just after finishing the second ramping)."*
        Restore_Feedback : string
            Is the feedback restored after TipShaper? 'yes' or 'no'
            
        *from TCP protocol manual

        """
        
        #### Use this section to convert from user inputs to TCP inputs ####
        
                
        #### Standard Command Setup ####
        # Returns the number of channels, and their names! 
        
        #Command setup:
        command_name = 'TipShaper.PropsGet'
        self.sending_infomation = {}
        self.sending_infomation_types = []
        self.sending_special_length_loc = []
        
        self.response_required_value = 'Yes'
        
        #Do you want to see the error?
        self.print_error = 'Yes'
        
        #How to deal with returned data:
        self.recieved_infomation = {'Switch Off Delay':'', 'Change Bias?':'', 'Bias (V)':'', 'Tip Lift (m)':'', 'Lift Time (s)':'', 'Bias Lift (V)':'', 'Bias Settling time (s)':'', \
                                    'Lift Height (m)':'', 'Lift Time 2 (s)':'', 'End Wait Time (s)':'', 'Retore Feedback?':''}
        self.recieved_infomation_types = ['float32', 'uint32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'uint32']
        self.special_length_loc = []
        list_return = list(self.recieved_infomation.keys())
        
        #Do we need it to convert for us?  
        convert_required= 'Yes'
        
        #Run the function to generate command        
        self.run_command_fun(command_name, convert_required)
        
        #### Use this section to return values if required ####
        Switch_Off_Delay = self.return_output_data[list_return[0]]
        
        if self.return_output_data[list_return[1]] == 0:
            Change_Bias = 'no'
        elif self.return_output_data[list_return[1]] == 1:
            Change_Bias = 'yes'
        
        Bias = self.return_output_data[list_return[2]]
        Tip_Lift = self.return_output_data[list_return[3]]
        Lift_Time =self.return_output_data[list_return[4]]
        Bias_Lift = self.return_output_data[list_return[5]]
        Bias_Settling_time = self.return_output_data[list_return[6]]
        Lift_Height = self.return_output_data[list_return[7]]
        Lift_Time_2 = self.return_output_data[list_return[8]]
        End_Wait_Time = self.return_output_data[list_return[9]]
        
        if self.return_output_data[list_return[10]] == 0:
            Restore_Feedback = 'no'
        elif self.return_output_data[list_return[10]] == 1:
            Restore_Feedback = 'yes'
        
        return Switch_Off_Delay, Change_Bias, Bias, Tip_Lift, Lift_Time, Bias_Lift, Bias_Settling_time, Lift_Height, Lift_Time_2, End_Wait_Time, Restore_Feedback
        
        #Uncomment if you want it to return values
        #return self.return_output_data[list_return[1]], self.return_output_data[list_return[2]]
        
    def TipShaper_PropsSet(self, Switch_Off_Delay, Change_Bias, Bias, Tip_Lift, Lift_Time, Bias_Lift, Bias_Settling_time, Lift_Height, Lift_Time_2, End_Wait_Time, Restore_Feedback):
        """
        (Switch_Off_Delay, Change_Bias, Bias, Tip_Lift, Lift_Time, Bias_Lift, Bias_Settling_time, Lift_Height, Lift_Time_2, End_Wait_Time, Restore_Feedback)
        Sets all relavant changeable values and checkboxes in the Tip Shaper window. NOTE: Tip Shaper window MUST be open. 

        Parameters
        ----------
        Switch_Off_Delay : float
            "Time during which the Z position is averaged right before switching the Z-Controller off"*
        Change_Bias : string
            Is the bias changed prior to first Z ramp? 'yes' or 'no' or kept the 'same'.
        Bias : float
            Value the bias is changed to if Change_Bias = 'yes'
        Tip_Lift : float
            "Defines the relative height the tip is going to ramp for the first time (from the current Z position)."*
        Lift_Time : float
            "Defines the time to ramp Z from the current Z position by the Tip Lift amount."*
        Bias_Lift : float
            "Bias voltage applied just after the first Z ramping."*
        Bias_Settling_time : float
            "Time to wait after applying the Bias Lift value, and it is also the time to wait after applying Bias (V) before ramping Z for the first time."*
        Lift_Height : float
            "Defines the height the tip is going to ramp for the second time"*
        Lift_Time_2 : float
            "Given time to ramp Z in the second ramping"*
        End_Wait_Time : float
            "Time to wait after restoring the initial Bias voltage (just after finishing the second ramping)."*
        Restore_Feedback : string
            Is the feedback restored after TipShaper? 'yes' or 'no' or kept the 'same'
        
        *from TCP protocol manual
        
        Returns
        -------
        None.

        """
        
        #### Use this section to convert from user inputs to TCP inputs ####
        if Change_Bias == 0 or Change_Bias == 1 or Change_Bias == 2:
            Change_Bias_value = Change_Bias
        elif Change_Bias.lower() == 'same':
            Change_Bias_value = 0
        elif Change_Bias.lower() == 'yes':
            Change_Bias_value = 1
        elif Change_Bias.lower() == 'no':
            Change_Bias_value = 2
        
        if Restore_Feedback == 0 or Restore_Feedback == 1 or Restore_Feedback == 2:
            Restore_Feedback_value = Restore_Feedback
        elif Restore_Feedback.lower() == 'same':
            Restore_Feedback_value = 0
        elif Restore_Feedback.lower() == 'yes':
            Restore_Feedback_value = 1
        elif Restore_Feedback.lower() == 'no':
            Restore_Feedback_value = 2
            
                
        #### Standard Command Setup ####
        # Returns the number of channels, and their names! 
        
        
        #Command setup:
        command_name = 'TipShaper.PropsSet'
        self.sending_infomation = {'Switch Off Delay':Switch_Off_Delay, 'Change Bias?':Change_Bias_value, 'Bias (V)':Bias, 'Tip Lift (m)':Tip_Lift, 'Lift Time (s)':Lift_Time, \
                                   'Bias Lift (V)':Bias_Lift, 'Bias Settling time (s)':Bias_Settling_time, 'Lift Height (m)':Lift_Height, 'Lift Time 2 (s)': Lift_Time_2, \
                                       'End Wait Time (s)':End_Wait_Time, 'Restore Feedback?':Restore_Feedback_value}
        self.sending_infomation_types = ['float32', 'uint32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'float32', 'uint32']
        self.sending_special_length_loc = []
        
        self.response_required_value = 'Yes'
        
        #Do you want to see the error?
        self.print_error = 'Yes'
        
        #How to deal with returned data:
        self.recieved_infomation = {}
        self.recieved_infomation_types = []
        self.special_length_loc = []
        list_return = list(self.recieved_infomation.keys())
        
        #Do we need it to convert for us?  
        convert_required= 'No'
        
        #Run the function to generate command        
        self.run_command_fun(command_name, convert_required)
        
        #### Use this section to return values if required ####
       
        
        #Uncomment if you want it to return values
        #return self.return_output_data[list_return[1]], self.return_output_data[list_return[2]]  
        
        
        ################################
        ######### Atom Tracking ########
        ################################
        
        
    def AtomTrack_CtrlSet(self, atom_control_type, on_off):
        """
        (atom_control_type, status)
        Which type of tracking within the Atom tracking software is given by the first arguement. 
        The status sets wether the type of tracking is turned on (1) or off (2).
        

        Parameters
        ----------
        atom_control_type : string or int
            Set to 'modulation' (or 0), 'controller' (or 1), 'drift' (or 2) to select that module to them turn on/off
        on_off : string or int
            Turn 'on' (or 1), or turn 'off' (or 0).

        Returns
        -------
        None.

        """
         
        #### Use this section to convert from user inputs to TCP inputs ####
        # Atom type track
        if atom_control_type == 0 or atom_control_type == 1 or atom_control_type == 2:
            atom_control_type_value = atom_control_type
        elif atom_control_type.lower() == 'modulation':
            atom_control_type_value = 0
        elif atom_control_type.lower() == 'controller':
            atom_control_type_value = 1
        elif atom_control_type.lower() == 'drift':
            atom_control_type_value = 2
        
        
        # ON or OFF
        if on_off == 0 or on_off == 1:
            on_off_value = on_off
        elif on_off.lower() == 'on':
            on_off_value = 1
        elif on_off.lower() == 'off':
            on_off_value = 0
        
        #### Standard Command Setup ####
        # Turns the selected Atom Tracking control (modulation = 0, controller = 1
        # or drift measurement = 2) On or Off.
        #Then for the status set Off (=0) or On (=1)
         
        #Command setup:
        command_name = 'AtomTrack.CtrlSet'
        self.sending_infomation = {'AT Control':atom_control_type_value, 'Status': on_off_value}
        self.sending_infomation_types = ['uint16', 'uint16']
        
        self.response_required_value = 'Yes'
        
        #Do you want to see the error?
        self.print_error = 'Yes'
        
        #How to deal with returned data:
        self.recieved_infomation = {}
        self.recieved_infomation_types = []
        list_return = list(self.recieved_infomation.keys())
        
        #Do we need it to convert for us?  
        convert_required= 'No'
         
        #Run the function to generate command        
        self.run_command_fun(command_name, convert_required)
        
        #### Use this section to return values if required ####
        
        #Uncomment if you want it to return values
        #return self.return_output_data[list_return[0]]
         
    def AtomTrack_StatusGet(self, atom_control_type):
        """
        (atom_control_type)
        Gets the status of whatever atom control type is selected. Returning 'on' or 'off'.

        Parameters
        ----------
        atom_control_type : string or int
            Set to 'modulation' (or 0), 'controller' (or 1), 'drift' (or 2) to select that module to determine state.

        Returns
        -------
        on_off : string
            Returns string for the selected atom control 'on' or 'off'.

        """
         
        #### Use this section to convert from user inputs to TCP inputs ####
        # Atom type track
        if atom_control_type == 0 or atom_control_type == 1 or atom_control_type == 2:
            atom_control_type_value = atom_control_type
        elif atom_control_type.lower() == 'modulation':
            atom_control_type_value = 0
        elif atom_control_type.lower() == 'controller':
            atom_control_type_value = 1
        elif atom_control_type.lower() == 'drift':
            atom_control_type_value = 2
        
        #### Standard Command Setup ####
        # Turns the selected Atom Tracking control (modulation = 0, controller = 1
        # or drift measurement = 2) On or Off.
        #Then for the status set Off (=0) or On (=1)
         
        #Command setup:
        command_name = 'AtomTrack.StatusGet'
        self.sending_infomation = {'AT Control':atom_control_type_value}
        self.sending_infomation_types = ['uint16']
        
        self.response_required_value = 'Yes'
        
        #Do you want to see the error?
        self.print_error = 'Yes'
        
        #How to deal with returned data:
        self.recieved_infomation = {'Status':''}
        self.recieved_infomation_types = ['uint16']
        list_return = list(self.recieved_infomation.keys())
        
        #Do we need it to convert for us?  
        convert_required= 'Yes'
        
        #Run the function to generate command        
        self.run_command_fun(command_name, convert_required)
        
        #### Use this section to return values if required ####
        if self.return_output_data[list_return[0]] == 1:
            return 'on'
        if self.return_output_data[list_return[0]] == 0:
            return 'off'
        
        #Uncomment if you want it to return values
        #return self.return_output_data[list_return[0]]     
         
     
    def AtomTrack_PropsSet(self, intergral_gain, frequency_mod, amplitude_mod, phase_shift, switch_off_delay):
        """
        (intergral_gain, frequency_mod, amplitude_mod, phase_shift, switch_off_delay)
        Set the various parameters of the Atom Tracking software. 

        Parameters
        ----------
        intergral_gain : float
            Sets in AtomTracking Controller the Integral gain. In SI units, so 1e-10 = 100pm... 
        frequency_mod : float
            Sets in AtomTracking Modulation the frequency. In SI units, so 10 = 10 Hz.
        amplitude_mod : float
            Sets in AtomTracking Modulation the amplitude. In SI units, so 1e-10 = 100pm...
        phase_shift : float
            Sets in AtomTracking Modulation the phase. Measured in degrees. 
        switch_off_delay : float
            Sets in AtomTracking Controller the switch off delay (max 5 seconds).

        Returns
        -------
        None.

        """
        #### Use this section to convert from user inputs to TCP inputs ####
        if switch_off_delay > 5: 
            print('Note that the max \'Switch-off delay(s)\' is 5 seconds. ')
        
        #### Standard Command Setup ####
        # Sets the Atom tracking parameters
   
        
        #Command setup:
        command_name = 'AtomTrack.PropsSet'
        self.sending_infomation = {'Intergral Gain': intergral_gain, 'Freq Mod (Hz)': frequency_mod, 'Amp Mod (m)': amplitude_mod, 'Phase (deg)':phase_shift, 'Switch off delay(s)': switch_off_delay }
        self.sending_infomation_types = ['float32','float32','float32','float32','float32']
        
        self.response_required_value = 'Yes'
        
        #Do you want to see the error?
        self.print_error = 'Yes'
        
        #How to deal with returned data:
        self.recieved_infomation = {}
        self.recieved_infomation_types = []
        list_return = list(self.recieved_infomation.keys())
        
        #Do we need it to convert for us?  
        convert_required= 'No'
        
        #Run the function to generate command        
        self.run_command_fun(command_name, convert_required)
        
        #### Use this section to return values if required ####
        
        #Uncomment if you want it to return values
        #return self.return_output_data[list_return[0]]      
         
    def AtomTrack_PropsGet(self):
        """
        ()
        Gets and returns the variables listed from the Atom Tracking controller.

        Returns
        -------
        intergral_gain : float
            Gets in AtomTracking Controller the Integral gain. In SI units, so 1e-10 = 100pm... 
        frequency_mod : float
            Gets in AtomTracking Modulation the frequency. In SI units, so 10 = 10 Hz.
        amplitude_mod : float
            Gets in AtomTracking Modulation the amplitude. In SI units, so 1e-10 = 100pm...
        phase_shift : float
            Gets in AtomTracking Modulation the phase. Measured in degrees. 
        switch_off_delay : float
            Gets in AtomTracking Controller the switch off delay (max 5 seconds).

        """
        
        #### Use this section to convert from user inputs to TCP inputs ####
        
        
        #### Standard Command Setup ####
        # Sets the Atom tracking parameters
   
        
        #Command setup:
        command_name = 'AtomTrack.PropsGet'
        self.sending_infomation = {}
        self.sending_infomation_types = []
        
        self.response_required_value = 'Yes'
        
        #Do you want to see the error?
        self.print_error = 'Yes'
        
        #How to deal with returned data:
        self.recieved_infomation = {'Intergral Gain': '' , 'Freq Mod (Hz)': '', 'Amp Mod (m)': '', 'Phase (deg)':'', 'Switch off delay(s)': '' }
        self.recieved_infomation_types = ['float32','float32','float32','float32','float32']
        list_return = list(self.recieved_infomation.keys())
        
        #Do we need it to convert for us?  
        convert_required= 'Yes'
        
        #Run the function to generate command        
        self.run_command_fun(command_name, convert_required)
        
        #### Use this section to return values if required ####
        
        #Uncomment if you want it to return values
        return self.return_output_data[list_return[0]], self.return_output_data[list_return[1]], self.return_output_data[list_return[2]], self.return_output_data[list_return[3]], self.return_output_data[list_return[4]]     
       
        
    def AtomTrack_DriftComp(self):
        """
        ()
        Applies the drift measurement to the drift compensation. By turning it ON. 
        Note: You cannot turn it off without turning of the feedback and manually turning this off. 

        Returns
        -------
        None.

        """
        
        #### Use this section to convert from user inputs to TCP inputs ####
        
        
        #### Standard Command Setup ####
        # Atom tracking - sets the measured drift value to the compensation feature
   
        
        #Command setup:
        command_name = 'AtomTrack.DriftComp'
        self.sending_infomation = {}
        self.sending_infomation_types = []
        
        self.response_required_value = 'Yes'
        
        #Do you want to see the error?
        self.print_error = 'Yes'
        
        #How to deal with returned data:
        self.recieved_infomation = {}
        self.recieved_infomation_types = []
        list_return = list(self.recieved_infomation.keys())
        
        #Do we need it to convert for us?  
        convert_required= 'No'
        
        #Run the function to generate command        
        self.run_command_fun(command_name, convert_required)
        
        #### Use this section to return values if required ####
        
        #Uncomment if you want it to return values
        #return self.return_output_data[list_return[0]]
   
           
    def AtomTrack_QuickComp(self, tilt_or_drift):
        """
        (tilt_or_drift)
        Turns on the 'Quick Compensation' feature in the Atom Tracking module. This

        Parameters
        ----------
        tilt_or_drift : string or int
            Turn on the 'Quick Compensation' for either, 'smartilt' (or 0), or 'drift' (or 1).

        Returns
        -------
        None.

        """
        
        #### Use this section to convert from user inputs to TCP inputs ####
        #Convert to number if required!
        if tilt_or_drift == 1 or tilt_or_drift == 0:
            tilt_or_drift_value = tilt_or_drift
        elif tilt_or_drift.lower() == 'smartilt':
            tilt_or_drift_value = 0
        elif tilt_or_drift.lower() == 'drift':
            tilt_or_drift_value = 1
        
        #### Standard Command Setup ####
        # Atom tracking - Enables quick compensation
   
        
        #Command setup:
        command_name = 'AtomTrack.QuickCompStart'
        self.sending_infomation = {'AT Control': tilt_or_drift_value}
        self.sending_infomation_types = ['uint16']
        
        self.response_required_value = 'Yes'
        
        #Do you want to see the error?
        self.print_error = 'Yes'
        
        #How to deal with returned data:
        self.recieved_infomation = {}
        self.recieved_infomation_types = []
        list_return = list(self.recieved_infomation.keys())
        
        #Do we need it to convert for us?  
        convert_required= 'No'
        
        #Run the function to generate command        
        self.run_command_fun(command_name, convert_required)
        
        #### Use this section to return values if required ####
        
        #Uncomment if you want it to return values
        #return self.return_output_data[list_return[0]]        
       
       
       
        
    def AtomTrackQuick_Check_Hold(self):
        """
        ()
        Using the self.AtomTrack_QuickComp(tilt_or_drift) is non latching. Use this function to 
        test for the end of the compensation. 

        Returns
        -------
        None.

        """
        
        start_time = time.time()
        print('Checking if the AtomTrack Quick Comp is running (every 2 seconds)... Will hold here until finished!')
        print('Started - '+ time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
        check = 0
        
        while check == 0:
            time.sleep(2)
            if self.AtomTrack_StatusGet(2) == 'on':
                print('AtomTrack Quick Comp still running... (Elapsed Time = '+str(time.strftime('%H:%M:%S',time.localtime(time.time()-start_time)))+')')
                
            elif self.AtomTrack_StatusGet(2) == 'off':
                print('')
                print('AtomTrack Quick Comp completed! (Elapsed Time = '+str(time.strftime('%H:%M:%S',time.localtime(time.time()-start_time)))+')')
                print('')
                check = 1    
                
                
    
        ################################
        ######### Piezo Section ########
        ################################
    
    
    def Piezo_DriftCompSet(self, comp_on_off, Vx, Vy, Vz, sat_limit):
        """
        (comp_on_off, Vx, Vy, Vz, sat_limit)
        Turn on/off drift comp and set values of X,Y,Z as well as the piezo saturation limit.
        NOTE: Only can be turned OFF when not in feedback. Also Sat_lime max is 50%.

        Parameters
        ----------
        comp_on_off : sting or int
            Turn 'on' (or 1), or turn 'off' (or 0) the drift comp setting. 
        Vx : float
            Sets the drift comp in X, measured in m/s. 
        Vy : float
            Sets the drift comp in X, measured in m/s..
        Vz : float
            Sets the drift comp in X, measured in m/s..
        sat_limit : float
            The saturatation limit (measured in %) of the piezo range, drift comp will disengage at these limits. Note: max value is 50 (%)

        Returns
        -------
        None.

        """
        
        #### Use this section to convert from user inputs to TCP inputs ####
        if sat_limit > 50:
            print('NOTE: Only 50% or lower is accepted! Only 50% will be applied! ')
        
        
        # ON or OFF
        if comp_on_off == 0 or comp_on_off == 1 or comp_on_off == -1:
            comp_on_off_value = comp_on_off
        elif comp_on_off.lower() == 'on':
            comp_on_off_value = 1
        elif comp_on_off.lower() == 'off':
            comp_on_off_value = 0
        elif comp_on_off.lower() == 'no change':
            comp_on_off_value = -1
                
        #### Standard Command Setup ####
        # Returns the number of channels, and their names! 
        
        #Command setup:
        command_name = 'Piezo.DriftCompSet'
        self.sending_infomation = {'Compensation on/off': comp_on_off_value, 'Vx Value': Vx, 'Vy Value': Vy, 'Vz Value': Vz, 'Saturation': sat_limit}
        self.sending_infomation_types = ['int', 'float32', 'float32', 'float32', 'float32']
        self.sending_special_length_loc = []
        
        self.response_required_value = 'No'
        
        #Do you want to see the error?
        self.print_error = 'Yes'
        
        #How to deal with returned data:
        self.recieved_infomation = {}
        self.recieved_infomation_types = []
        self.special_length_loc = []
        list_return = list(self.recieved_infomation.keys())
        
        #Do we need it to convert for us?  
        convert_required= 'No'
        
        #Run the function to generate command        
        self.run_command_fun(command_name, convert_required)
        
        #### Use this section to return values if required ####
        
        #Uncomment if you want it to return values
        #return self.return_output_data[list_return[1]], self.return_output_data[list_return[2]]
        
    def Piezo_DriftCompGet(self):
        """
        ()
        Returns the status of the piezo drift compensation, current settings and the saturatation of the compensation value. 

        Returns
        -------
        comp_on_off : sting or int
            Turn 'on' (or 1), or turn 'off' (or 0) the drift comp setting. 
        Vx : float
            Sets the drift comp in X, measured in m/s. 
        Vy : float
            Sets the drift comp in X, measured in m/s..
        Vz : float
            Sets the drift comp in X, measured in m/s..
        SatX : float
            The saturatation limit of X (measured in %) of the piezo range.
        SatY : float
            The saturatation limit of X (measured in %) of the piezo range.
        SatZ : float
            The saturatation limit of X (measured in %) of the piezo range.
        Total_sat_value : float
            The saturatation limit (measured in %) of the piezo range, drift comp will disengage at this limits. Note: max value is 50 (%)

        """
        #### Use this section to convert from user inputs to TCP inputs ####
                
        #### Standard Command Setup ####
        # Returns the number of channels, and their names! 
        
        #Command setup:
        command_name = 'Piezo.DriftCompGet'
        self.sending_infomation = {}
        self.sending_infomation_types = []
        self.sending_special_length_loc = []
        
        self.response_required_value = 'Yes'
        
        #Do you want to see the error?
        self.print_error = 'Yes'
        
        #How to deal with returned data:
        self.recieved_infomation = {'Comp status':'', 'Vx':'', 'Vy':'', 'Vz':'', 'Satx':'', 'Saty':'', 'Satz':'', 'Sat limit':''}
        self.recieved_infomation_types = ['uint32', 'float32', 'float32', 'float32', 'uint32', 'uint32', 'uint32', 'float32']
        self.special_length_loc = []
        list_return = list(self.recieved_infomation.keys())
        
        #Do we need it to convert for us?  
        convert_required= 'Yes'
        
        #Run the function to generate command        
        self.run_command_fun(command_name, convert_required)
        
        #### Use this section to return values if required ####
        if self.return_output_data[list_return[0]] == 0:
            comp_stat = 'off'
        elif self.return_output_data[list_return[0]] == 1:
            comp_stat = 'on'
            
        Vx = self.return_output_data[list_return[1]]
        Vy = self.return_output_data[list_return[2]]
        Vz = self.return_output_data[list_return[3]]
        
        SatX = self.return_output_data[list_return[4]]
        SatY = self.return_output_data[list_return[5]]
        SatZ = self.return_output_data[list_return[6]]
        
        Total_sat_value = self.return_output_data[list_return[7]]
        
        #Uncomment if you want it to return values
        return comp_stat, Vx, Vy, Vz, SatX, SatY, SatZ, Total_sat_value
        
        
        
    def Piezo_TiltGet(self):
        """
        ()
        Returns the X-Axis and Y-Axis tilt correction values. 

        Returns
        -------
        tiltX : float
            Returns the tilt setting of the X-Axis. Measured in deg. 
        tiltY : float
            Returns the tilt setting of the Y-Axis. Measured in deg. 

        """

        #### Use this section to convert from user inputs to TCP inputs ####
        
        #### Standard Command Setup ####
        # Returns the number of channels, and their names! 
        
        #Command setup:
        command_name = 'Piezo.TiltGet'
        self.sending_infomation = {}
        self.sending_infomation_types = []
        self.sending_special_length_loc = []
        
        self.response_required_value = 'Yes'
        
        #Do you want to see the error?
        self.print_error = 'Yes'
        
        #How to deal with returned data:
        self.recieved_infomation = {'TiltX(deg)':'', 'TiltY(deg)':''}
        self.recieved_infomation_types = ['float32', 'float32']
        self.special_length_loc = []
        list_return = list(self.recieved_infomation.keys())
        
        #Do we need it to convert for us?  
        convert_required= 'Yes'
        
        #Run the function to generate command        
        self.run_command_fun(command_name, convert_required)
        
        #### Use this section to return values if required ####
        tiltX = self.return_output_data[list_return[0]]
        tiltY = self.return_output_data[list_return[1]]
        
        #Uncomment if you want it to return values
        return tiltX, tiltY

    def Piezo_TiltSet(self, tiltX, tiltY):
        """
        (tiltX, tiltY)
        Set the tilt of X and Y axis of the image. MEasured in degrees. 

        Parameters
        ----------
        tiltX : float
            Sets the tilt setting of the X-Axis. Measured in deg. 
        tiltY : float
            Sets the tilt setting of the Y-Axis. Measured in deg. 

        Returns
        -------
        None.

        """

        #### Use this section to convert from user inputs to TCP inputs ####
        
        #### Standard Command Setup ####
        # Returns the number of channels, and their names! 
        
        #Command setup:
        command_name = 'Piezo.TiltSet'
        self.sending_infomation = {'TiltX(deg)':tiltX, 'TiltY(deg)':tiltY}
        self.sending_infomation_types = ['float32', 'float32']
        self.sending_special_length_loc = []
        
        self.response_required_value = 'Yes'
        
        #Do you want to see the error?
        self.print_error = 'Yes'
        
        #How to deal with returned data:
        self.recieved_infomation = {}
        self.recieved_infomation_types = []
        self.special_length_loc = []
        list_return = list(self.recieved_infomation.keys())
        
        #Do we need it to convert for us?  
        convert_required = 'Yes'
        
        #Run the function to generate command        
        self.run_command_fun(command_name, convert_required)
        
        #### Use this section to return values if required ####
        
        
        #Uncomment if you want it to return values
        
    
    
    
    
    
        
        ################################
        ######### Bias  Spectra ########
        ################################
        
    def Bias_SpectraWinOpen(self):
        """
        ()
        Opens the Bias spectroscopy window.         

        Returns
        -------
        None.

        """
        
        #### Use this section to convert from user inputs to TCP inputs ####
        
        
        #### Standard Command Setup ####
        # Opens the Bias spectra window.
   
        
        #Command setup:
        command_name = 'BiasSpectr.Open'
        self.sending_infomation = {}
        self.sending_infomation_types = []
        
        self.response_required_value = 'Yes'
        
        #Do you want to see the error?
        self.print_error = 'Yes'
        
        #How to deal with returned data:
        self.recieved_infomation = {}
        self.recieved_infomation_types = []
        list_return = list(self.recieved_infomation.keys())
        
        #Do we need it to convert for us?  
        convert_required= 'No'
        
        #Run the function to generate command        
        self.run_command_fun(command_name, convert_required)
        
        #### Use this section to return values if required ####
        
        #Uncomment if you want it to return values
        #return self.return_output_data[list_return[0]]
       

    def Bias_SpectraStart(self, base_name_spectra):
        """
        (base_name_spectra)
        Runs the Bias spectra with the pre-defined parameters in the Bias spectra window. Also calls to open the window to prevent errors. 
        Can manually set the base filename. You must press 'Auto-save' for the spectra to save automatically on the spectra window. 
        This is locking, in that you cannot run any other python code whilst the spectra is taking data. 
        
        Parameters
        ----------
        base_name_spectra : string
            Sets the base name for the file spectra.

        Returns
        -------
        None.

        """
        
        #Need to open too! 
        self.Bias_SpectraWinOpen()
        #print('Note: If spectra will take longer than 20 minutes, this will crash. I can increase it if needed. ')
        print('')
        print('Spectra now being taken.')
        
        #### Use this section to convert from user inputs to TCP inputs ####
        get_data = 0 #Not currently setup here to recive data.
        base_name_length = len(base_name_spectra)
        
        # Need to make this latching... so we need to increase time for timeout
        self.s.settimeout(1200) # Shouldn't need 10 minutes for 1 spectra
        
        #### Standard Command Setup ####
        # Starts the spectra
   
        
        #Command setup:
        command_name = 'BiasSpectr.Start'
        self.sending_infomation = {'Get Data': get_data, 'Base Name Length': base_name_length, 'Base Name': base_name_spectra  }
        self.sending_infomation_types = ['uint32', 'int', 'string']
        
        self.response_required_value = 'Yes'
        
        #Do you want to see the error?
        self.print_error = 'Yes'
        
        #How to deal with returned data:
        self.recieved_infomation = {}
        self.recieved_infomation_types = []
        list_return = list(self.recieved_infomation.keys())
        
        #Do we need it to convert for us?  
        convert_required= 'No'
        
        #Run the function to generate command        
        self.run_command_fun(command_name, convert_required)
        
        #### Use this section to return values if required ####
        print('Spectra completed.')
        print('')
        #Reset timeout to 30 seconds
        self.s.settimeout(30)
        #Uncomment if you want it to return values
        #return self.return_output_data[list_return[0]]        
       
       
        
    def Bias_SpectraStatusGet(self):
        """
        ()
        Gets if the spectra is running. Not sure it can be used on the same port as the command to start the spectra was sent.

        Returns
        -------
        status: string
            Should be 'off' if not running, or 'on' if running.

        """
        #### Use this section to convert from user inputs to TCP inputs ####

        
        #### Standard Command Setup ####
        #Make sure the window is open to check. 
        self.Bias_SpectraWinOpen() 
   
        
        #Command setup:
        command_name = 'BiasSpectr.StatusGet'
        self.sending_infomation = {}
        self.sending_infomation_types = []
        
        self.response_required_value = 'Yes'
        
        #Do you want to see the error?
        self.print_error = 'Yes'
        
        #How to deal with returned data:
        self.recieved_infomation = {'Status':''}
        self.recieved_infomation_types = ['uint32']
        list_return = list(self.recieved_infomation.keys())
        
        #Do we need it to convert for us?  
        convert_required= 'Yes'
        
        #Run the function to generate command        
        self.run_command_fun(command_name, convert_required)
        
        #### Use this section to return values if required ####
        
        if self.return_output_data[list_return[0]] == 0:
            return_value = 'off'
        elif self.return_output_data[list_return[0]] == 1:
            return_value = 'on'
        
        
        #Uncomment if you want it to return values
        return return_value    

        ################################
        ########### Z Spectra ##########
        ################################
        
    def Z_SpectraStatusGet(self):
        """
        ()
        Gets if the spectra is running. Not sure it can be used on the same port as the command to start the spectra was sent.
    
        Returns
        -------
        status: string
            Should be 'off' if not running, or 'on' if running.
    
        """
        #### Use this section to convert from user inputs to TCP inputs ####
    
        
        #### Standard Command Setup ####
        
       
        #Make sure the window is open to check. 
        self.Z_SpectraWinOpen() 
       
        #Command setup:
        command_name = 'ZSpectr.StatusGet'
        self.sending_infomation = {}
        self.sending_infomation_types = []
        
        self.response_required_value = 'Yes'
        
        #Do you want to see the error?
        self.print_error = 'Yes'
        
        #How to deal with returned data:
        self.recieved_infomation = {'Status':''}
        self.recieved_infomation_types = ['uint32']
        list_return = list(self.recieved_infomation.keys())
        
        #Do we need it to convert for us?  
        convert_required= 'Yes'
        
        #Run the function to generate command        
        self.run_command_fun(command_name, convert_required)
        
        #### Use this section to return values if required ####
        
        if self.return_output_data[list_return[0]] == 0:
            return_value = 'off'
        elif self.return_output_data[list_return[0]] == 1:
            return_value = 'on'
        
        
        #Uncomment if you want it to return values
        return return_value    



    def Z_SpectraWinOpen(self):
        """
        ()
        Opens the Z spectroscopy window.         

        Returns
        -------
        None.

        """
        
        #### Use this section to convert from user inputs to TCP inputs ####
        
        
        #### Standard Command Setup ####
        # Opens the Bias spectra window.
   
        
        #Command setup:
        command_name = 'ZSpectr.Open'
        self.sending_infomation = {}
        self.sending_infomation_types = []
        
        self.response_required_value = 'Yes'
        
        #Do you want to see the error?
        self.print_error = 'Yes'
        
        #How to deal with returned data:
        self.recieved_infomation = {}
        self.recieved_infomation_types = []
        list_return = list(self.recieved_infomation.keys())
        
        #Do we need it to convert for us?  
        convert_required= 'No'
        
        #Run the function to generate command        
        self.run_command_fun(command_name, convert_required)
        
        #### Use this section to return values if required ####
        
        #Uncomment if you want it to return values
        #return self.return_output_data[list_return[0]]
       

    def Z_SpectraStart(self, base_name_spectra):
        """
        (base_name_spectra)
        Runs the Z spectra with the pre-defined parameters in the Bias spectra window. Also calls to open the window to prevent errors. 
        Can manually set the base filename. You must press 'Auto-save' for the spectra to save automatically on the spectra window. 
        This is locking, in that you cannot run any other python code whilst the spectra is taking data. 
        
        Parameters
        ----------
        base_name_spectra : string
            Sets the base name for the file spectra.

        Returns
        -------
        None.

        """
        
        #Need to open too! 
        self.Z_SpectraWinOpen()
        #print('Note: If spectra will take longer than 20 minutes, this will crash. I can increase it if needed. ')
        print('')
        print('Spectra now being taken.')
        
        #### Use this section to convert from user inputs to TCP inputs ####
        get_data = 0 #Not currently setup here to recive data.
        base_name_length = len(base_name_spectra)
        
        # Need to make this latching... so we need to increase time for timeout
        self.s.settimeout(1200) # Shouldn't need 10 minutes for 1 spectra
        
        #### Standard Command Setup ####
        # Starts the spectra
   
        
        #Command setup:
        command_name = 'ZSpectr.Start'
        self.sending_infomation = {'Get Data': get_data, 'Base Name Length': base_name_length, 'Base Name': base_name_spectra  }
        self.sending_infomation_types = ['uint32', 'int', 'string']
        
        self.response_required_value = 'Yes'
        
        #Do you want to see the error?
        self.print_error = 'Yes'
        
        #How to deal with returned data:
        self.recieved_infomation = {}
        self.recieved_infomation_types = []
        list_return = list(self.recieved_infomation.keys())
        
        #Do we need it to convert for us?  
        convert_required= 'No'
        
        #Run the function to generate command        
        self.run_command_fun(command_name, convert_required)
        
        #### Use this section to return values if required ####
        print('Spectra completed.')
        print('')
        #Reset timeout to 30 seconds
        self.s.settimeout(30)
        #Uncomment if you want it to return values
        #return self.return_output_data[list_return[0]]        
       
       
        


        ################################
        ############ Pattern ###########
        ################################

    def Pattern_ExpOpen(self):
        """
        ()
        Opens the Pattern section on the Scan control window (and the current experiment)

        Returns
        -------
        None.

        """
        #### Use this section to convert from user inputs to TCP inputs ####
   
        #### Standard Command Setup ####
        # Opens the pattern exp window
   
        #Command setup:
        command_name = 'Pattern.ExpOpen'
        self.sending_infomation = {}
        self.sending_infomation_types = []
        
        self.response_required_value = 'Yes'
        
        #Do you want to see the error?
        self.print_error = 'Yes'
        
        #How to deal with returned data:
        self.recieved_infomation = {}
        self.recieved_infomation_types = []
        self.special_length_loc = ['','','']
        list_return = list(self.recieved_infomation.keys())
        
        #Do we need it to convert for us?  
        convert_required= 'No'
        
        #Run the function to generate command        
        self.run_command_fun(command_name, convert_required)
        
        #### Use this section to return values if required ####
        
        #Uncomment if you want it to return values
        #return self.return_output_data[list_return[0]]            
       
        
    def Pattern_CloudGet(self):
        """
        ()
        Gets the position of all point given by 'Pattern'->'Cloud' on the scan control window. 
        
        Returns
        -------
        number_of_point: int
            Returns the number of points in the cloud
        X_list: float
            Returns a list of all the X positions of the cloud points. In SI units, so 1e-9 = 1 nm. 
        Y_list: float
            Returns a list of all the Y positions of the cloud points. In SI units, so 1e-9 = 1 nm. 

        """
       
        #### Use this section to convert from user inputs to TCP inputs ####
   
        
        #### Standard Command Setup ####
        # Gets all the positions of the cloud and ouputs these!
   
        #Command setup:
        command_name = 'Pattern.CloudGet'
        self.sending_infomation = {}
        self.sending_infomation_types = []
        
        self.response_required_value = 'Yes'
        
        #Do you want to see the error?
        self.print_error = 'No'
        
        #How to deal with returned data:
        self.recieved_infomation = {'Number of points':'', 'X Coord':'', 'Y Coord':''}
        self.recieved_infomation_types = ['int','1Dfloat32','1Dfloat32']
        self.special_length_loc = ['',0,0]
        list_return = list(self.recieved_infomation.keys())
        
        #Do we need it to convert for us?  
        convert_required= 'Yes'
        
        #Run the function to generate command        
        self.run_command_fun(command_name, convert_required)
        
        #### Use this section to return values if required ####
        
        #Uncomment if you want it to return values
        return self.return_output_data[list_return[0]], self.return_output_data[list_return[1]], self.return_output_data[list_return[2]]
    
    
    
    def Pattern_CloudSet(self, X_coord, Y_coord, active_pattern = 1):
        """
        (X_coord, Y_coord, active_pattern = 1)
        Sets an array of 'dots' on the scan window to show you where your X and Y coordinates are on the surface.

        Parameters
        ----------
        X_coord : 1D array of float
            List of X positions on the surface.
        Y_coord : 1D array of float
            List of X positions on the surface.
        active_pattern : int (0 or 1), optional
            The default is 1. Sets the 'Pattern' window to cloud. 

        Raises
        ------
        Exception
            If the length of X_coord and Y_coord do not match. 

        Returns
        -------
        None.

        """
        
        #### Use this section to convert from user inputs to TCP inputs ####
        length_of_coord = np.size(X_coord)
        if np.size(X_coord) != np.size(Y_coord):
            raise Exception('Length of X and Y coordinates does not match.')
                
        #### Standard Command Setup ####
        # Returns the number of channels, and their names! 
        
        #Command setup:
        command_name = 'Pattern.CloudSet'
        self.sending_infomation = {'set active pattern':active_pattern, 'num of points':length_of_coord, 'Xcoord': X_coord,'Ycoord':Y_coord}
        self.sending_infomation_types = ['uint32','int','1Dfloat32','1Dfloat32']
        self.sending_special_length_loc = ['','',1,1]
        
        self.response_required_value = 'Yes'
        
        #Do you want to see the error?
        self.print_error = 'No'
        
        #How to deal with returned data:
        self.recieved_infomation = {}
        self.recieved_infomation_types = []
        self.special_length_loc = []
        list_return = list(self.recieved_infomation.keys())
        
        #Do we need it to convert for us?  
        convert_required= 'No'
        
        #Run the function to generate command        
        self.run_command_fun(command_name, convert_required)
        
        #### Use this section to return values if required ####
        
        
       
        
        #Uncomment if you want it to return values
        #return self.return_output_data[list_return[1]], self.return_output_data[list_return[2]]
        
        
       
    def Pattern_GridGet(self):
        """
        ()
        Returns the description for a grid setup on Nanonis. 

        Returns
        -------
        Number_of_points_in_X : int
            Number of spectra points in X.
        Number_of_points_in_Y : int
            Number of spectra points in Y.
        CentreX : float
            Position of X where the centre of the grid is situated. 
        CentreY : float
            Position of Y where the centre of the grid is situated.
        width : float
            Width of the grid. (along X axis assuming angle = 0 deg)
        height : float
            Height of the grid. (only Y axis assuming angle = 0 deg)
        angle : float
            Angle of the grid. 

        """
       
        #### Use this section to convert from user inputs to TCP inputs ####
       
        
        #### Standard Command Setup ####
        
       
        #Command setup:
        command_name = 'Pattern.GridGet'
        self.sending_infomation = {}
        self.sending_infomation_types = []
        
        self.response_required_value = 'Yes'
        
        #Do you want to see the error?
        self.print_error = 'No'
        
        #How to deal with returned data:
        self.recieved_infomation = {'Number of points in X':'', 'Number of points in Y':'',\
                                'CentreX':'', 'CentreY':'' ,'width':'', 'height':'', 'angle':''}
        self.recieved_infomation_types = ['int','int', 'float32', 'float32','float32','float32','float32']
        self.special_length_loc = []
        list_return = list(self.recieved_infomation.keys())
        
        #Do we need it to convert for us?  
        convert_required= 'Yes'
        
        #Run the function to generate command        
        self.run_command_fun(command_name, convert_required)
        
        #### Use this section to return values if required ####
        Number_of_points_in_X = self.return_output_data[list_return[0]]
        Number_of_points_in_Y = self.return_output_data[list_return[1]]
        CentreX = self.return_output_data[list_return[2]]
        CentreY = self.return_output_data[list_return[3]]
        width = self.return_output_data[list_return[4]]
        height = self.return_output_data[list_return[5]]
        angle = self.return_output_data[list_return[6]]
        
        #Uncomment if you want it to return values
        return Number_of_points_in_X, Number_of_points_in_Y, CentreX, CentreY, width, height, angle
        
    
    

        
       
        
        
        ################################
        ############ Lock-in ###########
        ################################
    
    
    def LockIn_ModOnOffSet(self, modulator_number, on_off):
        """
        (modulator_number, on_off)
        
        Turn 'on' or 'off' the modulator in the lock-in module. 

        Parameters
        ----------
        modulator_number : int
            Select which modulator to turn on or off. 
        on_off : string or int
            Turn 'on' (or 1), or turn 'off' (or 0) the modulator. 

        Returns
        -------
        None.

        """
       
        #### Use this section to convert from user inputs to TCP inputs ####
        
        if on_off == 0 or on_off == 1:
            on_off_value = on_off
        elif on_off.lower() == 'on':
            on_off_value = 1
        elif on_off.lower() == 'off':
            on_off_value = 0
        
        #### Standard Command Setup ####
        # Turns on/off the modulator... modulator number is specifed and begins at 1
   
        #Command setup:
        command_name = 'LockIn.ModOnOffSet'
        self.sending_infomation = {'Modulator Num': modulator_number , 'On/Off':on_off_value}
        self.sending_infomation_types = ['int','uint32']
        
        self.response_required_value = 'Yes'
        
        #Do you want to see the error?
        self.print_error = 'Yes'
        
        #How to deal with returned data:
        self.recieved_infomation = {}
        self.recieved_infomation_types = []
        self.special_length_loc = []
        list_return = list(self.recieved_infomation.keys())
        
        #Do we need it to convert for us?  
        convert_required= 'No'
        
        #Run the function to generate command        
        self.run_command_fun(command_name, convert_required)
        
        #### Use this section to return values if required ####
        
        #Uncomment if you want it to return values
        #return self.return_output_data[list_return[0]]
       

    def LockIn_ModOnOffGet(self, modulator_number):
        """
        (modulator_number)
        Gets the status of the modulator selected and returns an 'on' or 'off'.

        Parameters
        ----------
        modulator_number : int
            Select which modulator to determine the status of. 

        Returns
        -------
        on_off : str
            Is the selected modulator 'on' or 'off'.

        """
       
        #### Use this section to convert from user inputs to TCP inputs ####
                
        #### Standard Command Setup ####
        # Gets the status of the modulator... modulator number is specifed and begins at 1
   
        #Command setup:
        command_name = 'LockIn.ModOnOffGet'
        self.sending_infomation = {'Modulator Num': modulator_number}
        self.sending_infomation_types = ['int']
        
        self.response_required_value = 'Yes'
        
        #Do you want to see the error?
        self.print_error = 'Yes'
        
        #How to deal with returned data:
        self.recieved_infomation = {'Lockin on/off status':''}
        self.recieved_infomation_types = ['uint32']
        self.special_length_loc = ['']
        list_return = list(self.recieved_infomation.keys())
        
        #Do we need it to convert for us?  
        convert_required= 'Yes'
        
        #Run the function to generate command        
        self.run_command_fun(command_name, convert_required)
        
        #### Use this section to return values if required ####
        if self.return_output_data[list_return[0]] == 0:
            return 'off'
        elif self.return_output_data[list_return[0]] == 1:
            return 'on'
        
        #Uncomment if you want it to return values
        #return self.return_output_data[list_return[0]]


    def LockIn_ModSignalSet(self, mod_signal_index, modulator_number):
        """
        TO COMPLETE

        """
        print('Do not use yet!')
       
        '''
        #### Use this section to convert from user inputs to TCP inputs ####
                
        #### Standard Command Setup ####
        # Gets the status of the modulator... modulator number is specifed and begins at 1
   
        #Command setup:
        command_name = 'LockIn.ModSignalSet'
        self.sending_infomation = {'Modulator Num': modulator_number, 'mod signal index': mod_signal_index }
        self.sending_infomation_types = ['int', 'int']
        
        self.response_required_value = 'Yes'
        
        #Do you want to see the error?
        self.print_error = 'Yes'
        
        #How to deal with returned data:
        self.recieved_infomation = {}
        self.recieved_infomation_types = []
        self.special_length_loc = []
        list_return = list(self.recieved_infomation.keys())
        
        #Do we need it to convert for us?  
        convert_required= 'Yes'
        
        #Run the function to generate command        
        self.run_command_fun(command_name, convert_required)
        
        #### Use this section to return values if required ####
        
        
        #Uncomment if you want it to return values
        #return self.return_output_data[list_return[0]]
        '''
        
    def LockIn_DemodPhasSet(self, demodulator_number, phase):
        """
        (demodulator_number, phase)
        Sets the Ref. phase of the demodulate part of the lock-in for whichever demodulator_number you set. 

        Parameters
        ----------
        demodulator_number : int
            Which demodulator are you wanting to change the Ref. Phase of?
        phase : float
            The reference phase that the demodulator will be set at. 

        Returns
        -------
        None.

        """
       
        #### Use this section to convert from user inputs to TCP inputs ####
                
        #### Standard Command Setup ####
        # Gets the status of the modulator... modulator number is specifed and begins at 1
   
        #Command setup:
        command_name = 'LockIn.DemodPhasSet'
        self.sending_infomation = {'deModulator Num': demodulator_number, 'Phase (deg)': phase }
        self.sending_infomation_types = ['int','float32']
        
        self.response_required_value = 'No'
        
        #Do you want to see the error?
        self.print_error = 'No'
        
        #How to deal with returned data:
        self.recieved_infomation = {}
        self.recieved_infomation_types = []
        self.special_length_loc = []
        list_return = list(self.recieved_infomation.keys())
        
        #Do we need it to convert for us?  
        convert_required= 'No'
        
        #Run the function to generate command        
        self.run_command_fun(command_name, convert_required)
        
        #### Use this section to return values if required ####

        
        #Uncomment if you want it to return values
        #return self.return_output_data[list_return[0]]


    def LockIn_DemodPhasGet(self, demodulator_number):
        """
        (demodulator_number)
        Returns the reference phase of the demodulator number provided.
        
        Parameters
        ----------
        demodulator_number : int
            Which demodulator are you wanting return the Ref. Phase of?

        Returns
        -------
        phase_value : float
            The reference phase value of that demodulator.

        """
       
        #### Use this section to convert from user inputs to TCP inputs ####
                
        #### Standard Command Setup ####
        # Gets the status of the modulator... modulator number is specifed and begins at 1
   
        #Command setup:
        command_name = 'LockIn.DemodPhasGet'
        self.sending_infomation = {'deModulator Num': demodulator_number}
        self.sending_infomation_types = ['int']
        
        self.response_required_value = 'Yes'
        
        #Do you want to see the error?
        self.print_error = 'Yes'
        
        #How to deal with returned data:
        self.recieved_infomation = {'Phase (deg)': ''}
        self.recieved_infomation_types = ['float32']
        self.special_length_loc = []
        list_return = list(self.recieved_infomation.keys())
        
        #Do we need it to convert for us?  
        convert_required= 'Yes'
        
        #Run the function to generate command        
        self.run_command_fun(command_name, convert_required)
        
        #### Use this section to return values if required ####
        phase_value = self.return_output_data[list_return[0]]
        return phase_value
        
        #Uncomment if you want it to return values
        #return self.return_output_data[list_return[0]]
        
        
        
    def LockIn_ModAmpSet(self, amplitude, modulator_number = 1):
        """
        (amplitude, modulator_number = 1)
        Set the modulator output amplitude. 

        Parameters
        ----------
        amplitude : float
            Amplitude of the modulaing value. 
        modulator_number : int, optional
            Specifies which modulator to use. The default is 1. Unless you are doing somthing special, or have a specific setup, this is not required to be changed. 

        Returns
        -------
        None.

        """
        
        #### Use this section to convert from user inputs to TCP inputs ####
        
                
        #### Standard Command Setup ####
        # Returns the number of channels, and their names! 
        
        #Command setup:
        command_name = 'LockIn.ModAmpSet'
        self.sending_infomation = {'modulator number':modulator_number, 'amplitude': amplitude}
        self.sending_infomation_types = ['int','float32']
        self.sending_special_length_loc = []
        
        self.response_required_value = 'Yes'
        
        #Do you want to see the error?
        self.print_error = 'No'
        
        #How to deal with returned data:
        self.recieved_infomation = {}
        self.recieved_infomation_types = []
        self.special_length_loc = []
        list_return = list(self.recieved_infomation.keys())
        
        #Do we need it to convert for us?  
        convert_required= 'No'
        
        #Run the function to generate command        
        self.run_command_fun(command_name, convert_required)
        
        #### Use this section to return values if required ####
        
        
        #Uncomment if you want it to return values
        #return self.return_output_data[list_return[1]], self.return_output_data[list_return[2]]

    def LockIn_ModPhasFreqSet(self, frequency, modulator_number = 1):
        """
        (frequency, modulator_number = 1)
        Sets the modulator output frequency.

        Parameters
        ----------
        frequency : float
            Frequency of the modulaing value. 
        modulator_number : int, optional
            Specifies which modulator to use. The default is 1. Unless you are doing somthing special, or have a specific setup, this is not required to be changed. 

        Returns
        -------
        None.

        """
        
        #### Use this section to convert from user inputs to TCP inputs ####
        
                
        #### Standard Command Setup ####
        # Returns the number of channels, and their names! 
        
        #Command setup:
        command_name = 'LockIn.ModPhasFreqSet'
        self.sending_infomation = {'modulator number':modulator_number, 'freq': frequency}
        self.sending_infomation_types = ['int','float64']
        self.sending_special_length_loc = []
        
        self.response_required_value = 'Yes'
        
        #Do you want to see the error?
        self.print_error = 'No'
        
        #How to deal with returned data:
        self.recieved_infomation = {}
        self.recieved_infomation_types = []
        self.special_length_loc = []
        list_return = list(self.recieved_infomation.keys())
        
        #Do we need it to convert for us?  
        convert_required= 'No'
        
        #Run the function to generate command        
        self.run_command_fun(command_name, convert_required)
        
        #### Use this section to return values if required ####
        
        
        #Uncomment if you want it to return values
        #return self.return_output_data[list_return[1]], self.return_output_data[list_return[2]]


    def LockIn_ModPhasFreqGet(self, modulator_number = 1):
        """
        (modulator_number = 1)
        Returns the modulator output frequency

        Parameters
        ----------
        modulator_number : int, optional
            Specifies which modulator to use. The default is 1. Unless you are doing somthing special, or have a specific setup, this is not required to be changed. 

        Returns
        -------
        frequency : float
            Frequency of the modulaing value. 

        """
        
        #### Use this section to convert from user inputs to TCP inputs ####
        
                
        #### Standard Command Setup ####
        # Returns the number of channels, and their names! 
        
        #Command setup:
        command_name = 'LockIn.ModPhasFreqGet'
        self.sending_infomation = {'modulator number':modulator_number}
        self.sending_infomation_types = ['int']
        self.sending_special_length_loc = []
        
        self.response_required_value = 'Yes'
        
        #Do you want to see the error?
        self.print_error = 'No'
        
        #How to deal with returned data:
        self.recieved_infomation = {'Frequency':''}
        self.recieved_infomation_types = ['float64']
        self.special_length_loc = []
        list_return = list(self.recieved_infomation.keys())
        
        #Do we need it to convert for us?  
        convert_required= 'Yes'
        
        #Run the function to generate command        
        self.run_command_fun(command_name, convert_required)
        
        #### Use this section to return values if required ####
        frequency = self.return_output_data[list_return[0]]
        
        return frequency
        
        #Uncomment if you want it to return values
        #return self.return_output_data[list_return[1]], self.return_output_data[list_return[2]]


    ################################
    ############ Signals ###########
    ################################
    
    def Signals_NamesGet(self):
        """
        ()
        Returns the a list of all the siganls availible to view/use in Nanonis. Can use this list to determine which channel number you want to use. 

        Returns
        -------
        Number_of_names : int
            Total number of channels is returned.
        Signal_names : list of string
            Names of each channel. 

        """
        
        #### Use this section to convert from user inputs to TCP inputs ####
                
        #### Standard Command Setup ####
        # Returns the number of channels, and their names! 
        
        #Command setup:
        command_name = 'Signals.NamesGet'
        self.sending_infomation = {}
        self.sending_infomation_types = []
        
        self.response_required_value = 'Yes'
        
        #Do you want to see the error?
        self.print_error = 'No'
        
        #How to deal with returned data:
        self.recieved_infomation = {'Names size': '', 'Names Number': '', 'Signal Names':''}
        self.recieved_infomation_types = ['uint32', 'uint32' , '1Dstringarray']
        self.special_length_loc = ['','',0]
        list_return = list(self.recieved_infomation.keys())
        
        #Do we need it to convert for us?  
        convert_required= 'Yes'
        
        #Run the function to generate command        
        self.run_command_fun(command_name, convert_required)
        
        #### Use this section to return values if required ####
        #if self.return_output_data[list_return[0]] == 0:
        #    return 'off'
        #elif self.return_output_data[list_return[0]] == 1:
         #   return 'on'
        
        #Uncomment if you want it to return values
        return self.return_output_data[list_return[1]], self.return_output_data[list_return[2]]
    
    
    def Signals_ValGet(self, signal_channel_index, new_data = 0):
        """
        (signal_channel_index, new_data = 0)
        Returns the value of the channel given. 
        
        Example is channel 0 is commonly Current (I). This would provide the current value. 

        Parameters
        ----------
        signal_channel_index : int
            Channels numbered 0 to X (system dependent).
        new_data : int, optional
            "Selects whether the function returns the next available signal value or if it waits for a full period of new data." The default is 0.
            Please see manual for more details. Setting new_data = 0 is OK. 
            
        Returns
        -------
        value : float
            Value of the channel at that moment in time. 

        """
        
        #### Use this section to convert from user inputs to TCP inputs ####
                
        #### Standard Command Setup ####
        # Returns the number of channels, and their names! 
        
        #Command setup:
        command_name = 'Signals.ValGet'
        self.sending_infomation = {'Signal index': signal_channel_index, 'New data?': new_data}
        self.sending_infomation_types = ['int','uint32']
        
        self.response_required_value = 'Yes'
        
        #Do you want to see the error?
        self.print_error = 'No'
        
        #How to deal with returned data:
        self.recieved_infomation = {'Signal Value': ''}
        self.recieved_infomation_types = ['float32']
        self.special_length_loc = []
        list_return = list(self.recieved_infomation.keys())
        
        #Do we need it to convert for us?  
        convert_required= 'Yes'
        
        #Run the function to generate command        
        self.run_command_fun(command_name, convert_required)
        
        #### Use this section to return values if required ####
        value = self.return_output_data[list_return[0]]
        
        return value
        
        #Uncomment if you want it to return values
        #return self.return_output_data[list_return[1]], self.return_output_data[list_return[2]]

    def Signals_ValsGet(self, signal_channel_indexes, new_data = 0):
        """
        (signal_channel_indexes, new_data = 0)
        Returns the values of the list of channels given as a list in the same order. 
    

        Parameters
        ----------
        signal_channel_indexes : list of int
            Channels numbered 0 to X (system dependent). 
        new_data : int, optional
            "Selects whether the function returns the next available signal value or if it waits for a full period of new data." The default is 0.
            Please see manual for more details. Setting new_data = 0 is OK. 

        Returns
        -------
        signal_values : list of floats
            List of the channel's values at that moment in time. 

        """
        
        #### Use this section to convert from user inputs to TCP inputs ####
        size_of_array = len(signal_channel_indexes)
                
        #### Standard Command Setup ####
        # Returns the number of channels, and their names! 
        
        #Command setup:
        command_name = 'Signals.ValsGet'
        self.sending_infomation = {'Size of signal index array': size_of_array, 'Signal indexes': signal_channel_indexes, 'New data?': new_data}
        self.sending_infomation_types = ['int','1Dint','uint32']
        self.sending_special_length_loc = ['',0,'']
        
        self.response_required_value = 'Yes'
        
        #Do you want to see the error?
        self.print_error = 'No'
        
        #How to deal with returned data:
        self.recieved_infomation = {'Signal size array': '','Signal Value': ''}
        self.recieved_infomation_types = ['int', '1Dfloat32']
        self.special_length_loc = ['',0]
        list_return = list(self.recieved_infomation.keys())
        
        #Do we need it to convert for us?  
        convert_required= 'Yes'
        
        #Run the function to generate command        
        self.run_command_fun(command_name, convert_required)
        
        #### Use this section to return values if required ####
        signal_values = self.return_output_data[list_return[1]]
        
        return signal_values
        
        #Uncomment if you want it to return values
        #return self.return_output_data[list_return[1]], self.return_output_data[list_return[2]]




    ################################
    ######## Data Logger ###########
    ################################
    
    def Data_LoggerOpen(self):
        """
        ()
        Opens the data logger window.

        Returns
        -------
        None.

        """
       
        #### Use this section to convert from user inputs to TCP inputs ####
                
        #### Standard Command Setup ####
        # Opens the data logger
        
        #Command setup:
        command_name = 'DataLog.Open'
        self.sending_infomation = {}
        self.sending_infomation_types = []
        
        self.response_required_value = 'No'
        
        #Do you want to see the error?
        self.print_error = 'No'
        
        #How to deal with returned data:
        self.recieved_infomation = {}
        self.recieved_infomation_types = []
        self.special_length_loc = []
        list_return = list(self.recieved_infomation.keys())
        
        #Do we need it to convert for us?  
        convert_required= 'No'
        
        #Run the function to generate command        
        self.run_command_fun(command_name, convert_required)
        
        #### Use this section to return values if required ####
        #if self.return_output_data[list_return[0]] == 0:
        #    return 'off'
        #elif self.return_output_data[list_return[0]] == 1:
         #   return 'on'
        
        #Uncomment if you want it to return values
        #return self.return_output_data[list_return[0]]


    def Data_LoggerStart(self):
        """
        ()
        Starts the data logger recording data. You must make sure the settings are as you want them to be. 

        Returns
        -------
        None.

        """
       
        #### Use this section to convert from user inputs to TCP inputs ####
                
        #### Standard Command Setup ####
        # Opens the data logger
        
        #Command setup:
        command_name = 'DataLog.Start'
        self.sending_infomation = {}
        self.sending_infomation_types = []
        
        self.response_required_value = 'No'
        
        #Do you want to see the error?
        self.print_error = 'No'
        
        #How to deal with returned data:
        self.recieved_infomation = {}
        self.recieved_infomation_types = []
        self.special_length_loc = []
        list_return = list(self.recieved_infomation.keys())
        
        #Do we need it to convert for us?  
        convert_required= 'No'
        
        #Run the function to generate command        
        self.run_command_fun(command_name, convert_required)
        
        #### Use this section to return values if required ####
        #if self.return_output_data[list_return[0]] == 0:
        #    return 'off'
        #elif self.return_output_data[list_return[0]] == 1:
         #   return 'on'
        
        #Uncomment if you want it to return values
        #return self.return_output_data[list_return[0]]   

    def Data_LoggerStop(self):
        """
        ()
        This stops the data logger recording data. 
        
        Returns
        -------
        None.

        """
       
        #### Use this section to convert from user inputs to TCP inputs ####
                
        #### Standard Command Setup ####
        # Opens the data logger
        
        #Command setup:
        command_name = 'DataLog.Stop'
        self.sending_infomation = {}
        self.sending_infomation_types = []
        
        self.response_required_value = 'No'
        
        #Do you want to see the error?
        self.print_error = 'No'
        
        #How to deal with returned data:
        self.recieved_infomation = {}
        self.recieved_infomation_types = []
        self.special_length_loc = []
        list_return = list(self.recieved_infomation.keys())
        
        #Do we need it to convert for us?  
        convert_required= 'No'
        
        #Run the function to generate command        
        self.run_command_fun(command_name, convert_required)
        
        #### Use this section to return values if required ####
        #if self.return_output_data[list_return[0]] == 0:
        #    return 'off'
        #elif self.return_output_data[list_return[0]] == 1:
         #   return 'on'
        
        #Uncomment if you want it to return values
        #return self.return_output_data[list_return[0]]
        
    def Data_LoggerStatusGet(self):
        """
        ()
        Gets the parameters from the data logger.  

        Returns
        -------
        start_datetime: string
            The datetime that the last recording was started.
        run_hours: int
            The length of time in hours that the data logger last ran for.
        run_mins: int
            The length of time in mins that the data logger last ran for.
        run_seconds: float
            The length of time in seconds that the data logger last ran for.
        end_datetime: string
            The datetime that the last recording was ended.
        loc_of_recording_data: string
            Save path of the last data recording. 

        """
       
       
        #### Use this section to convert from user inputs to TCP inputs ####
                
        #### Standard Command Setup ####
        # Returns data logger status paramters
        # Returns
        # 0: Start datetime
        # 1:run Hours
        # 2:run min
        # 3:run sec
        # 4:run end datetim
        # 5:loc of the saved data file. 
        
        #Command setup:
        command_name = 'DataLog.StatusGet'
        self.sending_infomation = {}
        self.sending_infomation_types = []
        
        self.response_required_value = 'Yes'
        
        #Do you want to see the error?
        self.print_error = 'No'
        
        #How to deal with returned data:
        self.recieved_infomation = {'Start Time size':'', 'Start Time':'', 'Elapsed Hours':'','Elapsed Min':'','Elapsed Sec':''\
                                    , 'Stop time Size':'','Stop Time':'', 'Saved file path size': '', 'Saved file path':''\
                                        , 'Points counter':''}
        self.recieved_infomation_types = ['int','string','uint16','uint16','float32','int','string', 'int', 'string','int']
        self.special_length_loc = ['',0,'','','','',5,'',7,'']
        list_return = list(self.recieved_infomation.keys())
        
        #Do we need it to convert for us?  
        convert_required= 'Yes'
        
        #Run the function to generate command        
        self.run_command_fun(command_name, convert_required)
        
        #### Use this section to return values if required ####
        #if self.return_output_data[list_return[0]] == 0:
        #    return 'off'
        #elif self.return_output_data[list_return[0]] == 1:
         #   return 'on'
        
        #Uncomment if you want it to return values
        return self.return_output_data[list_return[1]], self.return_output_data[list_return[2]], self.return_output_data[list_return[3]], self.return_output_data[list_return[4]], self.return_output_data[list_return[6]], self.return_output_data[list_return[8]]  


    def Data_LoggerChsSet(self, channel_indexes):
        """
        (channel_indexes)
        Input the channel indexs as a list for the channels you wish to record with the data logger i.e [0,6,7,19], for channels 0, 6, 7, 19. 
        Can find the channel number using self.Signals_NamesGet().
        
        Parameters
        ----------
        channel_indexes : list with floats
            Input as a list the channel numbers to record with the data logger. 

        Returns
        -------
        None.

        """
       
        #### Use this section to convert from user inputs to TCP inputs ####

        num_of_channels = len(channel_indexes)
        #### Standard Command Setup ####
        # Selects the channels used in the data logger...
        #input [0,12,13,14] for Current, X,Y,Z. Use Signals_NamesGet to determine channel number

        #Command setup:
        command_name = 'DataLog.ChsSet'
        self.sending_infomation = {
            'Number of Channels': num_of_channels, 'Channel Indexes': channel_indexes}
        self.sending_infomation_types = ['int', '1Dint']
        self.sending_special_length_loc = ['', 0]

        self.response_required_value = 'No'

        #Do you want to see the error?
        self.print_error = 'No'

        #How to deal with returned data:
        self.recieved_infomation = {}
        self.recieved_infomation_types = []
        self.special_length_loc = []
        list_return = list(self.recieved_infomation.keys())

        #Do we need it to convert for us?
        convert_required = 'No'

        #Run the function to generate command
        self.run_command_fun(command_name, convert_required)

        #### Use this section to return values if required ####
        #if self.return_output_data[list_return[0]] == 0:
        #    return 'off'
        #elif self.return_output_data[list_return[0]] == 1:
        #   return 'on'

        #Uncomment if you want it to return values
        #return self.return_output_data[list_return[0]]


    def Data_LoggerPropsSet(self, basename_str, comment_str, averaging = 10):
        """
        (basename_str, comment_str, averaging = 10)
        Sets the important parameters of the data logger. I have stripped out most of the fucntionality, so that the values in the data logger parametrs stay roughly the same. 
        You can change the basename string, the comment string, and the number of averaging points the software takes for each point. Addtional functionality can be added easily. 

        Parameters
        ----------
        basename_str : string
            Sets the basename of the datalogger file, for additional files with the same basename, numbes will be automatically added
        comment_str : string
            Sets the comment within the data logger file. Can be useful to dicuss what you have changed. 
        averaging : int, optional
            The number of values averaged for a single point. The default is 10.

        Returns
        -------
        None.

        """
       
        #### Use this section to convert from user inputs to TCP inputs ####
                
        #### Standard Command Setup ####
        # Sets the data logger to:
        Aqu_mode = 1 #Continuous mode
        Aqu_hours = -1 #no chnage
        Aqu_min = -1 #no chnage
        Aqu_sec = -1 #no chnage
        
        basename_size = len(basename_str)
        comment_size = len(comment_str)
        
        size_of_modules = 0
        number_of_modules = 0
        list_of_modules = ''
        
        
        #Command setup:
        command_name = 'DataLog.PropsSet'
        self.sending_infomation = {'Acquisition mode':Aqu_mode ,'Acquisition duration(hours)':Aqu_hours \
                                   , 'Acquisition duration(min)':Aqu_min , 'Acquisition duration(sec)':Aqu_sec \
                                   , 'Averaging' :averaging , 'Basename size':basename_size , 'Basename':basename_str , 'Comment size':comment_size \
                                   , 'Comment':comment_str , 'Size of the list of modules':size_of_modules , 'Number of modules':number_of_modules \
                                   , 'List of Modules':list_of_modules}
            
        self.sending_infomation_types = ['uint16', 'int', 'int', 'float32', 'int', 'int', 'string', 'int','string', 'int', 'int','string']
        self.sending_special_length_loc = ['','','','','','',5,'',7]
        
        self.response_required_value = 'No'
        
        #Do you want to see the error?
        self.print_error = 'No'
        
        #How to deal with returned data:
        self.recieved_infomation = {}
        self.recieved_infomation_types = []
        self.special_length_loc = []
        list_return = list(self.recieved_infomation.keys())
        
        #Do we need it to convert for us?  
        convert_required= 'No'
        
        #Run the function to generate command        
        self.run_command_fun(command_name, convert_required)
        
        #### Use this section to return values if required ####
        #if self.return_output_data[list_return[0]] == 0:
        #    return 'off'
        #elif self.return_output_data[list_return[0]] == 1:
         #   return 'on'
        
        #Uncomment if you want it to return values
        #return self.return_output_data[list_return[0]]
        
        
        
    
    '''
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    USER DEFINED fuctions are here!
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    Note: Setup as a nested class, just to make it easier to 
    '''
        
    #######################################################################
    # Bias Functions
    #######################################################################
    
    def Bias_set_slow(self, bias, time_to_take_seconds=10, num_of_steps=50):
        """
        (bias, time_to_take_seconds=10, num_of_steps=50)
        Use this function to slowly change the bias from the current value to a new value. Useful where a large change may be a 'bias pulse'.
        

        Parameters
        ----------
        bias : set bias applied to sample (float)
            Applies this value to the bias controller.
        time_to_take_seconds : float or int, optional(set to 10s)
            How long the system will take to reach the bias set
        num_of_steps : int, optional(set to 50 values between current and final)
            The default is 50. the number of steps to take betwen current and final bias. 

        Returns
        -------
        None.

        """
        
        #Use this to change the bias slowly from the current value to a different one.
        if time_to_take_seconds < 5:
            print('Might take longer than 5 seconds...Due to request time')
        #Get current bias and setup an array of values that will act as intermediate value on the way.
        bias_start = self.Bias_get()
        print('Moving bias from '+str(bias_start)+' V to '+str(bias)+' V in '+str(time_to_take_seconds)+'s')
        
        bias_values = np.linspace(bias_start, bias, num_of_steps)
        
        sleep_time = time_to_take_seconds / num_of_steps
        
        for i in bias_values: 
            self.Bias_set(i)
            time.sleep(sleep_time)
            
        
        print('Completed slow bias move - Bias = ' + str(self.Bias_get())+' V')
        
        
        
    def Scan_Check_Hold(self):
        """
        ()
        Use this function to prevent continued execution of your python script, if you want to wait untill the end of the scan.
        Will check the scan status every 10 seconds. This can be reduced if required. 

        Returns
        -------
        None.

        """
        start_time = time.time()
        print('Checking if the Scan is running (every 10 seconds)... Will hold here until turned off!')
        print('Started - '+ time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
        check = 0
        
        #First check, do we even need this?
        if self.Scan_StatusGet() == 'off':
            print('Scan already not stopped, skipping this check!')
            check = 1
        
        while check == 0:
            time.sleep(10)
            if self.Scan_StatusGet() == 'on':
                print('Scan still running... (Elapsed Time = '+str(time.strftime('%H:%M:%S',time.localtime(time.time()-start_time)))+')')
                
            elif self.Scan_StatusGet() == 'off':
                if self.Z_SpectraStatusGet() == 'on' or self.Bias_SpectraStatusGet() == 'on':
                    print('Detected spectra running, still waiting for scan to complete!')
                else:    
                    print('')
                    print('Scan completed! (Elapsed Time = '+str(time.strftime('%H:%M:%S',time.localtime(time.time()-start_time)))+')')
                    print('')
                    check = 1
                
    def AutoApproach_Check_Hold(self):
        """
        ()
        Use this function to prevent continued execution of your python script, if you want to wait until the tip has reached the surface.
        Will check the status every 10 seconds. This can be reduced if required. 
        

        Returns
        -------
        None.

        """
        
        start_time = time.time()
        print('Checking if the Auto Approach is running (every 10 seconds)... Will hold here until turned off!')
        print('Started - '+ time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
        check = 0
        
        #First check, do we even need this?
        if self.AutoApproach_OnOffGet() == 'off':
            print('Auto Approach already not running, skipping this check!')
            check = 1
        
        while check == 0:
            time.sleep(10)
            if self.AutoApproach_OnOffGet() == 'on':
                print('Auto Approach still running... (Elapsed Time = '+str(time.strftime('%H:%M:%S',time.localtime(time.time()-start_time)))+')')
                
            elif self.AutoApproach_OnOffGet() == 'off':
                print('')
                print('Auto Approach completed! (Elapsed Time = '+str(time.strftime('%H:%M:%S',time.localtime(time.time()-start_time)))+')')
                print('')
                check = 1
                
                
    
    def Pattern_GridGetPoints(self): 
        #We want to get the points of the current grid! 
        [Number_of_points_in_X, Number_of_points_in_Y, CentreX, CentreY, width, height, angle] = self.Pattern_GridGet()
        
    
    def cart2pol(self, x, y):
        rho = np.sqrt(x**2 + y**2)
        deg = np.degrees(np.arctan2(y, x))
        return(rho, deg)
    
    def test_phase_lock_in(self, LI_1X_channel_number, LI_1Y_channel_number, time_to_record_phase):
        #Can find the XY values using self.lock_in_measured_values
        #Can find the R and deg using self.lock_in_measured_values_R_deg
        
        self.values_of_phase = []
        start_time = time.time()
        while time.time() - start_time < time_to_record_phase:
            #print(self.Signals_ValsGet([LI_1X_channel_number,LI_1Y_channel_number]))
            self.values_of_phase.append(self.Signals_ValsGet([LI_1X_channel_number,LI_1Y_channel_number]))
            
        #Convert to R and deg
        self.lock_in_measured_values = np.array(self.values_of_phase)
        self.lock_in_measured_values = [np.mean(self.lock_in_measured_values[:,0]),np.mean(self.lock_in_measured_values[:,1])]
        self.lock_in_measured_values_R_deg = self.cart2pol(self.lock_in_measured_values[0], self.lock_in_measured_values[1])
        
        
    def is_phase_OK(self, phase, phase_allowable_error, phase_desired = 0):
        
        if np.abs(phase) < np.abs(phase_allowable_error):
            #Test if phase is near 0.
            OK_test = True  
            phase_diff = phase
        else:
            # phase is not near zero here
            OK_test = False 
            phase_diff = phase
        
        return OK_test, phase_diff
            
        
    
    def lock_in_setup(self, frequency, bias_oscillation, retract_distance = 20e-9, feedback_on_after = 'yes', lock_in_signal = 1, time_to_record_phase = 5, change_current_gain_mode = 'auto', phase_allowable_error = 3):
        

        
        #First make sure that lock-in is off
        self.LockIn_ModOnOffSet(1,'off')
        
        
        #Assuming if the feeback is off... them you aren't in tunnelling.(but I will check anyway!)
        feedback_get = self.Z_feedback_get()
        time.sleep(0.5)
        current_get = self.Signals_ValGet(0)
        
        if feedback_get == 'off' and np.abs(current_get) < 20e-12:
            print('Feedback is off, and current is less than 20 pA.... OK to proceed!')
        else:
            
            print('In feedback, will retract %e m now.' % retract_distance)
            off_Z_pos = self.Z_pos_get() + retract_distance
            time.sleep(1) #NEEDS SLEEP OTHERWISE IT just ignores the command (bug?!)
            self.Z_feedback_set('off')
            time.sleep(1)
            self.Z_pos_set(off_Z_pos)
            print('Done!')
            print('')
        
        
        print('Setting up the lock-in amplifer.')
        
        print('Setting supplied frequency = %.1f, amplitude = %.1f and turning ON modulator!' % (frequency,bias_oscillation))
        #Change parameters to those given! 
        self.LockIn_ModAmpSet(bias_oscillation)
        self.LockIn_ModPhasFreqSet(frequency)
        
        #Turning on the lock-in following the setup
        self.LockIn_ModOnOffSet(1,'on')

        
        
        
        time.sleep(1) # Wait for this to happen
        
        
        #What is the current phase difference?
        #current_channel_number = self.find_signal_number_of_string('Current (A)')
        LI_1X_channel_number = self.find_signal_number_of_string('LI Demod 1 X (A)')
        LI_1Y_channel_number = self.find_signal_number_of_string('LI Demod 1 Y (A)')
        
        
        
        print('Testing the lock-in phase angle for '+str(time_to_record_phase)+' seconds...')
        self.test_phase_lock_in(LI_1X_channel_number, LI_1Y_channel_number, time_to_record_phase)
        #Note that the phase difference needs to be multiples of 90 deg to have all of the 
        phase = self.lock_in_measured_values_R_deg[1]
        print('Done! Found phase difference to be %.2f deg' % phase)
        
        
        #Check if the phase needs to be changed
        test_phase, phase_diff_measured = self.is_phase_OK(phase, phase_allowable_error)
        while test_phase == False:
            #This means that the phase is not zero here
            print('Need to change phase now. Attempting to do so...')
            
            current_ref_phase = self.LockIn_DemodPhasGet(1)
            if phase_diff_measured >0 and current_ref_phase<0:
                self.LockIn_DemodPhasSet(1, current_ref_phase+phase_diff_measured)
            elif phase_diff_measured >0 and current_ref_phase>0:
                self.LockIn_DemodPhasSet(1, current_ref_phase-phase_diff_measured)
                
                # take away if pos
                #self.LockIn_DemodPhasSet(1, current_ref_phase-phase_diff_measured)
            else:
                # add if neg
                self.LockIn_DemodPhasSet(1, current_ref_phase+phase_diff_measured)
                    
            
            self.test_phase_lock_in(LI_1X_channel_number, LI_1Y_channel_number, time_to_record_phase)
            phase = self.lock_in_measured_values_R_deg[1]
            
            test_phase, phase_diff_measured = self.is_phase_OK(phase, phase_allowable_error)
            print('Measured phase now = %.2f deg' % phase_diff_measured)
            
        print('Now tested OK!')
        
        print('')
        if feedback_on_after == 'yes':
            print('Now turning feedback back on')
            self.Z_feedback_set('on')
            
        print('Subtratcting 90 deg to phase now to get all capcitive coupling in Y as is convention')
        new_ref_phase = self.LockIn_DemodPhasGet(1) - 90
        time.sleep(1)
        self.LockIn_DemodPhasSet(1, new_ref_phase)
            
        print()
        print('All signal from surface should now be +ve in Lia1X')
        
       
        
        
    def find_signal_number_of_string(self, signal_name_str):
        
        #Collect all the names (as strings!)
        signals_names = [string_value.decode('utf-8') for string_value in self.Signals_NamesGet()[1]]
        
        try:
            channel_number = signals_names.index(signal_name_str)
            return channel_number
        
        except:
            print('\"\"' + signal_name_str + '\"\"'+ ' was not found in the channels on this system')
            
    
        
    
    
    
    def grid_maker(self, X_points, Y_points, X_centre, Y_centre, X_size, Y_size, angle):
        """
        
        To be completed
        angle relative to the +ve X axis
        """
        
        if X_points<2 or Y_points<2:
            raise Exception('This is not a number that is allow, use points higher than 2')
        
        X_centre_temp = 0
        Y_centre_temp = 0
        
        X_grid_pos = []
        Y_grid_pos = []
        
        #Let's start at the top LHS. 
        start_tempX = (X_centre_temp - X_size/2)
        start_tempY = (Y_centre_temp + Y_size/2)
        
        #Let's make sure that it would be centred! 
        #end_tempX = (X_centre_temp - X_size/2) + (X_points-1)*(X_size/X_points)
        #end_tempY = (Y_centre_temp + Y_size/2) - (Y_points-1)*(Y_size/Y_points)
        
        end_tempX = start_tempX + X_size
        end_tempY = start_tempY - Y_size
        
        #Now make start = end
        X_change = (start_tempX + end_tempX)/2
        Y_change = (start_tempY + end_tempY)/2
        
        start_tempX = start_tempX - X_change
        start_tempY = start_tempY - Y_change
        
        for ii in range(Y_points):
            # Y positions
            tempY = start_tempY - ii* (Y_size/(Y_points-1))
            for i in range(X_points):
                # X positions
                tempX = start_tempX + i* (X_size/(X_points-1))
            
                X_grid_pos.append(tempX)
                Y_grid_pos.append(tempY)
                
        #Get so we can add to it easily!
        X_grid_pos.reverse()
        Y_grid_pos.reverse()
        positions = np.rot90(np.array([X_grid_pos, Y_grid_pos]))
        
        angle_rad = np.deg2rad(angle)
        
        #Deal with angle now:
        for i in range(np.size(positions,0)):
            x_pos_temp = positions[i,0]
            y_pos_temp = positions[i,1]
            
            positions[i,0] = x_pos_temp*np.cos(angle_rad) - y_pos_temp*np.sin(angle_rad)
            positions[i,1] = y_pos_temp*np.cos(angle_rad) + x_pos_temp*np.sin(angle_rad)
        
        #Now deal with the centre
        positions[:,0] = positions[:,0] + X_centre
        positions[:,1] = positions[:,1] + Y_centre
        
        
            
        return positions[:,0], positions[:,1]

           

if __name__ == "__main__":
    #Only runs if you are running this as the main script file. 
    x = python_Nanonis_TCP()
    
    #x.Signals_ValsGet([86,87])
    #x.LockIn_ModPhasFreqSet(2200)
    
    
    #for i in range(200):
    #    [a,b] = x.Scan_FrameDataGrab(0)
    #    plt.clf()
    #    plt.imshow(b)
    #    plt.pause(1)

    #x.Scan_PropsGet()
    #x.Scan_PropsGet()
    #bias_now = x.Bias_get()
    
    #[a,b] = x.Scan_FrameDataGrab(30,'forwards')
    
    #x.Bias_SpectraStart('hello')
    #x.Pattern_CloudGet()
    #x.Signals_NamesGet()
    #x.Data_LoggerChsSet([0,12,13,14])


#x.command_convert_to_hex('ZCtrl.ZPosGet')
#x.form_header_code()
#x.body_forming()
#x.send_request_TCP()
#print(x.all_request)

#%%
#x.XY_tip_pos()
#x.XY_tip_set_pos(5e-9,5e-9)
#x.Bias_set(-6)
#x.Bias_get()
#x.Z_pos_get()

#plt.pause(1)


# need to look at the error
#x.Scan_action(0,1)

#x.close_socket()
#x.set_up_socket()


