import numpy as np
from numpy.polynomial import polynomial as P
import re
import io
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import struct
from datetime import datetime
import os
import glob



class output_data_spectra_dat():
    
    def get_file(self, pathname, filename, show_methods=0):

        #Find get the file name
        self.filename_path = pathname + '\\' + filename
        
        #Now we want to get the data and read it. 
        self.load_file(show_methods)
        
        if show_methods == 1:
            self.show_method_fun()
    
    def convert_time_to_epoch(self, time):
        dt = datetime.datetime.strptime(str(time), '%d.%m.%Y %H:%M:%S')
        return dt.timestamp()

    # def read_metadata_info(meta_data):
    #     """
    #     Note: some files will not have this info e.g. data logger files
    #     Can be extended/edited to access more/different metadata info.
    #     """

    #     # Extract position information:
    #     try: self.x_pos = float(meta_data.split('X (m)\t')[1].split('\n')[0][0:-1])
    #     except: pass
    #     try: self.y_pos = float(meta_data.split('Y (m)\t')[1].split('\n')[0][0:-1])
    #     except: pass
    #     try: self.z_pos = float(meta_data.split('Z (m)\t')[1].split('\n')[0][0:-1])
    #     except: pass

    #     # Extract the time information:
    #     try: self.start_time = self.convert_time_to_epoch(
    #         meta_data[meta_data.index('Start time') + 11:meta_data.index('Start time') + 11 + 19])
    #     except: pass

    #     # Extract comment
    #     try: self.comment = meta_data.split('Comment01\t')[1].split('\n')[0][0:-1]
    #     except: pass
    
    def load_file(self, show_methods):


        #Open the file. The data and headings start after the '[data]' section.
        f = open(self.filename_path, "r")
        f.seek(0)
        file_all_data = f.read()
        f.close()
        
        #Split at the '[data]' section. The second in the list is our data with the headers.
        file_data = re.split(r"\[DATA\]\n",file_all_data)[1]
        md = re.split(r"\[DATA\]\n", file_all_data)[0]

        #read_metadata_info(meta_data)

        #Load as df to make life easy
        self.df = pd.read_csv(io.StringIO(file_data),delimiter = '\t')
        self.meta_data = np.array(pd.read_csv(io.StringIO(md),delimiter = '\t'))
        
        self.list_of_methods = list(self.df)
     
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
                    method_number = count
                    break
            if method_number is not count:
                #We did not convert to an int value
                raise Exception("This input is not found in the list, check spelling is correct!")
                
        elif type(method_number) is not int: 
            raise Exception("Must be either and int or str variable")
        
        name_to_export = self.list_of_methods[method_number]
        
        exported_data = (self.df[name_to_export],self.list_of_methods[method_number])
        
        return exported_data
    


m=output_data_spectra_dat()

def get_metadata(pathname,filename):
    m.get_file(pathname,filename,show_methods=0)
    m.load_file(show_methods=0)
    meta_data = m.meta_data
    Datetime = meta_data[:,1][meta_data[:,0]=='Saved Date'][0]
    date = re.split(r' ',Datetime)[0]
    time = re.split(r' ',Datetime)[1]
    return date,time



def get_time_array(pathname,whichdata,howfar,timeaway):
    #list all files of a certain dataset and timeaway from molecule
    # print(pathname)
    all_files = os.listdir(pathname)
    matching_files = [f for f in all_files if f.startswith('Loop_1s_4.1K_') and f.endswith('.dat')]
    #print(matching_files)
    timedate_array=[]
    for i in range(0,100):
        try:
            numfiles = str('00050')
            
            filenum = str(int(i))
            filename0 = f'Loop_1s_4.1K_{filenum}_10nm_1_00001.dat'
            filename1 = f'Loop_1s_4.1K_{filenum}_10nm_1_{numfiles}.dat'
            # filename0 = 'Dataset_'+str(whichdata)+'_'+str(howfar)+'_'+str(timeaway)+'_'+str(filenum)+'_00001.dat'
            # filename1 = 'Dataset_'+str(whichdata)+'_'+str(howfar)+'_'+str(timeaway)+'_'+str(filenum)+'_'+str(numfiles)+'.dat'
            
            date0,time0 = get_metadata(pathname,filename0)
            date1,time1 = get_metadata(pathname,filename1)
            
            timedate_array.append([date0,date1,time0,time1])
            print(filename0)
        except Exception as e:
            print(f"Get_time_array ended at iteration {i}: {e}")
            break
        
    return timedate_array



def timeshift(time):
    timeparts = time.split(':')
    timeparts[0] = str(int(timeparts[0])+24)
    newtime = ':'.join(timeparts)
    return newtime


def time_to_seconds(t):
    h,m,s = map(int,t.split(':'))
    return h*3600 + m*60 + s
    
    
    
def import_temps(pathname):
    all_files = os.listdir(pathname)
    matching_files = [f for f in all_files if f.startswith('temp_data_2025') and f.endswith('.txt')]
    #print(matching_files)
    temparray=[]
    for fname in matching_files:
        try:
            print('Importing temperature data from file: '+str(fname))
            with open(str(pathname) + '/' + str(fname), "r") as f:
                content = f.read().splitlines()[1:]
            tempdata = np.array([re.split(r"[,T]", line) for line in content])
            # tempdata = np.array([re.split(r"[,]", line[1]) for line in tempdata[:,0]])
            # print(tempdata)
            
            for i in range(1, len(tempdata[:,0])):
                # if i % 50 == 0:
                    # print(tempdata[i][1])
                dataparts = tempdata[i][1].split(':')
                dataparts[-1] = round(float(dataparts[-1]))
                dataparts[-1] = "{:02}".format(dataparts[-1])
                tempdata[i][1] = ':'.join(dataparts)
            
                # date handling
                dt = datetime.strptime(tempdata[i][0], "%Y-%m-%d")
                tempdata[i][0] = dt.strftime("%d.%m.%Y")
                
            tempdata = np.delete(tempdata,0,axis=0)
            tempdata = np.delete(tempdata,[3,4,5],axis=1)
                
            temparray.append(tempdata)
        except Exception as e:
            print(f"Import_temps failed for file {fname}: {e}")
            # continue
        
    #now sort by date in ascending order using the first date entry
    temparray = sorted(temparray,key=lambda a: datetime.strptime(a[0, 0], "%d.%m.%Y"))    
    
    
    return temparray


def time_to_temp(timeval,temparr):
    temparrchop = list(temparr[:,1])
    temparrchop_seconds = np.array([time_to_seconds(t) for t in temparrchop])
    timeval_seconds = time_to_seconds(timeval)
    idx = np.abs(temparrchop_seconds-timeval_seconds).argmin()
    return temparr[idx][2]


def get_data_array(timedates,temparray):
    data_array = []
    
    startdate = timedates[0][0]
    
    #look through first entry in temparray[i] and find one that matches date to data (startdate)
    for i in range(len(temparray)):
        if temparray[i][0][0] == startdate:
            dayind = i #find dayindex to use to call the right day for temperature array. use i+1 later if we need
    temps0 = temparray[dayind]
    temps1 = temparray[dayind]
    for i in range(len(timedates)):
        date0 = timedates[i][0]
        date1 = timedates[i][1]
        time0 = timedates[i][2]
        time1 = timedates[i][3]
        
        # if the time spreads over to the next day, change temparray to next one
        if date0 != startdate:
            #time0 = timeshift(time0)
            temps0 = temparray[dayind+1]
            
        if date1 != startdate:
            #time1 = timeshift(time1)
            temps1 = temparray[dayind+1]
            
        #now find closest match from time array to temp array 
        T0 = time_to_temp(time0,temps0)
        T1 = time_to_temp(time1,temps1)
        
        data_array.append([float(T0),float(T1)])
            
    return data_array

    

def tempvstime(temparray,date,start_time,end_time):
    for i in range(len(temparray)):
        if temparray[i][0][0] == date:
            dayind = i
        
    temps = np.array(temparray[dayind][:,2],dtype=float)
    #print(temps)
    times = temparray[dayind][:,1]
    times_seconds = np.array([time_to_seconds(t) for t in times])
    
    #indstart = np.where(times == start_time)[0][0]
    #indend = np.where(times == end_time)[0][0]
    #print(indstart)
    temps_chop = temps[0:-1]
    times_seconds_chop = times_seconds[0:-1]
    times_seconds_chop = times_seconds_chop - times_seconds_chop[0]
    
    plt.scatter(times_seconds_chop,temps_chop)
    
    return temps_chop,times_seconds_chop

    
    
def import_counts(pathname_counts,filename_counts):
    counts=pd.read_csv(str(pathname_counts)+r'\\'+str(filename_counts))
    counts=np.array(counts)

    return counts        
    
#%%

############################### PATHNAME ################################
#########################################################################
timepath = r'E:\Nanoscience data\DBTF\November_2\21.11-1s-loops'

tempspath = r'E:\Nanoscience data\DBTF\November_2\Temps'

countspath = r'/home/ppyop1/PhD_stuff/DBTF/DBTF_counting_data'
#########################################################################
#########################################################################


### import and generate data ##
# This outputs an array of 3 values... Temperature : Count (by hand), Temperature error
def generate_raildata(tempspath, countspath, timepath, pos,datasetnum,time):
    
    timedates = get_time_array(timepath,datasetnum,'10nm',time)
    temparray = import_temps(str(tempspath))
    data_array = np.array(get_data_array(timedates,temparray))
    
    # counts = import_counts(str(countspath) , str(pos)+'_dataset'+str(datasetnum)+'_'+str(time)+'s.csv')

    # countsarr = np.full(len(timedates),np.nan)
    # countsarr[counts[:, 0].astype(int)] = counts[:, 1]

    xdata = np.mean(data_array,axis=1)
    #ydata = countsarr
    xerr = np.abs(np.mean(data_array,axis=1)- np.array([data_array[:,0],data_array[:,1]]))
    #print(xerr)

    return np.array([xdata,xerr[0]])
    #return [xdata,ydata,xerr]
    
# timetemps=[]
# for i in np.arange(50):
#     try: 
#         timedates = get_time_array(timepath,i,'10nm','0.1')
#         temparray = import_temps(str(tempspath))
#         data_array = np.array(get_data_array(timedates,temparray))
#         timetemps.append(data_array)
#         print('Imported dataset '+str(i))
#     except Exception:
#         print('No such thing as dataset '+str(i)+' you idiot')
        
        


#Inputs are: path for temps, path for counts, path for the timedata, prefix of filename (related to position on rail in nm)
#### dataset number (from filename), time of tip away from DeeBeeTeeEff (from filename)
# data27_0pt1_pos3 = generate_raildata(tempspath,countspath,timepath,'',27,'0.1')
# data27_0pt5_pos3 = generate_raildata(tempspath,countspath,timepath,'',27,'0.5')
# data28_0pt1_pos3 = generate_raildata(tempspath,countspath,timepath,'',28,'0.1')
# data28_1_pos3 = generate_raildata(tempspath,countspath,timepath,'',28,'1')
# data24_0pt1_pos2 = generate_raildata(tempspath,countspath,timepath,'',24,'0.1')
# data24_5_pos2 = generate_raildata(tempspath,countspath,timepath,'',24,'5')
# data25_0pt1_pos2 = generate_raildata(tempspath,countspath,timepath,'',25,'0.1')
# data25_0pt5_pos2 = generate_raildata(tempspath,countspath,timepath,'',25,'0.5')
# data26_0pt1_pos2 = generate_raildata(tempspath,countspath,timepath,'',26,'0.1')
# data26_1_pos2 = generate_raildata(tempspath,countspath,timepath,'',26,'1')


#%%
# pathsave=r'/home/ppyop1/Documents/DBTFdiffusion_xdata_4freddie'
# np.save(str(pathsave)+'/data27_0pt1_pos3.npy',np.array(data27_0pt1_pos3))
# np.save(str(pathsave)+'/data27_0pt5_pos3.npy',np.array(data27_0pt5_pos3))
# np.save(str(pathsave)+'/data28_0pt1_pos3.npy',np.array(data28_0pt1_pos3))
# np.save(str(pathsave)+'/data28_1_pos3.npy',np.array(data28_1_pos3))
# np.save(str(pathsave)+'/data24_0pt1_pos2.npy',np.array(data24_0pt1_pos2))
# np.save(str(pathsave)+'/data24_5_pos2.npy',np.array(data24_5_pos2))
# np.save(str(pathsave)+'/data25_0pt1_pos2.npy',np.array(data25_0pt1_pos2))
# np.save(str(pathsave)+'/data25_0pt5_pos2.npy',np.array(data25_0pt5_pos2))
# np.save(str(pathsave)+'/data26_0pt1_pos2.npy',np.array(data26_0pt1_pos2))
# np.save(str(pathsave)+'/data26_1_pos2.npy',np.array(data26_1_pos2))


#%% plotting shit. RUN THIS CELL ON ITS OWN IF YOU DONT WANT TO WAIT AGES FOR IMPORTING AGAIN
def generate_plot(data_to_plot,ax,colour,marker,label,title):
    x = data_to_plot[0]
    y = data_to_plot[1]
    y=np.ones_like(x)
    xerr = data_to_plot[1]
    
    ax.errorbar(x,y,xerr=xerr,yerr = (1/50) , capsize=2,c=colour,marker=marker,ls='',label=label,markersize=8)
    ax.legend(fontsize=15,loc='upper right')
    ax.set_title(str(title),fontsize=18,fontweight='bold')
    ax.tick_params(labelsize=15)
    plt.show()

# fig,ax=plt.subplots()
# generate_plot(data27_0pt1_pos3,ax,'r','s','lol','hehexd')




def main():
    input_path = r"C:\Users\ppxfc1\Downloads\Times for Phil (DS 25).txt"
    output_path = r"C:\Users\ppxfc1\Downloads\STM_cooldown_25.txt"

    df = pd.read_csv(
        input_path,
        skiprows=1,                 # Skip "Times for Phil (DS 24):"
        usecols=["Datetime", "STM"]
    )
    # df.columns = df.columns.str.strip()
    print(df.columns)

    df.to_csv(output_path, sep="\t", index=False)
    print(f"Saved Datetime and STM columns to: {output_path}")

    return 1



main()