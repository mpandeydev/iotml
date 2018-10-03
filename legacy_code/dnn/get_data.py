import numpy as np
import csv
import matplotlib.pyplot as plt
from scipy.signal import lfilter

# Selection lines
vehicle_list = ['A','B','C','D','E','F','G','H']
speed_list = ['20','30','40','50','60']
sensor_list = ['s001','s002','s003']

vehicle = 'A'
speed = '40'
sensor = 's001'
filter_param = 10

b = [1.0/filter_param]*filter_param
a = 1

root_path = '../mag_data/'
path = root_path+vehicle+speed+'/'+sensor+'.csv'

margin = 50
window_size = 500
jump = 10
#------------------------------------------------------------------------------

# Function definitions

def get_stats(data):
    std_dev = 0
    mean = 0
    mean = sum(data)/len(data)
    for i in data:
        dev = abs(i-mean)
        std_dev+=dev
    std_dev/=len(data)
    lower = 0
    higher = 0
    for i in range(1,len(data)):
        #print(abs(mag_z[i]-mean))
        #print(2*std_dev)
        #print()
        if abs(data[i]-mean)>10*std_dev:
            if lower > 0:
                higher = i
            else:
                lower = i
        if (higher-lower)>100:
            break
    #print(lower,higher)
    data = data[lower-margin:higher+margin]
    return mean,std_dev,lower,higher,data

def window_data(chosen_data, window_size):
    global jump
    # To be executed after get_pick_drop
    # Arguments : Type of data (Pick,Drop,Noise) and Window size
    # Returns : List of lists, each sublist being a data window of specified size
    data = [ int(x) for x in chosen_data ]
    windows = []
    index = 10
    while ((index+window_size)<len(data)):
        windows.append(data[index:(index + window_size)])
        index += jump
    return windows

def get_max_window(mag_data):
    global a,b,window_size
    mag_z_filtered = lfilter(b,a,mag_data)
    mag_z_windows = window_data(mag_z_filtered,window_size)
    
    to_plot = []
    flat_list = []
    std_list = []
    index = 0
    flag = 0
    old_index = 0
    for window in mag_z_windows:
        data_range = len(window)
        #plt.plot(range(0,len(mag_x)),mag_x)
        #plt.plot(range(0,len(mag_y)),mag_y)
        mean,std_dev,lower,higher,windowlet = get_stats(window)
        std_list.append(std_dev)
    
    windex = std_list.index(max(std_list))
    
    max_window = mag_z_windows[windex]
    return windex,max_window
#------------------------------------------------------------------------------

mag_x = []
mag_y = []
mag_z = []

for sel_vehicle in vehicle_list:
    for sel_speed in speed_list:
        for sel_sensor in sensor_list:
            mag_x = []
            mag_y = []
            mag_z = []
            path = root_path+sel_vehicle+sel_speed+'/'+sel_sensor+'.csv'
            #print(path)
            with open(path, 'r') as csvfile:
                reader = csv.DictReader(csvfile,fieldnames=['timestamp','x','y','z'])
                for row in reader:
                    mag_x.append(float(row['x']))
                    mag_y.append(float(row['y']))
                    mag_z.append(float(row['z']))
                    
            #mag_z = lfilter(b,a,mag_z)
            #mean, std_dev,lower,higher,mag_zya = get_stats(mag_z)
            plt.plot(range(0,len(mag_z)),mag_z)
            
            #print(mag_z)
            #mag_z = []
            #mag_z = lfilter(b,a,mag_x)
            #mean, std_dev,lower,higher,mag_zya = get_stats(mag_z)
            plt.plot(range(0,len(mag_x)),mag_x)
            
            #mag_z = []
            #mag_z = lfilter(b,a,mag_y)
            #mean, std_dev,lower,higher,mag_zya = get_stats(mag_z)
            plt.plot(range(0,len(mag_y)),mag_y)
            
            
            #plt.title(str(sel_vehicle)+'_'+str(sel_speed))
            plt.title(str(sel_vehicle)+'_'+str(sel_speed)+'_'+str(sel_sensor))
            plt.savefig('../mag_plots/raw/'+str(sel_vehicle)+'_'+str(sel_speed)+'_'+str(sel_sensor)+'.png')
            #plt.show()
            plt.close()

'''for sel_vehicle in vehicle_list:
    for sel_speed in speed_list:
        for sel_sensor in sensor_list:   
            
            mag_x = []
            mag_y = []
            mag_z = []
            path = root_path+sel_vehicle+sel_speed+'/'+sel_sensor+'.csv'
            #print(path)
            with open(path, 'r') as csvfile:
                reader = csv.DictReader(csvfile,fieldnames=['timestamp','x','y','z'])
                for row in reader:
                    mag_x.append(float(row['x']))
                    mag_y.append(float(row['y']))
                    mag_z.append(float(row['z']))
                        
            #mean, std_dev,lower,higher,mag_x = get_stats(mag_x)
            #mean, std_dev,lower,higher,mag_y = get_stats(mag_y)
            #mean, std_dev,lower,higher,mag_z = get_stats(mag_z)
            
            ind_x,max_x = get_max_window(mag_x)
            ind_y,max_y = get_max_window(mag_y)
            ind_z,max_z = get_max_window(mag_z)
            
            start_dex = min([ind_x,ind_y,ind_z])
            pad_x = ind_x - start_dex
            pad_y = ind_y - start_dex
            pad_z = ind_z - start_dex
                
            plt.plot(range(pad_z*jump,pad_z*jump+len(max_x)),max_z)
            plt.plot(range(pad_x*jump,pad_x*jump+len(max_y)),max_x)
            plt.plot(range(pad_y*jump,pad_y*jump+len(max_z)),max_y)
            
            
            #plt.title(str(sel_vehicle)+'_'+str(sel_speed))
            plt.title(str(std_dev))
            plt.savefig('../mag_plots/events/'+str(sel_vehicle)+'_'+str(sel_speed)+'_'+str(sel_sensor)+'.png')
            #plt.show()
            plt.close()                
            to_plot = []
            flat_list = []
            old_index = index'''
