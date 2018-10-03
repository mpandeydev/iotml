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

root_path = '../../mag_data/raw/'
path = root_path+vehicle+speed+'/'+sensor+'.csv'

margin = 50
window_size = 400
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
    index = 0
    while ((index+window_size)<len(data)):
        windows.append(data[index:(index + window_size)])
        index += window_size
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
                        
            x_windows = window_data(mag_x,window_size)
            y_windows = window_data(mag_y,window_size)
            z_windows = window_data(mag_z,window_size)
            
            windex = 0
            while windex<len(x_windows):
                
                x_plot = []
                y_plot = []
                z_plot = []
                
                mean,std_dev,lower,higher,data = get_stats(x_windows[windex])
                for element in x_windows[windex]:
                    x_plot.append(element-mean)
                
                mean,std_dev,lower,higher,data = get_stats(y_windows[windex])
                for element in y_windows[windex]:
                    y_plot.append(element-mean)
                
                mean,std_dev,lower,higher,data = get_stats(z_windows[windex])
                for element in z_windows[windex]:
                    z_plot.append(element-mean)
                
                x_plot = lfilter(b,a,x_plot)
                y_plot = lfilter(b,a,y_plot)
                z_plot = lfilter(b,a,z_plot)
                
                x_list = list(x_plot)
                y_list = list(y_plot)
                z_list = list(z_plot)
                veh_list = []
                veh_list.append(vehicle_list.index(sel_vehicle))
                spe_list = []
                spe_list.append(speed_list.index(sel_speed))
                sen_list = []
                sen_list.append(sensor_list.index(sel_sensor))
                
                name = str(sel_vehicle)+'_'+str(sel_speed)+'_'+str(sel_sensor)+'_'+str(windex)+'.csv'
                
                with open(name,'w') as csvfile:
                    writer = csv.writer(csvfile)
                    for x in x_list:
                        writer.writerow([x])
                    for y in y_list:
                        writer.writerow([y])
                    for z in z_list:
                        writer.writerow([z])
                    writer.writerow(veh_list)
                    writer.writerow(spe_list)
                    writer.writerow(sen_list)
                
                '''plt.plot(range(0,len(x_windows[windex])),x_plot)
                plt.plot(range(0,len(y_windows[windex])),y_plot)
                plt.plot(range(0,len(z_windows[windex])),z_plot)
                
                #plt.title(str(sel_vehicle)+'_'+str(sel_speed))
                plt.title(str(sel_vehicle)+'_'+str(sel_speed)+'_'+str(sel_sensor)+'_'+str(windex))
                plt.savefig('../../mag_plots/chunked/'+str(sel_vehicle)+'_'+str(sel_speed)+'_'+str(sel_sensor)+'_'+str(windex)+'.png')
                plt.close()                '''
                
                
                
                windex+=1
