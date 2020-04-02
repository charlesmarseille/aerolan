import numpy as np
#from matplotlib.dates import datestr2num
import matplotlib.pyplot as plt
from datetime import datetime as dt
import pandas as pd


#%% Function definitions
print("COSQM Data load function, returns data and dates in tuple")

def LoadData(path,cache={}):
    print (path)
    if path in cache:
        return cache[path]
    else:
        try:
            data_server = np.genfromtxt(path, usecols = list(np.arange(2,17)),invalid_raise=False)
            dates_str = np.genfromtxt(path, delimiter = ' ', usecols = [0,1], dtype = 'str')
            dates_cosqm = np.array([ dt.strptime( dates+times, '%Y-%m-%d%H:%M:%S' ).timestamp() for dates, times in dates_str ])
            cache[path] = (data_server, dates_cosqm)
            return data_server, dates_cosqm
        except:
            print('********error*********', path)


print("COSQM Data cloud correction")
print("puts  = 0 cosqm value if clouds present, detected by variance higher than threshold on timely adjacent measurement values")

def CloudCorr(cosqm_array, threshold, idx, idx_range):
    #check if data is at start or end of cosqm data, and correct range
    if idx<idx_range:
        left = 0
        right = idx_range
    elif idx>len(cosqm_array)-idx_range-1:
        left = idx_range
        right = 0
    else:
        left = int(idx_range/2)
        right = int(idx_range/2)

    #calculate variance of adjacent values
    var_val = np.var(cosqm_array[idx-left:idx+right])

    #put  = 0 if var bigger than threshold (clouds give a variance of approx 0.05mag)
    return 0 if var_val>threshold else cosqm_array[idx]


print("FIND CLOSEST TIME VALUES FROM AOD TO COSQM")

print("Define path from month and day of measurement on AOD")
def FindClosest(dates_aod,index):
    #define paths
    date = dt.fromtimestamp(dates_aod[index])
    date = dict(y = date.year, m = date.month, d = date.day)
    path = r'cosqm_izana/data/%(y)d/%(m)02d/%(y)d-%(m)02d-%(d)02d.txt' % date

    #Download data from this day (night)
    #try:
    data,dates_cosqm = LoadData(path)

    #find nearest time value for cosqm corresponding to aod measurement
    idx = np.abs(dates_cosqm-dates_aod[index]).argmin()
    delta_t = dates_cosqm[idx]-dates_aod[index]

    #except:
    #    delta_t = 1001

    #correct for errors of time matching (range of 1000 difference to trigger)
    if -1000<delta_t<1000:
        cosqm_value1 = data[idx]
        cosqm_value2 = np.copy(cosqm_value1)

        #Cloud correction
        for i in range(5):
            cosqm_value2[i] = CloudCorr(data[:,6+i],0.005,idx,40)
    else:
        cosqm_value2 = np.zeros(15)

    return cosqm_value2, delta_t

#%%---------------------
#################
#################
#
#       COSQM DATA
#   Variation as a function of whole year, day of week, month and season
#
#   Data taken from Martin Aubé's server:
#################
#################

#%% COSQM SANTA CRUZ
from glob import glob

path_santa='/Users/admin/Documents/physique/Maitrise/hiver_2020/cosqm_aod/aod/cosqm_santa_cruz/data/'

#find all paths of files in root directory
paths_santa=sorted(glob(path_santa+"*/*/*.txt"))

files=np.array([LoadData(path) for path in paths_santa])
cosqm_santa=np.concatenate(files[:,0])
dates_santa=np.concatenate(files[:,1])

dt_santa=[dt.fromtimestamp(date) for date in dates_santa]

#%% COSQM TEIDE ***BUG IN COSQM DATA FROM WGET COMMAND***
import pandas as pd

def LoadDataCorrupt(path,cache={}):
    print (path)
    if path in cache:
        return cache[path]
    else:
        try:
            data_server = np.array(pd.read_csv(path,sep=' ',dtype=None, error_bad_lines=False))
            data=data_server[:,2:].astype(float)
            dates_str=data_server[:,0:2].astype(str)
            dates = np.array([ dt.strptime( dates+times, '%Y-%m-%d%H:%M:%S' ).timestamp() for dates, times in dates_str ])
            print('data', np.shape(data), 'dates', np.shape(dates))
            dates1=dates.reshape(len(dates),1)
            output = np.concatenate((data,dates1),axis=1)
            print (np.shape(output))
            cache[path] = (data,dates)
            out=(data,dates)
            output=np.concatenate((out[0],out[1].reshape(np.shape(out[1])[0],1)),axis=1)
            return output
        except:
            print('********error*********', path, '\n')

path_teide='/Users/admin/Documents/physique/Maitrise/hiver_2020/cosqm_aod/aod/cosqm_teide/data/'
paths_teide=sorted(glob(path_teide+"*/*/*.txt"))
files=np.array([LoadDataCorrupt(path) for path in paths_teide])
valid=np.argwhere([type(file) != type(None) for file in files]).astype(int)
files1=files[valid]
cosqm_teide=np.concatenate(files1[:,0])
dates_teide=cosqm_teide[:,-1]
dt_teide=[dt.fromtimestamp(date) for date in dates_teide]

#%% COSQM IZANA AEMET  ***BUG IN COSQM DATA FROM WGET COMMAND***

path_izana='/Users/admin/Documents/physique/Maitrise/hiver_2020/cosqm_aod/aod/cosqm_izana/data/'
paths_izana=sorted(glob(path_izana+"*/*/*.txt"))
files=np.array([LoadData(path) for path in paths_izana])
cosqm_izana=np.concatenate(files[:,0])
dates_izana=np.concatenate(files[:,1])
dt_izana=[dt.fromtimestamp(date) for date in dates_izana]

#%% COSQM IZANA OBSERVATORY

path_obs='/Users/admin/Documents/physique/Maitrise/hiver_2020/cosqm_aod/aod/cosqm_obs/data/'
paths_obs=sorted(glob(path_obs+"*/*/*.txt"))
files=np.array([LoadData(path) for path in paths_obs])
cosqm_obs=np.concatenate(files[:,0])
dates_obs=np.concatenate(files[:,1])
dt_obs=[dt.fromtimestamp(date) for date in dates_obs]


#%% CALIMA observed on night of feburary 23rd 2020
first_day=1
last_day=28

### Calima at santa cruz for each band
calima_mask=(dates_santa>=dt(2020,2,first_day).timestamp())*(dates_santa<=dt(2020,2,last_day).timestamp())
calima_days=[dt.fromtimestamp(calima).date() for calima in dates_santa[calima_mask]]
inc=int(len(calima_days)/(last_day-first_day))
idxs=np.arange(0,len(calima_days),inc)
ticks_labels=calima_days[100:-1:inc]

color=['k','r','g','b','y']
for i in np.arange(5):
    if i ==0:
        plt.plot(np.arange(len(calima_days)), cosqm_santa[calima_mask,5+i],'.',markersize=2,color=color[i],label='Santa Cruz')
    else:
        plt.plot(np.arange(len(calima_days)), cosqm_santa[calima_mask,5+i],'.',markersize=2,color=color[i])
    plt.xlabel('Date')
    plt.ylabel('ZNSB (mag)')
    plt.xticks(idxs, ticks_labels, rotation=70)
    plt.legend()


#%% Calima at each location for clear band

santa_mask=(dates_santa>=dt(2020,2,first_day).timestamp())*(dates_santa<=dt(2020,2,last_day).timestamp())
santa_days=[dt.fromtimestamp(calima).date() for calima in dates_santa[santa_mask]]

izana_mask=(dates_izana>=dt(2020,2,first_day).timestamp())*(dates_izana<=dt(2020,2,last_day).timestamp())
izana_days=[dt.fromtimestamp(calima).date() for calima in dates_izana[izana_mask]]

obs_mask=(dates_obs>=dt(2020,2,first_day).timestamp())*(dates_obs<=dt(2020,2,last_day).timestamp())
obs_days=[dt.fromtimestamp(calima).date() for calima in dates_obs[obs_mask]]

teide_mask=(dates_teide>=dt(2020,2,first_day).timestamp())*(dates_teide<=dt(2020,2,last_day).timestamp())
teide_days=[dt.fromtimestamp(calima).date() for calima in dates_teide[teide_mask]]


inc=int(len(calima_days)/(last_day-first_day))
idxs=np.arange(0,len(calima_days),inc)
ticks_labels=calima_days[100:-1:inc]

plt.plot(np.arange(len(santa_days)), cosqm_santa[santa_mask,5],'.',markersize=5,color='w',label='Santa Cruz')
plt.plot(np.arange(len(izana_days)), cosqm_izana[izana_mask,5],'.',markersize=5,color='r',label='Izana')
plt.plot(np.arange(len(obs_days)), cosqm_obs[obs_mask,5],'.',markersize=5,color='g',label='Observatory (Izana)')
plt.plot(np.arange(len(teide_days)), cosqm_teide[teide_mask,5],'.',markersize=5,color='b',label='Pico del Teide')

plt.xlabel('Date')
plt.ylabel('ZNSB (mag)')
plt.xticks(idxs, ticks_labels, rotation=70)
plt.legend()


#%%--------------------
#
#       AOD Values from AERONET network
#   Values are from level 1.5, meaning there are cloud correction
#   but no atmospheric bands corrections.
#

#%% AOD
print(" SANTA CRUZ")
print("AOD LEVEL 1.5 (no clouds but no calib) data from aeronet for all of year 2019, all bands")
print("Day format changed from 06-11: full night to split at midnight")
print("read from columns 4 to 25 (aod 1640 to 340nm)")

cols = np.arange(4, 26)
path = 'cosqm_santa_cruz/20190101_20191231_Santa_Cruz_Tenerife.lev15'
header = 7
data_aod = np.genfromtxt(path, delimiter = ',', skip_header = header, usecols = cols)
data_aod[data_aod < 0] = 0

print("DATES")
dates_str = np.genfromtxt(path, delimiter = ',', skip_header = header, usecols = [0,1], dtype = str)
dates_aod = np.array([ dt.strptime( dates+times, '%d:%m:%Y%H:%M:%S' ).timestamp() for dates, times in dates_str ])

print("BANDS")
bands_aod = np.genfromtxt(path, delimiter = ',', skip_header = header-1, skip_footer = len(data_aod), usecols = cols, dtype = str)

print("find which bands have no data (take mean of bands and find indices diff. than 0)")
means = np.mean(data_aod, axis = 0)
non_empty_aod = np.array(np.nonzero(means))


#%% Load COSQM from AOD values

print("Load appropriate data from aod to corresponding cosqm values")
start = 5000
stop = 11400

indexes = range(start,stop)
cosqm_value = []
delta_ts = []

for i in indexes:
    data,delta_t = FindClosest(dates_aod,i)
    cosqm_value.append(data)
    delta_ts.append(delta_t)

cosqm_value = np.array(cosqm_value)

#%% Moon correction

print("Correction for moon rise and moon set (first and last values of aod recorded per night (day), going through midnight)")
moonrise_idx = np.zeros(len(dates_aod))


moonrise_idx=[(i if dt.utcfromtimestamp(dates_aod[i+1]).hour-dt.utcfromtimestamp(dates_aod[i]).hour>0
               and dt.utcfromtimestamp(dates_aod[i+1]).hour>15 else 0
               for i in range(len(dates_aod[start:stop])))]
moonrise_idx = np.array(np.nonzero(moonrise_idx)).astype(int)[0]

#%%Correlation plots

plot_start = 0
plot_end = 6400

plt.figure()
plt.plot(data_aod[start:stop,2],'.',markersize = 2)
plt.plot(moonrise_idx,data_aod[moonrise_idx+start,2],'.',color = 'r',markersize = 5)

plt.figure()
plt.plot(cosqm_value[moonrise_idx,5],'.',markersize = 2)


plt.figure()
plt.plot(data_aod[moonrise_idx+start,5],cosqm_value[moonrise_idx,5],'.',markersize = 2,color = 'w')
plt.plot(data_aod[moonrise_idx+start,5],cosqm_value[moonrise_idx,6],'.',markersize = 2,color = 'r')
plt.plot(data_aod[moonrise_idx+start,5],cosqm_value[moonrise_idx,7],'.',markersize = 2,color = 'g')
plt.plot(data_aod[moonrise_idx+start,5],cosqm_value[moonrise_idx,8],'.',markersize = 2,color = 'b')
plt.plot(data_aod[moonrise_idx+start,5],cosqm_value[moonrise_idx,9],'.',markersize = 2,color = 'y')


#%%AOD, TEIDE observatory
print(" TEIDE")
print("AOD LEVEL 1.5 (no clouds but no calib) data from aeronet for all of year 2019, all bands")
print("Day format changed from 06-11: full night to split at midnight")
print("read from columns 4 to 25 (aod 1640 to 340nm)")

cols = np.arange(4, 26)
path = 'cosqm_santa_cruz/20190101_20191231_Santa_Cruz_Tenerife.lev15'
header = 7
data_aod = np.genfromtxt(path, delimiter = ',', skip_header = header, usecols = cols)
data_aod[data_aod < 0] = 0

print("DATES")
dates_str = np.genfromtxt(path, delimiter = ',', skip_header = header, usecols = [0,1], dtype = str)
dates_aod = np.array([ dt.strptime( dates+times, '%d:%m:%Y%H:%M:%S' ).timestamp() for dates, times in dates_str ])

print("BANDS")
bands_aod = np.genfromtxt(path, delimiter = ',', skip_header = header-1, skip_footer = len(data_aod), usecols = cols, dtype = str)

print("find which bands have no data (take mean of bands and find indices diff. than 0)")
means = np.mean(data_aod, axis = 0)
non_empty_aod = np.array(np.nonzero(means))

