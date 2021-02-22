import numpy as np
#from matplotlib.dates import datestr2num
import matplotlib.pyplot as plt
from datetime import datetime as dt
from datetime import timezone
import pandas as pd
from glob import glob
from skyfield.api import load, Topos
from scipy import signal

##############################################################################
# Function definitions
#############################

# COSQM Data load function, returns data and dates in tuple
def LoadData(path,cache={}):
    print (path)
    if path in cache:
        return cache[path]
    else:
        try:
            data_server = np.genfromtxt(path, usecols = list(np.arange(2,17)),invalid_raise=False)
            dates_str = np.genfromtxt(path, delimiter = ' ', usecols = [0,1], dtype = 'str')
            dates_cosqm = np.array([ dt.strptime( dates+times+':+0000', '%Y-%m-%d%H:%M:%S:%z' ).timestamp() for dates, times in dates_str ])
            cache[path] = (data_server, dates_cosqm)
            return data_server, dates_cosqm
        except:
            print('********error*********', path)


# FIND CLOSEST TIME VALUES FROM AOD TO COSQM
# Define path from month and day of measurement on AOD
def FindClosest(dates_aod,index,path_in):
    # define paths
    date = dt.fromtimestamp(dates_aod[index])
    date = dict(y = date.year, m = date.month, d = date.day)
    path = path_in+r'/data/%(y)d/%(m)02d/%(y)d-%(m)02d-%(d)02d.txt' % date

    # Download data from this day (night)
    try:
    	data,dates_cosqm = LoadData(path)


    	# find nearest time value for cosqm corresponding to aod measurement
    	idx = np.abs(dates_cosqm-dates_aod[index]).argmin()
    	delta_t = dates_cosqm[idx]-dates_aod[index]

    except:
        delta_t = 1001

    # correct for errors of time matching (range of 1000 difference to trigger)
    if -1000<delta_t<1000:
        cosqm_value1 = data[idx]
        cosqm_value2 = np.copy(cosqm_value1)

        # Cloud correction
        #for i in range(5):
            #cosqm_value2[i] = CloudCorr(data[:,6+i],0.005,idx,40)
    else:
        cosqm_value2 = np.zeros(15)

    return cosqm_value2, delta_t

# Moon presence data reduction function and variables
ts = load.timescale()
planets = load('de421.bsp')
earth,moon,sun = planets['earth'],planets['moon'],planets['sun']

# Define variables for locations
izana_loc = earth + Topos('28.300398N', '16.512252W')
santa_loc = earth + Topos('28.472412500N', '16.247361500W')

# !! Time objects from dt need a timezone (aware dt object) to work with skyfield lib
# ex.: time = dt.now(timezone.utc), with an import of datetime.timezone
def MoonAngle(timestamp_array, location):
    datetime = np.array([ dt.fromtimestamp(timestamp, timezone.utc) for timestamp in timestamp_array ])
    t = ts.utc(datetime)
    astrometric = location.at(t).observe(moon)
    alt, _, _ = astrometric.apparent().altaz()
    return alt.degrees

def SunAngle(timestamp_array, location):
    datetime = np.array([ dt.fromtimestamp(timestamp, timezone.utc) for timestamp in timestamp_array ])
    t = ts.utc(datetime)
    astrometric = location.at(t).observe(sun)
    alt, _, _ = astrometric.apparent().altaz()
    return alt.degrees

# Cloud presence, analytical approach. Square moving filter of triplets, if higher than threshold, set to nan.
# def CloudTriplets(data_array, kernel_size):
# 	return np.array([np.convolve(col, np.ones(kernel_size)/kernel_size, mode='same') for col in data_array])

# def CloudTriplets(data_array, kernel_size):   ####not working####
# 	kernel = np.ones(kernel_size) / kernel_size
# 	conv_array = np.array([np.convolve(col,kernel, mode='same') / sum(kernel) for col in data_array.T]).T
# 	return  conv_array


def CloudDiff(data_array, threshold):
	diff_array_append = np.array([np.diff(col, append=0) for col in data_array.T]).T
	diff_array_prepend = np.array([np.diff(col, prepend=0) for col in data_array.T]).T
	filtered = data_array
	filtered[np.abs(diff_array_append)>threshold] = np.nan
	filtered[np.abs(diff_array_prepend)>threshold] = np.nan
	return filtered

a = np.copy(cosqm_santa_moon_sun[:,0])
diff = np.diff(a, append=0)
diff2 = np.diff(a, prepend=0)
threshold = 0.0001
a[np.abs(diff)>threshold] = 0
a[np.abs(diff2)>threshold] = 0
dates_a = dates_santa_moon_sun[a!=0]
a = a[a!=0]


##############################################################################
# COSQM DATA
# Variation as a function of whole year, day of week, month and season
# Data taken from Martin Aubé's server: http://dome.obsand.org:2080/DATA/CoSQM-Network/
#
# Data format: Date(YYYY-MM-DD), time(HH:MM:SS),
#             Latitude, Longitude, Elevation, SQM temperature, integration time, 
#             C(mag), R(mag), G(mag), B(mag), Y(mag),
#             C(watt),R(watt),G(watt),B(watt),Y(watt)  
#############################

# COSQM SANTA CRUZ
path_santa='cosqm_santa/data/'

# find all paths of files in root directory
paths_santa=sorted(glob(path_santa+"2019/*/*.txt"))

files=np.array([LoadData(path) for path in paths_santa])
cosqm_santa=np.concatenate(files[:,0])
dates_santa=np.concatenate(files[:,1])
dt_santa=np.array([dt.fromtimestamp(date, tz=timezone.utc) for date in dates_santa])

# Remove zeros from cosqm measurements (bugs from instruments)
zeros_mask = np.ones(cosqm_santa.shape[0], dtype=bool)
for i in range(5,10):
	zeros_mask[np.where(cosqm_santa[:,i]==0)[0]]=False
cosqm_santa = cosqm_santa[zeros_mask]
dates_santa = dates_santa[zeros_mask]
dt_santa = dt_santa[zeros_mask]

# Compute moon angles for each timestamp in COSQM data
# WARNING: LONG CALCULATION TIME! (10+ minutes for 80k points)
moon_angles = MoonAngle(dates_santa, santa_loc)
np.savetxt('cosqm_santa_moon_angles.txt', moon_angles)				#Save angles to reduce ulterior computing time
moon_angles = np.loadtxt('cosqm_santa_moon_angles.txt')					#Load already computed angles

# Mask values for higher angle than -18deg (astro twilight)
moon_min_angle = -18
moon_mask = np.ones(dates_santa.shape[0], bool)
moon_mask[np.where(moon_angles>+moon_min_angle)[0]] = False

dates_santa_moon = dates_santa[moon_mask]
cosqm_santa_moon = cosqm_santa[:,5:10][moon_mask]
dt_santa_moon = dt_santa[moon_mask]
#dates_days_since_start = np.array([(dt.fromtimestamp(date)-dt.fromtimestamp(dates[0])).days+1 for date in dates])

sun_angles = SunAngle(dates_santa_moon, santa_loc)
np.savetxt('cosqm_santa_sun_angles.txt', sun_angles)
sun_angles = np.loadtxt('cosqm_santa_sun_angles.txt')					#Load already computed angles

sun_min_angle = -18
sun_mask = np.ones(dates_santa_moon.shape[0], bool)
sun_mask[np.where(sun_angles>+sun_min_angle)[0]] = False

dates_santa_moon_sun = dates_santa_moon[sun_mask]
cosqm_santa_moon_sun = cosqm_santa_moon[sun_mask]
dt_santa_moon_sun = dt_santa_moon[sun_mask]

plt.scatter(dates_santa, cosqm_santa[:,5], s=50, label='cosqm_santa')
plt.scatter(dates_santa_moon, cosqm_santa_moon[:,0], label='moon below '+str(moon_min_angle),s=50)
plt.scatter(dates_santa_moon_sun, cosqm_santa_moon_sun[:,0], label='sun below '+str(sun_min_angle),s=8)
plt.scatter(dates_santa_moon_sun, sun_angles[sun_mask]/10, label='normalized sun angle')
plt.scatter(dates_santa_moon, moon_angles[moon_mask]/10, label='normalized moon angle')
plt.legend(loc=[0,0])


# Cloud removal with differential between points (if difference between 2 measurements is bigger than threshold, remove data)
cosqm_santa_diff = CloudDiff(cosqm_santa_moon_sun, 0.02)
plt.scatter(dates_santa_moon_sun, cosqm_santa_diff[:,0], s=10, c='k')



#########################
# Cloud removal from visual analysis: thumbnails are checked in a file browser and a folder is created with the clear
# skies data. The filename is of format YYYY-MM-DD_HH/MM/SS.jpg
######
# WARNING: filenames have SLASHES or colons, which can't be read in Microsoft Windows. You must use unix to replace / or : with _ in filenames so that
# the code works in Windows. The following 2 lines must be ran in unix in the folder showing the years of measurements (for string positions):
# ----    fnames = glob('*/*/webcam/*.jpg')
# ----    [os.rename(fname, fname[:28]+fname[29:31]+fname[32:]) for fname in fnames]
######

#santa_dates_noclouds_str = np.array(glob('cosqm_santa/data/*/*/webcam/*.jpg'))			# Load str from noclouds images
#np.savetxt('santa_cruz_noclouds_fnames.txt', santa_dates_noclouds_str, fmt='%s')
santa_dates_noclouds_str = pd.read_csv('santa_cruz_noclouds_fnames.txt', dtype='str').values
santa_noclouds_dates = np.array([ dt.strptime( date[0][-21:-4], '%Y-%m-%d_%H%M%S' ).timestamp() for date in santa_dates_noclouds_str ])		# Convert images time str to timestamps
santa_noclouds_days = np.array([ dt.strptime( date[0][-21:-11], '%Y-%m-%d' ).timestamp() for date in santa_dates_noclouds_str ])			# Convert images time str to timestamp days 

bins = np.arange(santa_noclouds_days.min(),santa_noclouds_days.max(), 24*60*60)		# define complete days for binning
santa_noclouds_days_hist, santa_noclouds_days_bins = np.histogram(santa_noclouds_days, np.arange(santa_noclouds_days.min(),santa_noclouds_days.max(), 24*60*60)) 		# count number of images per day
min_images = 20		# minimum number of non clouded images in a day to be considered
santa_noclouds_days_filtered = santa_noclouds_days_bins[np.argwhere(santa_noclouds_days_hist > min_images)][:,0]			# select only days that have at least min_images non-clouded images

# Mask days that were clouded
santa_days = np.array([ dt.strptime( date.strftime('%Y-%m-%d'), '%Y-%m-%d' ).timestamp() for date in dt_santa_moon_sun ])
cloud_mask = np.isin(santa_days, santa_noclouds_days_filtered)
dates_moon_sun_clouds = dates_moon_sun[cloud_mask]
cosqm_santa_moon_sun_clouds = cosqm_santa_moon_sun[cloud_mask]

# Plot cosqm_data filtered for clouds
plt.scatter(dates_moon_sun, cosqm_santa_moon_sun[:,0], s=1)
plt.scatter(dates_moon_sun_clouds, cosqm_santa_moon_sun_clouds[:,0], s=0.5)

#Apply threshold for mag values
cosqm_santa_moon_sun_clouds_thr = cosqm_santa_moon_sun_clouds
cosqm_santa_moon_sun_clouds_thr[cosqm_santa_moon_sun_clouds_thr<16]=np.nan


################
# Light pollution trends
################

# Per day of week
# dt.datetime.weekday() is: Monday=0, Tuesday=1... Sunday=6
d = dates_moon_sun_clouds
weekdays = np.array([ dt.fromtimestamp(timestamp, timezone.utc).weekday() for timestamp in d ])
hours = np.array([ dt.fromtimestamp(timestamp, timezone.utc).hour for timestamp in d ])
hours[hours>12]-=24
c = cosqm_santa_moon_sun_clouds

weekdays_str = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday','Sunday']
markers = ['.', '1', '2', '3', '4', 'o', 's']

plt.figure(figsize=[16,9])
for i in range(7):
	days_mask = np.ones(d.shape[0],dtype=bool)
	days_mask[np.where(weekdays != i)] = False
	plt.scatter(hours[days_mask],c[days_mask,0], s=30, label=weekdays_str[i], marker=markers[i])

plt.legend()
plt.title('ZNSB Santa-Cruz 2019 - day of week')


# Average per month
d = dates_moon_sun_clouds
months = np.array([ dt.fromtimestamp(timestamp, timezone.utc).month for timestamp in d ])
hours = np.array([ dt.fromtimestamp(timestamp, timezone.utc).hour for timestamp in d ])
hours[hours>12]-=24
c = cosqm_santa_moon_sun_clouds

months_str = ['January', 'Feburary', 'Mars', 'April', 'May', 'June', 'July', 'August', 'Septembre', 'october', 'November', 'December']
markers = ['.', '1', '2', '3', '4', 'o', 's', '*', '+', 'x', 'd', '|']

plt.figure(figsize=[16,9])
for month in np.unique(months):
	months_mask = np.ones(d.shape[0],dtype=bool)
	months_mask[np.where(months != month)] = False
	plt.scatter(hours[months_mask],c[months_mask,0], s=30, label=months_str[month], marker=markers[month])

plt.legend()
plt.title('ZNSB Santa-Cruz 2019 - measurement per hour of night')

plt.figure(figsize=[16,9])
months_avg = np.zeros(12)
for month in np.unique(months):
	months_mask = np.ones(d.shape[0],dtype=bool)
	months_mask[np.where(months != month)] = False
	months_avg[month] = c[months_mask,0].mean()

plt.scatter(np.unique(months),months_avg[np.unique(months)], s=30)
plt.title('ZNSB Santa-Cruz 2019 - average per month')

plt.figure(figsize=[16,9])
months_hours = []
months_avg_hours = []
for month in np.unique(months):
	months_mask = np.ones(d.shape[0],dtype=bool)
	months_mask[np.where(months != month)] = False
	unique_hours = np.unique(hours[months_mask])
	months_hours.append(np.array(unique_hours))
	months_avg_hours.append(np.array([ c[months_mask,0][np.where(hours[months_mask]==hour)].mean() for hour in unique_hours ]))

for i in range(len(months_hours)):
	plt.scatter(months_hours[i], months_avg_hours[i], s=500, marker=markers[i], label='-'+months_str[np.unique(months)[i]])
plt.title('ZNSB Santa-Cruz 2019 - average per month per hour of night')
plt.legend()



###########################
# Load AERONET data from corresponding site
###########################

#COSQM TEIDE ***BUG IN COSQM DATA FROM WGET COMMAND***
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

#COSQM IZANA AEMET
path_izana='/Users/admin/Documents/physique/Maitrise/hiver_2020/cosqm_aod/aod/cosqm_izana/data/'
paths_izana=sorted(glob(path_izana+"*/*/*.txt"))
files=np.array([LoadData(path) for path in paths_izana])
cosqm_izana=np.concatenate(files[:,0])
dates_izana=np.concatenate(files[:,1])
dt_izana=[dt.fromtimestamp(date) for date in dates_izana]

#COSQM IZANA OBSERVATORY
path_obs='/Users/admin/Documents/physique/Maitrise/hiver_2020/cosqm_aod/aod/cosqm_obs/data/'
paths_obs=sorted(glob(path_obs+"*/*/*.txt"))
files=np.array([LoadData(path) for path in paths_obs])
cosqm_obs=np.concatenate(files[:,0])
dates_obs=np.concatenate(files[:,1])
dt_obs=[dt.fromtimestamp(date) for date in dates_obs]


#################
#CALIMA
#################
# #CALIMA observed on night of feburary 23rd 2020
# first_day=1
# last_day=28

# ### Calima at santa cruz for each band
# calima_mask=(dates_santa>=dt(2020,2,first_day).timestamp())*(dates_santa<=dt(2020,2,last_day).timestamp())
# calima_days=[dt.fromtimestamp(calima).date() for calima in dates_santa[calima_mask]]
# inc=int(len(calima_days)/(last_day-first_day))
# idxs=np.arange(0,len(calima_days),inc)
# ticks_labels=calima_days[100:-1:inc]

# color=['k','r','g','b','y']
# for i in np.arange(5):
#     if i ==0:
#         plt.plot(np.arange(len(calima_days)), cosqm_santa[calima_mask,5+i],'.',markersize=2,color=color[i],label='Santa Cruz')
#     else:
#         plt.plot(np.arange(len(calima_days)), cosqm_santa[calima_mask,5+i],'.',markersize=2,color=color[i])
#     plt.xlabel('Date')
#     plt.ylabel('ZNSB (mag)')
#     plt.xticks(idxs, ticks_labels, rotation=70)
#     plt.legend()


# #%% Calima at each location for clear band

# santa_mask=(dates_santa>=dt(2020,2,first_day).timestamp())*(dates_santa<=dt(2020,2,last_day).timestamp())
# santa_days=[dt.fromtimestamp(calima).date() for calima in dates_santa[santa_mask]]

# izana_mask=(dates_izana>=dt(2020,2,first_day).timestamp())*(dates_izana<=dt(2020,2,last_day).timestamp())
# izana_days=[dt.fromtimestamp(calima).date() for calima in dates_izana[izana_mask]]

# obs_mask=(dates_obs>=dt(2020,2,first_day).timestamp())*(dates_obs<=dt(2020,2,last_day).timestamp())
# obs_days=[dt.fromtimestamp(calima).date() for calima in dates_obs[obs_mask]]

# teide_mask=(dates_teide>=dt(2020,2,first_day).timestamp())*(dates_teide<=dt(2020,2,last_day).timestamp())
# teide_days=[dt.fromtimestamp(calima).date() for calima in dates_teide[teide_mask]]


# inc=int(len(calima_days)/(last_day-first_day))
# idxs=np.arange(0,len(calima_days),inc)
# ticks_labels=calima_days[100:-1:inc]

# plt.plot(np.arange(len(santa_days)), cosqm_santa[santa_mask,5],'.',markersize=5,color='w',label='Santa Cruz')
# plt.plot(np.arange(len(izana_days)), cosqm_izana[izana_mask,5],'.',markersize=5,color='r',label='Izana')
# plt.plot(np.arange(len(obs_days)), cosqm_obs[obs_mask,5],'.',markersize=5,color='g',label='Observatory (Izana)')
# plt.plot(np.arange(len(teide_days)), cosqm_teide[teide_mask,5],'.',markersize=5,color='b',label='Pico del Teide')

# plt.xlabel('Date')
# plt.ylabel('ZNSB (mag)')
# plt.xticks(idxs, ticks_labels, rotation=70)
# plt.legend()

##########################################################

##########
# Aerosol phase function from brightness of low moon
##########
# By visual analysis, selection of nights with a full or near full moon. 
# slices = np.arange(16000,18500)
# angles_cut = np.array(angles[slices]).flatten()

# # Create mask
# phase_mask = np.argwhere((angles_cut<30) & (angles_cut>0))
# phase_angles_cut = angles_cut[phase_mask]

# plt.scatter(angles_cut[phase_mask], cosqm_obs[:,5][phase_mask],s=0.1, label='clear',c='k')
# plt.scatter(angles_cut[phase_mask], cosqm_obs[:,6][phase_mask],s=0.1, label='R',c='r')
# plt.scatter(angles_cut[phase_mask], cosqm_obs[:,7][phase_mask],s=0.1, label='G',c='g')
# plt.scatter(angles_cut[phase_mask], cosqm_obs[:,8][phase_mask],s=0.1, label='B',c='b')
# plt.scatter(angles_cut[phase_mask], cosqm_obs[:,9][phase_mask],s=0.1, label='Y',c='y')   
# plt.legend()





#%%--------------------
#
#       AOD Values from AERONET network
#   Values are from level 1.0, meaning there are no corrections
#

# COSQM Day format changed from 06-11: full night to split at midnight, so beware
# read from columns 4 to 25 (aod 1640 to 340nm)

cols = np.arange(4, 26)
path = 'cosqm_santa/20190611_20191231_Santa_Cruz_Tenerife.lev10'
header = 7
data_aod = np.genfromtxt(path, delimiter = ',', skip_header = header, usecols = cols)
data_aod[data_aod < 0] = 0

# DATES
dates_str = np.genfromtxt(path, delimiter = ',', skip_header = header, usecols = [0,1], dtype = str)
dates_aod = np.array([ dt.strptime( dates+times, '%d:%m:%Y%H:%M:%S' ).timestamp() for dates, times in dates_str ])

# BANDS AOD_1640nm', 'AOD_1020nm', 'AOD_870nm', 'AOD_865nm', 'AOD_779nm',
#       'AOD_675nm', 'AOD_667nm', 'AOD_620nm', 'AOD_560nm', 'AOD_555nm',
#       'AOD_551nm', 'AOD_532nm', 'AOD_531nm', 'AOD_510nm', 'AOD_500nm',
#       'AOD_490nm', 'AOD_443nm', 'AOD_440nm', 'AOD_412nm', 'AOD_400nm',
#       'AOD_380nm', 'AOD_340nm'],

bands_aod = np.genfromtxt(path, delimiter = ',', skip_header = header-1, skip_footer = len(data_aod), usecols = cols, dtype = str)

# Find which bands have no data (take mean of bands and find indices diff. than 0)
means = np.mean(data_aod, axis = 0)
non_empty_aod = np.array(np.nonzero(means))
data_aod = data_aod[:,non_empty_aod[0]]



#########
# GRAPHS
#########
# Plot each band of aod measurements for total data
[plt.scatter(dates_aod, data_aod[:,i], label=bands_aod[non_empty_aod[0,i]], s=0.2) for i in range(non_empty_aod[0].shape[0])]
plt.legend()
plt.show()

# Plot continuity between aod day values and cosqm night brightness values
plt.scatter(dates,cosqm_santa[:,0]/cosqm_santa[:,0].max(), s=1)
plt.scatter(dates_aod,data_aod[:,0]/data_aod[:,0].max()*20, s=1)





#%% Load COSQM from AOD values
# Load appropriate data from aod to corresponding cosqm values
start = 0
stop = cosqm_santa.shape[0]
#stop=100
indexes = range(start,stop)
cosqm_value = []
delta_ts = []

for i in indexes:
    data,delta_t = FindClosest(dates_aod,i,'cosqm_santa')
    cosqm_value.append(data)
    delta_ts.append(delta_t)

cosqm_value = np.array(cosqm_value)

#%%Correlation plots

plot_start = 0
plot_end = cosqm_santa.shape[0]

plt.figure()
plt.plot(data_aod[start:stop,2],'.',markersize = 2)
#plt.plot(moonrise_idx,data_aod[moonrise_idx+start,2],'.',color = 'r',markersize = 5)

#plt.figure()
#plt.plot(cosqm_value[moonrise_idx,5],'.',markersize = 2)


plt.figure()
plt.plot(data_aod[:,5],cosqm_value[moonrise_idx,5],'.',markersize = 2,color = 'w')
plt.plot(data_aod[:,5],cosqm_value[moonrise_idx,6],'.',markersize = 2,color = 'r')
plt.plot(data_aod[:,5],cosqm_value[moonrise_idx,7],'.',markersize = 2,color = 'g')
plt.plot(data_aod[:,5],cosqm_value[moonrise_idx,8],'.',markersize = 2,color = 'b')
plt.plot(data_aod[:,5],cosqm_value[moonrise_idx,9],'.',markersize = 2,color = 'y')


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

