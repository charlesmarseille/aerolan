import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime as dt
from datetime import timedelta as td
from datetime import timezone
import pandas as pd
from glob import glob
from skyfield.api import load, Topos, utc
from scipy import signal
from scipy.optimize import curve_fit
from matplotlib.colors import LogNorm

%matplotlib

##############################################################################
## Function definitions
#############################

## COSQM Data load function, returns data and dates in tuple
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


##COSQM TEIDE data load function ***BUG IN COSQM DATA FROM WGET COMMAND***
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
            cache[path] = (data,dates)
            out=(data,dates)
            output=np.concatenate((out[0],out[1].reshape(np.shape(out[1])[0],1)),axis=1)
            return output
        except:
            print('********error*********', path, '\n')

## FIND CLOSEST TIME VALUES FROM AOD TO COSQM
## Define path from month and day of measurement on AOD
def FindClosest(dates_aod,index,path_in):
    date = dt.fromtimestamp(dates_aod[index])
    date = dict(y = date.year, m = date.month, d = date.day)
    path = path_in+r'/data/%(y)d/%(m)02d/%(y)d-%(m)02d-%(d)02d.txt' % date

    ## Download data from this day (night)
    try:
    	data,dates_cosqm = LoadData(path)


    	## find nearest time value for cosqm corresponding to aod measurement
    	idx = np.abs(dates_cosqm-dates_aod[index]).argmin()
    	delta_t = dates_cosqm[idx]-dates_aod[index]

    except:
        delta_t = 1001

    ## correct for errors of time matching (range of 1000 difference to trigger)
    if -1000<delta_t<1000:
        cosqm_value1 = data[idx]
        cosqm_value2 = np.copy(cosqm_value1)

        ## Cloud correction
        #for i in range(5):
            #cosqm_value2[i] = CloudCorr(data[:,6+i],0.005,idx,40)
    else:
        cosqm_value2 = np.zeros(15)

    return cosqm_value2, delta_t

## Cloud screening function
def Cloudslidingwindow(data_array, window_size, threshold):
    da=np.copy(data_array)
    d = np.lib.stride_tricks.sliding_window_view(data_array, window_shape = window_size, axis=0).copy() #start and end values lost. size of array is data_array.shape[0]-2
    diffs = np.sum(np.abs(np.diff(d, axis=2)), axis=2)/(window_size-1)
    padd = np.full([1, da.shape[1]], np.nan)
    for i in range(window_size//2):
        diffs = np.insert(diffs, 0, padd, axis=0)                     #append nan to start to get same shape as input
        diffs = np.insert(diffs, diffs.shape[0], padd, axis=0)            #append nan to end to get same shape as input
    da[diffs>threshold] = np.nan
    return da


## Moon presence data reduction function and variables
ts = load.timescale()
planets = load('de421.bsp')
earth,moon,sun = planets['earth'],planets['moon'],planets['sun']

## Define variables for locations
izana_loc = earth + Topos('28.300398N', '16.512252W')
santa_loc = earth + Topos('28.472412500N', '16.247361500W')

## !! Time objects from dt need a timezone (aware dt object) to work with skyfield lib
## ex.: time = dt.now(timezone.utc), with an import of datetime.timezone
def MoonAngle(dt_array, location):
    t = ts.utc(dt_array)
    print(t.utc_strftime())
    astrometric = location.at(t).observe(moon)
    alt, _, _ = astrometric.apparent().altaz()
    return alt.degrees

def SunAngle(dt_array, location):
    t = ts.utc(dt_array)
    astrometric = location.at(t).observe(sun)
    alt, _, _ = astrometric.apparent().altaz()
    return alt.degrees


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

## COSQM SANTA CRUZ
path_santa='cosqm_santa/data/'
loc = santa_loc
loc_str = 'santa'

## find all paths of files in root directory
paths_santa=sorted(glob(path_santa+"*/*/*.txt"))

files=np.array([LoadData(path) for path in paths_santa])
cosqm_santa=np.concatenate(files[:,0])
dates_santa=np.concatenate(files[:,1])
dt_santa=np.array([dt.fromtimestamp(date, tz=timezone.utc)-td(hours=dt.fromtimestamp(dates_santa[0], tz=timezone.utc).hour) for date in dates_santa])
### FROM HERE, confirm that dt_santa[0] is at midnight. if not, change local time on computer to UTC.

## Remove zeros from cosqm measurements (bugs from instruments)
zeros_mask = np.ones(cosqm_santa.shape[0], dtype=bool)
for i in range(5,10):
	zeros_mask[np.where(cosqm_santa[:,i]==0)[0]]=False
cosqm_santa = cosqm_santa[zeros_mask][:,5:10]
dates_santa = dates_santa[zeros_mask]
dt_santa = dt_santa[zeros_mask]

## Cloud removal with differential between points (if difference between 2 measurements is bigger than threshold, remove data)
cosqm_santa_diff = Cloudslidingwindow(cosqm_santa, 5, 0.05)
plt.scatter(dt_santa, cosqm_santa_diff[:,0], s=10, c='k', label='derivative cloud screening')

## threshold from visual analysis (14mag seems reasonable)
#cosqm_santa_diff[cosqm_santa_diff<16.5] = np.nan

## Compute moon angles for each timestamp in COSQM data
print('moon_angles calculation')
moon_angles = MoonAngle(dt_santa, loc)
np.savetxt('cosqm_santa_moon_angles.txt', moon_angles)				#Save angles to reduce ulterior computing time
#moon_angles = np.loadtxt('cosqm_'+loc_str+'_moon_angles.txt')					#Load already computed angles

## Mask values for higher angle than -18deg (astro twilight)
moon_min_angle = -2
moon_mask = np.ones(dt_santa.shape[0], bool)
moon_mask[np.where(moon_angles>moon_min_angle)[0]] = False

dates_santa_moon = dates_santa[moon_mask]
cosqm_santa_moon = cosqm_santa_diff[moon_mask]
dt_santa_moon = dt_santa[moon_mask]
#dates_days_since_start = np.array([(dt.fromtimestamp(date)-dt.fromtimestamp(dates[0])).days+1 for date in dates])

## Compute sun angles for each timestamp in COSQM data
print('sun_angles calculation')
sun_angles = SunAngle(dt_santa_moon, santa_loc)
np.savetxt('cosqm_'+loc_str+'_sun_angles.txt', sun_angles)
#sun_angles = np.loadtxt('cosqm_'+loc_str+'_sun_angles.txt')					#Load already computed angles

sun_min_angle = -18
sun_mask = np.ones(dt_santa_moon.shape[0], bool)
sun_mask[np.where(sun_angles>sun_min_angle)[0]] = False

dates_santa_sun = dates_santa_moon[sun_mask]
cosqm_santa_sun = cosqm_santa_moon[sun_mask]
dt_santa_sun = dt_santa_moon[sun_mask]


plt.figure(figsize=[16,9])
plt.scatter(dt_santa, cosqm_santa[:,0], s=30, label='cosqm_santa')
plt.scatter(dt_santa_moon, cosqm_santa_moon[:,0], s=30, alpha=0.5, label='moon below '+str(moon_min_angle))
plt.scatter(dt_santa_sun, cosqm_santa_sun[:,0], s=15, label='sun below '+str(sun_min_angle), c='k')
#plt.scatter(dt_santa_moon, moon_angles[moon_mask]/10+23, s=10, label='moon angle')
#plt.scatter(dt_santa_moon_sun, sun_angles[sun_mask]/10+23, s=10, label='sun angle')
plt.legend(loc=[0,0])
plt.title('ZNSB Santa-Cruz - Filtered clear band')
plt.xlabel('date')
plt.ylabel('CoSQM magnitude (mag)')



#hist2d#
#raw without clouds
hours_float_raw = np.array([ dt.fromtimestamp(timestamp).hour+                    #WATCH OUT FOR TIMEZONE HERE!
    dt.fromtimestamp(timestamp).minute/60+
    dt.fromtimestamp(timestamp).second/3600 for timestamp in dates_santa ])
hours_float_raw[hours_float_raw>12]-=24

plt.figure(figsize=[12,8])
plt.hist2d(hours_float_raw, cosqm_santa[:,0], 200, cmap='inferno')
plt.hist2d(hours_float_raw, cosqm_santa[:,0], 200, cmap='inferno', norm=LogNorm())
plt.ylim(15,21)
plt.title('ZNSB - no filter - clear')
plt.xlabel('hour')
plt.ylabel('CoSQM magnitude')

#clouds filter
hours_float_diff = np.array([ dt.fromtimestamp(timestamp).hour+                    #WATCH OUT FOR TIMEZONE HERE!
    dt.fromtimestamp(timestamp).minute/60+
    dt.fromtimestamp(timestamp).second/3600 for timestamp in dates_santa ])
hours_float_diff[hours_float_diff>12]-=24

plt.figure(figsize=[12,8])
plt.hist2d(hours_float_diff[np.isfinite(cosqm_santa_diff)[:,0]], cosqm_santa_diff[:,0][np.isfinite(cosqm_santa_diff)[:,0]], 200, cmap='inferno')
plt.hist2d(hours_float_diff[np.isfinite(cosqm_santa_diff)[:,0]], cosqm_santa_diff[:,0][np.isfinite(cosqm_santa_diff)[:,0]], 200, cmap='inferno', norm=LogNorm())
plt.ylim(15,21)
plt.title('ZNSB - filter: clouds+variance - clear')
plt.xlabel('hour')
plt.ylabel('CoSQM magnitude')

#moon filter
hours_float_moon = np.array([ dt.fromtimestamp(timestamp).hour+                    #WATCH OUT FOR TIMEZONE HERE!
    dt.fromtimestamp(timestamp).minute/60+
    dt.fromtimestamp(timestamp).second/3600 for timestamp in dates_santa_moon ])
hours_float_moon[hours_float_moon>12]-=24

plt.figure(figsize=[12,8])
plt.hist2d(hours_float_moon[np.isfinite(cosqm_santa_moon)[:,0]], cosqm_santa_moon[:,0][np.isfinite(cosqm_santa_moon)[:,0]], 200, cmap='inferno')
plt.hist2d(hours_float_moon[np.isfinite(cosqm_santa_moon)[:,0]], cosqm_santa_moon[:,0][np.isfinite(cosqm_santa_moon)[:,0]], 200, cmap='inferno', norm=LogNorm())
plt.ylim(15,21)
plt.title('ZNSB - filter: moon - clear')
plt.xlabel('hour')
plt.ylabel('CoSQM magnitude')

#sun filter
hours_float_sun = np.array([ dt.fromtimestamp(timestamp).hour+                    #WATCH OUT FOR TIMEZONE HERE!
    dt.fromtimestamp(timestamp).minute/60+
    dt.fromtimestamp(timestamp).second/3600 for timestamp in dates_santa_sun ])
hours_float_sun[hours_float_sun>12]-=24

plt.figure(figsize=[12,8])
plt.hist2d(hours_float_sun[np.isfinite(cosqm_santa_sun)[:,0]], cosqm_santa_sun[:,0][np.isfinite(cosqm_santa_sun)[:,0]], 200, cmap='inferno')
plt.hist2d(hours_float_sun[np.isfinite(cosqm_santa_sun)[:,0]], cosqm_santa_sun[:,0][np.isfinite(cosqm_santa_sun)[:,0]], 200, cmap='inferno', norm=LogNorm())
plt.ylim(15,21)
plt.title('ZNSB - filter: sun - clear')
plt.xlabel('hour')
plt.ylabel('CoSQM magnitude')

#!!!following skipped for initial analysis!!!
#########################
## Cloud removal from visual analysis: thumbnails are checked in a file browser and a folder is created with the clear
## skies data. The filename is of format YYYY-MM-DD_HH/MM/SS.jpg
######
## WARNING: filenames have SLASHES or colons, which can't be read in Microsoft Windows. You must use unix to replace / or : with _ in filenames so that
## the code works in Windows. The following 2 lines must be ran in unix in the folder showing the years of measurements (for string positions):
## ----    fnames = glob('*/*/webcam/*.jpg')
## ----    [os.rename(fname, fname[:28]+fname[29:31]+fname[32:]) for fname in fnames]
######

#santa_dates_noclouds_str = np.array(glob('cosqm_santa/data/*/*/webcam/*.jpg'))			# Load str from noclouds images
#np.savetxt('santa_cruz_noclouds_fnames.txt', santa_dates_noclouds_str, fmt='%s')
# santa_dates_noclouds_str = pd.read_csv('santa_cruz_noclouds_fnames.txt', dtype='str').values
# santa_noclouds_dates = np.array([ dt.strptime( date[0][-21:-4], '%Y-%m-%d_%H%M%S' ).timestamp() for date in santa_dates_noclouds_str ])		# Convert images time str to timestamps
# santa_noclouds_days = np.array([ dt.strptime( date[0][-21:-11], '%Y-%m-%d' ).timestamp() for date in santa_dates_noclouds_str ])			# Convert images time str to timestamp days 

# bins = np.arange(santa_noclouds_days.min(),santa_noclouds_days.max(), 24*60*60)		# define complete days for binning
# santa_noclouds_days_hist, santa_noclouds_days_bins = np.histogram(santa_noclouds_days, np.arange(santa_noclouds_days.min(),santa_noclouds_days.max(), 24*60*60)) 		# count number of images per day
# min_images = 20		# minimum number of non clouded images in a day to be considered
# santa_noclouds_days_filtered = santa_noclouds_days_bins[np.argwhere(santa_noclouds_days_hist > min_images)][:,0]			# select only days that have at least min_images non-clouded images

# ## Mask days that were clouded
# santa_days = np.array([ dt.strptime( date.strftime('%Y-%m-%d'), '%Y-%m-%d' ).timestamp() for date in dt_santa_moon_sun ])
# cloud_mask = np.isin(santa_days, santa_noclouds_days_filtered)
# dates_santa_moon_sun_clouds = dates_santa_moon_sun[cloud_mask]
# dt_santa_moon_sun_clouds = dt_santa_moon_sun[cloud_mask]
# cosqm_santa_moon_sun_clouds = cosqm_santa_moon_sun[cloud_mask]

# ## Plot cosqm_data filtered for clouds
# plt.figure(figsize=[16,9])
# plt.scatter(dt_santa_moon_sun, cosqm_santa_moon_sun[:,0], s=30, c='b', label='moon and sun filtered')
# plt.scatter(dt_santa_moon_sun_clouds, cosqm_santa_moon_sun_clouds[:,0], s=20, c='r', label='cloud filter from pictures')
# plt.legend(loc=(0,0))
# plt.title('ZNSB Santa-Cruz')
# plt.xlabel('date')
# plt.ylabel('CoSQM magnitude (mag)')

################
#skipped section end


## set to nan values that are color higher than clear by visual analysis, followed by clouded nights by visual analysis
dates_color_str = np.array(['2019-07-21', '2019-07-23', '2019-07-25', '2019-07-26', '2019-07-27', '2019-07-29',
                    '2019-08-03', '2019-08-04', '2019-08-07', '2019-08-08', '2019-08-09', '2019-08-10', '2019-08-12', '2019-08-13', '2019-08-19', '2019-08-25','2019-08-26', 
                    '2019-09-23', '2019-09-24', '2019-09-25', '2019-09-26', '2019-09-27', '2019-09-28', '2019-09-29', '2019-09-30', 
                    '2019-10-01', '2019-10-02', '2019-10-03', '2019-10-04', '2019-10-05', '2019-10-07', '2019-10-30', '2019-10-31',  
                    '2019-11-01', '2019-11-26', '2019-11-27', '2019-11-28', '2019-11-29', '2019-11-30', 
                    '2019-12-01', '2019-12-02', '2019-12-04', '2019-12-25',
					'2020-02-23', '2020-02-24', 
                    '2020-05-20', 
                    '2020-08-27',
                    '2021-01-13', '2021-01-14', '2021-01-15', '2021-01-16'])

dates_cosqm_str = np.array([dt.strftime(date, '%Y-%m-%d') for date in dt_santa_sun])
dates_color_mask = np.ones(dates_cosqm_str.shape[0], dtype=bool)
dates_color_mask[np.isin(dates_cosqm_str, dates_color_str)] = False

dates_santa_sun = dates_santa_sun[dates_color_mask]
dt_santa_sun = dt_santa_sun[dates_color_mask]
cosqm_santa_sun = cosqm_santa_sun[dates_color_mask]

## Verify clear minus colors (to determine if measurement errors)
plt.scatter(dt_santa_sun, cosqm_santa_sun[:,0]-cosqm_santa_sun[:,1], c='r', s=10, label='clear-red')
plt.scatter(dt_santa_sun, cosqm_santa_sun[:,0]-cosqm_santa_sun[:,2], c='g', s=10, label='clear-green')
plt.scatter(dt_santa_sun, cosqm_santa_sun[:,0]-cosqm_santa_sun[:,3], c='b', s=10, label='clear-blue')
plt.scatter(dt_santa_sun, cosqm_santa_sun[:,0]-cosqm_santa_sun[:,4], c='y', s=10, label='clear-yellow')
plt.legend()
plt.title('ZNSB Santa-Cruz filtered data')
plt.xlabel('date')
plt.ylabel('CoSQM magnitude (mag)')


################
# Light pollution trends
################

## Every night normalized (substracted magnitude) with mean of 1am to 2am data
d = np.copy(dt_santa_sun)						#Attention, le timezone est en UTC, ce qui peut causer des problemes pour diviser les nuits ailleurs dans le monde
c = np.copy(cosqm_santa_sun)
c_norm = np.copy(c)
ddays_cosqm = np.array([(date.date()-d[0].date()).days for date in d])
hours = np.array([date.hour for date in d])
months = np.array([date.month for date in d])

for day in np.unique(ddays_cosqm):
	d_mask = np.zeros(ddays_cosqm.shape[0], dtype=bool)
	d_mask[(ddays_cosqm == day) & (hours < 12)] = True
	d_mask[(ddays_cosqm == day-1) & (hours > 12)] = True 
	d_mask_mean = d_mask.copy()
	d_mask_mean[hours != 1] = False
#	c_mean = np.ma.array(c, mask=np.isnan(c))[d_mask_mean].mean(axis=0)
	c_mean = np.nanmean(c[d_mask_mean],axis=0)
	c_norm[d_mask] -= c_mean

c_norm[c_norm > 1.8] = np.nan
plt.scatter(d,c_norm[:,0])


#Make plots
d = np.copy(dates_santa_sun)

weekdays_str = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday','Sunday']
months_str = ['January', 'Feburary', 'March', 'April', 'May', 'June', 'July', 'August', 'Septembre', 'October', 'November', 'December']
weekdays = np.array([ dt.fromtimestamp(timestamp, timezone.utc).weekday() for timestamp in d ])
markers = ['.', '1', '2', '3', '4', 'o', 's']


hours_float = np.array([ dt.fromtimestamp(timestamp).hour+                    #WATCH OUT FOR TIMEZONE HERE!
	dt.fromtimestamp(timestamp).minute/60+
	dt.fromtimestamp(timestamp).second/3600 for timestamp in d ])
hours_float[hours_float>12]-=24

bands = ['clear', 'red', 'green', 'yellow', 'blue']


## Normalized ZNSB for all filtered data
for b,band in enumerate(bands):
    fig, ax = plt.subplots()
    ax.scatter(hours_float,c_norm[:,b], s=10, label=band)
    ax.legend()
    plt.xlabel('hour from midnight (h)', fontsize=10)
    plt.ylabel('CoSQM Magnitude (mag)', fontsize=10)
    fig.suptitle(f'Normalized ZNSB Santa-Cruz - {band}', fontsize=15)
    plt.savefig(f'images/santa/trends/normalized_{band}.png')


#2d hist
plt.figure(figsize=[12,8])
plt.hist2d(hours_float[np.isfinite(c_norm)[:,0]], c_norm[:,0][np.isfinite(c_norm)[:,0]], 200, cmap='inferno')
plt.hist2d(hours_float[np.isfinite(c_norm)[:,0]], c_norm[:,0][np.isfinite(c_norm)[:,0]], 200, cmap='inferno', norm=LogNorm())
plt.ylim(-1,0.75)
plt.title('ZNSB - Normalized 1-2am - clear')
plt.xlabel('hour')
plt.ylabel('CoSQM normalized magnitude')

# Fitting of the normalized data to correct for night trend (does not change through year, month or day of week)
# 2nd order function for fit
def second_order(x,a,b,c):
    return a*x**2+b*x+c

fit_params_c, _ = curve_fit(second_order, hours_float[~np.isnan(c_norm[:,0])], c_norm[~np.isnan(c_norm[:,0])][:,0])
fit_params_r, _ = curve_fit(second_order, hours_float[~np.isnan(c_norm[:,1])], c_norm[~np.isnan(c_norm[:,1])][:,1])
fit_params_g, _ = curve_fit(second_order, hours_float[~np.isnan(c_norm[:,2])], c_norm[~np.isnan(c_norm[:,2])][:,2])
fit_params_b, _ = curve_fit(second_order, hours_float[~np.isnan(c_norm[:,3])], c_norm[~np.isnan(c_norm[:,3])][:,3])
fit_params_y, _ = curve_fit(second_order, hours_float[~np.isnan(c_norm[:,4])], c_norm[~np.isnan(c_norm[:,4])][:,4])
fit_params = np.array([fit_params_c, fit_params_r, fit_params_g, fit_params_b, fit_params_y]).T

xs = np.linspace(hours_float.min()-0.5, hours_float.max()+0.5, 1001)

fig, ax = plt.subplots()
ax.scatter(hours_float,c_norm[:,3], s=15, label='normalized ZNSB')
ax.plot(xs, second_order(xs, fit_params_b[0], fit_params_b[1], fit_params_b[2]), label='second order fit', c='k')
ax.legend()
plt.xlabel('hour from midnight (h)', fontsize=10)
plt.ylabel('CoSQM Magnitude (mag)', fontsize=10)
fig.suptitle(f'Normalized ZNSB Santa-Cruz - BLUE', fontsize=15)
plt.savefig(f'images/santa/trends/normalized_fitted_BLUE.png')

#hist 2d
plt.figure(figsize=[12,8])
plt.hist2d(hours_float[np.isfinite(c_norm)[:,0]], c_norm[:,0][np.isfinite(c_norm)[:,0]], 200, cmap='inferno')
plt.hist2d(hours_float[np.isfinite(c_norm)[:,0]], c_norm[:,0][np.isfinite(c_norm)[:,0]], 200, cmap='inferno', norm=LogNorm())
plt.plot(xs, second_order(xs, fit_params_c[0], fit_params_c[1], fit_params_c[2]), label='second order fit', c='c')
plt.ylim(-1,0.75)
plt.title('ZNSB - Normalized 1-2am - clear')
plt.xlabel('hour')
plt.ylabel('CoSQM normalized magnitude')

# Correct filtered data with fit curve
cosqm_santa_2nd = np.array([second_order(hours, fit_params_c[0], fit_params_c[1], fit_params_c[2]),
    second_order(hours_float, fit_params_r[0], fit_params_r[1], fit_params_r[2]),
    second_order(hours_float, fit_params_g[0], fit_params_g[1], fit_params_g[2]),
    second_order(hours_float, fit_params_b[0], fit_params_b[1], fit_params_b[2]),
    second_order(hours_float, fit_params_y[0], fit_params_y[1], fit_params_y[2])]).T

dt_santa_final = np.copy(dt_santa_sun)
cosqm_santa_final = np.copy(cosqm_santa_sun) - cosqm_santa_2nd

for b,band in enumerate(bands):
    fig, ax = plt.subplots()
    ax.scatter(hours_float, cosqm_santa_final[:,b], s=15, label='normalized ZNSB')
    ax.legend()
    plt.xlabel('hour from midnight (h)', fontsize=10)
    plt.ylabel('CoSQM Magnitude (mag)', fontsize=10)
    fig.suptitle(f'Normalized ZNSB Santa-Cruz - {band}', fontsize=15)
    plt.savefig(f'images/santa/trends/final_znsb_data_{band}.png')

#hist 2d
plt.figure(figsize=[12,8])
plt.hist2d(hours_float[np.isfinite(cosqm_santa_final)[:,2]], cosqm_santa_final[:,2][np.isfinite(cosqm_santa_final)[:,2]], 200, cmap='inferno')
plt.hist2d(hours_float[np.isfinite(cosqm_santa_final)[:,2]], cosqm_santa_final[:,2][np.isfinite(cosqm_santa_final)[:,2]], 200, cmap='inferno', norm=LogNorm())
#plt.ylim(-1,0.75)
plt.title('ZNSB - Normalized final - clear')
plt.xlabel('hour')
plt.ylabel('CoSQM normalized magnitude')


##########################
# AOD 
##########################

#%%--------------------
#
#       AOD Values from AERONET network
#   Values are from level 1.0, meaning there are no corrections
#
#######
# read from columns 4 to 25 (aod 1640 to 340nm)

cols = np.arange(4, 26)
path = 'cosqm_santa/20190601_20210131_Santa_Cruz_Tenerife.lev10'
header = 7
data_aod_raw = np.genfromtxt(path, delimiter = ',', skip_header = header, usecols = cols)

# Find which bands have no data (take mean of bands and find indices diff. than 0)
data_aod = data_aod_raw[:,np.where(data_aod_raw[0]>=0)[0]]
data_aod[data_aod < 0] = np.nan

dates_str = np.genfromtxt(path, delimiter = ',', skip_header = header, usecols = [0,1], dtype = str)
dates_aod = np.array([ dt.strptime( dates+times, '%d:%m:%Y%H:%M:%S' ).timestamp() for dates, times in dates_str ])
dt_aod = np.array([ dt.strptime( dates+times, '%d:%m:%Y%H:%M:%S' ) for dates, times in dates_str ])


#BANDS = ['AOD_1640nm', 'AOD_1020nm', 'AOD_870nm', 'AOD_865nm', 'AOD_779nm',
#    'AOD_675nm', 'AOD_667nm', 'AOD_620nm', 'AOD_560nm', 'AOD_555nm',
#    'AOD_551nm', 'AOD_532nm', 'AOD_531nm', 'AOD_510nm', 'AOD_500nm',
#    'AOD_490nm', 'AOD_443nm', 'AOD_440nm', 'AOD_412nm', 'AOD_400nm',
#    'AOD_380nm', 'AOD_340nm']

bands_aod = np.genfromtxt(path, delimiter = ',', skip_header = header-1, skip_footer = len(data_aod_raw), usecols = cols, dtype = str)[np.where(data_aod_raw[0]>=0)[0]]
#non empty bands:['AOD_1640nm', 'AOD_1020nm', 'AOD_870nm', 'AOD_675nm', 'AOD_500nm',
#       'AOD_440nm', 'AOD_380nm']


# Plot each band of aod measurements for total data
for i in range(bands_aod.shape[0]):
    plt.scatter(dt_aod, data_aod[:,i], label=bands_aod[i], s=0.2)
plt.legend()
plt.show()


#############
# CORRELATION
#############


# Selection of filtered and corrected ZNSB values: all data from noon to 10pm for aod_pm, all data from 3am till noon
cosqm_am = np.zeros((np.unique(ddays_cosqm).shape[0], 5))
cosqm_pm = np.zeros((np.unique(ddays_cosqm).shape[0], 5))


for i,day in enumerate(np.unique(ddays_cosqm)):
    d_mask_am = np.zeros(ddays_cosqm.shape[0], dtype=bool)
    d_mask_pm = np.zeros(ddays_cosqm.shape[0], dtype=bool)
    d_mask_am[(ddays_cosqm == day) & (hours >= 5) & (hours <= 10)] = True
    d_mask_pm[(ddays_cosqm == day) & (hours >= 16) & (hours <= 21)] = True
    inds_am = np.where(d_mask_am == True)[0]
    inds_pm = np.where(d_mask_pm == True)[0]
    cosqm_am[i] = np.nanmean(cosqm_santa_final[inds_am],axis=0)
    cosqm_pm[i] = np.nanmean(cosqm_santa_final[inds_pm],axis=0)
    cosqm_am[cosqm_am == 0] = np.nan                    # remove zeros values from sensor problem (no data?)
    cosqm_pm[cosqm_pm == 0] = np.nan                    # remove zeros values from sensor problem (no data?)


# Selection of AOD values: all data from 3pm to midnight for aod_pm, all data from midnight till 10am   
aod = np.copy(data_aod)
ddays_aod = np.array([(date.date()-dt_aod[0].date()).days for date in dt_aod])       #Attention, le timezone est en UTC, ce qui peut causer des problemes pour diviser les nuits ailleurs dans le monde
hours_aod = np.array([date.hour for date in dt_aod])
aod_am = np.zeros((np.unique(ddays).shape[0], aod.shape[1]))
aod_pm = np.zeros((np.unique(ddays).shape[0], aod.shape[1]))
hours_aodfloat = np.array([ time.hour + time.minute/60 + time.second/3600 for time in dt_aod])
hours_float[hours_float>12]-=24

for i,day in enumerate(np.unique(ddays_aod)):
    d_mask_am = np.zeros(ddays_aod.shape[0], dtype=bool)
    d_mask_pm = np.zeros(ddays_aod.shape[0], dtype=bool)
    d_mask_am[(ddays_aod == day) & (hours_aod <= 7)] = True
    d_mask_pm[(ddays_aod == day) & (hours_aod >= 18)] = True
    inds_am = np.where(d_mask_am == True)[0]
    inds_pm = np.where(d_mask_pm == True)[0]
    aod_am[i] =  np.nanmean(aod[inds_am],axis=0)
    aod_pm[i] =  np.nanmean(aod[inds_pm],axis=0)
    aod_am[aod_am <= 0] = np.nan                    # remove zeros values from sensor problem (no data?)
    aod_pm[aod_pm <= 0] = np.nan                    # remove zeros values from sensor problem (no data?)


# Find days in aod from cosqm
same_days_aod_inds = np.isin(np.unique(ddays_aod), np.unique(ddays_cosqm))
same_days_aod = np.unique(ddays_aod)[same_days_aod_inds]
same_days_cosqm_inds = np.isin(np.unique(ddays_cosqm), np.unique(ddays_aod))

plt.figure()
plt.title('ZNSB dusk and dawn values - Blue')
plt.scatter(np.unique(ddays_cosqm), cosqm_am[:,3], label='cosqm_am', s=15)
plt.scatter(np.unique(ddays_cosqm), cosqm_pm[:,3], label='cosqm_pm', s=15)
plt.xlabel('days from july 2019')
plt.ylabel('CoSQM Magnitudes (mag)')
plt.legend()

plt.figure()
plt.title('AOD dusk and dawn values')
plt.scatter(np.unique(ddays_aod), aod_am[:,3], label='aeronet_am', s=15)
plt.scatter(np.unique(ddays_aod), aod_pm[:,3], label='aeronet_pm', s=15)
plt.xlabel('days from july 2019')
plt.ylabel('AOD')
plt.legend()


#Correlation plots for the 4 color bands (R-629nm, G-546nm, B-514nm, Y-562nm)
plt.figure()
plt.scatter(aod_am[same_days_aod_inds][:,1], cosqm_am[same_days_cosqm_inds, 1], label='RED - AM')
plt.scatter(aod_pm[same_days_aod_inds][:,1], cosqm_pm[same_days_cosqm_inds, 1], label='RED - PM')
plt.title('AOD band - 675nm')
plt.legend()

plt.figure()
plt.scatter(aod_am[same_days_aod_inds][:,1], cosqm_am[same_days_cosqm_inds, 2], label='GREEN - AM')
plt.scatter(aod_pm[same_days_aod_inds][:,1], cosqm_pm[same_days_cosqm_inds, 2], label='GREEN - PM')
plt.title('AOD band - 675nm')
plt.legend()

plt.figure()
plt.scatter(aod_am[same_days_aod_inds][:,3], cosqm_am[same_days_cosqm_inds, 3], label='BLUE - AM')
plt.scatter(aod_pm[same_days_aod_inds][:,3], cosqm_pm[same_days_cosqm_inds, 3], label='BLUE - PM')
plt.title('AOD band - 675nm')
plt.legend()

plt.figure()
plt.scatter(aod_am[same_days_aod_inds][:,3], cosqm_am[same_days_cosqm_inds, 4], label='YELLOW - AM')
plt.scatter(aod_pm[same_days_aod_inds][:,3], cosqm_pm[same_days_cosqm_inds, 4], label='YELLOW - PM')
plt.title('AOD band - 675nm')
plt.legend()

####
plt.figure()
plt.scatter(aod_am[same_days_aod_inds][:,1], cosqm_am[same_days_cosqm_inds, 1], label='RED - AM')
plt.scatter(aod_pm[same_days_aod_inds][:,1], cosqm_pm[same_days_cosqm_inds, 1], label='RED - PM')
plt.title('AOD band - 500nm')
plt.legend()

plt.figure()
plt.scatter(aod_am[same_days_aod_inds][:,1], cosqm_am[same_days_cosqm_inds, 2], label='GREEN - AM')
plt.scatter(aod_pm[same_days_aod_inds][:,1], cosqm_pm[same_days_cosqm_inds, 2], label='GREEN - PM')
plt.title('AOD band - 500nm')
plt.legend()

plt.figure()
plt.scatter(aod_am[same_days_aod_inds][:,1], cosqm_am[same_days_cosqm_inds, 3], label='BLUE - AM')
plt.scatter(aod_pm[same_days_aod_inds][:,1], cosqm_pm[same_days_cosqm_inds, 3], label='BLUE - PM')
plt.title('AOD band - 500nm')
plt.legend()

plt.figure()
plt.scatter(aod_am[same_days_aod_inds][:,1], cosqm_am[same_days_cosqm_inds, 4], label='YELLOW - AM')
plt.scatter(aod_pm[same_days_aod_inds][:,1], cosqm_pm[same_days_cosqm_inds, 4], label='YELLOW - PM')
plt.title('AOD band - 500nm')
plt.legend()






##########################
##########################
########################## Not so important graphs

## Normalized ZNSB for each day of week in selected months to see for trends of days
for b,band in enumerate(bands):
    fig,axs = plt.subplots(3,4,sharex=True,sharey=True, figsize=(20,12))
    for month in range(1,13):
        for i in range(7):
            months_mask = np.ones(d.shape[0],dtype=bool)
            months_mask[np.where(months != month)] = False
            months_mask[np.where(weekdays != i)] = False
            axs[(month-1)//4,(month-1)%4].scatter(hours[months_mask],c_norm[months_mask,b], s=30, label=weekdays_str[i], marker=markers[i], alpha=0.6)
        #plt.xlim(-4.5,6.5)
        #plt.ylim(18,19.25)
        if month == 1:
            axs[(month-1)//4,(month-1)%4].legend(loc="upper left")
        axs[(month-1)//4,(month-1)%4].set_title(months_str[month-1])
    fig.suptitle(f'Normalized ZNSB Santa-Cruz - day of week, {band}', fontsize=24)

    fig.add_subplot(111,frame_on=False)
    plt.tick_params(labelcolor="none", bottom=False, left=False)
    plt.xlabel('hour from midnight (h)', fontsize=18)
    plt.ylabel('CoSQM Magnitude (mag)', fontsize=18)
    plt.tight_layout()
    plt.savefig(f'images/santa/trends/nomalized_months_and_days_trends_{band}.png')



## Normalized ZNSB for each day of week
for b,band in enumerate(bands):
    fig, ax = plt.subplots()
    for i in range(7):
        days_mask = weekdays == i
        ax.scatter(hours[days_mask],c_norm[days_mask,b], s=30, label=weekdays_str[i], marker=markers[i], alpha=0.6)
    
    ax.legend()
    plt.xlabel('hour from midnight (h)', fontsize=10)
    plt.ylabel('CoSQM Magnitude (mag)', fontsize=10)
    fig.suptitle(f'Normalized ZNSB Santa-Cruz - day of week, {band}', fontsize=15)
    plt.savefig(f'images/santa/trends/nomalized_weekdays_trends_{band}.png')


## Normalized ZNSB for each month
markers = ['.', '1', '2', '3', '4', 'o', 's', 'p', 'P', '*', 'D', 5]
for b,band in enumerate(bands):
    fig, ax = plt.subplots()
    for i in np.unique(months):
        months_mask = months == i
        ax.scatter(hours[months_mask],c_norm[months_mask,b], s=30, label=months_str[i-1], marker=markers[i-1], alpha=0.6)
    
    ax.legend()
    plt.xlabel('hour from midnight (h)', fontsize=10)
    plt.ylabel('CoSQM Magnitude (mag)', fontsize=10)
    fig.suptitle(f'Normalized ZNSB Santa-Cruz - month, {band}', fontsize=15)
    plt.savefig(f'images/santa/trends/nomalized_months_trends_{band}.png')


# Per day of week
# dt.datetime.weekday() is: Monday=0, Tuesday=1... Sunday=6
d = np.copy(dates_santa_moon_sun_clouds)
weekdays = np.array([ dt.fromtimestamp(timestamp, timezone.utc).weekday() for timestamp in d ])
hours = np.array([ dt.fromtimestamp(timestamp, timezone.utc).hour+
	dt.fromtimestamp(timestamp, timezone.utc).minute/60+
	dt.fromtimestamp(timestamp, timezone.utc).second/3600 for timestamp in d ])
hours[hours>12]-=24
c = np.copy(cosqm_santa_moon_sun_diff_clouds)

weekdays_str = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday','Sunday']
markers = ['.', '1', '2', '3', '4', 'o', 's']

plt.figure(figsize=[16,9])
for i in range(7):
	days_mask = np.ones(d.shape[0],dtype=bool)
	days_mask[np.where(weekdays != i)] = False
	plt.scatter(hours[days_mask],c[days_mask,0], s=30, label=weekdays_str[i], marker=markers[i], alpha=0.5)

plt.legend()
plt.title('ZNSB Santa-Cruz - day of week')
plt.xlabel('hour from midnight (h)')
plt.ylabel('CoSQM Magnitude (mag)')


# Average per month
markers = ['.', '1', '2', '3', '4', 'o', 's', '*', '+', 'x', 'd', '|']

plt.figure(figsize=[16,9])
for month in np.unique(months):
	months_mask = np.ones(d.shape[0],dtype=bool)
	months_mask[np.where(months != month)] = False
	plt.scatter(hours[months_mask],c[months_mask,0], s=30, label=months_str[month-1], marker=markers[month-1])

plt.legend()
plt.title('ZNSB Santa-Cruz - Clear band per hour of night')
plt.xlabel('hour from midnight (h)')
plt.ylabel('CoSQM Magnitude (mag)')


## ZNSB per month per hour of night
fig,axs = plt.subplots(6)
months_str = ['January', 'Feburary', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'october', 'November', 'December']
markers = ['.', '1', '2', '3', '4', 'o', 's', '*', '+', 'x', 'd', '|']

for month in np.unique(months)[:-6]:
	months_mask = np.ones(d.shape[0],dtype=bool)
	months_mask[np.where(months != month)] = False 
	axs[month-1].scatter(hours[months_mask],c_norm[months_mask,0], s=30, label=months_str[month-1], marker=markers[month-1])
	axs[month-1].set_ylabel(months_str[month-1])

plt.suptitle('ZNSB Santa-Cruz - Clear band per hour of night')

fig,axs = plt.subplots(6)
for month in np.unique(months)[5:]:
	months_mask = np.ones(d.shape[0],dtype=bool)
	months_mask[np.where(months != month)] = False
	axs[month-7].scatter(hours[months_mask],c_norm[months_mask,0], s=30, label=months_str[month-1], marker=markers[month-1])
	axs[month-7].set_ylabel(months_str[month-1])

plt.legend()
plt.suptitle('ZNSB Santa-Cruz - Clear band per hour of night')
plt.xlabel('hour from midnight (h)')
plt.ylabel('CoSQM Magnitude (mag)')



## ZNSB average per month
plt.figure(figsize=[16,9])
months_avg = np.zeros((12,5))
for month in np.unique(months):
	months_mask = np.ones(d.shape[0],dtype=bool)
	months_mask[np.where(months != month)] = False
	for i in range (5):
		months_avg[month-1,i] = c[months_mask,i][~np.isnan(c[months_mask,i])].mean()

months_avg[months_avg==0] = np.nan


colors = ['k','r','g','b','y']
bands = ['clear', 'red', 'green', 'blue', 'yellow']
for i in range (5):
	plt.scatter(months_str,months_avg[np.arange(12),i], s=30, c=colors[i], label=bands[i])
	plt.plot(months_str,months_avg[np.arange(12),i], linewidth=1, c=colors[i])

plt.title('ZNSB Santa-Cruz - average per month (06-2019 to 02-2021)')
plt.xlabel('month of year')
plt.ylabel('CoSQM Magnitude (mag)')
plt.legend()


hours = np.array([ dt.fromtimestamp(timestamp, timezone.utc).hour for timestamp in d ])
hours[hours>12]-=24
c = np.array(cosqm_santa_moon_sun_diff_clouds)


plt.figure(figsize=[16,9])
months_hours = []
months_avg_hours = []
for month in np.unique(months):
	months_mask = np.ones(d.shape[0],dtype=bool)
	months_mask[np.where(months != month)] = False
	unique_hours = np.unique(hours[months_mask])
	months_hours.append(np.array(unique_hours))
	months_avg_hours.append(np.array([ c[months_mask,0][np.where(hours[months_mask]==hour)][~np.isnan(c[months_mask,0][np.where(hours[months_mask]==hour)])].mean() for hour in unique_hours ]))


for i in range(len(months_hours)):
	plt.scatter(months_hours[i], months_avg_hours[i], s=100, marker=markers[i], label=months_str[np.unique(months)[i]-1])
	plt.plot(months_hours[i], months_avg_hours[i], linewidth=1)
plt.title('ZNSB Santa-Cruz 2019 - average per month per hour of night')
plt.xlabel('hour from midnight (h)')
plt.ylabel('CoSQM Magnitude (mag)')
plt.legend()


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
path = 'cosqm_santa/20190601_20210131_Santa_Cruz_Tenerife.lev10'
header = 7
data_aod = np.genfromtxt(path, delimiter = ',', skip_header = header, usecols = cols)
data_aod[data_aod < 0] = 0

# DATES
dates_str = np.genfromtxt(path, delimiter = ',', skip_header = header, usecols = [0,1], dtype = str)
dates_aod = np.array([ dt.strptime( dates+times, '%d:%m:%Y%H:%M:%S' ).timestamp() for dates, times in dates_str ])
dt_aod = np.array([ dt.strptime( dates+times, '%d:%m:%Y%H:%M:%S' ) for dates, times in dates_str ])


BANDS = ['AOD_1640nm', 'AOD_1020nm', 'AOD_870nm', 'AOD_865nm', 'AOD_779nm',
    'AOD_675nm', 'AOD_667nm', 'AOD_620nm', 'AOD_560nm', 'AOD_555nm',
    'AOD_551nm', 'AOD_532nm', 'AOD_531nm', 'AOD_510nm', 'AOD_500nm',
    'AOD_490nm', 'AOD_443nm', 'AOD_440nm', 'AOD_412nm', 'AOD_400nm',
    'AOD_380nm', 'AOD_340nm']

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

