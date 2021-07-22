import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime as dt
from datetime import timedelta as td
from datetime import timezone
import pandas as pd
from glob import glob
from skyfield.api import load, Topos, utc, Star
from skyfield.almanac import find_discrete, risings_and_settings
from scipy import signal
from scipy.optimize import curve_fit
from matplotlib.colors import LogNorm
from astropy import units as u
from astropy.coordinates import SkyCoord, EarthLocation
from astropy.time import Time

#%matplotlib

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
            data = pd.read_csv(path, delimiter=' ', names=['Date', 'Time', 'Lat', 'Lon', 'Alt', 'Temp', 'Wait', 'Sqm0', 'Sqm1', 'Sqm2', 'Sqm3', 'Sqm4', 'Sbcals0', 'Sbcals1', 'Sbcals2', 'Sbcals3', 'Sbcals4'])
            data['Datetime'] = pd.to_datetime(data['Date']+'T'+data['Time'], utc=True)        
            cache[path] = data
            return data
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


## Cloud screening function
def Cloudslidingwindow(data_array, window_size, threshold):
    da=np.copy(data_array)
    d = np.lib.stride_tricks.sliding_window_view(data_array, window_shape = window_size, axis=0).copy() #start and end values lost. size of array is data_array.shape[0]-2
    diffs = np.sum(np.abs(np.diff(d, axis=2)), axis=2)/(window_size-1)
    padd = np.full([1, da.shape[1]], np.nan)
    for i in range(window_size//2):
        diffs = np.insert(diffs, 0, padd, axis=0)                     #append nan to start to get same shape as input
        diffs = np.insert(diffs, diffs.shape[0], padd, axis=0)            #append nan to end to get same shape as input
    da[diffs>threshold/(window_size-1)] = np.nan
    return da


## Moon presence data reduction function and variables
ts = load.timescale()
planets = load('de421.bsp')
earth,moon,sun = planets['earth'],planets['moon'],planets['sun']
eph = load('de421.bsp')

## Define variables for locations
izana_loc = earth + Topos('28.300398N', '16.512252W')
santa_loc = earth + Topos('28.472412500N', '16.247361500W')

## !! Time objects from dt need a timezone (aware dt object) to work with skyfield lib
## ex.: time = dt.now(timezone.utc), with an import of datetime.timezone
def RaDecGal(dt_array, earth_location):
    sid_lon = Time(dt_array, location=earth_location).sidereal_time('mean')
    gal_coords = SkyCoord(sid_lon, loc_lat*u.degree, frame='icrs').galactic
    radec_gal = np.array((gal_coords.l.deg, gal_coords.b.deg)).T
    return radec_gal

def ObjectAngle(dt_array, object, location):
    t = ts.utc(dt_array)
    astrometric = location.at(t).observe(object)
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

#######
## Variables
path_cosqm='cosqm_santa/data/'
loc = santa_loc
loc_lat = 28.472412500
loc_lon = -16.247361500
eloc = EarthLocation(lat=loc_lat, lon=loc_lon)
loc_str = 'santa'
slide_threshold = 0.1
slide_window_size = 5
mw_min_angle = 40
moon_min_angle = -2
sun_min_angle = -18
cosqm_bands = np.array([667, 571, 503, 555])
#######


path_aod = 'cosqm_santa/20190601_20210131_Santa_Cruz_Tenerife.lev10'

## find all paths of files in root directory
paths_cosqm = sorted(glob(path_cosqm+"*/*/*.txt"))
files = pd.concat([LoadData(path) for path in paths_cosqm], ignore_index=True)
cosqm_santa = files[['Sqm0', 'Sqm1', 'Sqm2', 'Sqm3', 'Sqm4']].values
dt_santa_raw = files['Datetime']

#remove non datetime errors in cosqm files (NaT)
cosqm_santa = cosqm_santa[~pd.isnull(dt_santa_raw)]
dt_santa = dt_santa_raw[~pd.isnull(dt_santa_raw)]

### if day is wanted: dt_santa_day = dt_santa.dt.day
### if interval wanted: inds = (dt_santa.dt.date == np.array('2020-01-21',dtype='datetime64[D]')) | (dt_santa.dt.date == np.array('2020-01-22',dtype='datetime64[D]'))
### work on specific night interval: inds = (dt_santa.values > np.array('2020-01-21T12',dtype='datetime64[ns]')) 
##   & (dt_santa.values < np.array('2020-01-22T12',dtype='datetime64[ns]'))

## Remove zeros from cosqm measurements (bugs from instruments)
zeros_mask = (cosqm_santa!=0).all(1)
cosqm_santa = cosqm_santa[zeros_mask]
dt_santa = dt_santa[zeros_mask]

## Cloud removal with differential between points (if difference between 2 measurements is bigger than threshold, remove data)

# plt.scatter(dt_santa, cosqm_santa_diff[:,0], s=10, c='k', label='derivative cloud screening')

#milky way filter
print('milky way angles calculation')
mw_angles = RaDecGal(dt_santa, eloc)
mw_mask = mw_angles[:,1]>mw_min_angle
cosqm_santa_mw = cosqm_santa[mw_mask]
dt_santa_mw = dt_santa[mw_mask].reset_index(drop=True)

## Compute moon angles for each timestamp in COSQM data
print('moon angles calculation')
moon_angles = ObjectAngle(dt_santa_mw, moon, loc)
#np.savetxt('cosqm_santa_moon_angles.txt', moon_angles)				#Save angles to reduce ulterior computing time
#moon_angles = np.loadtxt('cosqm_'+loc_str+'_moon_angles.txt')					#Load already computed angles

## Mask values for higher angle than -18deg (astro twilight)

moon_mask = moon_angles<moon_min_angle
cosqm_santa_moon = cosqm_santa_mw[moon_mask]
dt_santa_moon = dt_santa_mw[moon_mask].reset_index(drop=True)
#dates_days_since_start = np.array([(dt.fromtimestamp(date)-dt.fromtimestamp(dates[0])).days+1 for date in dates])

## Compute sun angles for each timestamp in COSQM data
print('sun_angles calculation')
sun_angles = ObjectAngle(dt_santa_moon, sun, santa_loc)
#np.savetxt('cosqm_'+loc_str+'_sun_angles.txt', sun_angles)
#sun_angles = np.loadtxt('cosqm_'+loc_str+'_sun_angles.txt')					#Load already computed angles
sun_mask = sun_angles<sun_min_angle

cosqm_santa_sun_nodiff = cosqm_santa_moon[sun_mask]
dt_santa_sun_nodiff = dt_santa_moon[sun_mask]
dt_santa_sun = dt_santa_moon[sun_mask].reset_index(drop=True)

cosqm_santa_sun = Cloudslidingwindow(cosqm_santa_sun_nodiff, slide_window_size, slide_threshold)


## filter data when milky way is in the visible sky (5 degrees above horizon)
# (todo)

# plot filtering
# cosqm_band = 3
# plt.figure(figsize=[5,3], dpi=100)
# plt.scatter(dt_santa, cosqm_santa[:,cosqm_band], s=10, label='cosqm_santa')
# plt.scatter(dt_santa_mw, cosqm_santa_mw[:,cosqm_band], s=10, alpha=0.5, label='milky way below '+str(mw_min_angle))
# plt.scatter(dt_santa_moon, cosqm_santa_moon[:,cosqm_band], s=8, alpha=0.5, label='moon below '+str(moon_min_angle))
# plt.scatter(dt_santa_sun, cosqm_santa_sun[:,cosqm_band], s=6, label='sun below '+str(sun_min_angle), c='k')
# #plt.scatter(dt_santa_moon, moon_angles[moon_mask]/10+23, s=10, label='moon angle')
# #plt.scatter(dt_santa_moon_sun, sun_angles[sun_mask]/10+23, s=10, label='sun angle')
# plt.legend(loc=[0,0])
# plt.title('ZNSB Santa-Cruz - Filtered clear band')
# plt.xlabel('date')
# plt.ylabel('CoSQM magnitude (mag)')



#hist2d#
#raw without clouds
hours_float_raw = np.array([ date.hour+                    #WATCH OUT FOR TIMEZONE HERE!
    date.minute/60+
    date.second/3600 for date in dt_santa ])
hours_float_raw[hours_float_raw>12]-=24

# plt.figure(figsize=[12,8])
# plt.hist2d(hours_float_raw, cosqm_santa[:,1], 200, cmap='inferno')
# plt.hist2d(hours_float_raw, cosqm_santa[:,1], 200, cmap='inferno', norm=LogNorm())
# plt.ylim(15,21)
# plt.title('ZNSB - no filter - clear')
# plt.xlabel('hour')
# plt.ylabel('CoSQM magnitude')

#clouds filter
hours_float_diff = np.array([ date.hour+                    #WATCH OUT FOR TIMEZONE HERE!
    date.minute/60+
    date.second/3600 for date in dt_santa ])
hours_float_diff[hours_float_diff>12]-=24

#plt.figure(figsize=[12,8])
#plt.hist2d(hours_float_diff[np.isfinite(cosqm_santa_diff)[:,0]], cosqm_santa_diff[:,0][np.isfinite(cosqm_santa_diff)[:,0]], 200, cmap='inferno')
#plt.hist2d(hours_float_diff[np.isfinite(cosqm_santa_diff)[:,0]], cosqm_santa_diff[:,0][np.isfinite(cosqm_santa_diff)[:,0]], 200, cmap='inferno', norm=LogNorm())
#plt.ylim(15,21)
#plt.title('ZNSB - filter: clouds+variance - clear')
#plt.xlabel('hour')
#plt.ylabel('CoSQM magnitude')

#moon filter
hours_float_moon = np.array([ date.hour+                    #WATCH OUT FOR TIMEZONE HERE!
    date.minute/60+
    date.second/3600 for date in dt_santa_moon ])
hours_float_moon[hours_float_moon>12]-=24

# plt.figure(figsize=[12,8])
# plt.hist2d(hours_float_moon[np.isfinite(cosqm_santa_moon)[:,0]], cosqm_santa_moon[:,0][np.isfinite(cosqm_santa_moon)[:,0]], 200, cmap='inferno')
# plt.hist2d(hours_float_moon[np.isfinite(cosqm_santa_moon)[:,0]], cosqm_santa_moon[:,0][np.isfinite(cosqm_santa_moon)[:,0]], 200, cmap='inferno', norm=LogNorm())
# plt.ylim(15,21)
# plt.title('ZNSB - filter: moon - clear')
# plt.xlabel('hour')
# plt.ylabel('CoSQM magnitude')

#sun filter
hours_float_sun = np.array([ date.hour+                    #WATCH OUT FOR TIMEZONE HERE!
    date.minute/60+
    date.second/3600 for date in dt_santa_sun ])
hours_float_sun[hours_float_sun>12]-=24

# plt.figure(figsize=[12,8])
# plt.hist2d(hours_float_sun[np.isfinite(cosqm_santa_sun)[:,1]], cosqm_santa_sun[:,1][np.isfinite(cosqm_santa_sun)[:,1]], 200, cmap='inferno')
# plt.hist2d(hours_float_sun[np.isfinite(cosqm_santa_sun)[:,1]], cosqm_santa_sun[:,1][np.isfinite(cosqm_santa_sun)[:,1]], 200, cmap='inferno', norm=LogNorm())
# plt.ylim(15,21)
# plt.title('ZNSB - filter: sun - clear')
# plt.xlabel('hour')
# plt.ylabel('CoSQM magnitude')



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
dates_color_str = np.array(['2019-06-12', '2019-06-13', '2019-06-26',
                    '2019-07-01', '2019-07-02', '2019-07-21', '2019-07-23', '2019-07-24', '2019-07-25', '2019-07-26', '2019-07-27', '2019-07-29',
                    '2019-08-03', '2019-08-04', '2019-08-07', '2019-08-08', '2019-08-09', '2019-08-10', '2019-08-12', '2019-08-13', '2019-08-18', '2019-08-19', '2019-08-25','2019-08-26', '2019-08-30', '2019-08-31', 
                    '2019-09-06', '2019-09-23', '2019-09-24', '2019-09-25', '2019-09-26', '2019-09-27', '2019-09-28', '2019-09-29', '2019-09-30', 
                    '2019-10-01', '2019-10-02', '2019-10-03', '2019-10-04', '2019-10-05', '2019-10-07', '2019-10-25', '2019-10-26', '2019-10-30', '2019-10-31',  
                    '2019-11-01', '2019-11-22', '2019-11-23', '2019-11-25', '2019-11-26', '2019-11-27', '2019-11-28', '2019-11-29', '2019-11-30', 
                    '2019-12-01', '2019-12-02', '2019-12-04', '2019-12-17', '2019-12-19', '2019-12-20', '2019-12-21', '2019-12-23', '2019-12-24', '2019-12-25', '2019-12-26', '2019-12-28', '2019-12-29', '2019-12-30',
                    '2020-01-02', '2020-01-03', '2020-01-17', '2020-01-18', '2020-01-23', '2020-01-28', 
					'2020-02-12', '2020-02-13', '2020-02-14', '2020-02-17', '2020-02-18', '2020-02-19', '2020-02-20', 
                    '2020-03-13', '2020-03-16', '2020-03-18', '2020-03-20', '2020-03-21', '2020-03-22',
                    '2020-05-20', '2020-05-29', '2020-05-30', 
                    '2020-06-02', '2020-06-11', '2020-06-12', '2020-06-13', '2020-06-14', '2020-06-15', '2020-06-16', '2020-06-17', '2020-06-18', '2020-06-19', '2020-06-20',  '2020-06-26',
                    '2020-07-01', '2020-07-18', '2020-07-19',
                    '2020-08-01', '2020-08-12', '2020-08-13', '2020-08-21', '2020-08-22', '2020-08-27',
                    '2020-12-03', '2020-12-04', '2020-12-07', '2020-12-08', '2020-12-09', '2020-12-10', '2020-12-12', '2020-12-13', '2020-12-15', '2020-12-16', '2020-12-17', '2020-12-18', '2020-12-19', '2020-12-20', '2020-12-21', '2020-12-22', '2020-12-23', '2020-12-24', '2020-12-25',
                    '2021-01-13', '2021-01-14', '2021-01-15', '2021-01-16', '2021-01-17'])

dates_cosqm_str = np.array([dt.strftime(date, '%Y-%m-%d') for date in dt_santa_sun])
dates_color_mask = np.ones(dates_cosqm_str.shape[0], dtype=bool)
dates_color_mask[np.isin(dates_cosqm_str, dates_color_str)] = False

dt_santa_sun = dt_santa_sun[dates_color_mask]
cosqm_santa_sun = cosqm_santa_sun[dates_color_mask]

## Verify clear minus colors (to determine if measurement errors)
# plt.scatter(dt_santa_sun, cosqm_santa_sun[:,0]-cosqm_santa_sun[:,1], c='r', s=10, label='clear-red')
# plt.scatter(dt_santa_sun, cosqm_santa_sun[:,0]-cosqm_santa_sun[:,2], c='g', s=10, label='clear-green')
# plt.scatter(dt_santa_sun, cosqm_santa_sun[:,0]-cosqm_santa_sun[:,3], c='b', s=10, label='clear-blue')
# plt.scatter(dt_santa_sun, cosqm_santa_sun[:,0]-cosqm_santa_sun[:,4], c='y', s=10, label='clear-yellow')
# plt.legend()
# plt.title('ZNSB Santa-Cruz filtered data')
# plt.xlabel('date')
# plt.ylabel('CoSQM magnitude (mag)')


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

## Sigma clip of data
c_norm[c_norm > 1.8] = np.nan
# plt.scatter(d,c_norm[:,0])


#Make plots
d = np.copy(dt_santa_sun)

weekdays_str = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday','Sunday']
months_str = ['January', 'Feburary', 'March', 'April', 'May', 'June', 'July', 'August', 'Septembre', 'October', 'November', 'December']
weekdays = np.array([ date.weekday() for date in d ])
markers = ['.', '1', '2', '3', '4', 'o', 's']


hours_float = np.array([ date.hour+                    #WATCH OUT FOR TIMEZONE HERE!
	date.minute/60+
	date.second/3600 for date in d ])
hours_float[hours_float>12]-=24

bands = ['clear', 'red', 'green', 'yellow', 'blue']

#2d hist
# plt.figure(figsize=[12,8])
# plt.hist2d(hours_float[np.isfinite(c_norm)[:,0]], c_norm[:,0][np.isfinite(c_norm)[:,0]], 200, cmap='inferno')
# plt.hist2d(hours_float[np.isfinite(c_norm)[:,0]], c_norm[:,0][np.isfinite(c_norm)[:,0]], 200, cmap='inferno', norm=LogNorm())
# plt.ylim(-1,0.75)
# plt.title('ZNSB - Normalized 1-2am - clear')
# plt.xlabel('hour')
# plt.ylabel('CoSQM normalized magnitude')

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

# fig, ax = plt.subplots()
# ax.scatter(hours_float,c_norm[:,3], s=15, label='normalized ZNSB')
# ax.plot(xs, second_order(xs, fit_params_b[0], fit_params_b[1], fit_params_b[2]), label='second order fit', c='k')
# ax.legend()
# plt.xlabel('hour from midnight (h)', fontsize=10)
# plt.ylabel('CoSQM Magnitude (mag)', fontsize=10)
# plt.ylim(-0.6,0.5)
# fig.suptitle(f'Normalized ZNSB Santa-Cruz', fontsize=15)
# plt.savefig(f'images/santa/trends/normalized_fitted_BLUE.png')





#hist 2d
plt.figure(dpi=150,figsize=(7,4))
plt.hist2d(hours_float[np.isfinite(c_norm)[:,3]], c_norm[:,3][np.isfinite(c_norm)[:,3]], 80, cmap='inferno')
plt.hist2d(hours_float[np.isfinite(c_norm)[:,3]], c_norm[:,3][np.isfinite(c_norm)[:,3]], 80, cmap='inferno', norm=LogNorm())
plt.plot(xs, second_order(xs, fit_params_g[0], fit_params_g[1], fit_params_g[2]), label='second order fit', c='c')
plt.xlabel('Time from midnight (h)')
plt.ylabel('Normalized ZNSB (mag)')
cbar = plt.colorbar(format='%.0f', label='counts')
cbar.set_ticks(np.arange(1,11,1))
cbar.update_ticks()
cbar.set_ticklabels(np.arange(1,11,1))
plt.savefig('figures/trend/normalized_cosqm_503nm.png')

# Correct filtered data with fit curve
cosqm_santa_2nd = np.array([second_order(hours_float, fit_params_c[0], fit_params_c[1], fit_params_c[2]),
    second_order(hours_float, fit_params_r[0], fit_params_r[1], fit_params_r[2]),
    second_order(hours_float, fit_params_g[0], fit_params_g[1], fit_params_g[2]),
    second_order(hours_float, fit_params_b[0], fit_params_b[1], fit_params_b[2]),
    second_order(hours_float, fit_params_y[0], fit_params_y[1], fit_params_y[2])]).T

dt_santa_final = dt_santa_sun
cosqm_santa_final = np.copy(cosqm_santa_sun) - cosqm_santa_2nd
ddays_cosqm = np.array([(date.date()-d[0].date()).days for date in dt_santa_final])
dt_santa_date = dt_santa_final.dt.date


#Plot effect of corrected trends on specific nights
fig,ax = plt.subplots(dpi=150, figsize=(7,4))
ax.scatter(dt_santa_sun, cosqm_santa_sun[:,3], label='raw CoSQM ZNSB')
ax.scatter(dt_santa_final, cosqm_santa_final[:,3], label='Corrected nightly trend')
ax.set_xlim(pd.Timestamp('2020-05-22 21'), pd.Timestamp('2020-05-23 02'))
ax.set_ylim(19.38, 19.65)
ax.set_ylabel('ZNSB (mag)')
ax.legend()
#ax.set_yscale('log')
fig.savefig('figures/correction_single_night_blue.png')

c_tot = np.array((c_norm[:,1]+np.mean(cosqm_santa_sun[:,1][np.where((hours_float ==1) | (hours_float ==2))]),
c_norm[:,2]+np.mean(cosqm_santa_sun[:,2][np.where((hours_float ==1) | (hours_float ==2))]),
c_norm[:,3]+np.mean(cosqm_santa_sun[:,3][np.where((hours_float ==1) | (hours_float ==2))]),
c_norm[:,4]+np.mean(cosqm_santa_sun[:,4][np.where((hours_float ==1) | (hours_float ==2))]))).T


a = np.array((hours_float, cosqm_santa_2nd))
a.sort(axis=0)

x = np.arange(-2,6, 0.01)

#Plot corrected trend to show red goes darker throughout the night compared to other colors
plt.figure(dpi=150, figsize=(7,4))
plt.plot(x, second_order(x, fit_params_r[0], fit_params_r[1], fit_params_r[2]), c='r', linestyle='solid', linewidth=3, label=f'{cosqm_bands[0]} nm')
plt.plot(x, second_order(x, fit_params_g[0], fit_params_g[1], fit_params_g[2]), c='g', linestyle='dotted', linewidth=3, label=f'{cosqm_bands[1]} nm')
plt.plot(x, second_order(x, fit_params_b[0], fit_params_b[1], fit_params_b[2]), c='b', linestyle='dashed', linewidth=3, label=f'{cosqm_bands[2]} nm')
plt.plot(x, second_order(x, fit_params_y[0], fit_params_y[1], fit_params_y[2]), c='y', linestyle='dashdot', linewidth=3, label=f'{cosqm_bands[3]} nm')
plt.legend()
plt.xlabel('Time from midnight (h)')
plt.ylabel('Normalized ZNSB (mag)')
plt.savefig('figures/red_night_evolution.png')

# #hist 2d
# plt.figure(figsize=[12,8])
# plt.hist2d(hours_float[np.isfinite(cosqm_santa_corr)[:,0]], cosqm_santa_corr[:,0][np.isfinite(cosqm_santa_corr)[:,0]], 80, cmap='inferno')
# #plt.hist2d(hours_float[np.isfinite(cosqm_santa_corr)[:,0]], cosqm_santa_corr[:,0][np.isfinite(cosqm_santa_corr)[:,0]], 80, cmap='inferno', norm=LogNorm())
# plt.xlabel('hour')
# plt.ylabel('CoSQM magnitude')


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

data_aod_raw = pd.read_csv(path_aod, delimiter=',', header=6)
data_angstrom_raw = data_aod_raw['440-675_Angstrom_Exponent']

data_aod_raw['Datetime'] = pd.to_datetime(data_aod_raw['Date(dd:mm:yyyy)']+'T'+data_aod_raw['Time(hh:mm:ss)'], utc=True, format='%d:%m:%YT%H:%M:%S') 
dt_aod = data_aod_raw['Datetime']
dt_aod_date = data_aod_raw['Datetime'].dt.date
aod_bands = data_aod_raw.columns[(data_aod_raw.values[0]!=-999.0)]
#aod_bands = aod_bands[[column[:3] == 'AOD' for column in aod_bands]]
aod_bands = ['AOD_675nm', 'AOD_500nm', 'AOD_500nm', 'AOD_500nm']
data_aod_raw = data_aod_raw[aod_bands].values
mask = np.all(data_aod_raw>0, axis=1)
data_aod = data_aod_raw[mask]
data_angstrom = data_angstrom_raw[mask].values

dt_aod = dt_aod[mask]
dt_aod_date = dt_aod_date[mask]

#BANDS = [AOD_1640nm', 'AOD_1020nm', 'AOD_870nm', 'AOD_675nm', 'AOD_500nm',
#       'AOD_440nm', 'AOD_380nm']

#CoSQM bands matching: 
#R:629->AOD 675 index 3
#G:546->AOD 500 index 4
#B:514->AOD 500 index 4
#Y:562->AOD 500 index 4


# Plot each band of aod measurements for total data
#plt.figure()
#for i in range(aod_bands.shape[0]):
#    plt.scatter(dt_aod, data_aod[:,i], label=aod_bands[i], s=0.2)
#plt.legend()
#plt.show()

# find same days in each instrument
mask_aod = np.isin(dt_aod_date.values, np.unique(dt_santa_date))
dt_aod = dt_aod[mask_aod]
dt_aod_date = dt_aod_date[mask_aod]
data_aod = data_aod[mask_aod]
data_angstrom = data_angstrom[mask_aod]

mask_santa = np.isin(dt_santa_date.values, np.unique(dt_aod.dt.date))
dt_santa_final = dt_santa_final[mask_santa]
dt_santa_date = dt_santa_date[mask_santa]
cosqm_santa_final = cosqm_santa_final[mask_santa]
hours_float = hours_float[mask_santa]

dt_santa_corr = dt_santa_final
cosqm_santa_corr = cosqm_santa_final[:,1:]


#############
# CORRELATION
#############

# Selection of filtered and corrected ZNSB values: all data from noon to 10pm for aod_pm, all data from 3am till noon
cosqm_am = np.zeros((np.unique(dt_santa_date).shape[0], 5))
cosqm_pm = np.zeros((np.unique(dt_santa_date).shape[0], 5))
hours_cosqm = dt_santa_final.dt.hour

cosqm_mean_count = 1
cosqm_min_am_hour = 2
cosqm_max_pm_hour = 25

for i,day in enumerate(np.unique(dt_santa_date)):
#    d_mask_am = np.zeros(ddays_cosqm.shape[0], dtype=bool)
#    d_mask_pm = np.zeros(ddays_cosqm.shape[0], dtype=bool)
    d_mask_am = (dt_santa_date == day) & (hours_cosqm >= cosqm_min_am_hour) & (hours_cosqm < 12)
    d_mask_pm = (dt_santa_date == day) & (hours_cosqm >= 12) & (hours_cosqm < cosqm_max_pm_hour)
    inds_am = np.where(d_mask_am == True)[0]
    if inds_am.shape[0]>cosqm_mean_count:
        inds_am = inds_am[-inds_am.shape[0]:]
        cosqm_am[i] = np.nanmean(cosqm_santa_final[inds_am],axis=0)
    else:
        inds_am[inds_am == True] = False
        cosqm_am[i] = np.nan
    inds_pm = np.where(d_mask_pm == True)[0]
    if inds_pm.shape[0]>cosqm_mean_count:
        inds_pm = inds_pm[:inds_pm.shape[0]]
        cosqm_pm[i] = np.nanmean(cosqm_santa_final[inds_pm],axis=0)
    else:
        inds_pm[inds_pm == True] = False
        cosqm_pm[i] = np.nan
#    cosqm_am[cosqm_am == 0] = np.nan                    # remove zeros values from sensor problem (no data?)
#    cosqm_pm[cosqm_pm == 0] = np.nan                    # remove zeros values from sensor problem (no data?)

plt.figure()
plt.title('ZNSB dusk and dawn values - Blue')
plt.scatter(np.unique(dt_santa_date), cosqm_am[:,3], label='cosqm_am', s=15)
plt.scatter(np.unique(dt_santa_date), cosqm_pm[:,3], label='cosqm_pm', s=15)
plt.xlabel('days from july 2019')
plt.ylabel('CoSQM Magnitudes (mag)')
plt.legend()



# Selection of AOD values: all data from 3pm to midnight for aod_pm, all data from midnight till 10am   
aod_am = np.zeros((np.unique(dt_aod_date).shape[0], 4))
aod_pm = np.zeros((np.unique(dt_aod_date).shape[0], 4))
angstrom_am = np.zeros((np.unique(dt_aod_date).shape[0], 4))
angstrom_pm = np.zeros((np.unique(dt_aod_date).shape[0], 4))

hours_aod = dt_aod.dt.hour

aod_mean_count = 5
aod_max_am_hour = 10
aod_min_pm_hour = 14

for i,day in enumerate(np.unique(dt_aod_date)):
    d_mask_am = np.zeros(dt_aod_date.shape[0], dtype=bool)
    d_mask_pm = np.zeros(dt_aod_date.shape[0], dtype=bool)
    d_mask_am[(dt_aod_date == day) & (hours_aod < aod_max_am_hour)] = True
    d_mask_pm[(dt_aod_date == day) & (hours_aod >= aod_min_pm_hour)] = True
    inds_am = np.where(d_mask_am == True)[0]
    if inds_am.shape[0]>aod_mean_count:
        inds_am = inds_am[:inds_am.shape[0]]
        aod_am[i] =  np.nanmean(data_aod[inds_am], axis=0)
        angstrom_am[i] = np.nanmean(data_angstrom[inds_am], axis=0)
    else:
        inds_am[inds_am == True] = False
        aod_am[i] =  np.nan
        angstrom_am[i] = np.nan

    inds_pm = np.where(d_mask_pm == True)[0]
    if inds_pm.shape[0]>aod_mean_count:
        inds_pm = inds_pm[-inds_pm.shape[0]:]
        aod_pm[i] =  np.nanmean(data_aod[inds_pm],axis=0)
        angstrom_pm[i] = np.nanmean(data_angstrom[inds_pm], axis=0)
    else:
        inds_pm[inds_pm == True] = False
        aod_pm[i] =  np.nan
        angstrom_pm[i] = np.nan


wls1 = np.ones((aod_am.shape[0], 4))*cosqm_bands      # CoSQM center wavelenghts per filter
wls2 = np.ones((aod_am.shape[0], 4))*np.array([675, 500, 500, 500])     # CE318-T sunphoto center wavelenghts per filter

def aod_from_angstrom(aod2, wls1, wls2, alpha):
    return aod2*(wls1/wls2)**-alpha

def angstrom_from_aod(ind1, ind2):
    return -np.log(cosqm_aod_all[:,ind1]/cosqm_aod_all[:,ind2])/np.log(cosqm_bands[ind1]/cosqm_bands[ind2])


aod_am_corr = aod_from_angstrom(aod_am, wls1, wls2, angstrom_am)
aod_pm_corr = aod_from_angstrom(aod_pm, wls1, wls2, angstrom_pm)



plt.figure()
plt.title('ZNSB dusk and dawn values - Blue')
plt.scatter(np.unique(dt_santa_date), aod_am_corr[:,3], label='aod_am_corr', s=15)
plt.scatter(np.unique(dt_santa_date), aod_pm_corr[:,3], label='aod_pm_corr', s=15)
plt.xlabel('days from july 2019')
plt.ylabel('CoSQM Magnitudes (mag)')
plt.legend()

def fit_func(x, a,b):
    return -a*np.log(x/b)

def fit_func1(x, params):
    return fit_func(x, params[0], params[1])


#set limit to constant in second order equation to assure always positive values
param_bounds=([1,1],[30,50])
p0 = [7, 20]

#single fit functions for both dusk and dawn correlated points
cosqm = np.vstack([cosqm_am, cosqm_pm])[:,1:]
cosqm = np.array(cosqm, dtype=np.longdouble)
aod = np.vstack([aod_am, aod_pm])
valid = ~(np.isnan(cosqm) | (np.isnan(aod)))

corr_fitr, _ = curve_fit(fit_func, cosqm[:,0][valid[:,0]], aod[:,0][valid[:,0]], p0=p0, bounds=param_bounds)
corr_fitg, _ = curve_fit(fit_func, cosqm[:,1][valid[:,1]], aod[:,1][valid[:,1]], p0=p0, bounds=param_bounds)
corr_fitb, _ = curve_fit(fit_func, cosqm[:,2][valid[:,2]], aod[:,2][valid[:,2]], p0=p0, bounds=param_bounds)
corr_fity, _ = curve_fit(fit_func, cosqm[:,3][valid[:,3]], aod[:,3][valid[:,3]], p0=p0, bounds=param_bounds)

corr_single_list = [corr_fitr,corr_fitg,corr_fitb,corr_fity]
corr_single_round = [np.around(corr, decimals=2) for corr in corr_single_list]

#fit functions for dusk
valid = ~(np.isnan(cosqm_am[:,1:]) | (np.isnan(aod_am)))
corr_am_fitr, _ = curve_fit(fit_func, cosqm_am[:,1][valid[:,0]], aod_am[:,0][valid[:,0]], p0=p0, bounds=param_bounds)
corr_am_fitg, _ = curve_fit(fit_func, cosqm_am[:,2][valid[:,1]], aod_am[:,1][valid[:,1]], p0=p0, bounds=param_bounds)
corr_am_fitb, _ = curve_fit(fit_func, cosqm_am[:,3][valid[:,2]], aod_am[:,2][valid[:,2]], p0=p0, bounds=param_bounds)
corr_am_fity, _ = curve_fit(fit_func, cosqm_am[:,4][valid[:,3]], aod_am[:,3][valid[:,3]], p0=p0, bounds=param_bounds)


#fit functions for dawn
valid = ~(np.isnan(cosqm_pm[:,1:]) | (np.isnan(aod_pm)))
corr_pm_fitr, _ = curve_fit(fit_func, cosqm_pm[:,1][valid[:,0]], aod_pm[:,0][valid[:,0]], p0=p0, bounds=param_bounds)
corr_pm_fitg, _ = curve_fit(fit_func, cosqm_pm[:,2][valid[:,1]], aod_pm[:,1][valid[:,1]], p0=p0, bounds=param_bounds)
corr_pm_fitb, _ = curve_fit(fit_func, cosqm_pm[:,3][valid[:,2]], aod_pm[:,2][valid[:,2]], p0=p0, bounds=param_bounds)
corr_pm_fity, _ = curve_fit(fit_func, cosqm_pm[:,4][valid[:,3]], aod_pm[:,3][valid[:,3]], p0=p0, bounds=param_bounds)


corr_list = [corr_am_fitr,corr_pm_fitr,corr_am_fitg,corr_pm_fitg,corr_am_fitb,corr_pm_fitb,corr_am_fity,corr_pm_fity]
corr_round = [np.around(corr, decimals=2) for corr in corr_list]

cosqm_aod_r = fit_func1(cosqm_santa_corr[:,0], corr_fitr)
cosqm_aod_g = fit_func1(cosqm_santa_corr[:,1], corr_fitg)
cosqm_aod_b = fit_func1(cosqm_santa_corr[:,2], corr_fitb)
cosqm_aod_y = fit_func1(cosqm_santa_corr[:,3], corr_fity)

cosqm_aod_all = np.array((cosqm_aod_r, cosqm_aod_g, cosqm_aod_b, cosqm_aod_y)).T

#plt.figure()
#plt.title('AOD dusk and dawn values')
#plt.scatter(np.unique(dt_aod_date), aod_am[:,3], label='aeronet_am', s=15)
#plt.scatter(np.unique(dt_aod_date), aod_pm[:,3], label='aeronet_pm', s=15)
#plt.xlabel('days from july 2019')
#plt.ylabel('AOD')
#plt.legend()


#Correlation plots for the 4 color bands

#xs = np.arange(np.nanmin(cosqm_am),np.nanmax(cosqm_pm),0.001)
xs = np.arange(15,24,0.0001)
cosqm_santa_angstrom = angstrom_from_aod(1,2)

c1='#1f77b4'
c2='#ff7f0e'



#single fit function: correlation
fig, ax = plt.subplots(2,2, sharex=True, sharey=True, figsize=(10,6), dpi=150)
ax[0, 0].scatter(cosqm_am[:,1], aod_am[:,0], label='dusk')
ax[0, 0].scatter(cosqm_pm[:,1], aod_pm[:,0], label='dawn')
ax[0, 0].plot(xs, fit_func1(xs, corr_fitr))
ax[0, 0].plot(xs, xs*0, linestyle='--', linewidth=0.5, c='k')
ax[0, 1].scatter(cosqm_am[:,2], aod_am[:,1])
ax[0, 1].scatter(cosqm_pm[:,2], aod_pm[:,1])
ax[0, 1].plot(xs, fit_func1(xs, corr_fitg))
ax[0, 1].plot(xs, xs*0, linestyle='--', linewidth=0.5, c='k')
ax[1, 0].scatter(cosqm_am[:,3], aod_am[:,2])                                
ax[1, 0].scatter(cosqm_pm[:,3], aod_pm[:,2])
ax[1, 0].plot(xs, fit_func1(xs, corr_fitb))
ax[1, 0].plot(xs, xs*0, linestyle='--', linewidth=0.5, c='k')
ax[1, 1].scatter(cosqm_am[:,4], aod_am[:,3])
ax[1, 1].scatter(cosqm_pm[:,4], aod_pm[:,3])
ax[1, 1].plot(xs, fit_func1(xs, corr_fity))
ax[1, 1].plot(xs, xs*0, linestyle='--', linewidth=0.5, c='k')
ax[0, 0].set_yscale('log')
ax[0,0].set_xlim(16.29,21.1)
ax[0,0].set_ylim(0.01,1.75)
fig.text(0.5, 0.04, 'ZNSB (mag)', ha='center')
fig.text(0.04, 0.5, 'AOD', va='center', rotation='vertical')
#fig.text(0.5, 0.9, r'$AOD = -a*log\left(\frac{x}{b}\right)$', ha='center')
ax[0, 0].text(0.05,0.5, f'{cosqm_bands[0]} nm\na,b = {str(corr_single_round[0])[1:-1]}', horizontalalignment='left', verticalalignment='center', transform=ax[0,0].transAxes)
ax[0, 1].text(0.05,0.5, f'{cosqm_bands[1]} nm\na,b = {str(corr_single_round[1])[1:-1]}', horizontalalignment='left', verticalalignment='center', transform=ax[0,1].transAxes)
ax[1, 0].text(0.05,0.5, f'{cosqm_bands[2]} nm\na,b = {str(corr_single_round[2])[1:-1]}', horizontalalignment='left', verticalalignment='center', transform=ax[1,0].transAxes)
ax[1, 1].text(0.05,0.5, f'{cosqm_bands[3]} nm\na,b = {str(corr_single_round[3])[1:-1]}', horizontalalignment='left', verticalalignment='center', transform=ax[1,1].transAxes)
ax[0, 0].set_xlim(16.25,20.6)
ax[0, 0].set_ylim(0.01)
ax[0, 0].legend(loc='lower left', prop={'size': 8})
plt.savefig('figures/correlation/santa/correlation_single_fit_santa.png')


#single fit function: continuity 2020-05-21
msize=0.5
fig, ax = plt.subplots(2,2, sharex=True, sharey=True, dpi=150, figsize=(10,6))
plt.setp(ax, xticks=[pd.Timestamp('2020-05-22'), pd.Timestamp('2020-05-25'), pd.Timestamp('2020-05-28')], xticklabels=['2020-05-22', '2020-05-25', '2020-05-28'])
ax[0,0].scatter(dt_aod, data_aod[:,0], s=msize, label='CE318-T')
ax[0,0].scatter(dt_santa_corr, cosqm_aod_r, s=msize, label='CoSQM derived AOD')
ax[0,1].scatter(dt_aod, data_aod[:,1], s=msize)
ax[0,1].scatter(dt_santa_corr, cosqm_aod_g, s=msize)
ax[1,0].scatter(dt_aod, data_aod[:,2], s=msize)
ax[1,0].scatter(dt_santa_corr, cosqm_aod_b, s=msize)
ax[1,0].tick_params('x', labelrotation=0)
ax[1,1].scatter(dt_aod, data_aod[:,3], s=msize)
ax[1,1].scatter(dt_santa_corr, cosqm_aod_y, s=msize)
ax[1,1].tick_params('x', labelrotation=0)
ax[0, 0].set_yscale('log')
ax[0,0].set_xlim(pd.Timestamp('2020-05-21'), pd.Timestamp('2020-05-29'))
ax[0,0].set_ylim(0.02,0.81)
fig.text(0.04, 0.5, 'Correlated AOD', va='center', rotation='vertical', fontsize=10)
ax[0,0].legend()
ax[0, 0].text(0.5,0.15, f'{cosqm_bands[0]} nm', horizontalalignment='center', verticalalignment='center', transform=ax[0,0].transAxes, fontsize=10)
ax[0, 1].text(0.5,0.15, f'{cosqm_bands[1]} nm', horizontalalignment='center', verticalalignment='center', transform=ax[0,1].transAxes, fontsize=10)
ax[1, 0].text(0.5,0.15, f'{cosqm_bands[2]} nm', horizontalalignment='center', verticalalignment='center', transform=ax[1,0].transAxes, fontsize=10)
ax[1, 1].text(0.5,0.15, f'{cosqm_bands[3]} nm', horizontalalignment='center', verticalalignment='center', transform=ax[1,1].transAxes, fontsize=10)
plt.savefig('figures/continuity/continuity_aod_20200521.png')


#single fit function: continuity Angstrom 2020-05-21
fig, ax = plt.subplots(1,1, dpi=150, figsize=(10,6))
ax.scatter(dt_aod, data_angstrom, s=0.5, label='CE318-T derived 440-679nm AE')
ax.scatter(dt_santa_corr, cosqm_santa_angstrom, s=0.5, label='CoSQM derived 503-571nm AE')
ax.set_xlim(pd.Timestamp('2020-05-21'), pd.Timestamp('2020-05-29'))
ax.set_ylim(-1.34,3.73)
ax.set_ylabel('503-571nm Angstrom exponent')
ax.legend()
#ax.set_yscale('log')
plt.savefig('figures/continuity/continuity_angstrom_20200521.png')



#single fit function: continuity -> variance for uncertainty
fig, ax = plt.subplots(2,2, sharex=True, sharey=True, dpi=150, figsize=(10,6))
ax[0,0].scatter(dt_aod, data_aod[:,0], s=10, label='CE318-T')
ax[0,0].scatter(dt_santa_corr, cosqm_aod_r, s=10, label='CoSQM derived AOD')
ax[0,1].scatter(dt_aod, data_aod[:,1], s=10)
ax[0,1].scatter(dt_santa_corr, cosqm_aod_g, s=10)
ax[1,0].scatter(dt_aod, data_aod[:,2], s=10)
ax[1,0].scatter(dt_santa_corr, cosqm_aod_b, s=10)
ax[1,0].tick_params('x', labelrotation=45)
ax[1,1].scatter(dt_aod, data_aod[:,3], s=10)
ax[1,1].scatter(dt_santa_corr, cosqm_aod_y, s=10)
ax[1,1].tick_params('x', labelrotation=45)
ax[0, 0].set_yscale('log')
ax[0,0].set_xlim(pd.Timestamp('2020-05-23 12'), pd.Timestamp('2020-05-24 10'))
ax[0,0].set_ylim(0.12,0.22)
#fig.text(0.5, 0.04, 'Date', ha='center', fontsize=15)
fig.text(0.04, 0.5, 'Correlated AOD', va='center', rotation='vertical', fontsize=15)
ax[0,0].legend(loc='lower center')
ax[0, 0].text(0.25,0.75, f'{cosqm_bands[0]} nm', horizontalalignment='center', verticalalignment='center', transform=ax[0,0].transAxes, fontsize=15)
ax[0, 1].text(0.25,0.75, f'{cosqm_bands[1]} nm', horizontalalignment='center', verticalalignment='center', transform=ax[0,1].transAxes, fontsize=15)
ax[1, 0].text(0.25,0.75, f'{cosqm_bands[2]} nm', horizontalalignment='center', verticalalignment='center', transform=ax[1,0].transAxes, fontsize=15)
ax[1, 1].text(0.25,0.75, f'{cosqm_bands[3]} nm', horizontalalignment='center', verticalalignment='center', 
transform=ax[1,1].transAxes, fontsize=15)
plt.savefig('figures/continuity/continuity_santa_20200524_variance.png')


#single fit function: continuity Angstrom -> variance uncertainty
fig, ax = plt.subplots(1,1, dpi=150, figsize=(8,5))
ax.scatter(dt_aod, data_angstrom, s=10, label='CE318-T derived 440-679nm AE')
ax.scatter(dt_santa_corr, cosqm_santa_angstrom, s=10, label='CoSQM derived 503-571nm AE')
ax.tick_params('x', labelrotation=45)
#ax.set_xlim(pd.Timestamp('2020-05-23 12'), pd.Timestamp('2020-05-24 10'))
#ax.set_ylim(-0.4,0.5)
#ax.set_yscale('log')
#fig.text(0.5, 0.04, 'Date', ha='center', fontsize=15)
fig.text(0.04, 0.5, 'Angstrom exponent', va='center', rotation='vertical', fontsize=15)
plt.savefig('figures/continuity/continuity_santa_angstrom_variance.png')























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


