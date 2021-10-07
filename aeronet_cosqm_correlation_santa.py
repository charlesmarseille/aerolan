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
import pytz
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)


#%matplotlib

##############################################################################
## Function definitions
#############################

## COSQM Data load function, returns data and dates in tuple
def LoadData(path,cache={}):
	#print (path)
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
	dd = np.lib.stride_tricks.sliding_window_view(data_array, window_shape = window_size, axis=0).copy() #start and end values lost. size of array is data_array.shape[0]-2
	diffs = np.sum(np.abs(np.diff(dd, axis=2)), axis=2)/(window_size-1)
	padd = np.full([1, da.shape[1]], np.nan)
	for i in range(window_size//2):
		diffs = np.insert(diffs, 0, padd, axis=0)                     #append nan to start to get same shape as input
		diffs = np.insert(diffs, diffs.shape[0], padd, axis=0)            #append nan to end to get same shape as input
	da[diffs[:,0]>threshold/(window_size-1)] = np.nan
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
# 
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
path_aod = 'cosqm_santa/20190601_20210131_Santa_Cruz_Tenerife.lev10'
cosqm_bands = np.array([652, 599, 532, 588])
bands = ['clear', 'red', 'green', 'blue', 'yellow']

loc = santa_loc
loc_lat = 28.472412500
loc_lon = -16.247361500
eloc = EarthLocation(lat=loc_lat, lon=loc_lon)
loc_str = 'santa'
slide_threshold = 0.1
slide_window_size = 5
mw_min_angle = 30
moon_min_angle = -2
sun_min_angle = -18
cosqm_window_size = 7

cosqm_mean_count = 1        #number of averages on cosqm data for correlation
cosqm_min_am_hour = 2       #start of dusk cosqm measurements for correlation
cosqm_max_pm_hour = 25      #end of dawn cosqm measurements for correlation
aod_mean_count = 5          #number of averages on aeronet aod data for correlation
aod_max_am_hour = 10        #end of dusk aeronet aod data for correlation
aod_min_pm_hour = 14        #start of dawn aeronet aod data for correlation

normalized_cosqm_znsb_threshold = 1.8
cosqm_aod_threshold = 0.04
ae_min = 0
ae_max = 2

#######


## find all paths of files in root directory
print('Load cosqm data')
paths_cosqm = sorted(glob(path_cosqm+"*/*/*.txt"))
files = pd.concat([LoadData(path) for path in paths_cosqm], ignore_index=True)
cosqm_santa = files[['Sqm0', 'Sqm1', 'Sqm2', 'Sqm3', 'Sqm4']].values
dt_santa_raw = files['Datetime']

#remove non datetime errors in cosqm files (NaT)
cosqm_santa = cosqm_santa[~pd.isnull(dt_santa_raw)]
dt_santa = dt_santa_raw[~pd.isnull(dt_santa_raw)]

## Remove zeros from cosqm measurements (bugs from instruments)
print('Cleaning: remove zeros from cosqm measurements')
zeros_mask = (cosqm_santa!=0).all(1)
cosqm_santa = cosqm_santa[zeros_mask]
dt_santa = dt_santa[zeros_mask]

#Clouds sliding window filter
print('Filter: clouds sliding window filter')
cosqm_santa_cloud = Cloudslidingwindow(cosqm_santa, slide_window_size, slide_threshold)
dt_santa_cloud = dt_santa[~np.isnan(cosqm_santa_cloud[:,0])]
cosqm_santa_cloud = cosqm_santa_cloud[~np.isnan(cosqm_santa_cloud[:,0])]

#milky way filter
print('Filter: milky way angles calculation')
mw_angles = RaDecGal(dt_santa_cloud, eloc)
mw_mask = mw_angles[:,1]>mw_min_angle
cosqm_santa_mw = cosqm_santa_cloud[mw_mask]
dt_santa_mw = dt_santa_cloud[mw_mask].reset_index(drop=True)

## Compute moon angles for each timestamp in COSQM data
print('Filter: moon angles calculation')
moon_angles = ObjectAngle(dt_santa_mw, moon, loc)
moon_mask = moon_angles<moon_min_angle
cosqm_santa_moon = cosqm_santa_mw[moon_mask]
dt_santa_moon = dt_santa_mw[moon_mask].reset_index(drop=True)

## Compute sun angles for each timestamp in COSQM data
print('Filter: sun_angles calculation')
sun_angles = ObjectAngle(dt_santa_moon, sun, santa_loc)
sun_mask = sun_angles<sun_min_angle
cosqm_santa_sun = cosqm_santa_moon[sun_mask]
dt_santa_sun = dt_santa_moon[sun_mask].reset_index(drop=True)

# plot filtering
band = 3
plt.figure(figsize=[7,4], dpi=150)
plt.scatter(dt_santa, cosqm_santa[:,band], s=10, label='cosqm_santa')
plt.scatter(dt_santa_mw, cosqm_santa_mw[:,band], s=10, alpha=0.5, label='milky way below '+str(mw_min_angle))
plt.scatter(dt_santa_moon, cosqm_santa_moon[:,band], s=8, alpha=0.5, label='moon below '+str(moon_min_angle))
plt.scatter(dt_santa_sun, cosqm_santa_sun[:,band], s=6, label='sun below '+str(sun_min_angle), c='k')
plt.legend(loc=[0,0])
plt.ylabel(f'CoSQM {cosqm_bands[band-1]}nm magnitude (mag)')


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

dts_event = np.array([dt_santa.dt.date, dt_santa_cloud.dt.date, dt_santa_mw.dt.date, dt_santa_moon.dt.date, dt_santa_sun.dt.date])


fsize=10
sizem = 1
fig,ax = plt.subplots(figsize=(7,4), dpi=150)
ax.scatter(np.unique(dt_santa), 5*np.ones(np.unique(dt_santa).shape[0]), label='Raw', s=sizem, marker='s', alpha=0.5)
ax.scatter(np.unique(dt_santa_cloud), 4*np.ones(np.unique(dt_santa_cloud).shape[0]), label='Clouds', s=sizem, marker='s', alpha=0.5)
ax.scatter(np.unique(dt_santa_mw), 3*np.ones(np.unique(dt_santa_mw).shape[0]), label='Milky Way', s=sizem, marker='s', alpha=0.5)
ax.scatter(np.unique(dt_santa_moon), 2*np.ones(np.unique(dt_santa_moon).shape[0]), label='Moon', s=sizem, marker='s', alpha=0.5)
ax.scatter(np.unique(dt_santa_sun), np.ones(np.unique(dt_santa_sun).shape[0]), label='Sun', s=sizem, marker='s', alpha=0.5)
ax.set_yticks([])
ax.set_xticks([pd.Timestamp('2019-08-27'), pd.Timestamp('2020-05-01'), pd.Timestamp('2021-01-04')])
ax.set_xticklabels(['2019-09', '2020-05', '2021-01'])
ax.xaxis.set_minor_locator(MultipleLocator(31))
ax.xaxis.grid(ls='--', lw='0.5', which='both')
ax.set_ylim(0.8, 5.5)
ax.text(pd.Timestamp('2019-09-01'), 5+0.2, 'Raw', fontsize=fsize)
ax.text(pd.Timestamp('2019-09-01'), 4+0.2, 'Clouds removed', fontsize=fsize)
ax.text(pd.Timestamp('2019-09-01'), 3+0.2, 'Milky Way removed', fontsize=fsize)
ax.text(pd.Timestamp('2019-09-01'), 2+0.2, 'Moon removed', fontsize=fsize)
ax.text(pd.Timestamp('2019-09-01'), 1+0.2, 'Sun removed', fontsize=fsize)
plt.tight_layout()
#plt.savefig('figures/filtering_eventplot.png')



# # Make eventplot to graph filtering of dates
# lineoffsets1 = np.array([3, 6, 9, 12, 15])
# linelengths1 = np.ones(5)*2

# sizem = 2
# fsize = 7
# dts = np.array([pd.Timestamp(date).timestamp() for date in dt_santa.dt.date])
# dts_cloud = np.copy(dts)
# dts_cloud[np.isnan(cosqm_santa_cloud[:,0])] = False
# dts_mw = np.copy(dts)
# dts_mw[~mw_mask] = np.nan
# dts_moon = np.copy(dts)
# dts_moon[mw_mask][~moon_mask] = np.nan
# dts_sun = np.copy(dts)
# dts_sun[mw_mask][moon_mask][~sun_mask] = np.nan

# dts_event = np.vstack((dts,dts_cloud,dts_mw,dts_moon,dts_sun))
# fig, ax = plt.subplots(figsize=(12,8), dpi=150)
# ax.eventplot(dts_event, colors=['r', 'g', 'b', 'y', 'k'], orientation='horizontal', lineoffsets=lineoffsets1, linelengths=linelengths1, linewidths=0.1)
# ax.set_xlim(pd.Timestamp('2019-08-27').timestamp()-3600*24, pd.Timestamp('2021-01-04').timestamp()+3600*24)
# ax.set_ylim(0,20)
# ax.tick_params(left=False, labelleft = False )
# ax.set_xticks([pd.Timestamp('2019-08-27').timestamp(), pd.Timestamp('2020-05-01').timestamp(), pd.Timestamp('2021-01-04').timestamp()])
# ax.set_xticklabels(['2019-09', '2020-05', '2021-01']) 
# #ax.legend(loc=(pd.Timestamp('2019-08-27').timestamp(), 3.5), prop={'size': 6})
# ax.set_ylim(0.9,5.5)
# #ax.xaxis.set_minor_locator(MultipleLocator(31*24*1627660252))
# ax.xaxis.grid(linestyle = '--', linewidth = 0.5, which='both')
# ax.text(pd.Timestamp('2019-08-27').timestamp(), 5+0.2, 'Raw', fontsize=fsize )
# ax.text(pd.Timestamp('2019-08-27').timestamp(), 4+0.2, 'Clouds removed', fontsize=fsize )
# ax.text(pd.Timestamp('2019-08-27').timestamp(), 3+0.2, 'Milky Way removed', fontsize=fsize )
# ax.text(pd.Timestamp('2019-08-27').timestamp(), 2+0.2, 'Moon removed', fontsize=fsize )
# ax.text(pd.Timestamp('2019-08-27').timestamp(), 1+0.2, 'Sun removed', fontsize=fsize )
# plt.savefig('figures/filtering_eventplot.png')



## set to nan values that are color higher than clear by visual analysis, followed by clouded nights by visual analysis
print('Remove invalid cosqm data (R channel higher than C), and visual cloud screening')
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
cosqm_santa_sun_unfiltered = np.copy(cosqm_santa_sun)

d = np.copy(dt_santa_sun)

ddays_cosqm = np.array([(date.date()-d[0].date()).days for date in d])
hours = np.array([date.hour for date in d])


# Sliding window filter on all bands to smooth out rapid variations (mainly caused by R channel having too high variance since resolution of sqm)
print('Sliding window filter on all cosqm bands')
def Sliding_window_cosqm(data_array, window_size):
	da = np.copy(data_array)
	dslide = np.lib.stride_tricks.sliding_window_view(data_array, window_shape = window_size, axis=0).copy() #start and end values lost. size of array is data_array.shape[0]-2
	dd = np.nanmean(dslide, axis=2)
	padd = np.full([1, da.shape[1]], np.nan)
	for i in range(window_size//2):
		dd = np.insert(dd, 0, padd, axis=0)                     #append nan to start to get same shape as input
		dd = np.insert(dd, dd.shape[0], padd, axis=0)      	      #append nan to end to get same shape as input
	return dd

cosqm_santa_sun = Sliding_window_cosqm(cosqm_santa_sun, cosqm_window_size)
np.savetxt('cosqm_filtered_santa.csv', np.array([dt_santa_sun[~np.isnan(cosqm_santa_sun[:,0])], cosqm_santa_sun[:,0][~np.isnan(cosqm_santa_sun[:,0])], cosqm_santa_sun[:,1][~np.isnan(cosqm_santa_sun[:,0])], cosqm_santa_sun[:,2][~np.isnan(cosqm_santa_sun[:,0])], cosqm_santa_sun[:,3][~np.isnan(cosqm_santa_sun[:,0])], cosqm_santa_sun[:,4][~np.isnan(cosqm_santa_sun[:,0])]]).T, fmt="%s")


#print('remove artefact from sliding window filter ZNSB')
for day in np.unique(ddays_cosqm):
	d_mask = np.zeros(ddays_cosqm.shape[0], dtype=bool)
	d_mask[(ddays_cosqm == day) & (hours < 12)] = True
	d_mask[(ddays_cosqm == day-1) & (hours > 12)] = True 
	try:
		cosqm_santa_sun[np.where(d_mask==True)[0][0]] = np.nan
	except:
		print('0', day)
		pass
	try:
		cosqm_santa_sun[np.where(d_mask==True)[0][-1]] = np.nan
	except:
		print('-1', day)
		pass

# Plot variance filter effect on data
colors=['r', 'g', 'b', 'y']
plt.figure(dpi=150, figsize=(7,4))
for i in (0,2):
	plt.scatter(dt_santa_sun, cosqm_santa_sun_unfiltered[:,i+1], c=colors[i], s=8, label=f'{cosqm_bands[i]}nm')
	plt.scatter(dt_santa_sun, cosqm_santa_sun[:,i+1], c='k', marker='+', s=8)
plt.xlabel('Time of measurement (UTC)')
plt.ylabel('CoSQM ZNSB (MPSAS)')
plt.ylim(19.6)
plt.legend()
plt.xticks(ticks=[pd.Timestamp(f'2020-01-24 0{i+1}') for i in range(6)], labels=[str(i+1) for i in range(6)])
plt.xlim(pd.Timestamp('2020-01-24 02:30'), pd.Timestamp('2020-01-24 06:33'))
plt.tight_layout()
#plt.savefig('figures/sliding_window_cosqm.png')

#Find variance before and after cosqm sliding window 
#inds = np.where((dt_santa_date.values == pd.Timestamp('2020-01-24')) | (dt_santa_date.values == pd.Timestamp('2020-01-25')))[0]
#cosqm_santa_sun_unfiltered_var = np.nanvar(cosqm_santa_sun_unfiltered[inds], axis=0)
#cosqm_santa_sun_var = np.nanvar(cosqm_santa_sun[inds], axis=0)


################
# Light pollution trends
################
c = np.copy(cosqm_santa_sun)
c_norm = np.copy(c)
## Every night normalized (substracted magnitude) with mean of 1am to 2am data
## correct for timechange
print('Correct for timechange (DST)')
date2019 = pd.Timestamp('2019-10-27 02', tz='UTC')
date2020p = pd.Timestamp('2020-03-29 01', tz='UTC')
date2020 = pd.Timestamp('2020-10-25 02', tz='UTC')
date2021p = pd.Timestamp('2021-03-28 01', tz='UTC')
date2021 = pd.Timestamp('2021-10-31 02', tz='UTC')
d_local = np.copy(dt_santa_sun)
td = pd.Timedelta('01h')
d_local[(d_local>date2020p) & (d_local<date2020)] += td
d_local[(d_local>date2021p) & (d_local<date2021)] += td
hours_local = np.array([date.hour for date in d_local])

months = np.array([date.month for date in d])

print('Normalize ZNSB with 1-2am mean ZNSB per night')
for day in np.unique(ddays_cosqm):
	d_mask = np.zeros(ddays_cosqm.shape[0], dtype=bool)
	d_mask[(ddays_cosqm == day) & (hours_local < 12)] = True
	d_mask[(ddays_cosqm == day-1) & (hours_local > 12)] = True 
	d_mask_mean = d_mask.copy()
	d_mask_mean[hours_local != 1] = False
	c_mean = np.nanmean(c[d_mask_mean],axis=0)
	c_norm[d_mask] -= c_mean

## Threshold clip of data
print('Threshold clip of normalized cosqm ZNSB')
c_norm[c_norm > normalized_cosqm_znsb_threshold] = np.nan

# Fitting of the normalized data to correct for night trend (does not change through year, month or day of week)
# 2nd order function for fit
def third_order(x,a,b,c,d):
	return a*x**3+b*x**2+c*x+d

hours_float = np.array([ date.hour+                    #WATCH OUT FOR TIMEZONE HERE!
	date.minute/60+
	date.second/3600 for date in d_local ])
hours_float[hours_float>12]-=24

fit_params_c, _ = curve_fit(third_order, hours_float[~np.isnan(c_norm[:,0])], c_norm[~np.isnan(c_norm[:,0])][:,0])
fit_params_r, _ = curve_fit(third_order, hours_float[~np.isnan(c_norm[:,1])], c_norm[~np.isnan(c_norm[:,1])][:,1])
fit_params_g, _ = curve_fit(third_order, hours_float[~np.isnan(c_norm[:,2])], c_norm[~np.isnan(c_norm[:,2])][:,2])
fit_params_b, _ = curve_fit(third_order, hours_float[~np.isnan(c_norm[:,3])], c_norm[~np.isnan(c_norm[:,3])][:,3])
fit_params_y, _ = curve_fit(third_order, hours_float[~np.isnan(c_norm[:,4])], c_norm[~np.isnan(c_norm[:,4])][:,4])
fit_params = np.array([fit_params_c, fit_params_r, fit_params_g, fit_params_b, fit_params_y]).T

cosqm_santa_2nd_noweight = np.array([third_order(hours_float, fit_params_c[0], fit_params_c[1], fit_params_c[2], fit_params_c[3]),
	third_order(hours_float, fit_params_r[0], fit_params_r[1], fit_params_r[2], fit_params_r[3]),
	third_order(hours_float, fit_params_g[0], fit_params_g[1], fit_params_g[2], fit_params_g[3]),
	third_order(hours_float, fit_params_b[0], fit_params_b[1], fit_params_b[2], fit_params_b[3]),
	third_order(hours_float, fit_params_y[0], fit_params_y[1], fit_params_y[2], fit_params_y[3])]).T


# Add weighted fit function with weights computed from initial unweighted fit: sigma loop.
# Values bigger than one sigma are removed at each iteration until sigma converges. Once obtained, the final parameters are kept. The computation is made for each band seperately.
# std dev values for Santa Cruz ended up at R:0.078, B:0.072

cosqm_santa_2nd = np.copy(cosqm_santa_2nd_noweight)

for i in range(10):
	sigma = cosqm_santa_2nd - c_norm
	cw_mask = np.abs(sigma[:,0])<np.nanstd(sigma[:,0])
	rw_mask = np.abs(sigma[:,1])<np.nanstd(sigma[:,1])
	gw_mask = np.abs(sigma[:,2])<np.nanstd(sigma[:,2])
	bw_mask = np.abs(sigma[:,3])<np.nanstd(sigma[:,3])
	yw_mask = np.abs(sigma[:,4])<np.nanstd(sigma[:,4])
	print('Trends weighted fit')
	print(f'n_sigma:{i}, std_R:{np.nanstd(sigma[:,1])}, std_B:{np.nanstd(sigma[:,3])}')
	fit_params_weight_c, _ = curve_fit(third_order, hours_float[cw_mask], c_norm[:,0][cw_mask])
	fit_params_weight_r, _ = curve_fit(third_order, hours_float[rw_mask], c_norm[:,1][rw_mask])
	fit_params_weight_g, _ = curve_fit(third_order, hours_float[gw_mask], c_norm[:,2][gw_mask])
	fit_params_weight_b, _ = curve_fit(third_order, hours_float[bw_mask], c_norm[:,3][bw_mask])
	fit_params_weight_y, _ = curve_fit(third_order, hours_float[yw_mask], c_norm[:,4][yw_mask])
	fit_params_weight = np.array([fit_params_weight_c, fit_params_weight_r, fit_params_weight_g, fit_params_weight_b, fit_params_weight_y]).T

	cosqm_santa_2nd = np.array([third_order(hours_float, *fit_params_weight_c),
		third_order(hours_float, *fit_params_weight_r),
		third_order(hours_float, *fit_params_weight_g),
		third_order(hours_float, *fit_params_weight_b),
		third_order(hours_float, *fit_params_weight_y)]).T

# 	if i==0:
# 		xs = np.linspace(hours_float.min()-0.5, hours_float.max()+0.5, 1001)
# 		band = 1
# 		plt.figure(dpi=150,figsize=(7,4))
# 		plt.hist2d(hours_float[np.isfinite(c_norm)[:,band]], c_norm[:,band][np.isfinite(c_norm)[:,band]], 80, cmap='inferno')
# 		#plt.hist2d(hours_float[np.isfinite(c_norm)[:,band]], c_norm[:,band][np.isfinite(c_norm)[:,band]], 80, cmap='inferno', norm=LogNorm(), vmax=10)
# 		plt.plot(xs, third_order(xs, *fit_params[:, band]), label=f'third order fit {cosqm_bands[band-1]}nm', c='c')
# 		plt.xlabel('Local time from midnight (h)')
# 		plt.ylabel(f'Normalized {cosqm_bands[band-1]}nm ZNSB (MPSAS)')
# 		plt.ylim(-0.3)
# 		cbar = plt.colorbar(label='counts')
# 		#cbar.set_ticks(np.arange(1,12,1))
# 		#cbar.update_ticks()
# 		#cbar.set_ticklabels(np.arange(0,11,1))
# 		plt.tight_layout()
# 		#plt.savefig(f'figures/trend/normalized_cosqm_{cosqm_bands[band-1]}.png')
# 	plt.plot(xs, third_order(xs, *fit_params_weight[:,band]), label=f'third order weighted fit 644nm - n_sigma={i+1}', c='w', linestyle='--')
# plt.legend()



#hist 2d
weekdays_str = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday','Sunday']
months_str = ['January', 'Feburary', 'March', 'April', 'May', 'June', 'July', 'August', 'Septembre', 'October', 'November', 'December']
weekdays = np.array([ date.weekday() for date in d_local ])
markers = ['.', '1', '2', '3', '4', 'o', 's']

xs = np.linspace(hours_float.min()-0.5, hours_float.max()+0.5, 1001)
band = 1
plt.figure(dpi=150,figsize=(7,4))
plt.hist2d(hours_float[np.isfinite(c_norm)[:,band]], c_norm[:,band][np.isfinite(c_norm)[:,band]], 80, cmap='inferno')
#plt.hist2d(hours_float[np.isfinite(c_norm)[:,band]], c_norm[:,band][np.isfinite(c_norm)[:,band]], 80, cmap='inferno', norm=LogNorm(), vmax=10)
plt.plot(xs, third_order(xs, *fit_params[:, band]), label=f'third order fit {cosqm_bands[band-1]}nm', c='c')
plt.plot(xs, third_order(xs, *fit_params_weight[:,band]), label=f'third order weighted fit {cosqm_bands[band-1]}nm', c='w', linestyle='--')
plt.xlabel('Local time from midnight (h)')
plt.ylabel(f'Normalized {cosqm_bands[band-1]}nm ZNSB (MPSAS)')
plt.ylim(-0.3)
cbar = plt.colorbar(label='counts')
#cbar.set_ticks(np.arange(1,12,1))
#cbar.update_ticks()
#cbar.set_ticklabels(np.arange(0,11,1))
plt.tight_layout()
#plt.savefig(f'figures/trend/normalized_cosqm_{cosqm_bands[band-1]}.png')

xs = np.linspace(hours_float.min()-0.5, hours_float.max()+0.5, 1001)
band = 3
plt.figure(dpi=150,figsize=(7,4))
plt.hist2d(hours_float[np.isfinite(c_norm)[:,band]], c_norm[:,band][np.isfinite(c_norm)[:,band]], 80, cmap='inferno')
plt.plot(xs, third_order(xs, fit_params[0, band], fit_params[1, band], fit_params[2, band], fit_params[3, band]), label='third order fit 644nm', c='c')
plt.plot(xs, third_order(xs, fit_params_weight[0, band], fit_params_weight[1, band], fit_params_weight[2, band], fit_params_weight[3, band]), label='third order weighted fit 644nm', c='w', linestyle='--')
plt.xlabel('Local time from midnight (h)')
plt.ylabel(f'Normalized {cosqm_bands[band-1]}nm ZNSB (MPSAS)')
plt.ylim(-0.3)
cbar = plt.colorbar(label='counts')
#cbar.set_ticks(np.arange(1,12,1))
#cbar.update_ticks()
#cbar.set_ticklabels(np.arange(0,11,1))
plt.tight_layout()
#plt.savefig(f'figures/trend/normalized_cosqm_{cosqm_bands[band-1]}.png')

# Correct filtered data with fit curve
print('Correct filtered data with trend fits')
dt_santa_final = dt_santa_sun
cosqm_santa_final = np.copy(cosqm_santa_sun) - cosqm_santa_2nd
ddays_cosqm = np.array([(date.date()-d[0].date()).days for date in dt_santa_final])
dt_santa_date = dt_santa_final.dt.date
dt_santa_dt = dt_santa_final.dt.to_pydatetime()


#Plot effect of corrected trends on specific nights
band = 3
fig,ax = plt.subplots(dpi=150, figsize=(7,4))
plt.setp(ax, xticks=[pd.Timestamp('2020-05-22 23'), pd.Timestamp('2020-05-23 00'), pd.Timestamp('2020-05-23 01'), pd.Timestamp('2020-05-23 02')], xticklabels=['-1', '0', '1', '2'])
ax.scatter(d_local, cosqm_santa_sun[:,band], s=10, label=f'Filtered CoSQM {cosqm_bands[band-1]}nm ZNSB')
ax.scatter(d_local, cosqm_santa_final[:,band], s=10, label='Corrected for nightly trend')
ax.set_xlim(pd.Timestamp('2020-05-22 22h35'), pd.Timestamp('2020-05-23 03'))
ax.set_xlabel('Local time from midnight (h)')
ax.tick_params('x', labelrotation=0)
ax.set_ylim(19.40, 19.65)
ax.set_ylabel(f'{cosqm_bands[band-1]}nm CoSQM ZNSB (MPSAS)')
ax.legend()
plt.tight_layout()
#fig.savefig(f'figures/trend/correction_single_night_{cosqm_bands[band-1]}.png')

c_tot = np.array((c_norm[:,1]+np.mean(cosqm_santa_sun[:,1][np.where((hours_float ==1) | (hours_float ==2))]),
c_norm[:,2]+np.mean(cosqm_santa_sun[:,2][np.where((hours_float ==1) | (hours_float ==2))]),
c_norm[:,3]+np.mean(cosqm_santa_sun[:,3][np.where((hours_float ==1) | (hours_float ==2))]),
c_norm[:,4]+np.mean(cosqm_santa_sun[:,4][np.where((hours_float ==1) | (hours_float ==2))]))).T

x = np.arange(-2,6, 0.01)

#Plot corrected trend to show red goes darker throughout the night compared to other colors
plt.figure(dpi=200, figsize=(7,5))
plt.plot(x, third_order(x, fit_params_weight_r[0], fit_params_weight_r[1], fit_params_weight_r[2], fit_params_weight_r[3]), c='r', linestyle='solid', linewidth=2, label=f'{cosqm_bands[0]} nm')
plt.plot(x, third_order(x, fit_params_weight_g[0], fit_params_weight_g[1], fit_params_weight_g[2], fit_params_weight_g[3]), c='g', linestyle='dotted', linewidth=2, label=f'{cosqm_bands[1]} nm')
plt.plot(x, third_order(x, fit_params_weight_b[0], fit_params_weight_b[1], fit_params_weight_b[2], fit_params_weight_b[3]), c='b', linestyle='dashed', linewidth=2, label=f'{cosqm_bands[2]} nm')
plt.plot(x, third_order(x, fit_params_weight_y[0], fit_params_weight_y[1], fit_params_weight_y[2], fit_params_weight_y[3]), c='y', linestyle='dashdot', linewidth=2, label=f'{cosqm_bands[3]} nm')
plt.legend()
plt.xlabel('Local time from midnight (h)')
plt.ylabel('Normalized CoSQM ZNSB (MPSAS)')
plt.tight_layout()
#plt.savefig(f'figures/trend/trend_fits.png')

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

print('filter 0 values in aeronet AOD')
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
print('find same days that have measurements (aeronet vs cosqm)')
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

# Evaluation of distinction between cosqm bands -> to determine if bands are uncorrelated to extract valuable aod and AE
xs = np.arange(16,21,0.01)
fig, ax = plt.subplots(1,3, sharex=True, sharey=True, figsize=(10,7), dpi=200)
ax[0].scatter(cosqm_santa_corr[:,0], cosqm_santa_corr[:,1], s=10, c='g', label=f'{cosqm_bands[1]} nm')
ax[0].scatter(cosqm_santa_corr[:,0], cosqm_santa_corr[:,2], s=10, c='b', label=f'{cosqm_bands[2]} nm')
ax[0].scatter(cosqm_santa_corr[:,0], cosqm_santa_corr[:,3], s=10, c='y', label=f'{cosqm_bands[3]} nm')
ax[1].scatter(cosqm_santa_corr[:,2], cosqm_santa_corr[:,1], s=10, c='g', label=f'{cosqm_bands[1]} nm')
ax[1].scatter(cosqm_santa_corr[:,2], cosqm_santa_corr[:,3], s=10, c='y', label=f'{cosqm_bands[3]} nm')
ax[2].scatter(cosqm_santa_corr[:,1], cosqm_santa_corr[:,3], s=10, c='y', label=f'{cosqm_bands[3]} nm')
ax[0].set_xlabel(f'{cosqm_bands[0]} nm')
ax[1].set_xlabel(f'{cosqm_bands[2]} nm')
ax[2].set_xlabel(f'{cosqm_bands[1]} nm')
ax[0].legend()
ax[1].legend()
ax[2].legend()
fig.supxlabel('CoSQM ZNSB (MPSAS)', fontsize=10)
fig.supylabel('CoSQM ZNSB (MPSAS)', fontsize=10)
plt.tight_layout()
plt.savefig('figures/correlation/santa/correlation_znsb.png')

#############
# CORRELATION
#############

# Selection of filtered and corrected ZNSB values
print('selection of cosqm measurements for dusk and dawn correlation')
cosqm_am = np.zeros((np.unique(dt_santa_date).shape[0], 5))
cosqm_pm = np.zeros((np.unique(dt_santa_date).shape[0], 5))
hours_cosqm = dt_santa_final.dt.hour


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

# band = 3
# plt.figure()
# plt.scatter(np.unique(dt_santa_date), cosqm_am[:,band], label='cosqm_am', s=15)
# plt.scatter(np.unique(dt_santa_date), cosqm_pm[:,band], label='cosqm_pm', s=15)
# plt.xlabel('days from july 2019')
# plt.ylabel(f'CoSQM {cosqm_bands[band-1]}nm Magnitudes (mag)')
# plt.legend()



# Selection of AOD values  
print('selection of aeronet aod measurements for dusk and dawn correlation')
aod_am = np.zeros((np.unique(dt_aod_date).shape[0], 4))
aod_pm = np.zeros((np.unique(dt_aod_date).shape[0], 4))
angstrom_am = np.zeros((np.unique(dt_aod_date).shape[0], 4))
angstrom_pm = np.zeros((np.unique(dt_aod_date).shape[0], 4))
hours_aod = dt_aod.dt.hour

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
wls2 = np.ones((aod_am.shape[0], 4))*np.array([675, 500, 500, 500])     # AERONET daytime sunphoto center wavelenghts per filter

def aod_from_angstrom(aod2, wls1, wls2, alpha):
	return aod2*(wls1/wls2)**-alpha

def angstrom_from_aod(arr, bands, ind1, ind2):
	return -np.log(arr[:,ind1]/arr[:,ind2])/np.log(bands[ind1]/bands[ind2])

# Correct computed aod with matching band nominal wavelength
print('correct computed aod with matching band nominal wavelength')
aod_am_corr = aod_from_angstrom(aod_am, wls1, wls2, angstrom_am)
aod_pm_corr = aod_from_angstrom(aod_pm, wls1, wls2, angstrom_pm)

# band=3
# plt.figure()
# plt.scatter(np.unique(dt_santa_date), aod_am_corr[:,band], label='aod_am_corr', s=15)
# plt.scatter(np.unique(dt_santa_date), aod_pm_corr[:,band], label='aod_pm_corr', s=15)
# plt.xlabel('days from july 2019')
# plt.ylabel(f'CoSQM {cosqm_bands[band-1]}nm Magnitudes (mag)')
# plt.legend()


# Correlation fits
print('AERONET AOD-CoSQM ZNSB correlation calculations')

def fit_func(x, a,b):
	return -a*np.log(x/b)

def fit_func1(x, params):
	return fit_func(x, params[0], params[1])

#single fit functions for both dusk and dawn correlated points
param_bounds=([0,10],[15,25])
p0 = [7, 20]
cosqm = np.vstack([cosqm_am, cosqm_pm])[:,1:]
cosqm = np.array(cosqm, dtype=np.longdouble)

aod_unfiltered = np.vstack([aod_am, aod_pm])
aod_am[aod_am<0.04] = np.nan
aod_pm[aod_pm<0.04] = np.nan
aod = np.vstack([aod_am, aod_pm])
valid = ~(np.isnan(cosqm) | (np.isnan(aod)))

# make histogram of aod values
plt.figure(dpi=150, figsize=(12,8))
hist_aod_unfiltered = np.histogram(aod_unfiltered[~np.isnan(aod_unfiltered)], bins=100)
plt.scatter(hist_aod_unfiltered[1][:-1][hist_aod_unfiltered[0]>0], hist_aod_unfiltered[0][hist_aod_unfiltered[0]>0], c='b')
plt.xlim(0,1.5)
plt.ylim(0)
plt.xlabel('AOD values')
plt.ylabel('counts')

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






# # Threshold clip on correlated AOD (crazy AOD values from imperfect AOD-ZNSB fit)
# print('threshold clip on correlated cosqm aod (for aod values showing 50% error)')
# cosqm_aod_r[cosqm_aod_r<cosqm_aod_threshold] = np.nan
# cosqm_aod_g[cosqm_aod_g<cosqm_aod_threshold] = np.nan
# cosqm_aod_b[cosqm_aod_b<cosqm_aod_threshold] = np.nan
# cosqm_aod_y[cosqm_aod_y<cosqm_aod_threshold] = np.nan
# cosqm_aod_all = np.array((cosqm_aod_r, cosqm_aod_g, cosqm_aod_b, cosqm_aod_y)).T

# # fit AE for each 4 points cosqm measurement
# print('compute AE fit for each cosqm 4 values measurements')
# def FitAe(wl, k, alpha):
# 	return k*wl**-alpha

# def fit_except(data_array):                         # Some cosqm aod values do not converge while fitting
# 	try:
# 		return curve_fit(FitAe, cosqm_bands, data_array, p0=(1000, 0.5))[0]
# 	except RuntimeError:
# 		print(f"Error - curve_fit failed: cosqm aod values {data_array}")
# 		return np.array((None, None), dtype=object)

# ae_fit_params = np.zeros((cosqm_aod_all.shape[0], 2))
# ae_fit_params[:] = np.nan
# cosqm_nonnans = ~np.isnan(cosqm_aod_all).any(axis=1)
# ae_fit_params[cosqm_nonnans] = np.vstack([fit_except(cosqm_aods) for cosqm_aods in cosqm_aod_all[cosqm_nonnans]])
# ae_fit_succeed = ~np.isnan(ae_fit_params[:,0])

# cosqm_ae = np.array([(fit1, fit2) for fit1, fit2 in ae_fit_params])


cosqm_ae = angstrom_from_aod(np.array([cosqm_aod_r, cosqm_aod_b]).T, cosqm_bands, 0, 1)




#Filter bad AE values (negative and higher than threshold)
ae_mask = (cosqm_ae>ae_min) & (cosqm_ae<=ae_max)


# Plot fitted AE from cosqm_aod values
# dt_santa_final_date = np.array([date.to_datetime64() for date in dt_santa_final])
# x_ae = np.arange(500,700, 0.1)
# inds=np.arange(2550, 2555, 1)    # to find specific nights: np.where(dt_santa_date == pd.to_datetime('2020-02-24'))

# for ind in inds:
#     plt.scatter(cosqm_bands, cosqm_aod_all[ind], label=f'{dt_santa_final_date[ind]}')
#     plt.plot(x_ae, FitAe(x_ae, ae_fit_params[ind][0], ae_fit_params[ind][1]))
# plt.legend()

# # plot znsb to understand why alpha goes crazy for high ZNSB and small ZNSB differences between bands
# colors=['r', 'g', 'b', 'y']
# for i in range(4):
#     plt.scatter(dt_santa_dt, cosqm_santa_final[:, i+1], c=colors[i], s=10)


#single fit function: correlation
xs = np.arange(16,21,0.01)
fig, ax = plt.subplots(2,1, sharex=True, sharey=True, figsize=(7,7), dpi=200)
ax[0].scatter(cosqm_pm[:,1], aod_pm[:,0], label='dawn')
ax[0].scatter(cosqm_am[:,1], aod_am[:,0], label='dusk')
ax[0].plot(xs, fit_func1(xs, corr_fitr), c='k', linewidth=1)
ax[1].scatter(cosqm_pm[:,3], aod_pm[:,2])
ax[1].scatter(cosqm_am[:,3], aod_am[:,2])
ax[1].plot(xs, fit_func1(xs, corr_fitb), c='k', linewidth=1)
ax[0].set_xlim(17,20.7)
ax[0].set_yscale('log')
ax[0].set_ylim(0.028,3.75)
ax[0].text(0.05,0.2, f'{cosqm_bands[0]} nm\na,b = {str(corr_single_round[0])[1:-1]}', horizontalalignment='left', verticalalignment='center', transform=ax[0].transAxes)
ax[1].text(0.05,0.2, f'{cosqm_bands[2]} nm\na,b = {str(corr_single_round[2])[1:-1]}', horizontalalignment='left', verticalalignment='center', transform=ax[1].transAxes)
ax[0].legend(loc='center left', prop={'size': 10})
fig.supxlabel('CoSQM ZNSB (MPSAS)', fontsize=10)
fig.supylabel('AERONET daytime AOD', fontsize=10)
plt.tight_layout()
plt.savefig('figures/correlation/santa/correlation_single_fit_santa.png')


# Relative humidity data
msize = 5
hr = pd.read_csv('hr_2020.dat').values
hr[:,1] = np.around(hr[:,1].astype(float), decimals=0)
hr[:,3] = np.around(hr[:,3].astype(float), decimals=0)
fig,ax = plt.subplots(figsize=(6,3), dpi=200)
plt.hist(hr[:,1], bins=np.unique(hr[:,1]),  label='Mean (Dawn)', alpha=0.8)
plt.hist(hr[:,3], bins=np.unique(hr[:,3]), label='Mean (Dusk)', alpha=0.8)
plt.legend(loc='upper left', prop={'size': 9})
plt.xlabel('Relative humidity (%)')
plt.ylabel('Counts')
plt.tight_layout()
plt.savefig('figures/continuity/hr_2020.png')


#single fit function: continuity 2020-02-21
msize=1.5
fig, ax = plt.subplots(2,1, sharex=True, sharey=True, dpi=120, figsize=(12,8))
plt.setp(ax, xticks=[pd.Timestamp('2020-02-23'), pd.Timestamp('2020-03-02')], xticklabels=['2020-02-23', '2020-03-02'])
ax[0].scatter(dt_aod, data_aod[:,0], s=msize, label='AERONET daytime')
ax[0].scatter(dt_santa_corr[ae_mask], cosqm_aod_r[ae_mask], s=msize, label='CoSQM derived AOD')
ax[1].scatter(dt_aod, data_aod[:,2], s=msize)
ax[1].scatter(dt_santa_corr[ae_mask], cosqm_aod_b[ae_mask], s=msize)
ax[0].tick_params('x', labelrotation=0)
ax[1].tick_params('x', labelrotation=80)
ax[0].set_yscale('log')
ax[0].set_xlim(pd.Timestamp('2020-02-20'), pd.Timestamp('2020-03-04'))
ax[0].set_ylim(0.04,5.3)
ax[0].legend(prop={"size":6}, loc='upper right')
ax[0].text(0.35,0.1, f'{cosqm_bands[0]} nm', horizontalalignment='center', verticalalignment='center', transform=ax[0].transAxes, fontsize=10)
ax[1].text(0.35,0.1, f'{cosqm_bands[2]} nm', horizontalalignment='center', verticalalignment='center', transform=ax[1].transAxes, fontsize=10)
fig.supxlabel('Time (UTC)', fontsize=10)
fig.supylabel('AOD', fontsize=10)
plt.tight_layout()
plt.savefig('figures/continuity/continuity_aod_20200220.png')


#single fit function: continuity Angstrom 2020-02-21
fig, ax = plt.subplots(1,1, dpi=120, figsize=(10,6))
plt.setp(ax, xticks=[pd.Timestamp('2020-02-22'), pd.Timestamp('2020-02-27'), pd.Timestamp('2020-03-03')], xticklabels=['2020-02-22', '2020-02-27', '2020-03-03'])
ax.scatter(dt_aod, data_angstrom, s=1.5, label='AERONET daytime derived 440-679nm AE')
ax.scatter(dt_santa_corr[ae_mask], cosqm_ae[ae_mask], s=1.5, label=f'CoSQM derived fitted AE')
ax.plot(dt_aod, np.zeros(dt_aod.shape[0]), linewidth=1, linestyle='--', color='grey')
ax.set_xlim(pd.Timestamp('2020-02-20 16'), pd.Timestamp('2020-03-04'))
ax.set_ylim(-0.2,2)
ax.set_xlabel('Time - UTC')
ax.set_ylabel('Angstrom exponent')
ax.legend(prop={"size":10})
ax.xaxis.set_minor_locator(MultipleLocator(1))
#ax.set_yscale('log')
plt.tight_layout()
plt.savefig('figures/continuity/continuity_angstrom_20200220.png')


# Import Lidar data from AERONET on may 2020 (Africa provided the data)
mpl_may = pd.read_csv('wetransfer_lidar_data_2021-09-06_0658/Lidar_Data/aod_SCO.dat', delimiter=' ', skiprows=2, names=('Date', 'Time', 'aod'))[:-1]
mpl_may_aod = np.array([str(val)[:-1] for val in mpl_may['aod']], dtype=float)
mpl_may_dates = np.array([pd.Timestamp(date+'T'+time) for date, time in np.array([mpl_may['Date'].values, mpl_may['Time'].values]).T])

#single fit function: continuity 2020-05-21
msize=1.5
fig, ax = plt.subplots(2,2, sharex=True, sharey=True, dpi=120, figsize=(12,8))
plt.setp(ax, xticks=[pd.Timestamp('2020-05-23'), pd.Timestamp('2020-05-27')], xticklabels=['2020-05-23', '2020-05-27'])
ax[0].scatter(dt_aod, data_aod[:,0], s=msize, label='AERONET daytime')
ax[0].scatter(dt_santa_corr[ae_mask], cosqm_aod_r[ae_mask], s=msize, label='CoSQM derived AOD')
ax[0].scatter(mpl_may_dates, mpl_may_aod, c='k', marker='+', s=msize/2, label='MPL derived AOD')
ax[1].scatter(dt_aod, data_aod[:,2], s=msize)
ax[1].scatter(dt_santa_corr[ae_mask], cosqm_aod_b[ae_mask], s=msize)
ax[1].scatter(mpl_may_dates, mpl_may_aod, c='k', marker='+', s=msize/2, label='MPL derived AOD')
ax[1].tick_params('x', labelrotation=0)
ax[1,1].tick_params('x', labelrotation=0)
ax[0, 0].set_yscale('log')
ax[0,0].set_xlim(pd.Timestamp('2020-05-21'), pd.Timestamp('2020-05-29'))
ax[0,0].set_ylim(0.04,0.81)
ax[0,0].legend(prop={"size":10}, loc='upper left')
ax[0, 0].text(0.5,0.10, f'{cosqm_bands[0]} nm', horizontalalignment='center', verticalalignment='center', transform=ax[0,0].transAxes, fontsize=10)
ax[0, 1].text(0.5,0.10, f'{cosqm_bands[1]} nm', horizontalalignment='center', verticalalignment='center', transform=ax[0,1].transAxes, fontsize=10)
ax[1, 0].text(0.5,0.10, f'{cosqm_bands[2]} nm', horizontalalignment='center', verticalalignment='center', transform=ax[1,0].transAxes, fontsize=10)
ax[1, 1].text(0.5,0.10, f'{cosqm_bands[3]} nm', horizontalalignment='center', verticalalignment='center', transform=ax[1,1].transAxes, fontsize=10)
ax[1,1].xaxis.set_minor_locator(MultipleLocator(1))
fig.supxlabel('Time (UTC)', fontsize=10)
fig.supylabel('AOD', fontsize=10)
plt.tight_layout()
plt.savefig('figures/continuity/continuity_aod_20200521.png')


#single fit function: continuity Angstrom 2020-05-21
fig, ax = plt.subplots(1,1, dpi=120, figsize=(10,6))
plt.setp(ax, xticks=[pd.Timestamp('2020-05-22'), pd.Timestamp('2020-05-25'), pd.Timestamp('2020-05-28')], xticklabels=['2020-05-22', '2020-05-25', '2020-05-28'])
ax.scatter(dt_aod, data_angstrom, s=1.5, label='AERONET daytime derived 440-679nm AE')
ax.scatter(dt_santa_corr[ae_mask], cosqm_ae[ae_mask], s=1.5, label='CoSQM derived fitted AE')
ax.plot(dt_aod, np.zeros(dt_aod.shape[0]), linewidth=1, linestyle='--', color='grey')
ax.set_xlim(pd.Timestamp('2020-05-21'), pd.Timestamp('2020-05-29'))
ax.set_ylim(-0.2,2)
ax.set_xlabel('Time - UTC')
ax.set_ylabel('Angstrom exponent')
ax.legend(prop={"size":10})
ax.xaxis.set_minor_locator(MultipleLocator(1))
#ax.set_yscale('log')
plt.tight_layout()
plt.savefig('figures/continuity/continuity_angstrom_20200521.png')

# Import Lidar data from AERONET on july 2020 (Africa provided the data)
mpl_july = pd.read_csv('wetransfer_lidar_data_2021-09-06_0658/aod_sco_july_2020.dat', delimiter=' ', skiprows=2, names=('Date', 'Time', 'aod'))[:-1]
mpl_july_aod = np.array([str(val)[:-1] for val in mpl_july['aod']], dtype=float)
mpl_july_dates = np.array([pd.Timestamp(date+'T'+time) for date, time in np.array([mpl_july['Date'].values, mpl_july['Time'].values]).T])

#single fit function: continuity 2020-07-07
msize=1.5
fig, ax = plt.subplots(2,1, sharex=True, sharey=True, dpi=200, figsize=(10,10))
#plt.setp(ax, xticks=[pd.Timestamp('2020-02-23'), pd.Timestamp('2020-03-02')], xticklabels=['2020-02-23', '2020-03-02'])
ax[0].scatter(mpl_july_dates, mpl_july_aod, c='k', marker='+', s=msize/5, label='MPL derived AOD')
ax[0].scatter(dt_aod, data_aod[:,0], s=msize, label='AERONET daytime')
ax[0].scatter(dt_santa_corr[ae_mask], cosqm_aod_r[ae_mask], s=msize, label='CoSQM derived AOD')
ax[1].scatter(mpl_july_dates, mpl_july_aod, c='k', marker='+', s=msize/5, label='MPL derived AOD')
ax[1].scatter(dt_aod, data_aod[:,2], s=msize)
ax[1].scatter(dt_santa_corr[ae_mask], cosqm_aod_b[ae_mask], s=msize)
ax[1].tick_params('x', labelrotation=30)
ax[0].set_yscale('log')
ax[0].set_xlim(pd.Timestamp('2020-07-07'), pd.Timestamp('2020-07-18 12'))
ax[0].set_ylim(0.01,5.3)
ax[0].legend(prop={"size":6}, loc='lower right')
ax[0].text(0.1,0.1, f'{cosqm_bands[0]} nm', horizontalalignment='center', verticalalignment='center', transform=ax[0].transAxes, fontsize=10)
ax[1].text(0.1,0.1, f'{cosqm_bands[2]} nm', horizontalalignment='center', verticalalignment='center', transform=ax[1].transAxes, fontsize=10)
fig.supxlabel('Time (UTC)', fontsize=10)
fig.supylabel('AOD', fontsize=10)
plt.tight_layout()
plt.savefig('figures/continuity/continuity_aod_20200220.png')


#single fit function: continuity Angstrom 2020-07-07
fig, ax = plt.subplots(1,1, dpi=120, figsize=(10,6))
#plt.setp(ax, xticks=[pd.Timestamp('2020-02-22'), pd.Timestamp('2020-02-27'), pd.Timestamp('2020-03-03')], xticklabels=['2020-02-22', '2020-02-27', '2020-03-03'])
ax.scatter(dt_aod, data_angstrom, s=1.5, label='AERONET daytime derived 440-679nm AE')
ax.scatter(dt_santa_corr[ae_mask], cosqm_ae[ae_mask], s=1.5, label=f'CoSQM derived fitted AE')
ax.plot(dt_aod, np.zeros(dt_aod.shape[0]), linewidth=1, linestyle='--', color='grey')
ax.set_xlim(pd.Timestamp('2020-07-07'), pd.Timestamp('2020-07-18 12'))
ax.set_ylim(-0.2,2)
ax.set_xlabel('Time - UTC')
ax.set_ylabel('Angstrom exponent')
ax.legend(prop={"size":10})
ax.xaxis.set_minor_locator(MultipleLocator(1))
#ax.set_yscale('log')
plt.tight_layout()
plt.savefig('figures/continuity/continuity_angstrom_20200220.png')


msize=3
#single fit function: continuity -> variance for uncertainty
fig, ax = plt.subplots(2,1, sharex=True, sharey=True, dpi=200, figsize=(8,8))
plt.setp(ax, xticks=[pd.Timestamp('2020-05-23 17'), pd.Timestamp('2020-05-23 21'), pd.Timestamp('2020-05-24 01'), pd.Timestamp('2020-05-24 05')], xticklabels=['-7', '-3', '1', '5'])
ax[0].scatter(dt_aod, data_aod[:,0], s=msize, label='AERONET daytime')
ax[0].scatter(dt_santa_corr, cosqm_aod_r, s=msize, label='CoSQM derived AOD')
ax[1].scatter(dt_aod, data_aod[:,2], s=msize)
ax[1].scatter(dt_santa_corr, cosqm_aod_b, s=msize)
#ax[0, 0].set_yscale('log')
ax[0].set_xlim(pd.Timestamp('2020-05-23 12'), pd.Timestamp('2020-05-24 10'))
ax[0].set_ylim(0.12,0.22)
ax[0].legend(prop={"size":10}, loc='upper left')
ax[0].text(0.5,0.10, f'{cosqm_bands[0]} nm', horizontalalignment='center', verticalalignment='center', transform=ax[0,0].transAxes, fontsize=10)
ax[1].text(0.5,0.10, f'{cosqm_bands[2]} nm', horizontalalignment='center', verticalalignment='center', transform=ax[1,0].transAxes, fontsize=10)
fig.supxlabel('Time from midnight - UTC (h)', fontsize=10)
fig.supylabel('AOD', fontsize=10)
plt.tight_layout()
#plt.savefig('figures/continuity/continuity_aod_20200524_variance.png')


#single fit function: continuity Angstrom -> variance uncertainty
fig, ax = plt.subplots(1,1, dpi=120, figsize=(10,6))
plt.setp(ax, xticks=[pd.Timestamp('2020-05-23 17'), pd.Timestamp('2020-05-23 21'), pd.Timestamp('2020-05-24 01'), pd.Timestamp('2020-05-24 05')], xticklabels=['-7', '-3', '1', '5'])
ax.scatter(dt_aod, data_angstrom, s=msize, label='AERONET daytime derived 440-679nm AE')
ax.scatter(dt_santa_corr, cosqm_ae[:,1], s=msize, label='CoSQM derived 544-644nm AE')
ax.plot(dt_aod, np.zeros(dt_aod.shape[0]), linewidth=1, linestyle='--', color='grey')
ax.set_xlim(pd.Timestamp('2020-05-23 12'), pd.Timestamp('2020-05-24 10'))
ax.set_ylim(-0.73,0.54)
ax.legend(prop={"size":10})
ax.xaxis.set_minor_locator(MultipleLocator(1))
#ax.set_yscale('log')
fig.supxlabel('Time from midnight - UTC (h)', fontsize=10)
fig.supylabel('Angstrom exponent', fontsize=10)
plt.tight_layout()
#plt.savefig('figures/continuity/continuity_angstrom_20200524_variance.png')


#ZNSB for the same night
fig, ax = plt.subplots(1,1, figsize=(7,5), dpi=200)
plt.setp(ax, xticks=[pd.Timestamp('2020-05-23 22'), pd.Timestamp('2020-05-23 23'), pd.Timestamp('2020-05-24 00'), pd.Timestamp('2020-05-24 01'), pd.Timestamp('2020-05-24 02')], xticklabels=['-2', '-1', '0', '1', '2'])
cs = ['r', 'g', 'b', 'y']
for i in range(4):
	ax.scatter(dt_santa, cosqm_santa[:, i+1], s=msize*0.8, c=colors[i], alpha=0.4)
	ax.scatter(dt_santa, cosqm_santa[:, i+1], s=msize*0.1, c='k')
	ax.scatter(dt_santa_final, cosqm_santa_final[:, i+1], c=cs[i], s=msize, label=f'{cosqm_bands[i]} nm')

ax.set_xlim(pd.Timestamp('2020-05-23 21:50'), pd.Timestamp('2020-05-24 02:10'))
ax.set_ylim(18.5,20.35)
ax.set_xlabel('Time from midnight - UTC (h)', fontsize=10)
ax.set_ylabel('ZNSB (MPSAS)', fontsize=10)
ax.legend(prop={"size":10}, bbox_to_anchor=(0.1, 0.2))
ax.xaxis.set_minor_locator(MultipleLocator(1))
plt.tight_layout()
plt.savefig('figures/continuity/ZNSB_20200523.png')


# COSQM spectral responses: weighted mean of response by spectral functions of local lamp technologies (HPS, LED, MH)

cosqm_responses = pd.read_csv('spectra/sensitivity-including-atm-and-spectrum.csv', header=0)
cr = np.array([cosqm_responses[band].values/np.nanmax(cosqm_responses['Y']) for band in ['R','G','B','Y']]).T
cr_led = np.array([cosqm_responses[band].values/np.nanmax(cosqm_responses['Y-LED']) for band in ['R-LED','G-LED','B-LED','Y-LED']]).T
sky = cosqm_responses['sky'].values/np.nanmax(cosqm_responses['sky'])
sky_led = cosqm_responses['Sky-LED'].values/np.nanmax(cosqm_responses['sky'])

linewidth=1
plt.figure(figsize=(6,3.5))
plt.plot(cosqm_responses['WL'], cr[:,0], c='r', linewidth=linewidth, label=f'{cosqm_bands[0]} nm')
plt.plot(cosqm_responses['WL'], cr[:,1], c='g', linewidth=linewidth, label=f'{cosqm_bands[1]} nm')
plt.plot(cosqm_responses['WL'], cr[:,2], c='b', linewidth=linewidth, label=f'{cosqm_bands[2]} nm')
plt.plot(cosqm_responses['WL'], cr[:,3], c='y', linewidth=linewidth, label=f'{cosqm_bands[3]} nm')
plt.grid(b=True)
plt.legend()
plt.xlabel('Wavelenght (nm)')
plt.ylabel('Normalized spectral response')
plt.xlim(400,975)
plt.ylim(0)
plt.tight_layout()
plt.savefig('figures/spectra/cosqm_spectral_response_weighted.png')

plt.figure(figsize=(6,3.5))
plt.plot(cosqm_responses['WL'], cr_led[:,0], c='r', linewidth=linewidth, label=f'{cosqm_bands[0]} nm')
plt.plot(cosqm_responses['WL'], cr_led[:,1], c='g', linewidth=linewidth, label=f'{cosqm_bands[1]} nm')
plt.plot(cosqm_responses['WL'], cr_led[:,2], c='b', linewidth=linewidth, label=f'{cosqm_bands[2]} nm')
plt.plot(cosqm_responses['WL'], cr_led[:,3], c='y', linewidth=linewidth, label=f'{cosqm_bands[3]} nm')
plt.grid(b=True)
plt.legend()
plt.xlabel('Wavelenght (nm)')
plt.ylabel('Normalized spectral response')
plt.xlim(400,975)
plt.ylim(0)
plt.tight_layout()
plt.savefig('figures/spectra/cosqm_spectral_response_weighted_LED.png')

plt.figure(figsize=(6,3.5))
plt.plot(cosqm_responses['WL'], sky, linewidth=linewidth, label='HPS+MH+LED')
plt.plot(cosqm_responses['WL'], sky_led, linewidth=linewidth, label='LED')
plt.grid(b=True)
plt.legend(loc='upper right')
plt.xlabel('Wavelenght (nm)')
plt.ylabel('Normalized spectral contribution')
plt.xlim(400,975)
plt.ylim(0)
plt.tight_layout()
plt.savefig('figures/spectra/simulated_sky_spectrum.png')