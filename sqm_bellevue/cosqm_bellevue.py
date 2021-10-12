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

## sqm Data load function, returns data and dates in tuple
def LoadData(path,cache={}):
	if path in cache:
		return cache[path]
	else:
		try:
			data = pd.read_csv(path, delimiter=';', names=['utc', 'local_time', 'temp', 'volt', 'mpsas', 'meas_type'], header=37)
			data.utc = pd.to_datetime(data.utc, utc=True)
			data.local_time = pd.to_datetime(data.local_time, utc=False)
			cache[path] = data
			return data
		except:
			print('****error****', path)

# Following is for CoSQM

	# #print (path)
	# if path in cache:
	# 	return cache[path]
	# else:
	# 	try:
	# 		data = pd.read_csv(path, delimiter=' ', names=['Date', 'Time', 'Lat', 'Lon', 'Alt', 'Temp', 'Wait', 'Sqm0', 'Sqm1', 'Sqm2', 'Sqm3', 'Sqm4', 'Sbcals0', 'Sbcals1', 'Sbcals2', 'Sbcals3', 'Sbcals4'])
	# 		data['Datetime'] = pd.to_datetime(data['Date']+'T'+data['Time'], utc=True)        
	# 		cache[path] = data
	# 		return data
	# 	except:
	# 		print('********error*********', path)


##sqm TEIDE data load function ***BUG IN sqm DATA FROM WGET COMMAND***
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
bell_loc = earth + Topos('45.377505557984975N', '71.90920834543029W')

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
# sqm DATA
# 
# Data taken from Martin Aubé's server: http://dome.obsand.org:2080/DATA/sqm-Network/
#
# Data format: Date(YYYY-MM-DD), time(HH:MM:SS),
#             Latitude, Longitude, Elevation, SQM temperature, integration time, 
#             C(mag), R(mag), G(mag), B(mag), Y(mag),
#             C(watt),R(watt),G(watt),B(watt),Y(watt)  
#############################

#######
## Variables
path_sqm='data/'
sqm_bands = np.array([652, 599, 532, 588])
bands = ['clear', 'red', 'green', 'blue', 'yellow']

loc = bell_loc
loc_lat = 45.377505557984975
loc_lon = -71.90920834543029
eloc = EarthLocation(lat=loc_lat, lon=loc_lon)
loc_str = 'bellevue'
slide_threshold = 0.1
slide_window_size = 15
mw_min_angle = 30
moon_min_angle = -2
sun_min_angle = -18
sqm_window_size = 7
normalized_sqm_znsb_threshold = 1.8
#######


## find all paths of files in root directory
print('Load sqm data')
paths_sqm = sorted(glob(path_sqm+"*.dat"))
files = pd.concat([LoadData(path) for path in paths_sqm], ignore_index=True)
sqm_bell = np.array([files.mpsas, files.mpsas]).T
sqm_bell_raw = sqm_bell.copy()
dt_bell_raw = files.utc
dt_local_raw = files.local_time

#remove non datetime errors in sqm files (NaT)
sqm_bell = sqm_bell[~pd.isnull(dt_bell_raw)]
dt_bell = dt_bell_raw[~pd.isnull(dt_bell_raw)]
dt_local = dt_local_raw[~pd.isnull(dt_bell_raw)]


## Remove zeros from sqm measurements (bugs from instruments)
print('Cleaning: remove zeros from sqm measurements')
zeros_mask = (sqm_bell!=0).all(1)
sqm_bell = sqm_bell[zeros_mask]
dt_bell = dt_bell[zeros_mask]
dt_local = dt_local[zeros_mask]

#milky way filter
print('Filter: milky way angles calculation')
mw_angles = RaDecGal(dt_bell, eloc)
mw_mask = mw_angles[:,1]>mw_min_angle
sqm_bell_mw = sqm_bell[mw_mask]
dt_bell_mw = dt_bell[mw_mask].reset_index(drop=True)
dt_local_mw = dt_local[mw_mask].reset_index(drop=True)

## Compute moon angles for each timestamp in sqm data
print('Filter: moon angles calculation')
moon_angles = ObjectAngle(dt_bell_mw, moon, loc)
moon_mask = moon_angles<moon_min_angle
sqm_bell_moon = sqm_bell_mw[moon_mask]
dt_bell_moon = dt_bell_mw[moon_mask].reset_index(drop=True)
dt_local_moon = dt_local_mw[moon_mask].reset_index(drop=True)

## Compute sun angles for each timestamp in sqm data
print('Filter: sun_angles calculation')
sun_angles = ObjectAngle(dt_bell_moon, sun, bell_loc)
sun_mask = sun_angles<sun_min_angle
sqm_bell_sun = sqm_bell[sun_mask]
dt_bell_sun = dt_bell[sun_mask].reset_index(drop=True)
dt_local_sun = dt_local[sun_mask].reset_index(drop=True)

#Clouds sliding window filter
print('Filter: clouds sliding window filter')
sqm_bell_cloud = Cloudslidingwindow(sqm_bell_sun, slide_window_size, slide_threshold)
dt_bell_cloud = dt_bell_sun[~np.isnan(sqm_bell_cloud[:,0])]
dt_local_cloud = dt_local_sun[~np.isnan(sqm_bell_cloud[:,0])]
sqm_bell_cloud = sqm_bell_cloud[~np.isnan(sqm_bell_cloud[:,0])]

# plot filtering
print('make filtering plot')
band = 0
plt.figure(figsize=[14,11], dpi=200)
plt.scatter(dt_local, sqm_bell[:,band], s=10, label='Raw SQM data')
plt.scatter(dt_local_mw, sqm_bell_mw[:,band], c='r', s=10, label='milky way below '+str(mw_min_angle))
plt.scatter(dt_local_moon, sqm_bell_moon[:,band], c='m', s=8, label='moon below '+str(moon_min_angle))
plt.scatter(dt_local_sun, sqm_bell_sun[:,band], c='y', s=6, label='sun below '+str(sun_min_angle))
plt.scatter(dt_local_cloud, sqm_bell_cloud[:,band], c='k', s=10, label='cloud screening')
plt.legend(loc=[0,0])
plt.ylabel('sqm magnitude (MPSAS)')
plt.xlabel('Local date and time')
plt.tight_layout()
#plt.savefig('sqm_bellevue/all_data.png')


plt.figure(figsize=[14,11], dpi=200)
plt.scatter(dt_local, sqm_bell[:,band], s=10, label='Raw SQM data')
plt.scatter(dt_local_mw, sqm_bell_mw[:,band], c='r', s=10, label='milky way below '+str(mw_min_angle))
plt.scatter(dt_local_moon, sqm_bell_moon[:,band], c='m', s=8, label='moon below '+str(moon_min_angle))
plt.scatter(dt_local_sun, sqm_bell_sun[:,band], c='y', s=6, label='sun below '+str(sun_min_angle))
plt.scatter(dt_local_cloud, sqm_bell_cloud[:,band], c='k', s=10, label='cloud screening')
plt.ylim(15.5, 19.3)
plt.xlim(pd.Timestamp('2021-04-13 16:00'), pd.Timestamp('2021-04-15 08:00'))
plt.legend(loc=[0,0])
plt.ylabel('sqm magnitude (MPSAS)')
plt.xlabel('Local date and time')
plt.tight_layout()
#plt.savefig('sqm_bellevue/zoom_filtering.png')


day=14
datestr = '04/%s/21' %str(day)
datestr1 = '02/%s/21' %str(day+1)

sun_angles = ObjectAngle(dt_bell_moon, sun, bell_loc)
sun_mask_0 = sun_angles<0
dt_local_sun0 = dt_local[sun_mask_0].reset_index(drop=True)
dates0 = np.array([dt.strftime(val.date(), format='%D') for val in dt_local_sun0])
hours0 = np.array([val.hour for val in dt_local_sun0])
dt_sun0 = dt_local_sun0[(dates0 == datestr) & (hours0 > 12)].reset_index(drop=True)[0]
print(dt_sun0)

sun_mask = sun_angles<-18
dt_local_sun = dt_local[sun_mask].reset_index(drop=True)
dates = np.array([dt.strftime(val.date(), format='%D') for val in dt_local_sun])
hours = np.array([val.hour for val in dt_local_sun])
try:
	dt_sun = dt_local_sun[(dates == datestr) & (hours > 12)].reset_index(drop=True)[0]
except:
	dt_sun = dt_local_sun[(dates == datestr1) & (hours <12)].reset_index(drop=True)[0]
print(dt_sun)

moon_angles = ObjectAngle(dt_bell, moon, loc)
moon_mask = moon_angles<-18
dt_local_moon = dt_local[moon_mask].reset_index(drop=True)
dates = np.array([dt.strftime(val.date(), format='%D') for val in dt_local_moon])
hours = np.array([val.hour for val in dt_local_moon])
dt_moon = dt_local_moon[(dates == datestr) & (hours > 12)].reset_index(drop=True)[0]
print(dt_moon)

plt.figure(figsize=[14,11], dpi=200)
plt.scatter(dt_local_raw+td(hours=1), sqm_bell_raw[:,band], s=1, label='Raw SQM data')
#plt.plot(dt_local_raw, sqm_bell_raw[:,band], linewidth=0.1, label='Raw SQM data')
plt.axvline(dt_sun0+td(hours=1), linestyle='--', c='r', linewidth=0.6, label='Sun below horizon')
plt.axvline(dt_sun+td(hours=1), linestyle='--', c='k', linewidth=0.6, label='Sun 18$^{\circ}$ below horizon')
plt.axvline(dt_moon+td(hours=1), linestyle='--', c='purple', linewidth=0.6, label='Moon 18$^{\circ}$ below horizon')
plt.ylim(5, 20)
plt.xlim(pd.Timestamp('2021-04-14 19:00'), pd.Timestamp('2021-04-15 06:15'))
plt.legend(loc='lower right')
plt.ylabel('sqm magnitude (MPSAS)')
plt.xlabel('Local date and time')
plt.tight_layout()
plt.savefig('raw_night.png')


plt.figure(figsize=[14,11], dpi=200)
plt.scatter(dt_local_raw+td(hours=1), sqm_bell_raw[:,band], s=1, label='Raw SQM data')
#plt.plot(dt_local_raw, sqm_bell_raw[:,band], linewidth=0.1, label='Raw SQM data')
plt.axvline(dt_sun0+td(hours=1), linestyle='--', c='r', linewidth=0.6, label='Sun below horizon')
plt.axvline(dt_sun+td(hours=1), linestyle='--', c='k', linewidth=0.6, label='Sun 18$^{\circ}$ below horizon')
plt.axvline(dt_moon+td(hours=1), linestyle='--', c='purple', linewidth=0.6, label='Moon 18$^{\circ}$ below horizon')
plt.ylim(5, 20)
plt.xlim(pd.Timestamp('2021-04-14 19:00'), pd.Timestamp('2021-04-15 06:15'))
plt.legend(loc='upper right')
plt.ylabel('sqm magnitude (MPSAS)')
plt.xlabel('Local date and time')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('raw_night_invert.png')





















#hist2d#
#raw without clouds
hours_float_raw = np.array([ date.hour+                    #WATCH OUT FOR TIMEZONE HERE!
	date.minute/60+
	date.second/3600 for date in dt_bell ])
hours_float_raw[hours_float_raw>12]-=24

# plt.figure(figsize=[12,8])
# plt.hist2d(hours_float_raw, sqm_bell[:,1], 200, cmap='inferno')
# plt.hist2d(hours_float_raw, sqm_bell[:,1], 200, cmap='inferno', norm=LogNorm())
# plt.ylim(15,21)
# plt.title('ZNSB - no filter - clear')
# plt.xlabel('hour')
# plt.ylabel('sqm magnitude')

#clouds filter
hours_float_diff = np.array([ date.hour+                    #WATCH OUT FOR TIMEZONE HERE!
	date.minute/60+
	date.second/3600 for date in dt_bell ])
hours_float_diff[hours_float_diff>12]-=24

#plt.figure(figsize=[12,8])
#plt.hist2d(hours_float_diff[np.isfinite(sqm_bell_diff)[:,0]], sqm_bell_diff[:,0][np.isfinite(sqm_bell_diff)[:,0]], 200, cmap='inferno')
#plt.hist2d(hours_float_diff[np.isfinite(sqm_bell_diff)[:,0]], sqm_bell_diff[:,0][np.isfinite(sqm_bell_diff)[:,0]], 200, cmap='inferno', norm=LogNorm())
#plt.ylim(15,21)
#plt.title('ZNSB - filter: clouds+variance - clear')
#plt.xlabel('hour')
#plt.ylabel('sqm magnitude')

#moon filter
hours_float_moon = np.array([ date.hour+                    #WATCH OUT FOR TIMEZONE HERE!
	date.minute/60+
	date.second/3600 for date in dt_bell_moon ])
hours_float_moon[hours_float_moon>12]-=24

# plt.figure(figsize=[12,8])
# plt.hist2d(hours_float_moon[np.isfinite(sqm_bell_moon)[:,0]], sqm_bell_moon[:,0][np.isfinite(sqm_bell_moon)[:,0]], 200, cmap='inferno')
# plt.hist2d(hours_float_moon[np.isfinite(sqm_bell_moon)[:,0]], sqm_bell_moon[:,0][np.isfinite(sqm_bell_moon)[:,0]], 200, cmap='inferno', norm=LogNorm())
# plt.ylim(15,21)
# plt.title('ZNSB - filter: moon - clear')
# plt.xlabel('hour')
# plt.ylabel('sqm magnitude')

#sun filter
hours_float_sun = np.array([ date.hour+                    #WATCH OUT FOR TIMEZONE HERE!
	date.minute/60+
	date.second/3600 for date in dt_bell_sun ])
hours_float_sun[hours_float_sun>12]-=24

# plt.figure(figsize=[12,8])
# plt.hist2d(hours_float_sun[np.isfinite(sqm_bell_sun)[:,1]], sqm_bell_sun[:,1][np.isfinite(sqm_bell_sun)[:,1]], 200, cmap='inferno')
# plt.hist2d(hours_float_sun[np.isfinite(sqm_bell_sun)[:,1]], sqm_bell_sun[:,1][np.isfinite(sqm_bell_sun)[:,1]], 200, cmap='inferno', norm=LogNorm())
# plt.ylim(15,21)
# plt.title('ZNSB - filter: sun - clear')
# plt.xlabel('hour')
# plt.ylabel('sqm magnitude')

dts_event = np.array([dt_bell.dt.date, dt_bell_cloud.dt.date, dt_bell_mw.dt.date, dt_bell_moon.dt.date, dt_bell_sun.dt.date])


fsize=10
sizem = 1
fig,ax = plt.subplots(figsize=(7,4), dpi=150)
ax.scatter(np.unique(dt_bell), 5*np.ones(np.unique(dt_bell).shape[0]), label='Raw', s=sizem, marker='s', alpha=0.5)
ax.scatter(np.unique(dt_bell_cloud), 4*np.ones(np.unique(dt_bell_cloud).shape[0]), label='Clouds', s=sizem, marker='s', alpha=0.5)
ax.scatter(np.unique(dt_bell_mw), 3*np.ones(np.unique(dt_bell_mw).shape[0]), label='Milky Way', s=sizem, marker='s', alpha=0.5)
ax.scatter(np.unique(dt_bell_moon), 2*np.ones(np.unique(dt_bell_moon).shape[0]), label='Moon', s=sizem, marker='s', alpha=0.5)
ax.scatter(np.unique(dt_bell_sun), np.ones(np.unique(dt_bell_sun).shape[0]), label='Sun', s=sizem, marker='s', alpha=0.5)
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
plt.savefig('figures/filtering_eventplot.png')



# # Make eventplot to graph filtering of dates
# lineoffsets1 = np.array([3, 6, 9, 12, 15])
# linelengths1 = np.ones(5)*2

# sizem = 2
# fsize = 7
# dts = np.array([pd.Timestamp(date).timestamp() for date in dt_bell.dt.date])
# dts_cloud = np.copy(dts)
# dts_cloud[np.isnan(sqm_bell_cloud[:,0])] = False
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
print('Remove invalid sqm data (R channel higher than C), and visual cloud screening')
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

dates_sqm_str = np.array([dt.strftime(date, '%Y-%m-%d') for date in dt_bell_sun])
dates_color_mask = np.ones(dates_sqm_str.shape[0], dtype=bool)
dates_color_mask[np.isin(dates_sqm_str, dates_color_str)] = False

dt_bell_sun = dt_bell_sun[dates_color_mask]
sqm_bell_sun = sqm_bell_sun[dates_color_mask]
sqm_bell_sun_unfiltered = np.copy(sqm_bell_sun)

d = np.copy(dt_bell_sun)

ddays_sqm = np.array([(date.date()-d[0].date()).days for date in d])
hours = np.array([date.hour for date in d])


# Sliding window filter on all bands to smooth out rapid variations (mainly caused by R channel having too high variance since resolution of sqm)
print('Sliding window filter on all sqm bands')
def Sliding_window_sqm(data_array, window_size):
	da = np.copy(data_array)
	dslide = np.lib.stride_tricks.sliding_window_view(data_array, window_shape = window_size, axis=0).copy() #start and end values lost. size of array is data_array.shape[0]-2
	dd = np.nanmean(dslide, axis=2)
	padd = np.full([1, da.shape[1]], np.nan)
	for i in range(window_size//2):
		dd = np.insert(dd, 0, padd, axis=0)                     #append nan to start to get same shape as input
		dd = np.insert(dd, dd.shape[0], padd, axis=0)      	      #append nan to end to get same shape as input
	return dd

sqm_bell_sun = Sliding_window_sqm(sqm_bell_sun, sqm_window_size)
np.savetxt('sqm_filtered_bell.csv', np.array([dt_bell_sun[~np.isnan(sqm_bell_sun[:,0])], sqm_bell_sun[:,0][~np.isnan(sqm_bell_sun[:,0])], sqm_bell_sun[:,1][~np.isnan(sqm_bell_sun[:,0])], sqm_bell_sun[:,2][~np.isnan(sqm_bell_sun[:,0])], sqm_bell_sun[:,3][~np.isnan(sqm_bell_sun[:,0])], sqm_bell_sun[:,4][~np.isnan(sqm_bell_sun[:,0])]]).T, fmt="%s")


#print('remove artefact from sliding window filter ZNSB')
for day in np.unique(ddays_sqm):
	d_mask = np.zeros(ddays_sqm.shape[0], dtype=bool)
	d_mask[(ddays_sqm == day) & (hours < 12)] = True
	d_mask[(ddays_sqm == day-1) & (hours > 12)] = True 
	try:
		sqm_bell_sun[np.where(d_mask==True)[0][0]] = np.nan
	except:
		print('0', day)
		pass
	try:
		sqm_bell_sun[np.where(d_mask==True)[0][-1]] = np.nan
	except:
		print('-1', day)
		pass

# Plot variance filter effect on data
colors=['r', 'g', 'b', 'y']
plt.figure(dpi=150, figsize=(7,4))
for i in (0,2):
	plt.scatter(dt_bell_sun, sqm_bell_sun_unfiltered[:,i+1], c=colors[i], s=8, label=f'{sqm_bands[i]}nm')
	plt.scatter(dt_bell_sun, sqm_bell_sun[:,i+1], c='k', marker='+', s=8)
plt.xlabel('Time of measurement (UTC)')
plt.ylabel('sqm ZNSB (MPSAS)')
plt.ylim(19.6)
plt.legend()
plt.xticks(ticks=[pd.Timestamp(f'2020-01-24 0{i+1}') for i in range(6)], labels=[str(i+1) for i in range(6)])
plt.xlim(pd.Timestamp('2020-01-24 02:30'), pd.Timestamp('2020-01-24 06:33'))
plt.tight_layout()
plt.savefig('figures/sliding_window_sqm.png')

#Find variance before and after sqm sliding window 
#inds = np.where((dt_bell_date.values == pd.Timestamp('2020-01-24')) | (dt_bell_date.values == pd.Timestamp('2020-01-25')))[0]
#sqm_bell_sun_unfiltered_var = np.nanvar(sqm_bell_sun_unfiltered[inds], axis=0)
#sqm_bell_sun_var = np.nanvar(sqm_bell_sun[inds], axis=0)