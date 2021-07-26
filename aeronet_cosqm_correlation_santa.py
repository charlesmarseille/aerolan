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
cosqm_bands = np.array([667, 571, 503, 555])

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
cosqm_window_size = 7

cosqm_mean_count = 1        #number of averages on cosqm data for correlation
cosqm_min_am_hour = 2       #start of dusk cosqm measurements for correlation
cosqm_max_pm_hour = 25      #end of dawn cosqm measurements for correlation
aod_mean_count = 5          #number of averages on aeronet aod data for correlation
aod_max_am_hour = 10        #end of dusk aeronet aod data for correlation
aod_min_pm_hour = 14        #start of dawn aeronet aod data for correlation

normalized_cosqm_znsb_threshold = 1.8
cosqm_aod_threshold = 0.04
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

#milky way filter
print('Filter: milky way angles calculation')
mw_angles = RaDecGal(dt_santa, eloc)
mw_mask = mw_angles[:,1]>mw_min_angle
cosqm_santa_mw = cosqm_santa_cloud[mw_mask]
dt_santa_mw = dt_santa[mw_mask].reset_index(drop=True)

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

#Clouds sliding window filter
print('Filter: clouds sliding window filter')
cosqm_santa_sun = Cloudslidingwindow(cosqm_santa_sun, slide_window_size, slide_threshold)


# plot filtering
# band = 3
# plt.figure(figsize=[7,4], dpi=150)
# plt.scatter(dt_santa, cosqm_santa[:,band], s=10, label='cosqm_santa')
# plt.scatter(dt_santa_mw, cosqm_santa_mw[:,band], s=10, alpha=0.5, label='milky way below '+str(mw_min_angle))
# plt.scatter(dt_santa_moon, cosqm_santa_moon[:,band], s=8, alpha=0.5, label='moon below '+str(moon_min_angle))
# plt.scatter(dt_santa_sun, cosqm_santa_sun[:,band], s=6, label='sun below '+str(sun_min_angle), c='k')
# plt.legend(loc=[0,0])
# plt.ylabel(f'CoSQM {cosqm_bands[band-1]}nm magnitude (mag)')


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
		dd = np.insert(dd, dd.shape[0], padd, axis=0)            #append nan to end to get same shape as input
	return dd

cosqm_santa_sun = Sliding_window_cosqm(cosqm_santa_sun, cosqm_window_size)

print('remove artefact from sliding window filter ZNSB')
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
def second_order(x,a,b,c):
	return a*x**2+b*x+c

hours_float = np.array([ date.hour+                    #WATCH OUT FOR TIMEZONE HERE!
	date.minute/60+
	date.second/3600 for date in d_local ])
hours_float[hours_float>12]-=24

fit_params_c, _ = curve_fit(second_order, hours_float[~np.isnan(c_norm[:,0])], c_norm[~np.isnan(c_norm[:,0])][:,0])
fit_params_r, _ = curve_fit(second_order, hours_float[~np.isnan(c_norm[:,1])], c_norm[~np.isnan(c_norm[:,1])][:,1])
fit_params_g, _ = curve_fit(second_order, hours_float[~np.isnan(c_norm[:,2])], c_norm[~np.isnan(c_norm[:,2])][:,2])
fit_params_b, _ = curve_fit(second_order, hours_float[~np.isnan(c_norm[:,3])], c_norm[~np.isnan(c_norm[:,3])][:,3])
fit_params_y, _ = curve_fit(second_order, hours_float[~np.isnan(c_norm[:,4])], c_norm[~np.isnan(c_norm[:,4])][:,4])
fit_params = np.array([fit_params_c, fit_params_r, fit_params_g, fit_params_b, fit_params_y]).T

#hist 2d
weekdays_str = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday','Sunday']
months_str = ['January', 'Feburary', 'March', 'April', 'May', 'June', 'July', 'August', 'Septembre', 'October', 'November', 'December']
weekdays = np.array([ date.weekday() for date in d_local ])
markers = ['.', '1', '2', '3', '4', 'o', 's']
bands = ['clear', 'red', 'green', 'yellow', 'blue']

xs = np.linspace(hours_float.min()-0.5, hours_float.max()+0.5, 1001)
band = 1
plt.figure(dpi=150,figsize=(7,4))
plt.hist2d(hours_float[np.isfinite(c_norm)[:,band]], c_norm[:,band][np.isfinite(c_norm)[:,band]], 80, cmap='inferno')
plt.hist2d(hours_float[np.isfinite(c_norm)[:,band]], c_norm[:,band][np.isfinite(c_norm)[:,band]], 80, cmap='inferno', norm=LogNorm())
plt.plot(xs, second_order(xs, fit_params[0, band], fit_params[1, band], fit_params[2, band]), label='second order fit 667nm', c='c')
plt.xlabel('Local time from midnight (h)')
plt.ylabel(f'Normalized {cosqm_bands[band-1]}nm ZNSB (mag)')
plt.ylim(-0.5)
cbar = plt.colorbar(label='counts')
cbar.set_ticks(np.arange(1,11,1))
cbar.update_ticks()
cbar.set_ticklabels(np.arange(1,11,1))
plt.savefig(f'figures/trend/normalized_cosqm_{cosqm_bands[band-1]}.png')

# Correct filtered data with fit curve
print('Correct filtered data with trend fits')
cosqm_santa_2nd = np.array([second_order(hours_float, fit_params_c[0], fit_params_c[1], fit_params_c[2]),
	second_order(hours_float, fit_params_r[0], fit_params_r[1], fit_params_r[2]),
	second_order(hours_float, fit_params_g[0], fit_params_g[1], fit_params_g[2]),
	second_order(hours_float, fit_params_b[0], fit_params_b[1], fit_params_b[2]),
	second_order(hours_float, fit_params_y[0], fit_params_y[1], fit_params_y[2])]).T

dt_santa_final = dt_santa_sun
cosqm_santa_final = np.copy(cosqm_santa_sun) - cosqm_santa_2nd
ddays_cosqm = np.array([(date.date()-d[0].date()).days for date in dt_santa_final])
dt_santa_date = dt_santa_final.dt.date
dt_santa_dt = dt_santa_final.dt.to_pydatetime()


#Plot effect of corrected trends on specific nights
band = 3
fig,ax = plt.subplots(dpi=150, figsize=(7,4))
ax.scatter(d_local, cosqm_santa_sun[:,band], label=f'raw CoSQM {cosqm_bands[band-1]}nm ZNSB')
ax.scatter(d_local, cosqm_santa_final[:,band], label='Corrected nightly trend')
ax.set_xlim(pd.Timestamp('2020-05-22 22'), pd.Timestamp('2020-05-23 03'))
ax.set_xlabel('Local time (DST)')
ax.set_ylim(19.40, 19.65)
ax.set_ylabel('ZNSB (mag)')
ax.legend()
fig.savefig(f'figures/trend/correction_single_night_{bands[band]}.png')

c_tot = np.array((c_norm[:,1]+np.mean(cosqm_santa_sun[:,1][np.where((hours_float ==1) | (hours_float ==2))]),
c_norm[:,2]+np.mean(cosqm_santa_sun[:,2][np.where((hours_float ==1) | (hours_float ==2))]),
c_norm[:,3]+np.mean(cosqm_santa_sun[:,3][np.where((hours_float ==1) | (hours_float ==2))]),
c_norm[:,4]+np.mean(cosqm_santa_sun[:,4][np.where((hours_float ==1) | (hours_float ==2))]))).T

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

band = 3
plt.figure()
plt.scatter(np.unique(dt_santa_date), cosqm_am[:,band], label='cosqm_am', s=15)
plt.scatter(np.unique(dt_santa_date), cosqm_pm[:,band], label='cosqm_pm', s=15)
plt.xlabel('days from july 2019')
plt.ylabel(f'CoSQM {cosqm_bands[band-1]}nm Magnitudes (mag)')
plt.legend()



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
wls2 = np.ones((aod_am.shape[0], 4))*np.array([675, 500, 500, 500])     # CE318-T sunphoto center wavelenghts per filter

def aod_from_angstrom(aod2, wls1, wls2, alpha):
	return aod2*(wls1/wls2)**-alpha

def angstrom_from_aod(arr, bands, ind1, ind2):
	return -np.log(arr[:,ind1]/arr[:,ind2])/np.log(bands[ind1]/bands[ind2])

# Correct computed aod with matching band nominal wavelength
print('correct computed aod with matching band nominal wavelength')
aod_am_corr = aod_from_angstrom(aod_am, wls1, wls2, angstrom_am)
aod_pm_corr = aod_from_angstrom(aod_pm, wls1, wls2, angstrom_pm)

band=3
plt.figure()
plt.scatter(np.unique(dt_santa_date), aod_am_corr[:,band], label='aod_am_corr', s=15)
plt.scatter(np.unique(dt_santa_date), aod_pm_corr[:,band], label='aod_pm_corr', s=15)
plt.xlabel('days from july 2019')
plt.ylabel(f'CoSQM {cosqm_bands[band-1]}nm Magnitudes (mag)')
plt.legend()


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


# Threshold clip on correlated AOD (crazy AOD values from imperfect AOD-ZNSB fit)
print('threshold clip on correlated cosqm aod (for aod values showing 50% error)')
cosqm_aod_r[cosqm_aod_r<cosqm_aod_threshold] = np.nan
cosqm_aod_g[cosqm_aod_g<cosqm_aod_threshold] = np.nan
cosqm_aod_b[cosqm_aod_b<cosqm_aod_threshold] = np.nan
cosqm_aod_y[cosqm_aod_y<cosqm_aod_threshold] = np.nan
cosqm_aod_all = np.array((cosqm_aod_r, cosqm_aod_g, cosqm_aod_b, cosqm_aod_y)).T

# fit AE for each 4 points cosqm measurement
print('compute AE fit for each cosqm 4 values measurements')
def FitAe(wl, k, alpha):
	return k*wl**-alpha

def fit_except(data_array):                         # Some cosqm aod values do not converge while fitting
	try:
		return curve_fit(FitAe, cosqm_bands, data_array, p0=(1000, 0.5))[0]
	except RuntimeError:
		print(f"Error - curve_fit failed: cosqm aod values {data_array}")
		return np.array((None, None), dtype=object)

ae_fit_params = np.zeros((cosqm_aod_all.shape[0], 2))
ae_fit_params[:] = np.nan
cosqm_nonnans = ~np.isnan(cosqm_aod_all).any(axis=1)
ae_fit_params[cosqm_nonnans] = np.vstack([fit_except(cosqm_aods) for cosqm_aods in cosqm_aod_all[cosqm_nonnans]])
ae_fit_succeed = ~np.isnan(ae_fit_params[:,0])

cosqm_ae = np.array([(fit1, fit2) for fit1, fit2 in ae_fit_params])


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
xs = np.arange(17,21,0.01)
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
ax[0, 0].text(0.05,0.5, f'{cosqm_bands[0]} nm\na,b = {str(corr_single_round[0])[1:-1]}', horizontalalignment='left', verticalalignment='center', transform=ax[0,0].transAxes)
ax[0, 1].text(0.05,0.5, f'{cosqm_bands[1]} nm\na,b = {str(corr_single_round[1])[1:-1]}', horizontalalignment='left', verticalalignment='center', transform=ax[0,1].transAxes)
ax[1, 0].text(0.05,0.5, f'{cosqm_bands[2]} nm\na,b = {str(corr_single_round[2])[1:-1]}', horizontalalignment='left', verticalalignment='center', transform=ax[1,0].transAxes)
ax[1, 1].text(0.05,0.5, f'{cosqm_bands[3]} nm\na,b = {str(corr_single_round[3])[1:-1]}', horizontalalignment='left', verticalalignment='center', transform=ax[1,1].transAxes)
ax[0, 0].set_xlim(16.25,20.6)
ax[0, 0].set_ylim(0.01)
ax[0, 0].legend(loc='lower left', prop={'size': 8})
plt.savefig('figures/correlation/santa/correlation_single_fit_santa.png')


#single fit function: continuity 2020-02-21
msize=0.5
fig, ax = plt.subplots(2,2, sharex=True, sharey=True, dpi=150, figsize=(10,6))
#plt.setp(ax, xticks=[pd.Timestamp('2020-02-22'), pd.Timestamp('2020-02-27'), pd.Timestamp('2020-03-03')], xticklabels=['2020-02-22', '2020-02-27', '2020-03-03'])
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
ax[0,0].set_xlim(pd.Timestamp('2020-02-20'), pd.Timestamp('2020-03-04'))
ax[0,0].set_ylim(0.01,5.3)
fig.text(0.04, 0.5, 'Correlated AOD', va='center', rotation='vertical', fontsize=10)
ax[0,0].legend()
ax[0, 0].text(0.5,0.1, f'{cosqm_bands[0]} nm', horizontalalignment='center', verticalalignment='center', transform=ax[0,0].transAxes, fontsize=10)
ax[0, 1].text(0.5,0.1, f'{cosqm_bands[1]} nm', horizontalalignment='center', verticalalignment='center', transform=ax[0,1].transAxes, fontsize=10)
ax[1, 0].text(0.5,0.1, f'{cosqm_bands[2]} nm', horizontalalignment='center', verticalalignment='center', transform=ax[1,0].transAxes, fontsize=10)
ax[1, 1].text(0.5,0.1, f'{cosqm_bands[3]} nm', horizontalalignment='center', verticalalignment='center', transform=ax[1,1].transAxes, fontsize=10)
#plt.savefig('figures/continuity/continuity_aod_20200220.png')


#single fit function: continuity Angstrom 2020-02-21
fig, ax = plt.subplots(1,1, dpi=150, figsize=(10,6))
#plt.setp(ax, xticks=[pd.Timestamp('2020-02-22'), pd.Timestamp('2020-02-27'), pd.Timestamp('2020-03-03')], xticklabels=['2020-02-22', '2020-02-27', '2020-03-03'])
ax.scatter(dt_aod, data_angstrom, s=0.5, label='CE318-T derived 440-679nm AE')
ax.scatter(dt_santa_corr, cosqm_ae[:,1], s=0.5, label=f'CoSQM derived fitted AE')
ax.set_xlim(pd.Timestamp('2020-02-20'), pd.Timestamp('2020-03-04'))
#ax.set_ylim(-1.34,3.73)
ax.set_ylabel('503-667nm Angstrom exponent')
ax.legend()
#ax.set_yscale('log')
#plt.savefig('figures/continuity/continuity_angstrom_20200220.png')


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
plt.setp(ax, xticks=[pd.Timestamp('2020-05-22'), pd.Timestamp('2020-05-25'), pd.Timestamp('2020-05-28')], xticklabels=['2020-05-22', '2020-05-25', '2020-05-28'])
ax.scatter(dt_aod, data_angstrom, s=0.5, label='CE318-T derived 440-679nm AE')
ax.scatter(dt_santa_corr, cosqm_ae[:,1], s=0.5, label='CoSQM derived fitted AE')
ax.set_xlim(pd.Timestamp('2020-05-21'), pd.Timestamp('2020-05-29'))
ax.set_ylim(-1.34,3.73)
ax.set_ylabel('503-667nm Angstrom exponent')
ax.legend()
#ax.set_yscale('log')
plt.savefig('figures/continuity/continuity_angstrom_20200521.png')


#single fit function: continuity -> variance for uncertainty
fig, ax = plt.subplots(2,2, sharex=True, sharey=True, dpi=150, figsize=(10,6))
plt.setp(ax, xticks=[pd.Timestamp('2020-05-23 15'), pd.Timestamp('2020-05-24 07')], xticklabels=['2020-05-23 15 ', '2020-05-24 07'])
ax[0,0].scatter(dt_aod, data_aod[:,0], s=msize, label='CE318-T')
ax[0,0].scatter(dt_santa_corr, cosqm_aod_r, s=msize, label='CoSQM derived AOD')
ax[0,1].scatter(dt_aod, data_aod[:,1], s=msize)
ax[0,1].scatter(dt_santa_corr, cosqm_aod_g, s=msize)
ax[1,0].scatter(dt_aod, data_aod[:,2], s=msize)
ax[1,0].scatter(dt_santa_corr, cosqm_aod_b, s=msize)
ax[1,1].scatter(dt_aod, data_aod[:,3], s=msize)
ax[1,1].scatter(dt_santa_corr, cosqm_aod_y, s=msize)
#ax[0, 0].set_yscale('log')
ax[0,0].set_xlim(pd.Timestamp('2020-05-23 12'), pd.Timestamp('2020-05-24 10'))
ax[0,0].set_ylim(0.12,0.22)
#fig.text(0.5, 0.04, 'Date', ha='center', fontsize=15)
fig.text(0.04, 0.5, 'Correlated AOD', va='center', rotation='vertical', fontsize=10)
ax[0,0].legend(loc='upper center')
ax[0, 0].text(0.5,0.10, f'{cosqm_bands[0]} nm', horizontalalignment='center', verticalalignment='center', transform=ax[0,0].transAxes, fontsize=10)
ax[0, 1].text(0.5,0.10, f'{cosqm_bands[1]} nm', horizontalalignment='center', verticalalignment='center', transform=ax[0,1].transAxes, fontsize=10)
ax[1, 0].text(0.5,0.10, f'{cosqm_bands[2]} nm', horizontalalignment='center', verticalalignment='center', transform=ax[1,0].transAxes, fontsize=10)
ax[1, 1].text(0.5,0.10, f'{cosqm_bands[3]} nm', horizontalalignment='center', verticalalignment='center', transform=ax[1,1].transAxes, fontsize=10)
plt.savefig('figures/continuity/continuity_aod_20200524_variance.png')


#single fit function: continuity Angstrom -> variance uncertainty
fig, ax = plt.subplots(1,1, dpi=150, figsize=(8,5))
plt.setp(ax, xticks=[pd.Timestamp('2020-05-22 15'), pd.Timestamp('2020-05-24 00'), pd.Timestamp('2020-05-25 10')], xticklabels=['2020-05-22 15 ', '2020-05-24 00', '2020-05-25 12'])
ax.scatter(dt_aod, data_angstrom, s=msize, label='CE318-T derived 440-679nm AE')
ax.scatter(dt_santa_corr, cosqm_ae[:,1], s=msize, label='CoSQM derived 503-667nm AE')
ax.set_xlim(pd.Timestamp('2020-05-22 10'), pd.Timestamp('2020-05-25 12'))
ax.set_ylim(-0.5,0.798)
#ax.set_ylim(-0.4,0.5)
#ax.set_yscale('log')
#fig.text(0.5, 0.04, 'Date', ha='center', fontsize=15)
fig.text(0.04, 0.5, '503-571nm Angstrom exponent', va='center', rotation='vertical', fontsize=10)
plt.savefig('figures/continuity/continuity_angstrom_20200524_variance.png')