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
            data = pd.read_csv(path, delimiter=' ', error_bad_lines=False, names=['Date', 'Time', 'Lat', 'Lon', 'Alt', 'Temp', 'Wait', 'Sqm0', 'Sqm1', 'Sqm2', 'Sqm3', 'Sqm4', 'Sbcals0', 'Sbcals1', 'Sbcals2', 'Sbcals3', 'Sbcals4'])
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
teide_loc = earth + Topos('28.270233167N', '16.638741000W')
obs_loc = earth + Topos('28.3004167N', '16.512230555555554W')


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

## COSQM izana
path_cosqm='cosqm_obs/data/'
loc = obs_loc
loc_lat = 28.270233167
loc_lon = -16.638741000
eloc = EarthLocation(lat=loc_lat, lon=loc_lon)
loc_str = 'obs'
slide_threshold = 0.15
slide_window_size = 5
mw_min_angle = 40
moon_min_angle = -2
sun_min_angle = -15

path_aod = 'cosqm_obs/20190901_20210630_Izana.lev10'


## find all paths of files in root directory
paths_cosqm = sorted(glob(path_cosqm+"*/*/*.txt"))
files = pd.concat([LoadData(path) for path in paths_cosqm[70:]], ignore_index=True)
cosqm_obs = files[['Sqm0', 'Sqm1', 'Sqm2', 'Sqm3', 'Sqm4']].values
dt_obs = files['Datetime']

#remove parsing errors due to bad cosqm logging
#for i in range(5):
#    dtypes = np.array([type(data) for data in cosqm_obs[:,i]])
#    cosqm_obs = cosqm_obs[dtypes == float]
#    dt_obs = dt_obs[dtypes == float]

#remove non datetime errors in cosqm files (NaT)
cosqm_obs = cosqm_obs[~pd.isnull(dt_obs)]
dt_obs = dt_obs[~pd.isnull(dt_obs)]

### if day is wanted: dt_obs_day = dt_obs.dt.day
### if interval wanted: inds = (dt_obs.dt.date == np.array('2020-01-21',dtype='datetime64[D]')) | (dt_obs.dt.date == np.array('2020-01-22',dtype='datetime64[D]'))
### work on specific night interval: inds = (dt_obs.values > np.array('2020-01-21T12',dtype='datetime64[ns]')) 
##   & (dt_obs.values < np.array('2020-01-22T12',dtype='datetime64[ns]'))

## Remove zeros from cosqm measurements (bugs from instruments)
zeros_mask = ((cosqm_obs>1).all(1)) & ((cosqm_obs<25).all(1))
cosqm_obs = np.array(cosqm_obs[zeros_mask], dtype=float)
dt_obs = dt_obs[zeros_mask]



## Cloud removal with differential between points (if difference between 2 measurements is bigger than threshold, remove data)

print('cloud sliding window calculation')
cosqm_obs_diff = Cloudslidingwindow(cosqm_obs, slide_window_size, slide_threshold)
# plt.scatter(dt_obs, cosqm_obs_diff[:,0], s=10, c='k', label='derivative cloud screening')

#milky way filter
print('milky way angles calculation')
mw_angles = RaDecGal(dt_obs, eloc)
mw_mask = np.abs(mw_angles[:,1])>mw_min_angle
cosqm_obs_mw = cosqm_obs_diff[mw_mask]
dt_obs_mw = dt_obs[mw_mask].reset_index(drop=True)

## Compute moon angles for each timestamp in COSQM data
print('moon angles calculation')
moon_angles = ObjectAngle(dt_obs_mw, moon, loc)
#np.savetxt('cosqm_obs_moon_angles.txt', moon_angles)             #Save angles to reduce ulterior computing time
#moon_angles = np.loadtxt('cosqm_'+loc_str+'_moon_angles.txt')                  #Load already computed angles

## Mask values for higher angle than -18deg (astro twilight)

moon_mask = moon_angles<moon_min_angle
cosqm_obs_moon = cosqm_obs_mw[moon_mask]
dt_obs_moon = dt_obs_mw[moon_mask].reset_index(drop=True)
#dates_days_since_start = np.array([(dt.fromtimestamp(date)-dt.fromtimestamp(dates[0])).days+1 for date in dates])

## Compute sun angles for each timestamp in COSQM data
print('sun angles calculation')
sun_angles = ObjectAngle(dt_obs_moon, sun, obs_loc)
#np.savetxt('cosqm_'+loc_str+'_sun_angles.txt', sun_angles)
#sun_angles = np.loadtxt('cosqm_'+loc_str+'_sun_angles.txt')                    #Load already computed angles
sun_mask = sun_angles<sun_min_angle

cosqm_obs_sun = cosqm_obs_moon[sun_mask]
dt_obs_sun = dt_obs_moon[sun_mask].reset_index(drop=True)

## filter data when milky way is in the visible sky (5 degrees above horizon)
# (todo)

# plot filtering
plt.figure(figsize=[16,9])
plt.scatter(dt_obs, cosqm_obs[:,0], s=30, label='cosqm_obs')
plt.scatter(dt_obs_mw, cosqm_obs_mw[:,0], s=30, alpha=0.5, label='milky way below '+str(mw_min_angle))
plt.scatter(dt_obs_moon, cosqm_obs_moon[:,0], s=30, alpha=0.5, label='moon below '+str(moon_min_angle))
plt.scatter(dt_obs_sun, cosqm_obs_sun[:,0], s=15, label='sun below '+str(sun_min_angle), c='k')
#plt.scatter(dt_obs_moon, moon_angles[moon_mask]/10+23, s=10, label='moon angle')
#plt.scatter(dt_obs_moon_sun, sun_angles[sun_mask]/10+23, s=10, label='sun angle')
plt.legend(loc=[0,0])
plt.title(f'ZNSB {loc_str} - Filtered clear band')
plt.xlabel('date')
plt.ylabel('CoSQM magnitude (mag)')



#hist2d#
#raw without clouds
hours_float_raw = np.array([ date.hour+                    #WATCH OUT FOR TIMEZONE HERE!
    date.minute/60+
    date.second/3600 for date in dt_obs ])
hours_float_raw[hours_float_raw>12]-=24

plt.figure(figsize=[12,8])
plt.hist2d(hours_float_raw, cosqm_obs[:,0], 200, cmap='inferno')
plt.hist2d(hours_float_raw, cosqm_obs[:,0], 200, cmap='inferno', norm=LogNorm())
plt.title(f'{loc_str} - ZNSB - no filter - clear')
plt.xlabel('hour')
plt.ylabel('CoSQM magnitude')

#clouds filter
hours_float_diff = np.array([ date.hour+                    #WATCH OUT FOR TIMEZONE HERE!
    date.minute/60+
    date.second/3600 for date in dt_obs ])
hours_float_diff[hours_float_diff>12]-=24

plt.figure(figsize=[12,8])
plt.hist2d(hours_float_diff[np.isfinite(cosqm_obs_diff)[:,0]], cosqm_obs_diff[:,0][np.isfinite(cosqm_obs_diff)[:,0]], 200, cmap='inferno')
plt.hist2d(hours_float_diff[np.isfinite(cosqm_obs_diff)[:,0]], cosqm_obs_diff[:,0][np.isfinite(cosqm_obs_diff)[:,0]], 200, cmap='inferno', norm=LogNorm())
plt.title(f'{loc_str} - ZNSB - filter: clouds+variance - clear')
plt.xlabel('hour')
plt.ylabel('CoSQM magnitude')

#moon filter
hours_float_moon = np.array([ date.hour+                    #WATCH OUT FOR TIMEZONE HERE!
    date.minute/60+
    date.second/3600 for date in dt_obs_moon ])
hours_float_moon[hours_float_moon>12]-=24

plt.figure(figsize=[12,8])
plt.hist2d(hours_float_moon[np.isfinite(cosqm_obs_moon)[:,0]], cosqm_obs_moon[:,0][np.isfinite(cosqm_obs_moon)[:,0]], 200, cmap='inferno')
plt.hist2d(hours_float_moon[np.isfinite(cosqm_obs_moon)[:,0]], cosqm_obs_moon[:,0][np.isfinite(cosqm_obs_moon)[:,0]], 200, cmap='inferno', norm=LogNorm())
plt.title(f'{loc_str} - ZNSB - filter: moon - clear')
plt.xlabel('hour')
plt.ylabel('CoSQM magnitude')

#sun filter
hours_float_sun = np.array([ date.hour+                    #WATCH OUT FOR TIMEZONE HERE!
    date.minute/60+
    date.second/3600 for date in dt_obs_sun ])
hours_float_sun[hours_float_sun>12]-=24

plt.figure(figsize=[12,8])
plt.hist2d(hours_float_sun[np.isfinite(cosqm_obs_sun)[:,0]], cosqm_obs_sun[:,0][np.isfinite(cosqm_obs_sun)[:,0]], 200, cmap='inferno')
plt.hist2d(hours_float_sun[np.isfinite(cosqm_obs_sun)[:,0]], cosqm_obs_sun[:,0][np.isfinite(cosqm_obs_sun)[:,0]], 200, cmap='inferno', norm=LogNorm())
plt.title(f'{loc_str} - ZNSB - filter: sun - clear')
plt.xlabel('hour')
plt.ylabel('CoSQM magnitude')

## set to nan values that are color higher than clear by visual analysis, followed by clouded nights by visual analysis
dates_color_str = np.array(['2020-01-21', '2020-01-26',
    '2020-05-14', '2020-05-15', '2020-05-16', '2020-05-17', '2020-05-18', '2020-05-19', '2020-05-20', '2020-05-21', '2020-05-22', '2020-05-23'])

dates_cosqm_str = np.array([dt.strftime(date, '%Y-%m-%d') for date in dt_obs_sun])
dates_color_mask = np.ones(dates_cosqm_str.shape[0], dtype=bool)
dates_color_mask[np.isin(dates_cosqm_str, dates_color_str)] = False

dt_obs_sun = dt_obs_sun[dates_color_mask]
cosqm_obs_sun = cosqm_obs_sun[dates_color_mask]

################
# Light pollution trends
################

## Every night normalized (substracted magnitude) with mean of 1am to 2am data
d = np.copy(dt_obs_sun)                       #Attention, le timezone est en UTC, ce qui peut causer des problemes pour diviser les nuits ailleurs dans le monde
c = np.copy(cosqm_obs_sun)
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
#   c_mean = np.ma.array(c, mask=np.isnan(c))[d_mask_mean].mean(axis=0)
    c_mean = np.nanmean(c[d_mask_mean],axis=0)
    c_norm[d_mask] -= c_mean

c_norm[c_norm > 1.8] = np.nan
# plt.scatter(d,c_norm[:,0])


#Make plots
d = np.copy(dt_obs_sun)

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

# Correct filtered data with fit curve
cosqm_obs_2nd = np.array([second_order(hours_float, fit_params_c[0], fit_params_c[1], fit_params_c[2]),
    second_order(hours_float, fit_params_r[0], fit_params_r[1], fit_params_r[2]),
    second_order(hours_float, fit_params_g[0], fit_params_g[1], fit_params_g[2]),
    second_order(hours_float, fit_params_b[0], fit_params_b[1], fit_params_b[2]),
    second_order(hours_float, fit_params_y[0], fit_params_y[1], fit_params_y[2])]).T

dt_obs_final = dt_obs_sun
cosqm_obs_final = np.copy(cosqm_obs_sun) - cosqm_obs_2nd
ddays_cosqm = np.array([(date.date()-d[0].date()).days for date in dt_obs_final])
dt_obs_date = dt_obs_final.dt.date


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
data_aod_raw['Datetime'] = pd.to_datetime(data_aod_raw['Date(dd:mm:yyyy)']+'T'+data_aod_raw['Time(hh:mm:ss)'], utc=True, format='%d:%m:%YT%H:%M:%S') 
dt_aod = data_aod_raw['Datetime']
dt_aod_date = data_aod_raw['Datetime'].dt.date
aod_bands = data_aod_raw.columns[(data_aod_raw.values[0]!=-999.0)]
aod_bands = aod_bands[[column[:3] == 'AOD' for column in aod_bands]]
data_aod_raw = data_aod_raw[aod_bands].values
mask = np.all(data_aod_raw>0, axis=1)
data_aod = data_aod_raw[mask]
dt_aod = dt_aod[mask]
dt_aod_date = dt_aod_date[mask]
#BANDS = ['AOD_1640nm', 'AOD_1020nm', 'AOD_870nm', 'AOD_865nm', 'AOD_779nm',
#    'AOD_675nm', 'AOD_667nm', 'AOD_620nm', 'AOD_560nm', 'AOD_555nm',
#    'AOD_551nm', 'AOD_532nm', 'AOD_531nm', 'AOD_510nm', 'AOD_500nm',
#    'AOD_490nm', 'AOD_443nm', 'AOD_440nm', 'AOD_412nm', 'AOD_400nm',
#    'AOD_380nm', 'AOD_340nm']

# Plot each band of aod measurements for total data
plt.figure()
for i in range(aod_bands.shape[0]):
    plt.scatter(dt_aod, data_aod[:,i], label=aod_bands[i], s=0.2)
#plt.legend()
plt.show()

# find same days in each instrument
mask_aod = np.isin(dt_aod_date.values, np.unique(dt_obs_date))
dt_aod = dt_aod[mask_aod]
dt_aod_date = dt_aod_date[mask_aod]
data_aod = data_aod[mask_aod]

mask_obs = np.isin(dt_obs_date.values, np.unique(dt_aod.dt.date))
dt_obs_final = dt_obs_final[mask_obs]
cosqm_obs_final = cosqm_obs_final[mask_obs]
dt_obs_date = dt_obs_date[mask_obs]



#############
# CORRELATION
#############

# Selection of filtered and corrected ZNSB values: all data from noon to 10pm for aod_pm, all data from 3am till noon
cosqm_am = np.zeros((np.unique(dt_obs_date).shape[0], 5))
cosqm_pm = np.zeros((np.unique(dt_obs_date).shape[0], 5))
hours_cosqm = dt_obs_final.dt.hour

cosqm_mean_count = 10
cosqm_min_am_hour = 0
cosqm_max_pm_hour = 24

for i,day in enumerate(np.unique(dt_obs_date)):
#    d_mask_am = np.zeros(ddays_cosqm.shape[0], dtype=bool)
#    d_mask_pm = np.zeros(ddays_cosqm.shape[0], dtype=bool)
    d_mask_am = (dt_obs_date == day) & (hours_cosqm >= cosqm_min_am_hour) & (hours_cosqm < 4)
    d_mask_pm = (dt_obs_date == day) & (hours_cosqm >= 20) & (hours_cosqm < cosqm_max_pm_hour)
    inds_am = np.where(d_mask_am == True)[0]
    if inds_am.shape[0]>cosqm_mean_count:
        inds_am = inds_am[-cosqm_mean_count:]
        cosqm_am[i] = np.nanmean(cosqm_obs_final[inds_am],axis=0)
    else:
        inds_am[inds_am == True] = False
        cosqm_am[i] = np.nan
    inds_pm = np.where(d_mask_pm == True)[0]
    if inds_pm.shape[0]>cosqm_mean_count:
        inds_pm = inds_pm[:cosqm_mean_count]
        cosqm_pm[i] = np.nanmean(cosqm_obs_final[inds_pm],axis=0)
    else:
        inds_pm[inds_pm == True] = False
        cosqm_pm[i] = np.nan
#    cosqm_am[cosqm_am == 0] = np.nan                    # remove zeros values from sensor problem (no data?)
#    cosqm_pm[cosqm_pm == 0] = np.nan                    # remove zeros values from sensor problem (no data?)

plt.figure()
plt.title('ZNSB dusk and dawn values - Blue')
plt.scatter(np.unique(dt_obs_date), cosqm_am[:,1], label='cosqm_am', s=15)
plt.scatter(np.unique(dt_obs_date), cosqm_pm[:,1], label='cosqm_pm', s=15)
plt.xlabel('days from july 2019')
plt.ylabel('CoSQM Magnitudes (mag)')
plt.legend()



# Selection of AOD values: all data from 3pm to midnight for aod_pm, all data from midnight till 10am   
aod_am = np.zeros((np.unique(dt_aod_date).shape[0], aod_bands.shape[0]))
aod_pm = np.zeros((np.unique(dt_aod_date).shape[0], aod_bands.shape[0]))
hours_aod = dt_aod.dt.hour

aod_mean_count = 1
aod_max_am_hour = 12
aod_min_pm_hour = 12

for i,day in enumerate(np.unique(dt_aod_date)):
    d_mask_am = np.zeros(dt_aod_date.shape[0], dtype=bool)
    d_mask_pm = np.zeros(dt_aod_date.shape[0], dtype=bool)
    d_mask_am[(dt_aod_date == day) & (hours_aod < aod_max_am_hour)] = True
    d_mask_pm[(dt_aod_date == day) & (hours_aod >= aod_min_pm_hour)] = True
    inds_am = np.where(d_mask_am == True)[0]
    if inds_am.shape[0]>aod_mean_count:
        inds_am = inds_am[:aod_mean_count]
        aod_am[i] =  np.nanmean(data_aod[inds_am],axis=0)
    else:
        inds_am[inds_am == True] = False
        aod_am[i] =  np.nan

    inds_pm = np.where(d_mask_pm == True)[0]
    if inds_pm.shape[0]>aod_mean_count:
        inds_pm = inds_pm[-aod_mean_count:]
        aod_pm[i] =  np.nanmean(data_aod[inds_pm],axis=0)
    else:
        inds_pm[inds_pm == True] = False
        aod_pm[i] =  np.nan

plt.figure()
plt.title('AOD dusk and dawn values')
plt.scatter(np.unique(dt_aod_date), aod_am[:,3], label='aeronet_am', s=15)
plt.scatter(np.unique(dt_aod_date), aod_pm[:,3], label='aeronet_pm', s=15)
plt.xlabel('days from july 2019')
plt.ylabel('AOD')
plt.legend()


#Correlation plots for the 4 color bands (R-629nm, G-546nm, B-514nm, Y-562nm)
cosqm_bands = ['0', '629', '546', '514', '562']
j = 0
for i in range(aod_bands.shape[0]):
    fig, ax = plt.subplots(1,1)
    ax.scatter(aod_am[:,i], cosqm_am[:, j], label='AM')
    ax.scatter(aod_pm[:,i], cosqm_pm[:, j], label='PM')
    ax.set_xscale('log')
    #ax.set_yscale('log')
    ax.set_title(f'{loc_str}, AOD {aod_bands[i][4:]}, CoSQM {cosqm_bands[j]}nm')
    ax.legend()