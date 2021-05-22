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
            data = pd.read_csv(path, delimiter=' ', names=['Date', 'Time', 'Lat', 'Lon', 'Alt', 'Temp', 'Wait', 'Sqm0', 'Sqm1', 'Sqm2', 'Sqm3', 'Sqm4', 'Sbcals0', 'Sbcals1', 'Sbcals2', 'Sbcals3', 'Sbcals4'])
            data['Datetime'] = pd.to_datetime(data['Date']+'T'+data['Time'], utc=True)        
            cache[path] = data
            return data
        except:
            print('********error*********', path)

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
milkyway = Star(ra_hours=(17, 45, 40.04), dec_degrees=(-29, 0, 28.1))

## Define variables for locations
izana_loc = earth + Topos('28.300398N', '16.512252W')
santa_loc = earth + Topos('28.472412500N', '16.247361500W')

## !! Time objects from dt need a timezone (aware dt object) to work with skyfield lib
## ex.: time = dt.now(timezone.utc), with an import of datetime.timezone
def MoonAngle(dt_array, location):
    t = ts.utc(dt_array)
    astrometric = location.at(t).observe(moon)
    alt, _, _ = astrometric.apparent().altaz()
    return alt.degrees

def SunAngle(dt_array, location):
    t = ts.utc(dt_array)
    astrometric = location.at(t).observe(sun)
    alt, _, _ = astrometric.apparent().altaz()
    return alt.degrees

def MilkywayAngle(dt_array, location):
    t = ts.utc(dt_array)
    astrometric = location.at(t).observe(milkyway)
    alt, _, _ = astrometric.apparent().altaz()
    return alt.degrees

#%%--------------------
# COSQM DATA
# Variation as a function of whole year, day of week, month and season
# Data taken from Martin Aubé's server: http://dome.obsand.org:2080/DATA/CoSQM-Network/
#
# Data format: Date(YYYY-MM-DD), time(HH:MM:SS),
#             Latitude, Longitude, Elevation, SQM temperature, integration time, 
#             C(mag), R(mag), G(mag), B(mag), Y(mag),
#             C(watt),R(watt),G(watt),B(watt),Y(watt)  
# 

## COSQM location variables
path_santa='cosqm_izana/data/'
loc = izana_loc
loc_str = 'izana'

## find all paths of files in root directory
paths_santa = sorted(glob(path_santa+"*/*/*.txt"))
files = pd.concat([LoadData(path) for path in paths_santa], ignore_index=True)
cosqm_santa = files[['Sqm0', 'Sqm1', 'Sqm2', 'Sqm3', 'Sqm4']].values
dt_santa = files['Datetime']

#remove non datetime errors in cosqm files (NaT)
cosqm_santa = cosqm_santa[~pd.isnull(dt_santa)]
dt_santa = dt_santa[~pd.isnull(dt_santa)]

### if day is wanted: dt_santa_day = dt_santa.dt.day
### if interval wanted: inds = (dt_santa.dt.date == np.array('2020-01-21',dtype='datetime64[D]')) | (dt_santa.dt.date == np.array('2020-01-22',dtype='datetime64[D]'))
### work on specific night interval: inds = (dt_santa.values > np.array('2020-01-21T12',dtype='datetime64[ns]')) 
##   & (dt_santa.values < np.array('2020-01-22T12',dtype='datetime64[ns]'))

## Remove zeros from cosqm measurements (bugs from instruments)
zeros_mask = (cosqm_santa!=0).all(1)
cosqm_santa = cosqm_santa[zeros_mask]
dt_santa = dt_santa[zeros_mask]

## Cloud removal with differential between points (if difference between 2 measurements is bigger than threshold, remove data)
slide_threshold = 0.1
slide_window_size = 5
cosqm_santa_diff = Cloudslidingwindow(cosqm_santa, slide_window_size, slide_threshold)

## Compute moon angles for each timestamp in COSQM data
print('moon_angles calculation')
moon_angles = MoonAngle(dt_santa, loc)

## Mask values for higher angle than -18deg (astro twilight)
moon_min_angle = -2
moon_mask = moon_angles<moon_min_angle

cosqm_santa_moon = cosqm_santa_diff[moon_mask]
dt_santa_moon = dt_santa[moon_mask].reset_index(drop=True)
#dates_days_since_start = np.array([(dt.fromtimestamp(date)-dt.fromtimestamp(dates[0])).days+1 for date in dates])

## Compute sun angles for each timestamp in COSQM data
print('sun_angles calculation')
sun_angles = SunAngle(dt_santa_moon, santa_loc)
np.savetxt('cosqm_'+loc_str+'_sun_angles.txt', sun_angles)
#sun_angles = np.loadtxt('cosqm_'+loc_str+'_sun_angles.txt')					#Load already computed angles

sun_min_angle = -18
sun_mask = sun_angles<sun_min_angle

cosqm_santa_sun = cosqm_santa_moon[sun_mask]
dt_santa_sun = dt_santa_moon[sun_mask].reset_index(drop=True)

## filter data when milky way is in the visible sky (5 degrees above horizon)
# (todo)

# plot filtering
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
cosqm_santa_2nd = np.array([second_order(hours_float, fit_params_c[0], fit_params_c[1], fit_params_c[2]),
    second_order(hours_float, fit_params_r[0], fit_params_r[1], fit_params_r[2]),
    second_order(hours_float, fit_params_g[0], fit_params_g[1], fit_params_g[2]),
    second_order(hours_float, fit_params_b[0], fit_params_b[1], fit_params_b[2]),
    second_order(hours_float, fit_params_y[0], fit_params_y[1], fit_params_y[2])]).T

dt_santa_final = dt_santa_sun
cosqm_santa_final = np.copy(cosqm_santa_sun) - cosqm_santa_2nd
dt_santa_date = dt_santa_final.dt.date

#%%--------------------
#
#   AOD Values from AERONET network
#   Values are from level 1.0, meaning there are no corrections
#

# read from columns 4 to 25 (aod 1640 to 340nm)
path = 'cosqm_izana/20191201_20201130_Izana.lev10'
data_aod_raw = pd.read_csv(path, delimiter=',', header=6)
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
#non empty bands:['AOD_1640nm', 'AOD_1020nm', 'AOD_870nm', 'AOD_675nm', 'AOD_500nm', 'AOD_440nm', 'AOD_380nm']

# Plot each band of aod measurements for total data
plt.figure()
for i in range(aod_bands.shape[0]):
    plt.scatter(dt_aod, data_aod[:,i], s=20)
#plt.legend()




# find same days in each instrument
mask_aod = np.isin(dt_aod_date.values, np.unique(dt_santa_date))
dt_aod = dt_aod[mask_aod]
dt_aod_date = dt_aod_date[mask_aod]
data_aod = data_aod[mask_aod]

mask_santa = np.isin(dt_santa_date.values, np.unique(dt_aod.dt.date))
dt_santa_final = dt_santa_final[mask_santa]
cosqm_santa_final = cosqm_santa_final[mask_santa]
dt_santa_date = dt_santa_date[mask_santa]


#%%--------------------
# 
#  AOD-ZNSB CORRELATION
#
#


# Selection of filtered and corrected ZNSB values: all data from noon to 10pm for aod_pm, all data from 3am till noon
cosqm_am = np.zeros((np.unique(dt_santa_date).shape[0], 5))
cosqm_pm = np.zeros((np.unique(dt_santa_date).shape[0], 5))
hours_cosqm = dt_santa_final.dt.hour


for i,day in enumerate(np.unique(dt_santa_date)):
#    d_mask_am = np.zeros(ddays_cosqm.shape[0], dtype=bool)
#    d_mask_pm = np.zeros(ddays_cosqm.shape[0], dtype=bool)
    d_mask_am = (dt_santa_date == day) & (hours_cosqm >= 0) & (hours_cosqm < 12)
    d_mask_pm = (dt_santa_date == day) & (hours_cosqm >= 12) & (hours_cosqm < 24)
    inds_am = np.where(d_mask_am == True)[0]
    if inds_am.shape[0]>5:
        inds_am = inds_am[-20:]
    inds_pm = np.where(d_mask_pm == True)[0]
    if inds_pm.shape[0]>5:
        inds_pm = inds_pm[:20]
    print('day: ', day, 'am: ', inds_am.shape, 'pm: ', inds_pm.shape)
    cosqm_am[i] = np.nanmean(cosqm_santa_final[inds_am],axis=0)
    cosqm_pm[i] = np.nanmean(cosqm_santa_final[inds_pm],axis=0)
#    dt_cosqm_am = dt_santa_final[inds_am]
#    dt_cosqm_pm = dt_santa_final[inds_pm]
    cosqm_am[cosqm_am == 0] = np.nan                    # remove zeros values from sensor problem (no data?)
    cosqm_pm[cosqm_pm == 0] = np.nan                    # remove zeros values from sensor problem (no data?)




# Selection of AOD values: all data from 3pm to midnight for aod_pm, all data from midnight till 10am   
aod_am = np.zeros((np.unique(dt_aod_date).shape[0], aod_bands.shape[0]))
aod_pm = np.zeros((np.unique(dt_aod_date).shape[0], aod_bands.shape[0]))
hours_aod = dt_aod.dt.hour


for i,day in enumerate(np.unique(dt_aod_date)):
    d_mask_am = np.zeros(dt_aod_date.shape[0], dtype=bool)
    d_mask_pm = np.zeros(dt_aod_date.shape[0], dtype=bool)
    d_mask_am[(dt_aod_date == day) & (hours_aod < 12)] = True
    d_mask_pm[(dt_aod_date == day) & (hours_aod >= 12)] = True
    inds_am = np.where(d_mask_am == True)[0]
    if inds_am.shape[0]>5:
        inds_am = inds_am[:50]
    inds_pm = np.where(d_mask_pm == True)[0]
    if inds_pm.shape[0]>5:
        inds_pm = inds_pm[-50:]
    print('day: ', day, 'am: ', inds_am.shape, 'pm: ', inds_pm.shape)
    aod_am[i] =  np.nanmean(data_aod[inds_am],axis=0)
    aod_pm[i] =  np.nanmean(data_aod[inds_pm],axis=0)
#    dt_aod_am = dt_aod[inds_am]
#    dt_aod_pm = dt_aod[inds_pm]
    aod_am[aod_am <= 0] = np.nan                    # remove zeros values from sensor problem (no data?)
    aod_pm[aod_pm <= 0] = np.nan                    # remove zeros values from sensor problem (no data?)


plt.figure()
plt.title('ZNSB dusk and dawn values - Blue')
plt.scatter(np.unique(dt_santa_date), cosqm_am[:,3], label='cosqm_am', s=15)
plt.scatter(np.unique(dt_santa_date), cosqm_pm[:,3], label='cosqm_pm', s=15)
plt.xlabel('days from july 2019')
plt.ylabel('CoSQM Magnitudes (mag)')
plt.legend()

plt.figure()
plt.title('AOD dusk and dawn values')
plt.scatter(np.unique(dt_aod_date), aod_am[:,3], label='aeronet_am', s=15)
plt.scatter(np.unique(dt_aod_date), aod_pm[:,3], label='aeronet_pm', s=15)
plt.xlabel('days from july 2019')
plt.ylabel('AOD')
plt.legend()


#Correlation plots for the 4 color bands (R-629nm, G-546nm, B-514nm, Y-562nm)
cosqm_bands = ['629', '546', '514', '562']
j = 2
for i in range(7):
    plt.figure()
    plt.scatter(aod_am[:,i], cosqm_am[:, j], label='AM')
    plt.scatter(aod_pm[:,i], cosqm_pm[:, j], label='PM')
    plt.xscale('log')
    plt.yscale('log')
    plt.title(f'IZANA atmo, AOD {aod_bands[i][4:]}, CoSQM {cosqm_bands[j]}nm')
    plt.legend()