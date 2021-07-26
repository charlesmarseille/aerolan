#########################
## Cloud removal from visual analysis: thumbnails are checked in a file browser and a folder is created with the clear
## skies data. The filename is of format YYYY-MM-DD_HH/MM/SS.jpg
######
## WARNING: filenames have SLASHES or colons, which can't be read in Microsoft Windows. You must use unix to replace / or : with _ in filenames so that
## the code works in Windows. The following 2 lines must be ran in unix in the folder showing the years of measurements (for string positions):
## ----    fnames = glob('*/*/webcam/*.jpg')
## ----    [os.rename(fname, fname[:28]+fname[29:31]+fname[32:]) for fname in fnames]
######

santa_dates_noclouds_str = np.array(glob('cosqm_santa/data/*/*/webcam/*.jpg'))         # Load str from noclouds images
np.savetxt('santa_cruz_noclouds_fnames.txt', santa_dates_noclouds_str, fmt='%s')
santa_dates_noclouds_str = pd.read_csv('santa_cruz_noclouds_fnames.txt', dtype='str').values
santa_noclouds_dates = np.array([ dt.strptime( date[0][-21:-4], '%Y-%m-%d_%H%M%S' ).timestamp() for date in santa_dates_noclouds_str ])       # Convert images time str to timestamps
santa_noclouds_days = np.array([ dt.strptime( date[0][-21:-11], '%Y-%m-%d' ).timestamp() for date in santa_dates_noclouds_str ])          # Convert images time str to timestamp days 

bins = np.arange(santa_noclouds_days.min(),santa_noclouds_days.max(), 24*60*60)       # define complete days for binning
santa_noclouds_days_hist, santa_noclouds_days_bins = np.histogram(santa_noclouds_days, np.arange(santa_noclouds_days.min(),santa_noclouds_days.max(), 24*60*60))      # count number of images per day
min_images = 20       # minimum number of non clouded images in a day to be considered
santa_noclouds_days_filtered = santa_noclouds_days_bins[np.argwhere(santa_noclouds_days_hist > min_images)][:,0]          # select only days that have at least min_images non-clouded images

## Mask days that were clouded
santa_days = np.array([ dt.strptime( date.strftime('%Y-%m-%d'), '%Y-%m-%d' ).timestamp() for date in dt_santa_moon_sun ])
cloud_mask = np.isin(santa_days, santa_noclouds_days_filtered)
dates_santa_moon_sun_clouds = dates_santa_moon_sun[cloud_mask]
dt_santa_moon_sun_clouds = dt_santa_moon_sun[cloud_mask]
cosqm_santa_moon_sun_clouds = cosqm_santa_moon_sun[cloud_mask]

## Plot cosqm_data filtered for clouds
plt.figure(figsize=[16,9])
plt.scatter(dt_santa_moon_sun, cosqm_santa_moon_sun[:,0], s=30, c='b', label='moon and sun filtered')
plt.scatter(dt_santa_moon_sun_clouds, cosqm_santa_moon_sun_clouds[:,0], s=20, c='r', label='cloud filter from pictures')
plt.legend(loc=(0,0))
plt.title('ZNSB Santa-Cruz')
plt.xlabel('date')
plt.ylabel('CoSQM magnitude (mag)')