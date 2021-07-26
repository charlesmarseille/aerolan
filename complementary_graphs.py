

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


