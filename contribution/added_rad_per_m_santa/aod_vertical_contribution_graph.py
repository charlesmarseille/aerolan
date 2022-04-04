import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def get_contrib(fname):
	f = pd.read_csv(fname, delimiter='\n')
	f = f.values.flatten()
	verts_mask = ['Vertical dist.' in line for line in f]
	ar_mask = ['Added radiance' in line for line in f]
	ra_mask = ['Radiance accumulated' in line for line in f]
	verts = np.array([line[35:42] for line in f[verts_mask]], dtype=float)
	added_rad = np.array([line[20:] for line in f[ar_mask]], dtype=float)[1:-1]
	rad_accu = np.array([line[26:] for line in f[ra_mask]], dtype=float)[1:-1]
	thicks = (verts[1:-1]-verts[:-2])/2 + (verts[2:]-verts[1:-1])/2
	return np.array([verts[1:-1], thicks, added_rad, rad_accu])


# H=2
h2_0 = get_contrib('santa/st_cruz_2019_aerosol_optical_depth_0.13-wavelength_530.0_H2-layer_0.out')
h2_1 = get_contrib('santa/st_cruz_2019_aerosol_optical_depth_0.13-wavelength_530.0_H2-layer_1.out')
h2_0_na = get_contrib('santa/st_cruz_2019_aerosol_optical_depth_0-wavelength_530.0_H2-layer_0.out')
h2_1_na = get_contrib('santa/st_cruz_2019_aerosol_optical_depth_0-wavelength_530.0_H2-layer_1.out')


plt.figure()
plt.scatter(h2_0[0], (h2_0[2]+h2_1[2]-(h2_0_na[2]+h2_1_na[2]))/h2_0[1], marker='*', label='aerosols')
plt.scatter(h2_0[0], (h2_0_na[2]+h2_1_na[2])/h2_0[1], marker='d', label='molecules')
plt.scatter(h2_0[0], (h2_0[2]+h2_1[2])/h2_0[1], marker='+', label='aerosols+molecules')
plt.xscale('log')
#plt.yscale('log')
plt.ylim(-5e-13, 1.1e-11)
plt.legend(loc='upper right')
plt.xlabel('Vertical distance (m)')
plt.ylabel('Added radiance per meter ($W/str/m^3/nm$)')
plt.text(1e2, 1e-14, '532nm, H=2km')
props = dict(facecolor='white', lw=0.01)
plt.text(0.13, 0.92, '$10^{-11}$', fontsize=11, transform=plt.gcf().transFigure, bbox=props)
plt.tight_layout()
#plt.savefig('santa/added_rad_per_m_532nm_H2.png')

# H=10 
h10_0 = get_contrib('santa/st_cruz_2019_aerosol_optical_depth_0.13-wavelength_530.0_H10-layer_0_corr.out')
h10_1 = get_contrib('santa/st_cruz_2019_aerosol_optical_depth_0.13-wavelength_530.0_H10-layer_1.out')
h10_0_na = get_contrib('santa/st_cruz_2019_aerosol_optical_depth_0-wavelength_530.0_H10-layer_0.out')
h10_1_na = get_contrib('santa/st_cruz_2019_aerosol_optical_depth_0-wavelength_530.0_H10-layer_1.out')

plt.figure()
plt.scatter(h10_0[0], (h10_0[2]+h10_1[2]-(h10_0_na[2]+h10_1_na[2]))/h10_0[1], marker='*', label='aerosols')
plt.scatter(h10_0[0], (h10_0_na[2]+h10_1_na[2])/h10_0[1], marker='d', label='molecules')
plt.scatter(h10_0[0], (h10_0[2]+h10_1[2])/h10_0[1], marker='+', label='aerosols+molecules')
plt.xscale('log')
#plt.yscale('log')
plt.ylim(-5e-13,1.1e-11)
plt.legend(loc='upper right')
plt.xlabel('Vertical distance (m)')
plt.ylabel('Added radiance per meter ($W/str/m^3/nm$)')
plt.text(1e2, 1e-15, '532nm, H=10km')
props = dict(facecolor='white', lw=0.01)
plt.text(0.13, 0.92, '$10^{-11}$', fontsize=11, transform=plt.gcf().transFigure, bbox=props)
plt.tight_layout()
#plt.savefig('santa/added_rad_per_m_532nm_H10.png')

# H2 vs H10 aerosols

plt.figure()
plt.scatter(h2_0[0], (h2_0[2]+h2_1[2]-(h2_0_na[2]+h2_1_na[2]))/h2_0[1], marker='*', label='aerosols H=2')
plt.scatter(h10_0[0], (h10_0[2]+h10_1[2]-(h10_0_na[2]+h10_1_na[2]))/h10_0[1], marker='*', label='aerosols H=10')
plt.xscale('log')
#plt.yscale('log')
plt.ylim(-5e-13,0.8e-11)
plt.legend(loc='upper right')
plt.xlabel('Vertical distance (m)')
plt.ylabel('Added radiance per meter ($W/str/m^3/nm$)')
plt.tight_layout()
#plt.savefig('santa/added_rad_per_m_532nm_H2_vs_H10.png')


# Compute percentage of sky brightness from aerosols higher than x km altitude (to answer: are we restrained to PBL effects?)
# contrib = (Total SB - accumulated SB until 2km) / Total SB
ind_2km = 22
ind_5km = 29
aer2 = (h2_0+h2_1)-(h2_0_na+h2_1_na)  # aerosols = Total - no aerosols
aer2[0] = h2_0[0]
aer10 = (h10_0+h2_1)-(h10_0_na+h10_1_na)  # aerosols = Total - no aerosols
aer10[0] = h10_0[0]

h2_accu_aer_2km = np.sum(aer2[2,ind_2km:][aer2[2,ind_2km:]>0])/(h2_0[3,-1]+h2_1[3,-1])*100 	# 2km H aerosols
h2_accu_aer_5km = np.sum(aer2[2,ind_5km:][aer2[2,ind_5km:]>0])/(h2_0[3,-1]+h2_1[3,-1])*100

h10_accu_aer_2km = np.sum(aer10[2,ind_2km:][aer10[2,ind_2km:]>0])/(h2_0[3,-1]+h2_1[3,-1])*100 	# 10km H aerosols
h10_accu_aer_5km = np.sum(aer10[2,ind_5km:][aer10[2,ind_5km:]>0])/(h2_0[3,-1]+h2_1[3,-1])*100

print(h2_accu_aer_2km, h2_accu_aer_5km)
print(h10_accu_aer_2km, h10_accu_aer_5km)


aer2_accu = (h2_0[3]+h2_1[3]-(h2_0_na[3]+h2_1_na[3]))
mol2_accu = (h2_0_na[3]+h2_1_na[3])
sum2_accu = (h2_0[3]+h2_1[3])

aer10_accu = (h10_0[3]+h10_1[3]-(h10_0_na[3]+h10_1_na[3]))
mol10_accu = (h10_0_na[3]+h10_1_na[3])
sum10_accu = (h10_0[3]+h10_1[3])

# Plot radiance accumulated
plt.figure()
plt.scatter(h2_0[0], aer2_accu/aer2_accu.max(), marker='*', label='aerosols')
plt.scatter(h2_0[0], mol2_accu/aer2_accu.max(), marker='d', label='molecules')
plt.scatter(h2_0[0], sum2_accu/aer2_accu.max(), marker='+', label='aerosols+molecules')
plt.xscale('log')
plt.yscale('log')
#plt.ylim(-5e-13, 1.1e-11)
plt.grid(b=True, which='major', color='0.8', linestyle='-')
plt.grid(b=True, which='minor', color='0.8', linestyle='--')
plt.legend(loc='upper left')
plt.xlabel('Vertical distance (m)')
plt.ylabel('Normalized accumulated radiance ($1/str/m^2/nm$)')
plt.text(60, 1.5, '532nm, H=2km')
props = dict(facecolor='white', lw=0.01)
#plt.text(0.13, 0.92, '$10^{-11}$', fontsize=11, transform=plt.gcf().transFigure, bbox=props)
plt.tight_layout()
plt.savefig('accu_rad_per_m_532nm_H2.png')

plt.figure()
plt.scatter(h10_0[0], aer10_accu/aer10_accu.max(), marker='*', label='aerosols')
plt.scatter(h10_0[0], mol10_accu/aer10_accu.max(), marker='d', label='molecules')
plt.scatter(h10_0[0], sum10_accu/aer10_accu.max(), marker='+', label='aerosols+molecules')
plt.xscale('log')
plt.yscale('log')
plt.grid(b=True, which='major', color='0.8', linestyle='-')
plt.grid(b=True, which='minor', color='0.8', linestyle='--')
#plt.ylim(-5e-13, 1.1e-11)
plt.legend(loc='upper left')
plt.xlabel('Vertical distance (m)')
plt.ylabel('Normalized accumulated radiance ($1/str/m^2/nm$)')
plt.text(60, 1.5, '532nm, H=10km')
props = dict(facecolor='white', lw=0.01)
#plt.text(0.13, 0.92, '$10^{-11}$', fontsize=11, transform=plt.gcf().transFigure, bbox=props)
plt.tight_layout()
plt.savefig('accu_rad_per_m_532nm_H10.png')

