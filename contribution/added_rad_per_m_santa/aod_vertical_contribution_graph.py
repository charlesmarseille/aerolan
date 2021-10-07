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
h2_0 = get_contrib('santa/st_cruz_2019_aerosol_optical_depth_0.13-wavelength_505.0_H2-layer_0.out')
h2_1 = get_contrib('santa/st_cruz_2019_aerosol_optical_depth_0.13-wavelength_505.0_H2-layer_1.out')
h2_0_na = get_contrib('santa/st_cruz_2019_aerosol_optical_depth_0-wavelength_505.0_H2-layer_0.out')
h2_1_na = get_contrib('santa/st_cruz_2019_aerosol_optical_depth_0-wavelength_505.0_H2-layer_1.out')

plt.figure()
plt.scatter(h2_0[0], (h2_0[2]+h2_1[2]-(h2_0_na[2]+h2_1_na[2]))/h2_0[1], marker='*', label='aerosols')
plt.scatter(h2_0[0], (h2_0_na[2]+h2_1_na[2])/h2_0[1], marker='d', label='molecules')
plt.scatter(h2_0[0], (h2_0[2]+h2_1[2])/h2_0[1], marker='+', label='aerosols+molecules')
plt.xscale('log')
plt.yscale('log')
plt.ylim(-5e-13, 1.5e-11)
plt.legend(loc='upper right')
plt.xlabel('Vertical distance (m)')
plt.ylabel('Added radiance per meter ($W/str/m^3/nm$)')
plt.text(1e2, 1e-14, '505nm, H=2km')
props = dict(facecolor='white', lw=0.01)
plt.text(0.13, 0.92, '$10^{-11}$', fontsize=11, transform=plt.gcf().transFigure, bbox=props)
plt.tight_layout()
plt.savefig('../../figures/contribution/added_rad_per_m_505nm_H2_log.png')

# H=10 
h10_0 = get_contrib('santa/st_cruz_2019_aerosol_optical_depth_0.13-wavelength_530.0_H10-layer_0.out')
h10_1 = get_contrib('santa/st_cruz_2019_aerosol_optical_depth_0.13-wavelength_530.0_H10-layer_1.out')
h10_0_na = get_contrib('santa/st_cruz_2019_aerosol_optical_depth_0-wavelength_530.0_H10-layer_0.out')
h10_1_na = get_contrib('santa/st_cruz_2019_aerosol_optical_depth_0-wavelength_530.0_H10-layer_1.out')

plt.figure()
plt.scatter(h10_0[0], (h10_0[2]+h10_1[2]-(h10_0_na[2]+h10_1_na[2]))/h10_0[1], marker='*', label='aerosols')
plt.scatter(h10_0[0], (h10_0_na[2]+h10_1_na[2])/h10_0[1], marker='d', label='molecules')
plt.scatter(h10_0[0], (h10_0[2]+h10_1[2])/h10_0[1], marker='+', label='aerosols+molecules')
plt.xscale('log')
plt.yscale('log')
plt.ylim(-5e-13,1.5e-11)
plt.legend(loc='upper right')
plt.xlabel('Vertical distance (m)')
plt.ylabel('Added radiance per meter ($W/str/m^3/nm$)')
plt.text(1e2, 1e-15, '505nm, H=10km')
props = dict(facecolor='white', lw=0.01)
plt.text(0.13, 0.92, '$10^{-11}$', fontsize=11, transform=plt.gcf().transFigure, bbox=props)
plt.tight_layout()
plt.savefig('../../figures/contribution/added_rad_per_m_505nm_H10_log.png')

# H2 vs H10 aerosols

plt.figure()
plt.scatter(h2_0[0], (h2_0[2]+h2_1[2]-(h2_0_na[2]+h2_1_na[2]))/h2_0[1], marker='*', label='aerosols H=2')
plt.scatter(h10_0[0], (h10_0[2]+h10_1[2]-(h10_0_na[2]+h10_1_na[2]))/h10_0[1], marker='*', label='aerosols H=10')
plt.xscale('log')
#plt.yscale('log')
plt.ylim(-5e-13,0.4e-11)
plt.legend(loc='upper right')
plt.xlabel('Vertical distance (m)')
plt.ylabel('Added radiance per meter ($W/str/m^3/nm$)')
plt.tight_layout()
plt.savefig('santa/added_rad_per_m_505nm_H2_vs_H10.png')
