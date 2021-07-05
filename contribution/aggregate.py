#!/usr/bin/env python2

#import fiona
import pandas as pd
import geopandas as gpd
from geopandas.tools import sjoin
from shapely.geometry import Point
import MultiScaleData as MSD
import numpy as np
from glob import glob

#fiona.drvsupport.supported_drivers['KML'] = 'rw'


fnames = glob('*.hdf5')
fname = fnames[5]

contrib = []
hdf = MSD.Open(fname)
srs = hdf._attrs['srs']
for i in range(len(hdf)):
    ys,xs = np.where(hdf[i])
    ps = hdf.pixel_size(i)
    xmin = hdf._attrs['layers'][i]['xmin']
    ymin = hdf._attrs['layers'][i]['ymin']

    vs = hdf[i][ys,xs]
    xs = (xs+0.5)*ps + xmin
    ys = (ys+0.5)*ps + ymin
    contrib.extend(np.transpose([xs,ys,vs]))


points = pd.DataFrame(data=contrib,columns=['x','y','val'])
points = gpd.GeoDataFrame(
    points.drop(['x','y'],axis=1),
    crs={"init":srs},
    geometry=[Point(xy) for xy in zip(points.x,points.y)]
)
points['geometry'] = points['geometry'].to_crs(epsg=4326)
polys = gpd.read_file("zip://gobcan_unidades-estadisticas_municipios-generalizados-shp.zip")
polys['geometry'] = polys['geometry'].to_crs(epsg=4326)

pointsInPolys = sjoin(points,polys,how='left')
grouped = pointsInPolys.groupby("index_right")
contrib_total = grouped['val'].sum()

indices = contrib_total.index.values
values = contrib_total.values
