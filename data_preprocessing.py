
# coding: utf-8

# Precipitation Data
# 
# Author: Madhura Killedar
# Date   : 01/02/2018
# 

from __future__ import print_function
import sys
import os
import datetime
import numpy as np
import pandas as pd
from math import asin, acos, sqrt, sin, cos, radians, degrees, asinh
from matplotlib import pyplot as plt
import seaborn as sns
import xarray as xr
import query_paleogeography as qpg
import csv

from matplotlib import rc
font = {'family': 'serif',
        'serif' : ['Palatino'],
        'weight': 'bold',
        'size'  : 20}
rc('font', **font)
rc('text', usetex=True)
from matplotlib import rcParams
fig_dpi = 150
rcParams['figure.dpi']= fig_dpi

print("Hello Precipitation Modelling")

# ## Input/Output Datasets
data_folder = "data/"
data_deposits_filename = data_folder+"LithData_PaleoXY_Matthews2016_20180226.csv"
key_deposits_filename  = data_folder+"LithologyCodes.csv"
data_rainfall_filename = data_folder+"PRECT_Average_annum.grd"

images_folder = data_folder + 'images/'
print("The output path is %s"%(images_folder))
if not os.path.isdir(images_folder):
    os.makedirs(images_folder)


# ## Set up grids
lon_min = -180.
lon_max =  180.
lat_min =  -90.
lat_max =   90.
lon_spacing = 3.

lon_coords = np.arange(lon_min,lon_max,lon_spacing) # left edge of grid
nlonbins = len(lon_coords)
print("Number of Longitudinal bins = %i"%nlonbins)

nlatbins = int(nlonbins * lat_max / 280.) # rough guess to allow a square bin at equator
print("Number of Latitudinal bins = %i"%nlatbins)
lat_coords = np.empty(nlatbins) # lower edge of grid
lat_spacing = np.empty(nlatbins)
for ilat in np.arange(nlatbins):
    coslat = 1. - 2.*ilat/float(nlatbins)
    lat_rad = acos(coslat)
    lat_coords[ilat] = degrees(lat_rad) - 90.
for ilat in np.arange(nlatbins-1):
    lat_spacing[ilat] = lat_coords[ilat+1] - lat_coords[ilat]
lat_spacing[nlatbins-1] = 90. - lat_coords[nlatbins-1]

print("Check central latitude bin-edges ...")
print(lat_coords[nlatbins/2-1:nlatbins/2+2])

# remove polar grid-bins
nlatbins = nlatbins-2
lat_coords = lat_coords[1:-1]
lat_spacing = lat_spacing[1:-1]

    
# ## Read & plot sedimentary deposits data
print("Deposits...")
data_deposits = pd.read_csv(data_deposits_filename)
key_deposits  = pd.read_csv(key_deposits_filename)
dict_deposits = dict(zip(key_deposits.loc[:,'LithologyCode'], key_deposits.loc[:,'LithologyType']))
time_deposits = data_deposits.loc[:,'ReconstructionTime'].copy()
miocene_rows_deposits = (5.33 < time_deposits) & (time_deposits < 23.03)
miocene_deposits = data_deposits.loc[miocene_rows_deposits,:].copy()

fig = plt.figure(figsize=(16,8))
ax_deposits = fig.add_subplot(111)
miocene_deposits_grouped = miocene_deposits.groupby('LithologyCode')
for deposit_type, deposit in miocene_deposits_grouped:
    ax_deposits.plot(deposit.Paleolongitude, deposit.Paleolatitude, marker='o', linestyle='', ms=3, label=dict_deposits[deposit_type])
ax_deposits.set_xlabel('Paleolongitude')
ax_deposits.set_ylabel('Paleolatitude')
ax_deposits.set_title('Climate-Sensitive Lithologic Deposits from the Miocene')
ax_deposits.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax_deposits.axis('scaled')
fig.savefig(images_folder+"scatter_deposits_miocene.png", dpi=fig_dpi, bbox_inches='tight', pad_inches=0.3)


# ### Coal
coal_rows = miocene_deposits.loc[:,'LithologyCode']=='C'
miocene_deposits_coal = miocene_deposits.loc[coal_rows,:].copy()
mapbin_deposits_coal = np.zeros((nlatbins,nlonbins))
lon_all = miocene_deposits_coal.loc[:,'Paleolongitude'].copy()
lat_all = miocene_deposits_coal.loc[:,'Paleolatitude'].copy()
for ilon,lon in enumerate(lon_coords):
    for ilat,lat in enumerate(lat_coords):
        this_paleolongitude = (lon<lon_all) & (lon_all<(lon+lon_spacing))
        this_paleolatitude  = (lat<lat_all) & (lat_all<(lat+lat_spacing[ilat]))
        if(this_paleolongitude & this_paleolatitude).any():
            mapbin_deposits_coal[nlatbins-ilat-1,ilon] = 1
fig = plt.figure(figsize=(16,8))
ax_deposits_coal = sns.heatmap(mapbin_deposits_coal, square=True, cbar=False, cmap='Greens', xticklabels=False, yticklabels=False)
ax_deposits_coal.set_xlabel('Paleolongitude')
ax_deposits_coal.set_ylabel('Paleolatitude')
ax_deposits_coal.set_title('Presence of coal deposits during Miocene')
fig.savefig(images_folder+"map_deposits_coal_miocene.png", bbox_inches='tight', pad_inches=0.3)

# ### Evaporites
evaporites_rows = miocene_deposits.loc[:,'LithologyCode']=='E'
miocene_deposits_evaporites = miocene_deposits.loc[evaporites_rows,:].copy()
mapbin_deposits_evaporites = np.zeros((nlatbins,nlonbins))
lon_all = miocene_deposits_evaporites.loc[:,'Paleolongitude'].copy()
lat_all = miocene_deposits_evaporites.loc[:,'Paleolatitude'].copy()
for ilon,lon in enumerate(lon_coords):
    for ilat,lat in enumerate(lat_coords):
        this_paleolongitude = (lon<lon_all) & (lon_all<(lon+lon_spacing))
        this_paleolatitude  = (lat<lat_all) & (lat_all<(lat+lat_spacing[ilat]))
        if(this_paleolongitude & this_paleolatitude).any():
            mapbin_deposits_evaporites[nlatbins-ilat-1,ilon] = 1
fig = plt.figure(figsize=(16,8))
ax_deposits_evaporites = sns.heatmap(mapbin_deposits_evaporites, square=True, cbar=False, cmap='Oranges', xticklabels=False, yticklabels=False)
ax_deposits_evaporites.set_xlabel('Paleolongitude')
ax_deposits_evaporites.set_ylabel('Paleolatitude')
ax_deposits_evaporites.set_title('Presence of evaporite deposits during Miocene')
fig.savefig(images_folder+"map_deposits_evaporites_miocene.png", bbox_inches='tight', pad_inches=0.3)

# ### Glacial
glacial_rows = (miocene_deposits.loc[:,'LithologyCode']=='T') | (miocene_deposits.loc[:,'LithologyCode']=='G')
miocene_deposits_glacial = miocene_deposits.loc[glacial_rows,:].copy()
mapbin_deposits_glacial = np.zeros((nlatbins,nlonbins))
lon_all = miocene_deposits_glacial.loc[:,'Paleolongitude'].copy()
lat_all = miocene_deposits_glacial.loc[:,'Paleolatitude'].copy()
for ilon,lon in enumerate(lon_coords):
    for ilat,lat in enumerate(lat_coords):
        this_paleolongitude = (lon<lon_all) & (lon_all<(lon+lon_spacing))
        this_paleolatitude  = (lat<lat_all) & (lat_all<(lat+lat_spacing[ilat]))
        if(this_paleolongitude & this_paleolatitude).any():
            mapbin_deposits_glacial[nlatbins-ilat-1,ilon] = 1
fig = plt.figure(figsize=(16,8))
ax_deposits_glacial = sns.heatmap(mapbin_deposits_glacial, square=True, cbar=False, cmap='Blues', xticklabels=False, yticklabels=False)
ax_deposits_glacial.set_xlabel('Paleolongitude')
ax_deposits_glacial.set_ylabel('Paleolatitude')
ax_deposits_glacial.set_title('Presence of glacial deposits during Miocene')
fig.savefig(images_folder+"map_deposits_glacial_miocene.png", bbox_inches='tight', pad_inches=0.3)


# ## Read continental and topological data
miocene_time = time_deposits.loc[miocene_rows_deposits].iloc[0]
at_times = [miocene_time] * nlonbins * nlatbins
at_longitudes = list(lon_coords+0.5*lon_spacing) * nlatbins
at_latitudes = []
for ilat, lat in enumerate(lat_coords):
    at_latitudes = at_latitudes + [lat+0.5*lat_spacing[ilat]] * nlonbins 
at_latitudes = at_latitudes[::-1]

overland  = np.zeros((nlonbins*nlatbins))
shoredist = np.zeros((nlonbins*nlatbins))
shoredirc = np.zeros((nlonbins*nlatbins))
shoreline_results = qpg.query_paleo_shoreline_at_points(at_latitudes, at_longitudes, at_times)
print('Shoreline ...')
for idx, result in enumerate(shoreline_results):
    overland[idx]  = result[0]
    shoredist[idx] = result[1]
    shoredirc[idx] = result[2]
map_overland_binary = overland.reshape((nlatbins,nlonbins))
map_shore_distance  = shoredist.reshape((nlatbins,nlonbins))
map_shore_direction = shoredirc.reshape((nlatbins,nlonbins))

fig = plt.figure(figsize=(16,8))
ax_overland = fig.add_subplot(111)
ax_overland = sns.heatmap(map_overland_binary, cmap='coolwarm', square=True, cbar=False, xticklabels=False, yticklabels=False)
ax_overland.set_xlabel('Paleolongitude')
ax_overland.set_ylabel('Paleolatitude')
ax_overland.set_title('Distribution of land-mass during Miocene')
fig.savefig(images_folder+"map_overland_miocene.png", bbox_inches='tight', pad_inches=0.3)


# MASKING
unmask_overland = map_overland_binary.astype(bool)
unmask_deposits = (mapbin_deposits_coal.astype(bool) | mapbin_deposits_evaporites.astype(bool)) | mapbin_deposits_glacial.astype(bool)
unmask_include = unmask_overland | unmask_deposits
mask_exclude = np.invert(unmask_include)

unmask_deposits_shallowmarine = unmask_deposits & np.invert(unmask_overland)
map_shore_distance[unmask_deposits_shallowmarine] = 0
map_shore_direction[unmask_deposits_shallowmarine] = map_shore_direction[unmask_deposits_shallowmarine] - 180.


# REPLOT (Distance, Deposits)
fig = plt.figure(figsize=(16,8))
cmap_shoredist = sns.cubehelix_palette(8, start=2., rot=-.3, dark=0.3, light=0.7, reverse=True, as_cmap=True)
vmin=0.
vmax=3000.
ax_shoredist = fig.add_subplot(111)
ax_shoredist = sns.heatmap(map_shore_distance, cmap=cmap_shoredist, vmin=vmin, vmax=vmax, square=True, cbar=True, cbar_kws={"shrink": .75}, xticklabels=False, yticklabels=False, mask=mask_exclude)
ax_shoredist.set_xlabel('Paleolongitude')
ax_shoredist.set_ylabel('Paleolatitude')
ax_shoredist.set_title('Distance to shoreline during Miocene')
fig.savefig(images_folder+"map_shoredist_miocene.png", bbox_inches='tight', pad_inches=0.3)


# ## PRECIPITATION
print("Precipitation...")
data_rain = xr.open_dataset(data_rainfall_filename)
lon_list  = data_rain.variables["lon"].values
lat_list  = data_rain.variables["lat"].values
prc_array = data_rain.variables["z"].values
if np.isnan(np.sum(prc_array)):
    print("Warning: NaN in prc_array")
med_precipitation = np.median(prc_array) # just for a sensible color-scale
map_precipitation = np.ones((nlatbins,nlonbins)) * med_precipitation
map_log10precip   = np.zeros((nlatbins,nlonbins))
for ilon,lon in enumerate(lon_coords):
    for ilat,lat in enumerate(lat_coords):
        this_paleolongitude = (lon<lon_list) & (lon_list<(lon+lon_spacing))
        this_paleolatitude  = (lat<lat_list) & (lat_list<(lat+lat_spacing[ilat]))
        try:
            prc_here = prc_array[this_paleolatitude , this_paleolongitude]
            map_precipitation[nlatbins-ilat-1,ilon] = np.mean(prc_here)
            map_log10precip[nlatbins-ilat-1,ilon] = np.log10(np.mean(prc_here))
        except:
            map_precipitation[nlatbins-ilat-1,ilon] = None
            map_log10precip[nlatbins-ilat-1,ilon]   = None
if np.isnan(np.sum(map_precipitation)):
    print("Warning: NaN in map_precipitation")


fig, ax_rainfall_mask = plt.subplots(figsize=(16,8))            
ax_rainfall_mask = sns.heatmap(map_precipitation, cmap='RdBu', cbar=True, cbar_kws={"shrink": .75}, square=True, xticklabels=False, yticklabels=False, mask=mask_exclude)
ax_rainfall_mask.set_xlabel('Paleolongitude')
ax_rainfall_mask.set_ylabel('Paleolatitude')
ax_rainfall_mask.set_title('Annual Precipitation during Miocene')
fig.savefig(images_folder+"map_precipitation_miocene.png", bbox_inches='tight', pad_inches=0.3)

fig, ax_lograinfall = plt.subplots(figsize=(16,8))            
cmap = sns.cubehelix_palette(8, start=0.65, rot=-0.9, light=0.9, as_cmap=True)
ax_lograinfall = sns.heatmap(map_log10precip, cmap=cmap, cbar=True, cbar_kws={"shrink": .75}, square=True, xticklabels=False, yticklabels=False, mask=mask_exclude)
ax_lograinfall.set_xlabel('Paleolongitude')
ax_lograinfall.set_ylabel('Paleolatitude')
ax_lograinfall.set_title('Annual Log10 Precipitation during Miocene')
fig.savefig(images_folder+"map_log10precip_miocene.png", bbox_inches='tight', pad_inches=0.3)



elevation_arr = np.zeros((nlonbins*nlatbins))
topography_results = qpg.query_paleo_topography_at_points(at_latitudes, at_longitudes, at_times)
print('Topography...')
for idx, height in enumerate(topography_results):
    elevation_arr[idx] = height
map_elevation = elevation_arr.reshape((nlatbins,nlonbins))
map_elevation[unmask_deposits_shallowmarine] = 0
cmap_elevation = sns.diverging_palette(240, 20, s=85, l=28, n=9)

fig, ax_elevation_mask = plt.subplots(figsize=(16,8))            
ax_elevation_mask = sns.heatmap(map_elevation, cmap=cmap_elevation, cbar=True, cbar_kws={"shrink": .75}, mask=mask_exclude, square=True, xticklabels=False, yticklabels=False)
ax_elevation_mask.set_xlabel('Paleolongitude')
ax_elevation_mask.set_ylabel('Paleolatitude')
ax_elevation_mask.set_title('Height/Depth above/below sea-level during Miocene')
fig.savefig(images_folder+"map_elevation_miocene.png", bbox_inches='tight', pad_inches=0.3)




# ## Reformat data for GP modelling code
with open(data_folder+'precipitation_gpmodel_data.csv','wb') as csvfile:
    wr = csv.writer(csvfile)
    wr.writerow(['region_id',
                 'centroid_x', 'centroid_y', 'Precipitation', 'Log Precipitation', 
                 'Coal Deposits', 'Evaporites Deposits', 'Glacial Deposits', 
                 'Latitude', 'Longitude', 'Abs Latitude', 'Elevation', 'Sqrt Elevation', 'ASinh Elevation Sc50', 'Dist to Shore', 'Sqrt Dist', 'ASinh Dist Sc50', 'Sin Angle Shore', 'Cos Angle Shore'])
    ridx = 0
    for ilon,lon in enumerate(lon_coords):
        lonm = lon + 0.5*lon_spacing
        for ilat,lat in enumerate(lat_coords):
            latm = lat + 0.5*lat_spacing[ilat]
            land_here = unmask_include[nlatbins-ilat-1,ilon]
            if(land_here):
                prec = map_precipitation[nlatbins-ilat-1,ilon]
                logp = map_log10precip[nlatbins-ilat-1,ilon]
                ablt = abs(latm)
                coal = mapbin_deposits_coal[nlatbins-ilat-1,ilon]
                evap = mapbin_deposits_evaporites[nlatbins-ilat-1,ilon]
                glac = mapbin_deposits_glacial[nlatbins-ilat-1,ilon]
                elev = map_elevation[nlatbins-ilat-1,ilon]
                if elev<0:
                    elev = 0.
                dist = map_shore_distance[nlatbins-ilat-1,ilon]
                dird = map_shore_direction[nlatbins-ilat-1,ilon]
                dirr = radians(dird)
                row_fpos  = [latm, lonm, prec, logp]
                row_ideps = [coal, evap, glac]
                row_fcovs = [latm, lonm, ablt, elev, sqrt(elev), asinh(elev/50.), dist, sqrt(dist), asinh(dist/50.), sin(dirr), cos(dirr)]
                row_check = row_fpos + row_ideps + row_fcovs
                if not np.isnan(row_check).any():
                    ridx = ridx + 1
                    row_fpos_short  = map(lambda num: "%.2f"%num , row_fpos)
                    row_fcovs_short = map(lambda num: "%.2f"%num , row_fcovs)
                    row_info = [ridx] + row_fpos_short + row_ideps + row_fcovs_short
                    wr.writerow(row_info)
