
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
lon_spacing = 2.51

lon_coords = np.arange(lon_min,lon_max-0.5*lon_spacing,lon_spacing) # left edge of grid
nlonbins = len(lon_coords)
print("Number of Longitudinal bins = %i"%nlonbins)

nlatbins = 2 * int(nlonbins * lat_max / 561.) # rough guess to allow a square bin at equator
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
print("Edge latitude bins")
print(lat_coords[0],lat_coords[-1])
print("Edge longitude bins")
print(lon_coords[0],lon_coords[-1])

# remove polar grid-bins
nlatbins = nlatbins-2
lat_coords = lat_coords[1:-1]
lat_spacing = lat_spacing[1:-1]

print("All Latitude Coordinates:")
print(lat_coords)
    
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
plt.close(fig)

mapbin_deposits_all = np.zeros((nlatbins,nlonbins))


# ### Coal
coal_rows = [False]*miocene_deposits.shape[0]
coal_codes = ['C','PA','M','CR','B','L','K','O']
for code in coal_codes:
    this_code_rows = miocene_deposits.loc[:,'LithologyCode']==code
    coal_rows = coal_rows | this_code_rows
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
            mapbin_deposits_all[nlatbins-ilat-1,ilon] = mapbin_deposits_all[nlatbins-ilat-1,ilon] + 1
fig = plt.figure(figsize=(16,8))
ax_deposits_coal = sns.heatmap(mapbin_deposits_coal, square=True, cbar=False, cmap='Greens', xticklabels=False, yticklabels=False)
ax_deposits_coal.set_xlabel('Paleolongitude')
ax_deposits_coal.set_ylabel('Paleolatitude')
ax_deposits_coal.set_title('Presence of coal deposits during Miocene')
fig.savefig(images_folder+"map_deposits_coal_miocene.png", bbox_inches='tight', pad_inches=0.3)
plt.close(fig)

# ### Evaporites
evaporite_specific_rows = miocene_deposits.loc[:,'LithologyCode']=='E'
calcrete_rows = miocene_deposits.loc[:,'LithologyCode']=='CA'
evaporites_rows = evaporite_specific_rows | calcrete_rows
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
            mapbin_deposits_all[nlatbins-ilat-1,ilon] = mapbin_deposits_all[nlatbins-ilat-1,ilon] + 2
fig = plt.figure(figsize=(16,8))
ax_deposits_evaporites = sns.heatmap(mapbin_deposits_evaporites, square=True, cbar=False, cmap='Oranges', xticklabels=False, yticklabels=False)
ax_deposits_evaporites.set_xlabel('Paleolongitude')
ax_deposits_evaporites.set_ylabel('Paleolatitude')
ax_deposits_evaporites.set_title('Presence of evaporite deposits during Miocene')
fig.savefig(images_folder+"map_deposits_evaporites_miocene.png", bbox_inches='tight', pad_inches=0.3)
plt.close(fig)

# ### Glacial
glacial_rows = [False]*miocene_deposits.shape[0]
glacial_codes = ['T','G','D']
for code in glacial_codes:
    this_code_rows = miocene_deposits.loc[:,'LithologyCode']==code
    glacial_rows = glacial_rows | this_code_rows
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
            mapbin_deposits_all[nlatbins-ilat-1,ilon] = mapbin_deposits_all[nlatbins-ilat-1,ilon] + 3
fig = plt.figure(figsize=(16,8))
ax_deposits_glacial = sns.heatmap(mapbin_deposits_glacial, square=True, cbar=False, cmap='Blues', xticklabels=False, yticklabels=False)
ax_deposits_glacial.set_xlabel('Paleolongitude')
ax_deposits_glacial.set_ylabel('Paleolatitude')
ax_deposits_glacial.set_title('Presence of glacial deposits during Miocene')
fig.savefig(images_folder+"map_deposits_glacial_miocene.png", bbox_inches='tight', pad_inches=0.3)
plt.close(fig)

# ### Deposit types
fig = plt.figure(figsize=(16,8))
ax_deposits_all = sns.heatmap(mapbin_deposits_all, square=True, cbar=False, cmap='Set1', xticklabels=False, yticklabels=False)
ax_deposits_all.set_xlabel('Paleolongitude')
ax_deposits_all.set_ylabel('Paleolatitude')
ax_deposits_all.set_title('Deposit-types during Miocene')
fig.savefig(images_folder+"map_deposits_all_miocene.png", bbox_inches='tight', pad_inches=0.3)
plt.close(fig)

# ### Number of Deposits
mapbin_deposits_number = mapbin_deposits_coal + mapbin_deposits_evaporites + mapbin_deposits_glacial
fig = plt.figure(figsize=(16,8))
ax_deposits_number = sns.heatmap(mapbin_deposits_number, square=True, cbar=False, cmap='Purples', xticklabels=False, yticklabels=False)
ax_deposits_number.set_xlabel('Paleolongitude')
ax_deposits_number.set_ylabel('Paleolatitude')
ax_deposits_number.set_title('Number of deposit-types during Miocene')
fig.savefig(images_folder+"map_deposits_number_miocene.png", bbox_inches='tight', pad_inches=0.3)
plt.close(fig)


# ## Read continental and topological data
#miocene_time = time_deposits.loc[miocene_rows_deposits].iloc[0]
#miocene_time = 15.
#at_times = [miocene_time] * nlonbins * nlatbins
at_longitudes = list(lon_coords+0.5*lon_spacing) * nlatbins
at_latitudes = []
for ilat, lat in enumerate(lat_coords):
    at_latitudes = at_latitudes + [lat+0.5*lat_spacing[ilat]] * nlonbins 
at_latitudes = at_latitudes[::-1]

overland  = np.zeros((nlonbins*nlatbins))
shoredist = np.zeros((nlonbins*nlatbins))
shoredirc = np.zeros((nlonbins*nlatbins))
#shoreline_results = qpg.query_paleo_shoreline_at_points(at_latitudes, at_longitudes, at_times)
shoreline_results = qpg.query_mid_miocene_shoreline_at_points(at_latitudes, at_longitudes)
print('Shoreline ...')
for idx, result in enumerate(shoreline_results):
    overland[idx]  = result[0]
    shoredist[idx] = result[1]
    shoredirc[idx] = result[2]
map_overland_binary = overland.reshape((nlatbins,nlonbins))
map_shoredistance  = shoredist.reshape((nlatbins,nlonbins))
map_shoredirection = shoredirc.reshape((nlatbins,nlonbins))

fig = plt.figure(figsize=(16,8))
ax_overland = fig.add_subplot(111)
ax_overland = sns.heatmap(map_overland_binary, cmap='coolwarm', square=True, cbar=False, xticklabels=False, yticklabels=False)
ax_overland.set_xlabel('Paleolongitude')
ax_overland.set_ylabel('Paleolatitude')
ax_overland.set_title('Distribution of land-mass during Miocene')
fig.savefig(images_folder+"map_overland_miocene.png", bbox_inches='tight', pad_inches=0.3)
plt.close(fig)


# MASKING
unmask_overland = map_overland_binary.astype(bool)
unmask_deposits = (mapbin_deposits_coal.astype(bool) | mapbin_deposits_evaporites.astype(bool)) | mapbin_deposits_glacial.astype(bool)
mask_nodeposits = np.invert(unmask_deposits)
unmask_include = unmask_deposits.copy() #(unmask_overland | unmask_deposits) & unmask_deposits
mask_exclude = np.invert(unmask_include)
unmask_deposits_shallowmarine = unmask_deposits & np.invert(unmask_overland)
map_shoredistance[unmask_deposits_shallowmarine] = 0
map_shoredirection[unmask_deposits_shallowmarine] = map_shoredirection[unmask_deposits_shallowmarine] - 180.
map_shoredistance_sqrt  = np.sqrt(map_shoredistance)


# REPLOT (Distance, Deposits)
fig = plt.figure(figsize=(16,8))
cmap_shoredist = sns.cubehelix_palette(8, start=2., rot=-.3, dark=0.3, light=0.7, reverse=True, as_cmap=True)
vmin=0.
vmax=1850.
ax_shoredist = fig.add_subplot(111)
ax_shoredist = sns.heatmap(map_shoredistance, cmap=cmap_shoredist, vmin=vmin, vmax=vmax, square=True, cbar=True, cbar_kws={"shrink": .75}, xticklabels=False, yticklabels=False, mask=mask_exclude)
ax_shoredist.set_xlabel('Paleolongitude')
ax_shoredist.set_ylabel('Paleolatitude')
ax_shoredist.set_title('Distance to shoreline during Miocene')
fig.savefig(images_folder+"map_shoredist_miocene.png", bbox_inches='tight', pad_inches=0.3)
plt.close(fig)


# ## PRECIPITATION
print("Precipitation...")
data_rain = xr.open_dataset(data_rainfall_filename)
lon_data  = data_rain.variables["lon"].values
lat_data  = data_rain.variables["lat"].values
prc_array = data_rain.variables["z"].values
if np.isnan(np.sum(prc_array)):
    print("Warning: NaN in prc_array")
med_precipitation = np.median(prc_array) # just for a sensible color-scale
map_precipitation = np.ones((nlatbins,nlonbins)) * med_precipitation
map_log10precip   = np.zeros((nlatbins,nlonbins))
for ilon,lon_low in enumerate(lon_coords):
    for ilat,lat_low in enumerate(lat_coords):
        lon_high = lon_low+lon_spacing
        lat_high = lat_low+lat_spacing[ilat]
        this_paleolongitude = (lon_low<lon_data) & (lon_data<lon_high)
        this_paleolatitude  = (lat_low<lat_data) & (lat_data<lat_high)
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
plt.close(fig)

fig, ax_lograinfall = plt.subplots(figsize=(16,8))            
cmap_logprecip = sns.cubehelix_palette(8, start=0.65, rot=-0.9, light=0.9, as_cmap=True)
ax_lograinfall = sns.heatmap(map_log10precip, cmap=cmap_logprecip, cbar=True, cbar_kws={"shrink": .75}, square=True, xticklabels=False, yticklabels=False, mask=mask_exclude)
ax_lograinfall.set_xlabel('Paleolongitude')
ax_lograinfall.set_ylabel('Paleolatitude')
ax_lograinfall.set_title('Annual Log10 Precipitation during Miocene')
fig.savefig(images_folder+"map_log10precip_miocene.png", bbox_inches='tight', pad_inches=0.3)
plt.close(fig)



elevation_arr = np.zeros((nlonbins*nlatbins))
#topography_results = qpg.query_paleo_topography_at_points(at_latitudes, at_longitudes, at_times)
topography_results = qpg.query_mid_miocene_topography_at_points(at_latitudes, at_longitudes)
print('Topography...')
for idx, height in enumerate(topography_results):
    elevation_arr[idx] = height
map_elevation      = elevation_arr.reshape((nlatbins,nlonbins))
map_elevation[map_elevation<0] = 0. # recalibrate to ignore shallow areas underwater
map_elevation_sqrt = map_elevation.copy()
map_elevation_sqrt[map_elevation>=0] = np.sqrt(map_elevation_sqrt[map_elevation>=0])
#map_elevation[unmask_deposits_shallowmarine] = 0
#cmap_elevation = sns.diverging_palette(240, 20, s=85, l=28, n=9)
cmap_elevation = sns.cubehelix_palette(8, start=0.5, rot=-.3, dark=0.3, light=0.7, as_cmap=True)


vmin=0.
vmax=5000.
fig, ax_elevation_mask = plt.subplots(figsize=(16,8))            
ax_elevation_mask = sns.heatmap(map_elevation, cmap=cmap_elevation, cbar=True, cbar_kws={"shrink": .75}, mask=mask_exclude, square=True, xticklabels=False, yticklabels=False)
ax_elevation_mask.set_xlabel('Paleolongitude')
ax_elevation_mask.set_ylabel('Paleolatitude')
ax_elevation_mask.set_title('Height above sea-level during Miocene')
fig.savefig(images_folder+"map_elevation_miocene.png", bbox_inches='tight', pad_inches=0.3)
plt.close(fig)

vmin=0.
vmax=70.
fig, ax_elevation_mask = plt.subplots(figsize=(16,8))            
ax_elevation_mask = sns.heatmap(map_elevation_sqrt, cmap=cmap_elevation, cbar=True, cbar_kws={"shrink": .75}, mask=mask_exclude, square=True, xticklabels=False, yticklabels=False)
ax_elevation_mask.set_xlabel('Paleolongitude')
ax_elevation_mask.set_ylabel('Paleolatitude')
ax_elevation_mask.set_title('Sqrt height above sea-level during Miocene')
fig.savefig(images_folder+"map_elevationsqrt_miocene.png", bbox_inches='tight', pad_inches=0.3)
plt.close(fig)



# SCATTERPLOT DEPENDENCIES: feature versus precipitation
print("Scatterplots...")

target  = map_log10precip[unmask_include].flatten()
target_label = 'Log10 Precipitation'
target_suffix = 'log10precip'
#target  = map_precipitation[unmask_include].flatten()
#target_label = 'Precipitation'
#target_suffix = 'precip'

feature = map_elevation[unmask_include].flatten()
fig, ax_precip_elev = plt.subplots(figsize=(7,7))            
ax_precip_elev.plot(feature, target, marker='o', linestyle='', ms=2)
ax_precip_elev.set_xlabel('Elevation')
ax_precip_elev.set_ylabel(target_label)
ax_precip_elev.set_title('Precipitation versus Elevation during Miocene')
fig.savefig(images_folder+"scatter_elevation_"+target_suffix+".png", bbox_inches='tight', pad_inches=0.2)
plt.close(fig)

feature = map_elevation_sqrt[unmask_include].flatten()
fig, ax_precip_elev = plt.subplots(figsize=(7,7))            
ax_precip_elev.plot(feature, target, marker='o', linestyle='', ms=2)
ax_precip_elev.set_xlabel('Sqrt Elevation')
ax_precip_elev.set_ylabel(target_label)
ax_precip_elev.set_title('Precipitation versus Elevation during Miocene')
fig.savefig(images_folder+"scatter_elevationsqrt_"+target_suffix+".png", bbox_inches='tight', pad_inches=0.2)
plt.close(fig)

feature = map_shoredistance[unmask_include].flatten()
fig, ax_precip_dist = plt.subplots(figsize=(7,7))            
ax_precip_dist.plot(feature, target, marker='o', linestyle='', ms=2)
ax_precip_dist.set_xlabel('Distance to Shore')
ax_precip_dist.set_ylabel(target_label)
ax_precip_dist.set_title('Precipitation versus Distance to Shore during Miocene')
fig.savefig(images_folder+"scatter_shoredist_"+target_suffix+".png", bbox_inches='tight', pad_inches=0.2)
plt.close(fig)

feature = map_shoredistance_sqrt[unmask_include].flatten()
fig, ax_precip_dist = plt.subplots(figsize=(7,7))            
ax_precip_dist.plot(feature, target, marker='o', linestyle='', ms=2)
ax_precip_dist.set_xlabel('Sqrt Distance to Shore')
ax_precip_dist.set_ylabel(target_label)
ax_precip_dist.set_title('Precipitation versus Distance to Shore during Miocene')
fig.savefig(images_folder+"scatter_shoredistsqrt"+target_suffix+".png", bbox_inches='tight', pad_inches=0.2)
plt.close(fig)

feature = np.sin(np.radians(map_shoredirection[unmask_include].flatten()))
fig, ax_precip_sind = plt.subplots(figsize=(7,7))            
ax_precip_sind.plot(feature, target, marker='o', linestyle='', ms=2)
ax_precip_sind.set_xlabel('Sin Direction to Shore')
ax_precip_sind.set_ylabel(target_label)
ax_precip_sind.set_title('Precipitation versus Sin of Direction to Shore during Miocene')
fig.savefig(images_folder+"scatter_shoresind_"+target_suffix+".png", bbox_inches='tight', pad_inches=0.2)
plt.close(fig)

feature = np.cos(np.radians(map_shoredirection[unmask_include].flatten()))
fig, ax_precip_cosd = plt.subplots(figsize=(7,7))            
ax_precip_cosd.plot(feature, target, marker='o', linestyle='', ms=2)
ax_precip_cosd.set_xlabel('Cos Direction to Shore')
ax_precip_cosd.set_ylabel(target_label)
ax_precip_cosd.set_title('Precipitation versus Cos of Direction to Shore during Miocene')
fig.savefig(images_folder+"scatter_shorecosd_"+target_suffix+".png", bbox_inches='tight', pad_inches=0.2)
plt.close(fig)

feature = mapbin_deposits_coal[unmask_include].flatten()
presence = ["Present" if f>0.5 else "Not found" for f in feature]
fig, ax_precip_coal = plt.subplots(figsize=(7,7))            
sns.violinplot(x=presence, y=target, ax=ax_precip_coal)
ax_precip_coal.set_xlabel('Coal')
ax_precip_coal.set_ylabel(target_label)
ax_precip_coal.set_title('Precipitation versus Presence of Coal during Miocene')
fig.savefig(images_folder+"violin_coal_"+target_suffix+".png", bbox_inches='tight', pad_inches=0.2)
plt.close(fig)

feature = mapbin_deposits_evaporites[unmask_include].flatten()
presence = ["Present" if f>0.5 else "Not found" for f in feature]
fig, ax_precip_evaporites = plt.subplots(figsize=(7,7))            
sns.violinplot(x=presence, y=target, ax=ax_precip_evaporites)
ax_precip_evaporites.set_xlabel('Evaporites')
ax_precip_evaporites.set_ylabel(target_label)
ax_precip_evaporites.set_title('Precipitation versus Presence of Evaporites during Miocene')
fig.savefig(images_folder+"violin_evaporites_"+target_suffix+".png", bbox_inches='tight', pad_inches=0.2)
plt.close(fig)

feature = mapbin_deposits_glacial[unmask_include].flatten()
presence = ["Present" if f>0.5 else "Not found" for f in feature]
fig, ax_precip_glacial = plt.subplots(figsize=(7,7))            
sns.violinplot(x=presence, y=target, ax=ax_precip_glacial)
ax_precip_glacial.set_xlabel('Glacial')
ax_precip_glacial.set_ylabel(target_label)
ax_precip_glacial.set_title('Precipitation versus Presence of Glacial during Miocene')
fig.savefig(images_folder+"violin_glacial_"+target_suffix+".png", bbox_inches='tight', pad_inches=0.2)
plt.close(fig)

# 2D scatterplot
cmap_logprecip = sns.cubehelix_palette(8, start=0.65, rot=-0.9, light=0.7, as_cmap=True)

feature1 = map_elevation[unmask_include].flatten()
feature2 = map_shoredistance[unmask_include].flatten()
fig, ax_precip_featfeat = plt.subplots(figsize=(7,7))
sc = ax_precip_featfeat.scatter(feature1, feature2, c=target, cmap=cmap_logprecip, alpha=0.4)
plt.colorbar(sc)
ax_precip_featfeat.set_xlabel('Elevation')
ax_precip_featfeat.set_ylabel('Distance to Shore')
ax_precip_featfeat.set_title(target_label + ' vs Predictors')
fig.savefig(images_folder+"scatter_elevdist_"+target_suffix+".png", bbox_inches='tight', pad_inches=0.3)
plt.close(fig)

feature1 = map_elevation_sqrt[unmask_include].flatten()
feature2 = map_shoredistance_sqrt[unmask_include].flatten()
fig, ax_precip_featfeat = plt.subplots(figsize=(7,7))
sc = ax_precip_featfeat.scatter(feature1, feature2, c=target, cmap=cmap_logprecip, alpha=0.4)
plt.colorbar(sc)
ax_precip_featfeat.set_xlabel('Sqrt Elevation')
ax_precip_featfeat.set_ylabel('Sqrt Distance to Shore')
ax_precip_featfeat.set_title(target_label + ' vs Predictors')
fig.savefig(images_folder+"scatter_elevdistsqrt_"+target_suffix+".png", bbox_inches='tight', pad_inches=0.3)
plt.close(fig)

feature1 = np.sin(np.radians(map_shoredirection[unmask_include].flatten()))
feature2 = map_shoredistance[unmask_include].flatten()
fig, ax_precip_featfeat = plt.subplots(figsize=(7,7))
sc = ax_precip_featfeat.scatter(feature1, feature2, c=target, cmap=cmap_logprecip, alpha=0.4)
plt.colorbar(sc)
ax_precip_featfeat.set_xlabel('Sin Direction to Shore')
ax_precip_featfeat.set_ylabel('Distance to Shore')
ax_precip_featfeat.set_title(target_label + ' vs Predictors')
fig.savefig(images_folder+"scatter_sinddist_"+target_suffix+".png", bbox_inches='tight', pad_inches=0.3)
plt.close(fig)

feature1 = np.cos(np.radians(map_shoredirection[unmask_include].flatten()))
feature2 = map_shoredistance[unmask_include].flatten()
fig, ax_precip_featfeat = plt.subplots(figsize=(7,7))
sc = ax_precip_featfeat.scatter(feature1, feature2, c=target, cmap=cmap_logprecip, alpha=0.4)
plt.colorbar(sc)
ax_precip_featfeat.set_xlabel('Cos Direction to Shore')
ax_precip_featfeat.set_ylabel('Distance to Shore')
ax_precip_featfeat.set_title(target_label + ' vs Predictors')
fig.savefig(images_folder+"scatter_cosddist_"+target_suffix+".png", bbox_inches='tight', pad_inches=0.3)
plt.close(fig)



# ## Reformat data for GP modelling code
print("Reformatting data for GP modelling...")
with open(data_folder+'precipitation_gpmodel_data.csv','wb') as csvfile:
    wr = csv.writer(csvfile)
    wr.writerow(['region_id',
                 'centroid_x', 'centroid_y', 'Precipitation', 'Log10 Precipitation', 
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
                dist = map_shoredistance[nlatbins-ilat-1,ilon]
                dird = map_shoredirection[nlatbins-ilat-1,ilon]
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
