
# coding: utf-8

# # Precipitation Predict
# 
# ## Madhura Killedar
# ## date   : 27/04/2018
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
import query_paleogeography as qpg
import csv
import random

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
data_folder = 'data/'
data_deposits_filename = data_folder+'LithData_PaleoXY_Matthews2016_20180226.csv'
key_deposits_filename  = data_folder+'LithologyCodes.csv'

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

# remove polar grid-bins
nlatbins = nlatbins-2
lat_coords = lat_coords[1:-1]
lat_spacing = lat_spacing[1:-1]

# ## READ sedimentary deposits data
print("Deposits...")
data_deposits = pd.read_csv(data_deposits_filename)
key_deposits  = pd.read_csv(key_deposits_filename)
dict_deposits = dict(zip(key_deposits.loc[:,'LithologyCode'], key_deposits.loc[:,'LithologyType']))
time_deposits = data_deposits.loc[:,'ReconstructionTime'].copy()

all_epochs_int = []
for ep in pd.to_numeric(time_deposits).tolist(): 
    #if((6<ep) and (ep<395)):
    if((6<ep) and (ep<251.9)): # lower bound due to GPlates code, upper bound Permian
        all_epochs_int.append(int(ep))
all_epochs_int = set(all_epochs_int)
print(all_epochs_int)


# ## GET sedimentary deposits data
def get_deposits(epoch_time_min, epoch_time_max, time_deposits, data_deposits):
    epoch_rows_deposits = (epoch_time_min < time_deposits) & (time_deposits < epoch_time_max)
    epoch_deposits = data_deposits.loc[epoch_rows_deposits,:].copy()
    print("Number of deposits at this epoch = ",sum(epoch_rows_deposits))
    return epoch_deposits


# ## PLOT sedimentary deposits data
def plot_deposits_scatter(epoch_deposits, epoch_name, dict_deposits, output_folder):
    fig = plt.figure(figsize=(16,8))
    ax_deposits = fig.add_subplot(111)
    epoch_deposits_grouped = epoch_deposits.groupby('LithologyCode')
    for deposit_code, deposit in epoch_deposits_grouped:
        if deposit_code in dict_deposits:
            ax_deposits.plot(deposit.Paleolongitude, deposit.Paleolatitude, marker='o', linestyle='', ms=3, label=dict_deposits[deposit_code])
    ax_deposits.set_xlabel('Paleolongitude')
    ax_deposits.set_ylabel('Paleolatitude')
    ax_deposits.set_title('Climate-Sensitive Lithologic Deposits from the '+epoch_name)
    ax_deposits.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax_deposits.axis('scaled')
    fig.savefig(output_folder+"scatter_deposits_"+epoch_name+".png", dpi=fig_dpi, bbox_inches='tight', pad_inches=0.3)
    plt.close(fig)

def map_deposits_gridded(epoch_deposits, epoch_name, output_folder, lon_coords, lat_coords, lon_spacing, lat_spacing, nlatbins, nlonbins):
    # ### Coal
    coal_rows = [False]*epoch_deposits.shape[0]
    coal_codes = ['C','PA','M','CR','B','L','K','O']
    for code in coal_codes:
        this_code_rows = epoch_deposits.loc[:,'LithologyCode']==code
        coal_rows = coal_rows | this_code_rows
    epoch_deposits_coal = epoch_deposits.loc[coal_rows,:].copy()
    mapbin_deposits_coal = np.zeros((nlatbins,nlonbins))
    lon_all = epoch_deposits_coal.loc[:,'Paleolongitude'].copy()
    lat_all = epoch_deposits_coal.loc[:,'Paleolatitude'].copy()
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
    ax_deposits_coal.set_title('Presence of coal deposits during '+epoch_name)
    fig.savefig(output_folder+"map_deposits_coal_"+epoch_name+".png", bbox_inches='tight', pad_inches=0.3)
    plt.close(fig)
    # ### Evaporites
    evaporites_rows = [False]*epoch_deposits.shape[0]
    evaporites_codes = ['E','CA']
    for code in evaporites_codes:
        this_code_rows = epoch_deposits.loc[:,'LithologyCode']==code
        evaporites_rows = evaporites_rows | this_code_rows
    epoch_deposits_evaporites = epoch_deposits.loc[evaporites_rows,:].copy()
    mapbin_deposits_evaporites = np.zeros((nlatbins,nlonbins))
    lon_all = epoch_deposits_evaporites.loc[:,'Paleolongitude'].copy()
    lat_all = epoch_deposits_evaporites.loc[:,'Paleolatitude'].copy()
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
    ax_deposits_evaporites.set_title('Presence of evaporite deposits during '+epoch_name)
    fig.savefig(output_folder+"map_deposits_evaporites_"+epoch_name+".png", bbox_inches='tight', pad_inches=0.3)
    plt.close(fig)
    # ### Glacial
    glacial_rows = [False]*epoch_deposits.shape[0]
    glacial_codes = ['G','T','D']
    for code in glacial_codes:
        this_code_rows = epoch_deposits.loc[:,'LithologyCode']==code
        glacial_rows = glacial_rows | this_code_rows
    epoch_deposits_glacial = epoch_deposits.loc[glacial_rows,:].copy()
    mapbin_deposits_glacial = np.zeros((nlatbins,nlonbins))
    lon_all = epoch_deposits_glacial.loc[:,'Paleolongitude'].copy()
    lat_all = epoch_deposits_glacial.loc[:,'Paleolatitude'].copy()
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
    ax_deposits_glacial.set_title('Presence of glacial deposits during '+epoch_name)
    fig.savefig(output_folder+"map_deposits_glacial_"+epoch_name+".png", bbox_inches='tight', pad_inches=0.3)
    plt.close(fig)
    # return
    return {'coal':mapbin_deposits_coal, 'evap':mapbin_deposits_evaporites, 'glac':mapbin_deposits_glacial}



# ## READ continental and topological data
def get_qpg_results(epoch_time, nlonbins, nlatbins, lon_coords, lat_coords, lon_spacing, lat_spacing):
    at_times = [epoch_time] * nlonbins * nlatbins
    at_longitudes = list(lon_coords+0.5*lon_spacing) * nlatbins
    at_latitudes = []
    for ilat, lat in enumerate(lat_coords):
        at_latitudes = at_latitudes + [lat+0.5*lat_spacing[ilat]] * nlonbins 
    at_latitudes = at_latitudes[::-1]
    shoreline_results = qpg.query_paleo_shoreline_at_points(at_latitudes, at_longitudes, at_times)
    topography_results = qpg.query_paleo_topography_at_points(at_latitudes, at_longitudes, at_times)
    return shoreline_results, topography_results



def make_masks(nlonbins, nlatbins, shoreline_results, output_folder, epoch_name, mapbins):
    overland  = np.zeros((nlonbins*nlatbins))
    for idx, result in enumerate(shoreline_results):
        overland[idx]  = result[0]
    map_overland_binary = overland.reshape((nlatbins,nlonbins))
    fig = plt.figure(figsize=(18,6))
    ax_overland = fig.add_subplot(111)
    ax_overland = sns.heatmap(map_overland_binary, cmap='binary', square=True, cbar=False, xticklabels=False, yticklabels=False)
    ax_overland.set_xlabel('Paleolongitude')
    ax_overland.set_ylabel('Paleolatitude')
    ax_overland.set_title('Distribution of land-mass during '+epoch_name)
    fig.savefig(output_folder+"map_overland_"+epoch_name+".png", bbox_inches='tight', pad_inches=0.3)
    plt.close(fig)
    mapbin_deposits_coal, mapbin_deposits_evaporites, mapbin_deposits_glacial = mapbins['coal'], mapbins['evap'], mapbins['glac']
    unmask_deposits = (mapbin_deposits_coal.astype(bool) | mapbin_deposits_evaporites.astype(bool)) | mapbin_deposits_glacial.astype(bool)
    unmask_overland = map_overland_binary.astype(bool)
    unmask_LandOrDeposits = unmask_overland | unmask_deposits
    unmask_deposits_shallow = unmask_deposits & np.invert(unmask_overland)
    return unmask_deposits_shallow, unmask_deposits, unmask_LandOrDeposits



def make_map_shore(nlonbins, nlatbins, shoreline_results, unmask_deposits_shallow, mask_exclude, output_folder, epoch_name):
    shoredist = np.zeros((nlonbins*nlatbins))
    shoredirc = np.zeros((nlonbins*nlatbins))
    print('Shoreline ...')
    for idx, result in enumerate(shoreline_results):
        shoredist[idx] = result[1]
        shoredirc[idx] = result[2]
    map_shore_distance  = shoredist.reshape((nlatbins,nlonbins))
    map_shore_direction = shoredirc.reshape((nlatbins,nlonbins))
    # fix shallow areas
    map_shore_distance[unmask_deposits_shallow] = 0
    map_shore_direction[unmask_deposits_shallow] = map_shore_direction[unmask_deposits_shallow] - 180.
    # PLOT Distance
    cmap_shoredist = sns.cubehelix_palette(8, start=2., rot=-.3, dark=0.3, light=0.7, reverse=True, as_cmap=True)
    fig = plt.figure(figsize=(18,6))
    ax_shoredist = fig.add_subplot(111)
    ax_shoredist = sns.heatmap(map_shore_distance, cmap=cmap_shoredist, vmin=0., vmax=3000., square=True, cbar=True, cbar_kws={"shrink": .75},
                               xticklabels=False, yticklabels=False, mask=mask_exclude)
    ax_shoredist.set_xlabel('Paleolongitude')
    ax_shoredist.set_ylabel('Paleolatitude')
    ax_shoredist.set_title('Distance to shoreline during '+epoch_name)
    fig.savefig(output_folder+"map_shoredist_"+epoch_name+".png", bbox_inches='tight', pad_inches=0.3)
    plt.close(fig)
    return map_shore_distance, map_shore_direction



def make_map_elevation(nlonbins, nlatbins, topography_results, unmask_deposits_shallow, mask_exclude, output_folder, epoch_name):
    print('Topography/Elevation...')
    elevation_arr = np.zeros((nlonbins*nlatbins))
    for idx, height in enumerate(topography_results):
        elevation_arr[idx] = height
    map_elevation = elevation_arr.reshape((nlatbins,nlonbins))
    # fix shallow areas
    map_elevation[unmask_deposits_shallow] = 0
    # PLOT Elevation
    cmap_elevation = sns.cubehelix_palette(8, start=1.7, rot=.35, dark=0.4, light=0.9, reverse=True, as_cmap=True)
    fig, ax_elevation_mask = plt.subplots(figsize=(18,6))            
    ax_elevation_mask = sns.heatmap(map_elevation, vmin=0., vmax=5000., cmap=cmap_elevation, cbar=True, cbar_kws={"shrink": .75},
                                    mask=mask_exclude, square=True, xticklabels=False, yticklabels=False)
    ax_elevation_mask.set_xlabel('Paleolongitude')
    ax_elevation_mask.set_ylabel('Paleolatitude')
    ax_elevation_mask.set_title('Height/Depth above/below sea-level during '+epoch_name)
    fig.savefig(output_folder+"map_elevation_"+epoch_name+".png", bbox_inches='tight', pad_inches=0.3)
    plt.close(fig)
    return map_elevation



def writeout_data(data_folder, epoch_name, data_subset, lon_coords, lat_coords, mask_exclude, mapbins, map_elevation, map_shore_distance, map_shore_direction):
    # ## Reformat data for GP predicting code
    print("Reformatting data for GP modelling...")
    mapbin_deposits_coal, mapbin_deposits_evaporites, mapbin_deposits_glacial = mapbins['coal'], mapbins['evap'], mapbins['glac']
    with open(data_folder+'predictor_data_'+epoch_name+'_'+data_subset+'.csv','wb') as csvfile:
        wr = csv.writer(csvfile)
        wr.writerow(['region_id', 'centroid_x', 'centroid_y',
                     'Coal Deposits', 'Evaporites Deposits', 'Glacial Deposits', 
                     'Elevation', 'Sqrt Elevation', 'Dist to Shore', 'Sqrt Dist', 'Sin Angle Shore', 'Cos Angle Shore'])
        ridx = 0
        for ilon,lon in enumerate(lon_coords):
            lonm = lon + 0.5*lon_spacing
            for ilat,lat in enumerate(lat_coords):
                latm = lat + 0.5*lat_spacing[ilat]
                include_point = np.invert(mask_exclude[nlatbins-ilat-1,ilon])
                if(include_point):
                    coal = mapbin_deposits_coal[nlatbins-ilat-1,ilon]
                    evap = mapbin_deposits_evaporites[nlatbins-ilat-1,ilon]
                    glac = mapbin_deposits_glacial[nlatbins-ilat-1,ilon]
                    elev = max(map_elevation[nlatbins-ilat-1,ilon],0)
                    dist = map_shore_distance[nlatbins-ilat-1,ilon]
                    dird = map_shore_direction[nlatbins-ilat-1,ilon]
                    dirr = radians(dird)
                    row_fpos  = [latm, lonm]
                    row_ideps = [coal, evap, glac]
                    row_fcovs = [elev, sqrt(elev), dist, sqrt(dist), sin(dirr), cos(dirr)]
                    row_check = row_fpos + row_ideps + row_fcovs
                    if not np.isnan(row_check).any():
                        ridx = ridx + 1
                        row_fpos_short  = map(lambda num: "%.2f"%num , row_fpos)
                        row_fcovs_short = map(lambda num: "%.2f"%num , row_fcovs)
                        row_info = [ridx] + row_fpos_short + row_ideps + row_fcovs_short
                        wr.writerow(row_info)




# CALL FUNCTIONS
for epoch_time in all_epochs_int:
    epoch_time_min = epoch_time - 1
    epoch_time_max = epoch_time + 1
    epoch_name = str(epoch_time)+'Ma'
    print(epoch_name)
    epoch_deposits = get_deposits(epoch_time_min, epoch_time_max, time_deposits, data_deposits)
    plot_deposits_scatter(epoch_deposits, epoch_name, dict_deposits, images_folder)
    mapbins = map_deposits_gridded(epoch_deposits, epoch_name, images_folder, lon_coords, lat_coords, lon_spacing, lat_spacing, nlatbins, nlonbins)
    shoreline_results, topography_results = get_qpg_results(epoch_time, nlonbins, nlatbins, lon_coords, lat_coords, lon_spacing, lat_spacing)
    unmask_deposits_shallow, unmask_deposits, unmask_LandOrDeposits = make_masks(nlonbins, nlatbins, shoreline_results, images_folder, epoch_name, mapbins)
    for data_subset in ['land','deposit']:
        if data_subset == 'land': # choose only locations over land or shallow-marine areas with deposits
            mask_exclude = np.invert(unmask_LandOrDeposits) # larger dataset
        elif data_subset == 'deposit': # choose only locations where some type of deposit has been found
            mask_exclude = np.invert(unmask_deposits) # smaller dataset
        else:
            raise ValueError('I dont recognise that data subset!')
        map_shore_distance, map_shore_direction = make_map_shore(nlonbins, nlatbins, shoreline_results, unmask_deposits_shallow, mask_exclude, images_folder, epoch_name)
        map_elevation = make_map_elevation(nlonbins, nlatbins, topography_results, unmask_deposits_shallow, mask_exclude, images_folder, epoch_name)
        writeout_data(data_folder, epoch_name, data_subset, lon_coords, lat_coords, mask_exclude, mapbins, map_elevation, map_shore_distance, map_shore_direction)

