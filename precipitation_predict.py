
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
#from mpl_toolkits.basemap import Basemap

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

GP_folder = 'GPresults/'
sys.path.insert(0, GP_folder)
from settings import x_feature_names
print('Using these covariates: ',x_feature_names)
MCMC_sampler_filename = GP_folder+'sampler_chain.npy'
tnow = datetime.datetime.now()
datestring = tnow.strftime('%Y-%m-%d')

output_folder = 'Predictions/'
print("The output path is %s"%(output_folder))
if not os.path.isdir(output_folder):
    os.makedirs(output_folder)


# ## Set up grids
lon_min = -180.
lon_max =  180.
lat_min =  -90.
lat_max =   90.
lon_spacing = 2.

lon_coords = np.arange(lon_min,lon_max,lon_spacing) # left edge of grid
nlonbins = len(lon_coords)
print("Number of Longitudinal bins = %i"%nlonbins)

nlatbins = int(nlonbins * lat_max / 280.) # rough guess to allow a square bin at equator
#nlatbins = int(2./(1.-cos(radians(lon_spacing + 90.))))
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


# Geological time period of interest
epoch_name = 'Miocene'


def get_time(epoch_name):
    if epoch_name=='Miocene':
        epoch_time_min = 5.33
        epoch_time_max = 23.03
        epoch_time_mid = 14.15
    elif epoch_name=='Oligocene':
        epoch_time_min = 23.03
        epoch_time_max = 33.9
        epoch_time_mid = 27.8
    elif epoch_name=='Eocene':
        epoch_time_min = 33.9
        epoch_time_max = 56.0
        epoch_time_mid = 47.8
    elif epoch_name=='Paleocene':
        epoch_time_min = 56.0
        epoch_time_max = 66.0
        epoch_time_mid = 61.0
    elif epoch_name=='LateCretaceous':
        epoch_time_min = 66.0
        epoch_time_max = 100.5
        epoch_time_mid = 86.3
    else:
        raise ValueError('I dont recognise that epoch name!')
    return epoch_time_min, epoch_time_max, epoch_time_mid


# ## GET sedimentary deposits data
def get_deposits(epoch_time_min, epoch_time_max, time_deposits, data_deposits):
    epoch_rows_deposits = (epoch_time_min < time_deposits) & (time_deposits < epoch_time_max)
    epoch_deposits = data_deposits.loc[epoch_rows_deposits,:].copy()
    return epoch_deposits
#try:
#    epoch_time = epoch_time_mid
#except:
#    epoch_time = time_deposits.loc[epoch_rows_deposits].iloc[0]


# ## PLOT sedimentary deposits data
def plot_deposits_scatter(epoch_deposits, epoch_name, output_folder):
    fig = plt.figure(figsize=(16,8))
    ax_deposits = fig.add_subplot(111)
    epoch_deposits_grouped = epoch_deposits.groupby('LithologyCode')
    for deposit_type, deposit in epoch_deposits_grouped:
        ax_deposits.plot(deposit.Paleolongitude, deposit.Paleolatitude, marker='o', linestyle='', ms=3, label=dict_deposits[deposit_type])
    ax_deposits.set_xlabel('Paleolongitude')
    ax_deposits.set_ylabel('Paleolatitude')
    ax_deposits.set_title('Climate-Sensitive Lithologic Deposits from the '+epoch_name)
    ax_deposits.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax_deposits.axis('scaled')
    fig.savefig(output_folder+"scatter_deposits_"+epoch_name+".png", dpi=fig_dpi, bbox_inches='tight', pad_inches=0.3)

def map_deposits_gridded(epoch_deposits, epoch_name, output_folder, lon_coords, lat_coords, lon_spacing, lat_spacing, nlatbins, nlonbins):
    # ### Coal
    coal_rows = epoch_deposits.loc[:,'LithologyCode']=='C'
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
    # ### Evaporites
    evaporites_rows = epoch_deposits.loc[:,'LithologyCode']=='E'
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
    # ### Glacial
    glacial_rows = (epoch_deposits.loc[:,'LithologyCode']=='T') | (epoch_deposits.loc[:,'LithologyCode']=='G')
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
    mapbin_deposits_coal, mapbin_deposits_evaporites, mapbin_deposits_glacial = mapbins['coal'], mapbins['evap'], mapbins['glac']
    unmask_deposits = (mapbin_deposits_coal.astype(bool) | mapbin_deposits_evaporites.astype(bool)) | mapbin_deposits_glacial.astype(bool)
    unmask_overland = map_overland_binary.astype(bool)
    unmask_include = unmask_overland | unmask_deposits
    mask_exclude = np.invert(unmask_include)
    unmask_deposits_shallowmarine = unmask_deposits & np.invert(unmask_overland)
    return mask_exclude, unmask_deposits_shallowmarine



def map_shore(nlonbins, nlatbins, shoreline_results, unmask_deposits_shallowmarine, mask_exclude, output_folder, epoch_name):
    shoredist = np.zeros((nlonbins*nlatbins))
    shoredirc = np.zeros((nlonbins*nlatbins))
    print('Shoreline ...')
    for idx, result in enumerate(shoreline_results):
        shoredist[idx] = result[1]
        shoredirc[idx] = result[2]
    map_shore_distance  = shoredist.reshape((nlatbins,nlonbins))
    map_shore_direction = shoredirc.reshape((nlatbins,nlonbins))
    # fix shallow areas
    map_shore_distance[unmask_deposits_shallowmarine] = 0
    map_shore_direction[unmask_deposits_shallowmarine] = map_shore_direction[unmask_deposits_shallowmarine] - 180.
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
    return map_shore_distance, map_shore_direction




# CALL FUNCTIONS
epoch_time_min, epoch_time_max, epoch_time = get_time(epoch_name)
epoch_deposits = get_deposits(epoch_time_min, epoch_time_max, time_deposits, data_deposits)
plot_deposits_scatter(epoch_deposits, epoch_name, output_folder)
mapbins = map_deposits_gridded(epoch_deposits, epoch_name, output_folder, lon_coords, lat_coords, lon_spacing, lat_spacing, nlatbins, nlonbins)
shoreline_results, topography_results = get_qpg_results(epoch_time, nlonbins, nlatbins, lon_coords, lat_coords, lon_spacing, lat_spacing)
mask_exclude, unmask_deposits_shallowmarine = make_masks(nlonbins, nlatbins, shoreline_results, output_folder, epoch_name, mapbins)
map_shore_distance, map_shore_direction = map_shore(nlonbins, nlatbins, shoreline_results, unmask_deposits_shallowmarine, mask_exclude, output_folder, epoch_name)



# ## PLOT Elevation
elevation_arr = np.zeros((nlonbins*nlatbins))
print('Topography...')
for idx, height in enumerate(topography_results):
    elevation_arr[idx] = height
map_elevation = elevation_arr.reshape((nlatbins,nlonbins))
map_elevation[unmask_deposits_shallowmarine] = 0
cmap_elevation = sns.cubehelix_palette(8, start=1.7, rot=.35, dark=0.4, light=0.9, reverse=True, as_cmap=True)
fig, ax_elevation_mask = plt.subplots(figsize=(18,6))            
ax_elevation_mask = sns.heatmap(map_elevation, vmin=0., vmax=5000., cmap=cmap_elevation, cbar=True, cbar_kws={"shrink": .75},
                                mask=mask_exclude, square=True, xticklabels=False, yticklabels=False)
ax_elevation_mask.set_xlabel('Paleolongitude')
ax_elevation_mask.set_ylabel('Paleolatitude')
ax_elevation_mask.set_title('Height/Depth above/below sea-level during '+epoch_name)
fig.savefig(output_folder+"map_elevation_"+epoch_name+".png", bbox_inches='tight', pad_inches=0.3)



# ## READ Linear Coefficient & GP hyperparameter MCMC samples
sampler_chain = np.load(MCMC_sampler_filename)
ndim     = sampler_chain.shape[2]
MCMC_samples  = sampler_chain[:, :, :].reshape((-1, ndim))
nsamples = MCMC_samples.shape[0]
alpha_samples = MCMC_samples[:,0]
sigma_samples = MCMC_samples[:,1]
beta_samples  = MCMC_samples[:,2:-2]
gphp_samples  = MCMC_samples[:,-2:]

minmaxscale_BLR = pd.read_csv(GP_folder+"minmaxscale_BLR.csv")
minmaxscale_GP  = pd.read_csv(GP_folder+"minmaxscale_GP.csv")
BLRpar_min   = minmaxscale_BLR.loc[:, 'minimum'].copy()
BLRpar_range = minmaxscale_BLR.loc[:, 'range'].copy()
GPpar_min    = minmaxscale_GP.loc[ :, 'minimum']
GPpar_range  = minmaxscale_GP.loc[ :, 'range']



# # PRECIPITATION
nsubset = 1000 # try 1000 atleast!
map_logprecip_samples = np.zeros((nlatbins,nlonbins,nsubset))
map_logprecip_mean    = np.zeros((nlatbins,nlonbins))
map_logprecip_16pc    = np.zeros((nlatbins,nlonbins))
map_logprecip_84pc    = np.zeros((nlatbins,nlonbins))
map_logprecip_unct    = np.zeros((nlatbins,nlonbins))
list_mean = []
list_16pc = []
list_84pc = []
list_unct = []
mapbin_deposits_coal, mapbin_deposits_evaporites, mapbin_deposits_glacial = mapbins['coal'], mapbins['evap'], mapbins['glac']

for ilon,lon in enumerate(lon_coords):
    lonm = lon + 0.5*lon_spacing
    lon_scaled = (lonm - GPpar_min.loc['X coords']) / GPpar_range.loc['X coords']
    for ilat,lat in enumerate(lat_coords):
        latm = lat + 0.5*lat_spacing[ilat]
        lat_scaled = (latm - GPpar_min.loc['Y coords']) / GPpar_range.loc['Y coords']
        land_here = not mask_exclude[nlatbins-ilat-1,ilon]
        if(land_here):
            par_dict = {}
            par_dict['Coal Deposits']       = mapbin_deposits_coal[nlatbins-ilat-1,ilon]
            par_dict['Evaporites Deposits'] = mapbin_deposits_evaporites[nlatbins-ilat-1,ilon]
            par_dict['Glacial Deposits']    = mapbin_deposits_glacial[nlatbins-ilat-1,ilon]
            elev = map_elevation[nlatbins-ilat-1,ilon]
            if elev<0:
                elev = 0.
            par_dict['Elevation'] = elev
            dist = map_shore_distance[nlatbins-ilat-1,ilon]
            par_dict['Dist to Shore'] = dist
            dird = map_shore_direction[nlatbins-ilat-1,ilon]
            dirr = radians(dird)
            par_dict['Sin Angle Shore'] = sin(dirr)
            par_dict['Cos Angle Shore'] = cos(dirr)
            xpars_check = np.array([par_dict[name] for name in x_feature_names])
            # [latm, lonm, coal, evap, glac, elev, sqrt(elev), asinh(elev/50.), asinh(elev/70.), dist, sqrt(dist), asinh(dist/50.), asinh(dist/70.), sin(dirr), cos(dirr)]
            if np.isnan(xpars_check).any():
                mask_exclude[nlatbins-ilat-1,ilon] = True
            else:
                xpars_scaled = (xpars_check-BLRpar_min)/BLRpar_range
                for isub,isamp in enumerate(random.sample(range(nsamples),nsubset)):
                    alpha_i = alpha_samples[isamp]
                    beta_i  = beta_samples[isamp]
                    sigma_i = sigma_samples[isamp]
                    gphp_i  = gphp_samples[isamp]
                    mu_i    = alpha_i + np.dot(beta_i, xpars_scaled)
                    #kern.set_parameter_vector(gphp_i)
                    log_precip = random.normalvariate(mu_i, sigma_i)
                    map_logprecip_samples[nlatbins-ilat-1,ilon,isub] = log_precip
                map_logprecip_mean[nlatbins-ilat-1,ilon] = np.mean(map_logprecip_samples[nlatbins-ilat-1,ilon,:])
                map_logprecip_16pc[nlatbins-ilat-1,ilon] = np.percentile((map_logprecip_samples[nlatbins-ilat-1,ilon,:]),16)
                map_logprecip_84pc[nlatbins-ilat-1,ilon] = np.percentile((map_logprecip_samples[nlatbins-ilat-1,ilon,:]),84)
                map_logprecip_unct[nlatbins-ilat-1,ilon] = map_logprecip_84pc[nlatbins-ilat-1,ilon] - map_logprecip_16pc[nlatbins-ilat-1,ilon]
                list_mean.append(map_logprecip_mean[nlatbins-ilat-1,ilon])
                list_16pc.append(map_logprecip_84pc[nlatbins-ilat-1,ilon])
                list_84pc.append(map_logprecip_16pc[nlatbins-ilat-1,ilon])
                list_unct.append(map_logprecip_unct[nlatbins-ilat-1,ilon])


# ## PLOT Precipitation
print("minimum value is "+str(np.ma.array(list(map_logprecip_16pc),mask=mask_exclude).min()))
print("maximum value is "+str(np.ma.array(list(map_logprecip_84pc),mask=mask_exclude).max()))

fig, (ax_prec_mean, ax_prec_16pc, ax_prec_84pc) = plt.subplots(3,1,figsize=(17,17))   
fig.tight_layout()
fig.subplots_adjust(right=0.9)
cmap = sns.cubehelix_palette(8, start=.65, rot=-.9, light=0.9, as_cmap=True)
vmin = -1.5
vmax =  1.5
cbar_ax = fig.add_axes([0.92, 0.05, 0.03, 0.9])
sns.heatmap(map_logprecip_mean, vmin=vmin, vmax=vmax, cmap=cmap, cbar=False, square=True, xticklabels=False, yticklabels=False, mask=mask_exclude, ax=ax_prec_mean)
ax_prec_mean.set_title('Average Log Annual Precipitation (m/yr) during '+epoch_name)
ax_prec_mean.set_ylabel('Paleolatitude')
sns.heatmap(map_logprecip_16pc, vmin=vmin, vmax=vmax, cmap=cmap, cbar=True,  square=True, xticklabels=False, yticklabels=False, mask=mask_exclude, ax=ax_prec_16pc, cbar_ax=cbar_ax)
ax_prec_16pc.set_title('16th Perc Log Annual Precipitation (m/yr) during '+epoch_name)
ax_prec_16pc.set_ylabel('Paleolatitude')
sns.heatmap(map_logprecip_84pc, vmin=vmin, vmax=vmax, cmap=cmap, cbar=False, square=True, xticklabels=False, yticklabels=False, mask=mask_exclude, ax=ax_prec_84pc)
ax_prec_84pc.set_title('84th Perc Log Annual Precipitation (m/yr) during '+epoch_name)
ax_prec_84pc.set_xlabel('Paleolongitude')
ax_prec_84pc.set_ylabel('Paleolatitude')
fig.savefig(output_folder+"map_LogPrecipitation_percentiles_"+epoch_name+".png", pad_inches=0.6)


print("minimum mean prediction is "+str(np.ma.array(list(map_logprecip_mean),mask=mask_exclude).min()))
print("maximum mean prediction is "+str(np.ma.array(list(map_logprecip_mean),mask=mask_exclude).max()))
fig, ax_prec_pred = plt.subplots(figsize=(18,6))
fig.tight_layout()
fig.subplots_adjust(right=0.9)
cmap = sns.cubehelix_palette(8, start=0.65, rot=-0.9, light=0.9, as_cmap=True)
vmin = -0.8
vmax = 0.5
cbar_ax = fig.add_axes([0.92, 0.05, 0.03, 0.9])
ax_prec_pred.set_title('Predicted Log Annual Precipitation (m/yr) during '+epoch_name)
sns.heatmap(map_logprecip_mean, vmin=vmin, vmax=vmax, cmap=cmap, cbar=True,  square=True, xticklabels=False, yticklabels=False, mask=mask_exclude, ax=ax_prec_pred, cbar_ax=cbar_ax)
ax_prec_pred.set_xlabel('Paleolongitude', labelpad=10)
ax_prec_pred.set_ylabel('Paleolatitude',  labelpad=10)
fig.savefig(output_folder+"map_LogPrecipitation_predict_"+epoch_name+".png", pad_inches=0.6)
fig.savefig(output_folder+"map_LogPrecipitation_predict_"+epoch_name+".eps", pad_inches=0.6)



print("minimum uncertainty is "+str(np.ma.array(list(map_logprecip_unct),mask=mask_exclude).min()))
print("maximum uncertainty is "+str(np.ma.array(list(map_logprecip_unct),mask=mask_exclude).max()))
fig, ax_prec_unct = plt.subplots(figsize=(18,6))
fig.tight_layout()
fig.subplots_adjust(right=0.9)
cmap = sns.cubehelix_palette(8, start=0., rot=0.4, as_cmap=True)
vmin = 1.0
vmax = 1.4
cbar_ax = fig.add_axes([0.92, 0.05, 0.03, 0.9])
ax_prec_unct.set_title('Uncertainty in Log Annual Precipitation (m/yr) during '+epoch_name)
sns.heatmap(map_logprecip_unct, vmin=vmin, vmax=vmax, cmap=cmap, cbar=True,  square=True, xticklabels=False, yticklabels=False, mask=mask_exclude, ax=ax_prec_unct, cbar_ax=cbar_ax)
ax_prec_unct.set_xlabel('Paleolongitude', labelpad=10)
ax_prec_unct.set_ylabel('Paleolatitude',  labelpad=10)
fig.savefig(output_folder+"map_LogPrecipitation_uncert_"+epoch_name+".png", pad_inches=0.6)


fig, ax_unct_mean = plt.subplots(figsize=(10,10))
ax_unct_mean.plot(list_mean, list_unct, marker='o', linestyle='', ms=3)
ax_unct_mean.set_xlabel("Predicted Log Precipitation")
ax_unct_mean.set_ylabel("Uncertainty in Predicted Log Precipitation")
fig.savefig(output_folder+"scatter_uncertVmeanLogPrecip_"+epoch_name+".png", pad_inches=0.6)

fig, ax_bound_mean = plt.subplots(figsize=(10,10))
ax_bound_mean.plot(list_mean, list_84pc, 'bo', ms=2.5)
ax_bound_mean.plot(list_mean, list_16pc, 'ro', ms=2.5)
ax_bound_mean.plot([-1.5, 1.5], [-1.5,1.5], 'k-')
ax_bound_mean.set_xlabel("Predicted Log Precipitation")
ax_bound_mean.set_ylabel("Uncertainty in Predicted Log Precipitation")
fig.savefig(output_folder+"scatter_boundVmeanLogPrecip_"+epoch_name+".png", pad_inches=0.6)



# ## Draw Precipitation output over World Map
"""
fig = plt.figure(figsize=(18,9))
map = Basemap(projection='cyl', lat_0 = 57, lon_0 = 0, resolution = 'i', area_thresh = 0.1, llcrnrlon=-180.0, llcrnrlat=-60.0, urcrnrlon=180.0, urcrnrlat=60.0)
#map = Basemap(projection='tmerc', lat_0 = 57, lon_0 = -135, resolution = 'i', area_thresh = 0.1, llcrnrlon=-180.0, llcrnrlat=-60.0, urcrnrlon=180.0, urcrnrlat=60.0)
map.imshow(map_logprecip, cmap="PuBu")
#map.contour(lat_list, lon_list, prc_array)
map.drawcoastlines()
map.drawcountries()
#map.drawparallels(np.arange(-90,90,15.0),labels=[1,1,0,1])
#map.drawmeridians(np.arange(-180,180,15),labels=[1,1,0,1])
map.drawmapboundary()
#map.scatter(x=epoch_deposits_coal['Paleolongitude'], y=epoch_deposits_coal['Paleolatitude'])
#plt.imshow(map_logprecip)
"""
