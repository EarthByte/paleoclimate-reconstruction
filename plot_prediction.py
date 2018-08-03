
# coding: utf-8

# # Plot Predicted Precipitation 
# 
# ## Madhura Killedar
# ## date   : 20/07/2018
# 

from __future__ import print_function
import sys
import os
import numpy as np
import pandas as pd
from math import asin, acos, sqrt, sin, cos, radians, degrees, asinh
from matplotlib import pyplot as plt
import seaborn as sns
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

print("Hello Prediction Plotting")



# ## Input/Output Folders
predict_folder = 'predictions/'
print("Prediction folder path: %s"%(predict_folder))

# USER CHOICES
if(len(sys.argv)!=3):
    sys.exit('Usage: python plot_prediction.py <input file> <one-word subject of prediction>')
#prediction_file = 'predictions/glacial_results.csv' # example only
prediction_file = str(sys.argv[1])
subject = str(sys.argv[2]) # e.g. 'glacial'

# read file
predictions = pd.read_csv(prediction_file, header=None)
#latitudes     = predictions.loc[:,'Latitude']
#longitudes    = predictions.loc[:,'Longitude']
#precipitation = predictions.loc[:,'Predicted Precipitation']
latitudes     = predictions.loc[:,0]
longitudes    = predictions.loc[:,1]




# ## SET UP GRIDS
lon_min = -180.
lon_max =  180.
lat_min =  -90.
lat_max =   90.
lon_spacing = 3. #2.51

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




# # PRECIPITATION
map_predict_mean = np.zeros((nlatbins,nlonbins))
map_predict_low  = np.zeros((nlatbins,nlonbins))
map_predict_high = np.zeros((nlatbins,nlonbins))
map_predict_unct = np.zeros((nlatbins,nlonbins))
mask_exclude     = np.zeros((nlatbins,nlonbins))
list_mean = []
list_low  = []
list_high = []
list_unct = []

for ilon,lon_low in enumerate(lon_coords):
    lon_high = lon_low + lon_spacing
    for ilat,lat_low in enumerate(lat_coords):
        lat_high = lat_low + lat_spacing[ilat]
        this_paleolongitude = (lon_low<longitudes) & (longitudes<lon_high)
        this_paleolatitude  = (lat_low<latitudes) & (latitudes<lat_high)
        here = this_paleolongitude & this_paleolatitude
        if(here.any()):
            map_predict_mean[nlatbins-ilat-1,ilon] = predictions.loc[here,3]
            map_predict_low[nlatbins-ilat-1,ilon]  = predictions.loc[here,4]
            map_predict_high[nlatbins-ilat-1,ilon] = predictions.loc[here,5]
            map_predict_unct[nlatbins-ilat-1,ilon] = map_predict_high[nlatbins-ilat-1,ilon] - map_predict_low[nlatbins-ilat-1,ilon]
            list_mean.append(map_predict_mean[nlatbins-ilat-1,ilon])
            list_low.append(map_predict_high[nlatbins-ilat-1,ilon])
            list_high.append(map_predict_low[nlatbins-ilat-1,ilon])
            list_unct.append(map_predict_unct[nlatbins-ilat-1,ilon])
        else:
            mask_exclude[nlatbins-ilat-1,ilon] = True



# ## PLOT Prediction

print("minimum mean prediction is "+str(np.ma.array(list(map_predict_mean),mask=mask_exclude).min()))
print("maximum mean prediction is "+str(np.ma.array(list(map_predict_mean),mask=mask_exclude).max()))
fig, ax_prec_pred = plt.subplots(figsize=(18,6))
fig.tight_layout()
fig.subplots_adjust(right=0.9)
cmap = sns.cubehelix_palette(8, start=0.65, rot=-0.9, light=0.9, as_cmap=True)
cbar_ax = fig.add_axes([0.92, 0.05, 0.03, 0.9])
ax_prec_pred.set_title('Prediction for '+subject)
sns.heatmap(map_predict_mean, cmap=cmap, cbar=True,  square=True, xticklabels=False, yticklabels=False, mask=mask_exclude, ax=ax_prec_pred, cbar_ax=cbar_ax)
ax_prec_pred.set_xlabel('Paleolongitude', labelpad=10)
ax_prec_pred.set_ylabel('Paleolatitude',  labelpad=10)
fig.savefig(predict_folder+"map_prediction_"+subject+".pdf", pad_inches=0.6)



print("minimum uncertainty is "+str(np.ma.array(list(map_predict_unct),mask=mask_exclude).min()))
print("maximum uncertainty is "+str(np.ma.array(list(map_predict_unct),mask=mask_exclude).max()))
fig, ax_prec_unct = plt.subplots(figsize=(18,6))
fig.tight_layout()
fig.subplots_adjust(right=0.9)
cmap = sns.cubehelix_palette(8, start=0., rot=0.4, as_cmap=True)
cbar_ax = fig.add_axes([0.92, 0.05, 0.03, 0.9])
ax_prec_unct.set_title('Uncertainty in Prediction for '+subject)
sns.heatmap(map_predict_unct, cmap=cmap, cbar=True,  square=True, xticklabels=False, yticklabels=False, mask=mask_exclude, ax=ax_prec_unct, cbar_ax=cbar_ax)
ax_prec_unct.set_xlabel('Paleolongitude', labelpad=10)
ax_prec_unct.set_ylabel('Paleolatitude',  labelpad=10)
fig.savefig(predict_folder+"map_prediction_uncert_"+subject+".pdf", pad_inches=0.6)
