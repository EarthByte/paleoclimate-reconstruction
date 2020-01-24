
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
import cartopy.crs as ccrs


from mpl_toolkits.basemap import Basemap
from scipy.ndimage import filters
from scipy.ndimage import interpolation
import cv2
from matplotlib.patches import Polygon
#from matplotlib.collections import PatchCollectioninput

import shapefile

def readTopologyPlatepolygonFile(filename):
    '''
    Reads shapefiles and returns the all the data fields
    '''
    shapeRead = shapefile.Reader(filename)

    recs    = shapeRead.records()
    shapes  = shapeRead.shapes()
    fields  = shapeRead.fields
    Nshp    = len(shapes)
    
    return(recs,shapes,fields,Nshp)


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




# USER CHOICES
if(len(sys.argv)!=4):
    sys.exit('Usage: python plot_prediction.py <input file> <one-word subject of folder where data input resides  and outputs will be kept>')

 

predict_folder = str(sys.argv[1])


predict_filename = str(sys.argv[2])


subject = str(sys.argv[3])

directory_plot =  predict_folder+'/'+subject


if not os.path.exists(directory_plot):
    os.makedirs(directory_plot) 


directory_plot =  predict_folder+'/'+subject + "/map_prediction_"


if not os.path.exists(directory_plot):
    os.makedirs(directory_plot) 

directory_plot =  predict_folder+'/'+subject + "/map_prediction_uncert_"


if not os.path.exists(directory_plot):
    os.makedirs(directory_plot) 

directory_plot =  predict_folder+'/'+subject + "/map_actual_"


if not os.path.exists(directory_plot):
    os.makedirs(directory_plot) 



directory_plot =  predict_folder+'/'+subject

print(predict_filename , ' predict_filename')

prediction_file = predict_folder +'/' + predict_filename  # example only

print("Prediction file path: %s"%(prediction_file))
 
  
  
predictions = pd.read_csv(prediction_file, header=None)
 
  
latitudes     = predictions.loc[:,0]
longitudes    = predictions.loc[:,1]

#xt    = predictions.loc[:,3]
 


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

map_predict_actual = np.zeros((nlatbins,nlonbins))
map_predict_mean = np.zeros((nlatbins,nlonbins))
map_predict_low  = np.zeros((nlatbins,nlonbins))
map_predict_high = np.zeros((nlatbins,nlonbins))
map_predict_unct = np.zeros((nlatbins,nlonbins))
mask_exclude     = np.zeros((nlatbins,nlonbins))
list_mean = []
list_low  = []
list_high = []
list_unct = []

list_actual = []

if subject == 'coal':
    actual_col =  15
    mean_col  = 6 
    high_col =  12
    low_col =  9

    print(subject, ' is subject')


if subject == 'glacial':
    actual_col =   17
    mean_col  =  8
    high_col = 14
    low_col =  11


    print(subject, ' is subject')




if subject == 'evaporites':
    actual_col = 16
    mean_col  =  7  
    high_col =  13
    low_col =  10


    print(subject, ' is subject')


 
 

# what happens if there are (fixed by David Kohn)
# multiple longitudes and latitudes
# within the specified grid area?
# we take the mean of the predictions
# for all the latitudes and longitudes
# that are within that grid
take_pred_mean = True

# pred_wrapper is the wrapper put around predicted values
# i.e. if multiple values, take the mean
pred_wrapper = lambda x: x
if take_pred_mean: 
    pred_wrapper = lambda x: np.mean(x)

for ilon,lon_low in enumerate(lon_coords):
    lon_high = lon_low + lon_spacing
    for ilat,lat_low in enumerate(lat_coords):
        lat_high = lat_low + lat_spacing[ilat]
        this_paleolongitude = (lon_low<longitudes) & (longitudes<lon_high)
        this_paleolatitude  = (lat_low<latitudes) & (latitudes<lat_high)
 
        here = (this_paleolongitude & this_paleolatitude)
 
        if(here.any()):
            map_predict_actual[nlatbins-ilat-1,ilon] = pred_wrapper(predictions.loc[here,actual_col].values)
            map_predict_mean[nlatbins-ilat-1,ilon] = pred_wrapper(predictions.loc[here,mean_col].values)
            map_predict_low[nlatbins-ilat-1,ilon]  = pred_wrapper(predictions.loc[here,low_col].values)
            map_predict_high[nlatbins-ilat-1,ilon] = pred_wrapper(predictions.loc[here,high_col].values)
            map_predict_unct[nlatbins-ilat-1,ilon] = pred_wrapper(map_predict_high[nlatbins-ilat-1,ilon] - map_predict_low[nlatbins-ilat-1,ilon])
            list_mean.append(map_predict_mean[nlatbins-ilat-1,ilon])
            list_low.append(map_predict_high[nlatbins-ilat-1,ilon])
            list_high.append(map_predict_low[nlatbins-ilat-1,ilon])
            list_unct.append(map_predict_unct[nlatbins-ilat-1,ilon])
            list_actual.append(map_predict_actual[nlatbins-ilat-1,ilon])
        else:
            mask_exclude[nlatbins-ilat-1,ilon] = True



# ## PLOT Prediction

#print( map_predict_mean)


lons, lats = np.meshgrid(lon_coords,lat_coords)





fig = plt.figure(figsize=(16,12),dpi=150)
ax = fig.add_subplot(111)

pmap = Basemap(projection='moll', lat_0=0, lon_0=0, resolution='l')
pmap.drawmapboundary(fill_color='white')

#Load in plate polygons for plotting
#topologyFile='/Users/nbutter/Projects/GEO/Eocene/paleoclimate-reconstruction/data/PaleomagneticReferenceFrame/reconcoast/reconstructed_'+time1+'.00Ma.shp'
#[recs,shapes,fields,Nshp]=readTopologyPlatepolygonFile(topologyFile)
#patches   = []
'''for i, nshp in enumerate(range(Nshp)):
    #These are the plates that cross the dateline and cause 
        #banding errors
        polygonShape=shapes[nshp].points
        poly=np.array(polygonShape)
        testnum=max(poly[:,0])-min(poly[:,0])
        if testnum < 180.0:
            #lon_unwrapped = np.rad2deg(np.unwrap(np.deg2rad(poly[:,0])))
            #print(poly[:,0])
            #print(lon_unwrapped)
            a=pmap(poly[:,0],poly[:,1])
            #a=pmap(lon_unwrapped,poly[:,1])
            b=np.array([a[0],a[1]])
            patches.append(Polygon(b.T,closed=False,color='dimgrey'))
            #pc = PatchCollection(patches, match_original=True, edgecolor='k', linewidths=1., zorder=2)
            #ax.add_collection(pc)
            #xs, ys = pmap(poly[:,0], poly[:,1])
            #pmap.plot(xs, ys, 'k',zorder=1)
        else:
            print(testnum, recs[nshp][8],nshp)
        

pc = PatchCollection(patches, color='dimgrey', linewidths=1, zorder=1)
ax.add_collection(pc)'''
        
intensity = np.ma.masked_where(np.isnan(map_predict_mean), map_predict_mean)
im1 = pmap.pcolormesh(lons,-lats,intensity,shading='flat',cmap=plt.cm.gist_earth_r,latlon=True)



#pip install matplotlib==2.0.2  (this resolved error: MatplotlibDeprecationWarning: The axesPatch function was deprecated in version 2.1. Use Axes.patch instead)


cb = pmap.colorbar(im1,"right", size="3%", pad="5%",ticks=[0,1],fraction=0.001)
plt.clim(0,1)
cb.set_label('Prediction for '+subject,labelpad=10,size=20)


xh, yh = pmap(actual_lon, actual_lat)
line,=pmap.plot(xh,yh,'ob',label='Actual '+subject)
plt.legend(loc=4,prop={'size': 20})
#pmap.imshow(intensity,extent=[-180,180,-90,90],origin='upper',interpolation='bilinear',cmap=plt.cm.binary)
plt.text(33000000,16000000,time1+" Ma",size=20)


pmap.drawmeridians(np.arange(0, 360, 60),color='grey',dashes=[1,0],linewidth=0.2)
pmap.drawparallels(np.arange(-90, 90, 45),color='grey',dashes=[1,0],linewidth=0.2)



plt.show()




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
fig.savefig( directory_plot+"/map_prediction_/"+ predict_filename+".pdf", pad_inches=0.6)
fig.clf()



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
fig.savefig(directory_plot+"/map_prediction_uncert_/"+ predict_filename+".pdf", pad_inches=0.6)
fig.clf()


fig, ax_prec_unct = plt.subplots(figsize=(18,6))
fig.tight_layout()
fig.subplots_adjust(right=0.9)
cmap = sns.cubehelix_palette(8, start=0., rot=0.4, as_cmap=True)
cbar_ax = fig.add_axes([0.92, 0.05, 0.03, 0.9])
ax_prec_unct.set_title('Actual  '+subject)
sns.heatmap(map_predict_actual, cmap=cmap, cbar=True,  square=True, xticklabels=False, yticklabels=False, mask=mask_exclude, ax=ax_prec_unct, cbar_ax=cbar_ax)
ax_prec_unct.set_xlabel('Paleolongitude', labelpad=10)
ax_prec_unct.set_ylabel('Paleolatitude',  labelpad=10)
fig.savefig(directory_plot+"/map_actual_/"+predict_filename+".pdf", pad_inches=0.6)
fig.clf()
