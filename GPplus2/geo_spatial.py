"""
Class for running spatial geoprocessing
Version 0.1
Author: Sebastian Haan
"""
from __future__ import division, print_function
import matplotlib.pylab as plt
plt.style.use('ggplot')
import os
import numpy as np
import math
import pandas as pd
import csv
# Try to import Geopandas, only really neccessary for shapefile processing:
try:
    import geopandas as gpd   # use latest git version rather than conda or pip installation!
except:
    print('Warning: error while loading Geopandas. Will continue without geopandas...')



class Shape:
    """This class processes shapefiles and combines with crime/demographic data:
    Both input files, model features and spatial dataset, require a column "region_id" for merging both datasets.
    import of polygon shapefiles in geopandas and cleaning
    calculation of centroids of polygons and optionally distance matrix
    creation of cartesian grid with geomtric centroid of NSW as reference point (0,0)
    final export of NSW SA2 index code with centroid in lat/lng as well as cartesian grid ccordinates as csv
    visualistaion of shapes and centroids
    creates html data maps with folium
    export all in GeoJSON file format

    Requirements in addition to standard libraries:
    geopandas
    shapely
    geopy (optional)
    folium (optional, required for html map making in 'create_map')
    """



    def __init__(self, data_path, filename, outpath, calc_center):
        """
        :param data_path: Main data directory path for input files
        :param filename: path plus filename for shapefile
        :param outpath: path to direcory for outfile
        :param calc_center: boolean True if center has to be calculated from shapefiles, otherwise False
        """
        try:
            from geopy.distance import vincenty # for distance calculation
            self.dist_manual = False
        except:
            self.dist_manual = True
        self.xy_sa2 = []  # cartesian corrdinates of cnetroid of NSW SA2 regions
        self.cen_sa2 = []   # lat/lng corrdinates of centroid of NSW SA2 regions
        self.area = []   # surface area of SA2 regions as defined by polygon shapes
        self.dist_matrix = []  # distance matrix for NSW SA2 regions
        self.data_path = data_path + '/'
        self.shape_filename = filename
        self.outpath = outpath

        if calc_center and not os.path.exists(self.data_path+self.shape_filename):
            raise ValueError("Error preprocess_data: Shape file not found "+self.data_path+self.shape_filename)
        if calc_center and not os.path.exists(self.outpath):
            raise ValueError("Error: Path for Output files does not exist: "+self.outpath)


    def calc_dist(self, coord1, coord2):
        """ calculate approx distance in meters using vincenty inverse calculation, input: long, lat in degrees
        alternatively use geopy.distance.vincenty (if installed, should be faster since written in C++)

        :param coord1: first coordinate in format [Longitude, Latitude] in degrees
        :param coord2: second coordinate in format [Longitude, Latitude] in degrees
        :return: returns distance in meters
        """
        #--- CONSTANTS ------------------------------------+
        
        a=6378137.0                             # radius at equator in meters (WGS-84)
        f=1/298.257223563                       # flattening of the ellipsoid (WGS-84)
        b=(1-f)*a
        tol=10**-11
        maxIter = 100

        phi_1,L_1,=coord1                       
        phi_2,L_2,=coord2                  

        u_1=math.atan((1-f)*math.tan(math.radians(phi_1)))
        u_2=math.atan((1-f)*math.tan(math.radians(phi_2)))

        L=math.radians(L_2-L_1)

        Lambda=L                                # set initial value of lambda to L

        sin_u1=math.sin(u_1)
        cos_u1=math.cos(u_1)
        sin_u2=math.sin(u_2)
        cos_u2=math.cos(u_2)

        #--- BEGIN ITERATIONS -----------------------------+
        self.iters=0
        for i in range(0,maxIter):
            self.iters+=1
            
            cos_lambda=math.cos(Lambda)
            sin_lambda=math.sin(Lambda)
            sin_sigma=math.sqrt((cos_u2*math.sin(Lambda))**2+(cos_u1*sin_u2-sin_u1*cos_u2*cos_lambda)**2)
            cos_sigma=sin_u1*sin_u2+cos_u1*cos_u2*cos_lambda
            sigma=math.atan2(sin_sigma,cos_sigma)
            sin_alpha=(cos_u1*cos_u2*sin_lambda)/sin_sigma
            cos_sq_alpha=1-sin_alpha**2
            cos2_sigma_m=cos_sigma-((2*sin_u1*sin_u2)/cos_sq_alpha)
            C=(f/16)*cos_sq_alpha*(4+f*(4-3*cos_sq_alpha))
            Lambda_prev=Lambda
            Lambda=L+(1-C)*f*sin_alpha*(sigma+C*sin_sigma*(cos2_sigma_m+C*cos_sigma*(-1+2*cos2_sigma_m**2)))

            # successful convergence
            diff=abs(Lambda_prev-Lambda)
            if diff<=tol:
                break
            
        u_sq=cos_sq_alpha*((a**2-b**2)/b**2)
        A=1+(u_sq/16384)*(4096+u_sq*(-768+u_sq*(320-175*u_sq)))
        B=(u_sq/1024)*(256+u_sq*(-128+u_sq*(74-47*u_sq)))
        delta_sig=B*sin_sigma*(cos2_sigma_m+0.25*B*(cos_sigma*(-1+2*cos2_sigma_m**2)-(1/6)*B*cos2_sigma_m*(-3+4*sin_sigma**2)*(-3+4*cos2_sigma_m**2)))

        distance=b*A*(sigma-delta_sig)                 # output distance in meters
        return distance

    
    def calc_grid(self, position, clat, clng):
        """ converts latitude/longitude to cartesian grid in meters with reference center point to be (0,0)
        trigonometric approximation based on fact that latitude to meter doesn't change much, so
        ONLY USE FOR VERY SMALL AREAS WITH RESPECT TO TOTAL EARTH
        :param position: (2,ndim) array for coordinate array in format [[Lat1, Lng1],[Lat2, Lng2], ...]
        :param clat: latitude of origin(center) coordinate of cartesian grid
        :param clng: longitude of origin(center) coordinate of cartesian grid
        :return: cartesian grid coordinates [x,y].
        """
        lat = position[0]
        lng = position[1]
        dlat = (lat - clat) * 111000.
        if not self.dist_manual:
            d = vincenty(position, [clat, clng]).meters
        else:
            d = self.calc_dist(position, [clat, clng])
        if (d**2 > dlat**2):
            dlng = np.sqrt(d**2 - dlat**2) * np.sign(lng - clng)
        else:
            dlng = 0. # for the few entries where distance is similar to latitude within a few hundert meters
        return np.asarray([dlat, dlng])


    def create_geometry(self, clat, clng, store_data = True, region = None):
        """ Reads in shapefile data into geopandas and calculates cartesian grid points, distances, and areas
        :param clat: latitude of origin(center) coordinate of cartesian grid
        :param clng: longitude of origin(center) coordinate of cartesian grid
        :param store_data: boolean (stores csv file of cartesian grid coordinates)
        :param region: string of region to extract from shapefile with columnname "STE_NAME11", default None
        :return: no return data, saves data in csv files and class variable.
        """
        print("Read in shapefile data ...")
        geodata = gpd.GeoDataFrame.from_file(self.data_path + self.shape_filename)
        # Select only NSW data
        if region:
            shapedata = geodata[geodata.STE_NAME11 == region].copy()
        else:
            shapedata = geodata.copy()
        n_sa2 = shapedata.shape[0]
        self.cen_sa2 = np.zeros((n_sa2,2))
        self.xy_sa2 = np.zeros((n_sa2, 2))
        self.dist_matrix = np.zeros((n_sa2,n_sa2))

        # Calculate center of polygon
        print("Calculating centroids ...")
        for i in range(n_sa2):
            try:
                self.cen_sa2[i] = np.flipud(np.array(shapedata.ix[i].geometry.centroid))
            except:
                self.cen_sa2[i] = np.NaN
                print("Warning: Error in calculating center for row: ", i)

        # Reproject to cartesian grid in meters
        print("Calculating cartesian grid points ...")
        for i in range(n_sa2):
            if np.isnan(self.cen_sa2[i,0]) == False:
                self.xy_sa2[i] = self.calc_grid(self.cen_sa2[i], clat, clng)
            else:
                self.xy_sa2[i] = np.NaN
        shapedata['centroid_lat'] = self.cen_sa2[:,0]
        shapedata['centroid_lng'] = self.cen_sa2[:,1]
        shapedata['centroid_x'] = self.xy_sa2[:, 0]
        shapedata['centroid_y'] = self.xy_sa2[:, 1]
        self.index_sa2 = np.asarray(shapedata.SA2_MAIN11)
        shapedata['region_id'] = self.index_sa2.astype(int)
        shapedata.dropna(inplace = True)

        # Calculate distance in meters
        calc_dist=False
        if calc_dist:
            print("Calculating distance matrix ...")
            for i in range(n_sa2):
                if np.isnan(self.cen_sa2[i,0]) == False :
                    for j in range(n_sa2):
                        if np.isnan(self.cen_sa2[j,0]) == False :
                            if not self.dist_manual:
                                self.dist_matrix[i,j] = vincenty(self.cen_sa2[i], self.cen_sa2[j]).meters
                            else:
                                self.dist_matrix[i,j] = self.calc_dist(self.cen_sa2[i], self.cen_sa2[j])
                        else:
                            self.dist_matrix[i,j] = np.NaN
                else:
                    self.dist_matrix[i, :] = np.NaN

        # Calculating area of SA2 regions
        print("Calculating Area ...")
        shapedata_m = shapedata.copy()
        shapedata_m = shapedata_m.to_crs({'init': 'epsg:3395'})
        shapedata['shape_area_sqkm'] = shapedata_m['geometry'].area / 10e6   # in square km

        # Put all in one dataset
        self.shapedata = shapedata[np.isfinite(shapedata['centroid_x'])]
        header = ['region_id', 'centroid_x', 'centroid_y']
        if store_data:
            self.shapedata.to_csv(self.outpath + 'loc_data.csv', columns = header)


        # Storing data as csv
        # if store_data:
        #     f = open(self.outpath + 'loc_data.csv',"w")
        #     field_names=['region_id']
        #     field_names.append('centroid_x')
        #     field_names.append('centroid_y')
        #     writer = csv.DictWriter(f,fieldnames=field_names)
        #     writer.writeheader()
        #     row = dict()
        #     for i in range(len(self.shapedata['region_id'])):
        #         row['region_id'] = self.shapedata['region_id'][i]
        #         row['centroid_x'] = self.shapedata['centroid_x'].values[i]
        #         row['centroid_y'] = self.shapedata['centroid_y'].values[i]
        #         writer.writerow(row)
        #     f.close()

        check_centroid = False
        if check_centroid:
            ax = shapedata.ix[0:4].plot()
            shapedata.ix[0:4].geometry.centroid.plot(ax=ax, markersize=20)
            plt.show()

    def read_csvdata(self, csvpath):
        """ Read csv data (crime rates, demographics, residuals with region code) in pandas frame
        :param csvpath: path to inout directory for csv data
        :return: no return data, stores data in class variable
        """
        if not os.path.exists(csvpath):
            raise ValueError("Error preprocess_data: CSV file not found "+csvpath)
        self.data_df = pd.read_csv(csvpath)

    def combine_data(self):
        #combine geopandas with pandas data
        self.data_comb = self.shapedata.merge(self.data_df, on='region_id')

    def create_maps(self, clat, clng):
        """ Creates an interactive html map with folium and GeoJSON outfile
        :param clat: latitude of origin(center) coordinate of cartesian grid
        :param clng: longitude of origin(center) coordinate of cartesian grid
        :return: no return data, saves maps
        """
        import folium
        print("Creating Crime and Residual html maps ...")
        # set width and height in folium.Map for smaller file
        path=self.outpath + "SA2_spatial_crime.json"
        try:
            self.data_comb.to_file(path, 'GeoJSON') #write GeoJSON file
        except:
            if os.path.exists(path):
                print("File already exist, will not overwrite: "+path)
            else:
                print("Error in writing file: "+path)
        map = folium.Map(location=[clat, clng], zoom_start=7, tiles='cartodbpositron')
        map.choropleth(geo_path=path, data=self.data_comb,
                       key_on='feature.properties.region_id',
                       columns=['region_id','residual_crime_rate'],
                       fill_color='YlGnBu',
                       fill_opacity=0.7,
                       line_opacity=0.2,
                       legend_name='Residual Crime Rate')
        map.save(self.outpath+'map_BLRresidual_crimerate_DV.html')
        map = folium.Map(location=[clat, clng], zoom_start=7, tiles='cartodbpositron')
        map.choropleth(geo_path=path, data=self.data_comb,
                       key_on='feature.properties.region_id',
                       columns=['region_id', 'residual_crime_rate'],
                       fill_color='YlGnBu',
                       fill_opacity=0.7,
                       line_opacity=0.2,
                       legend_name='Log Crime Rate')
        map.save(self.outpath + 'map_log_crimerate.html')

    def create_maps2(self, outdir, columnstr, clat, clng):
        """ Creates an interactive html map with folium and pandas data
        :param outdir: path to output directory
        :param columnstr: string of column name od pandas dataframe
        :return: no return data, saves maps
        """
        import folium
        print('Creating '+columnstr+' maps ...')
        # set width and height in folium.Map for smaller file
        path = self.outpath + "SA2_spatial_crime.json"
        map = folium.Map(location=[clat, clng], zoom_start=7, tiles='cartodbpositron')
        map.choropleth(geo_path=path, data=self.data_comb,
                       key_on='feature.properties.region_id',
                       columns=['region_id', columnstr],
                       fill_color='YlGnBu',
                       fill_opacity=0.7,
                       line_opacity=0.2,
                       legend_name=columnstr)
        map.save(outdir + 'map_'+columnstr+'.html')