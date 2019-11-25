
from __future__ import print_function
import sys
import os
import numpy as np
import pandas as pd
from math import asin, acos, sqrt, sin, cos, radians, degrees, asinh
from matplotlib import pyplot as plt
import seaborn as sns
import csv


from numpy import genfromtxt
my_data = genfromtxt('results_all.csv', delimiter=',')

era_num = [14, 38, 39, 28, 51, 61, 77, 101, 129, 154, 182, 219, 242 ]

 
for i in range(len(era_num)): 

	era_index = np.where(my_data[:, 21]==era_num[i])[0]

	print(era_index.shape)

	era = my_data[era_index, :]

	np.savetxt( 'data_prediction_nov2019/results_depositsprecip/'+'era'+str(era_num[i])+'results.csv', era, delimiter=',')

	print(era.shape)

