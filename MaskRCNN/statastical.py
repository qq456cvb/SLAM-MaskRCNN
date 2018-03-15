import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt

from os import listdir
from os.path import isfile, join
from skimage.feature import match_template

image_dir = '/home/yzn/CV/Project_3d/rgbd_dataset_freiburg2_coke'


rgb_dir = join(image_dir,'rgb')
depth_dir = join(image_dir,'depth')
rgb_files = sorted(listdir(rgb_dir))
depth_files = sorted(listdir(depth_dir))

rgb_dict = {}
depth_dict = {}

for rgb_file in rgb_files:
	s = rgb_file.split('.')[0]
	if s in rgb_dict.keys():
		rgb_dict[s] = rgb_dict[s] + 1
	else:
		rgb_dict[s] = 1

for depth_file in depth_files:
	s = depth_file.split('.')[0]
	if s in depth_dict.keys():
		depth_dict[s] = depth_dict[s] + 1
	else:
		depth_dict[s] = 1

numberfile = open('test_log/image_number.txt', 'w')

for k,v in rgb_dict.items():
	numberfile.write("{}: {}\n".format(k,v))
	#print (k,v)
numberfile.write("-------------------------\n")


for k,v in depth_dict.items():
	numberfile.write("{}: {}\n".format(k,v))
	#print (k,v)
