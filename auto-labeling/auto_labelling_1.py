#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 17:05:38 2022

@author: root

"""
# customize the following name
data_dir = "/home/oem/ScottLin/project/darknet/data/"
dir_name =  data_dir + "final/auto-label/0715"
train_txt = "train_cell_0715.txt"

import os
image_files = []
os.chdir(dir_name)
for filename in os.listdir(os.getcwd()):
    if filename.endswith(".jpg"):
        image_files.append(dir_name+"/" + filename)
os.chdir(data_dir)
with open(train_txt, "w") as outfile:
    for image in image_files:
        outfile.write(image)
        outfile.write("\n")
    outfile.close()