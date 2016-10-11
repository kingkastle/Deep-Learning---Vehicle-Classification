# Deep-Learning Vehicle Classification


This project is a proof of concept (POC) solution where deep learning techniques are applied to vehicle recognition tasks, this is particularly important task in the area of traffic control and management, for example, companies operating road tolls to detect fraud actions since different fees are applied with regards to vehicle types. Images used to train neural nets are obtained from the [Imagenet](http://image-net.org/) dataset, which publicly distributes images URLs for hundreds of categories. Since the whole experiment is performed on a personal computer with limited computational resources, POC scope is also limited to the simple classification of two different kinds of vehicles: Trailer Trucks versus Sports Cars. Main POC's goal is to determine the maximum accuracy (percent of times model was correct on its predictions) different neural nets with basic architectures can reach using a limited set of images (less than 700) for training.

<img src="images/truck_versus_car.png" alt="Drawing" style="width: 500px;">

## COMPLETE REPORT: 
Report with full descriptions of motivations, methodology, results, etc. [Deep Learning Vehicle Classification-Project_writeup](https://github.com/kingkastle/Deep-Learning---Vehicle-Classification/blob/master/Capstone%20Project_writeup.md)


## Requirements:

Following libraries are necessary:

```
# local scripts:
from scripts import configuration  # includes paths and parameters configurations
from scripts import models # includes the different models
from scripts import Dataset_wrangling # includes scripts from downloading pics to generate datasets


# standard libraries
import os
import h5py
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D,MaxPooling2D
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from datetime import datetime
import configuration
import pickle
import multiprocessing
import logging
import urllib2
import download_imagenet_images
import pandas
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from IPython.core.display import HTML,display

```

Project is enterely written in Python 2.7.

## Instructions:

Please follow the instructions given in ```Vehicles_Categorization.ipynb```


Enjoy! 
