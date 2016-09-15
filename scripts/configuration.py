'''
Created on Aug 27, 2016

@author: rafaelcastillo

This file includes all configuration parameters used along the project
'''

## Configurations during the wrangling phase:

path_dataset = '/home/rafaelcastillo/MLND/Project5/DeepLearning/Dataset/' # path to dataset in local directory

dataset_train_test = '/home/rafaelcastillo/MLND/Project5/DeepLearning/Dataset_train_test' # path to dataset in the model required structure

sizes = [.6,.3,.1] # sizes of the different sets: train/test/validation

## Classes:
classes = ['n04467665','n04285008']

## Dimensions of the augmented pictures: (vehicles)
height = 405
width = 460

## models path:
model_path = '/home/rafaelcastillo/MLND/Project5/DeepLearning/models'

## models input pics dimensions:
model_height = 150
model_width = 150
