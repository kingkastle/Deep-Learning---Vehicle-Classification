'''
Created on Aug 21, 2016

@author: rafaelcastillo

This script is used to built the final dataset from the pictures downloaded
'''

import pandas
import numpy as np
from os import listdir,walk
from os.path import isfile, join
from keras.preprocessing.image import ImageDataGenerator
from scipy import ndimage
import seaborn as sns
sns.set(style="white")
import matplotlib.pyplot as plt
import os, shutil
import random
import Image



plt.style.use('seaborn-muted') # Using ggplot style for visualizations


def resize_pic(img_path,width,height):
    '''
    This function is used to resize pictures to the desired shape.
    
    Args:
        * img_path: path to local image (in jpg format) to convert
        * width: desired width of the output picture
        * height: desired height of the output picture
        
    Return:
        0/1 depending of the conversion status.
        Generated picture is stored under a different name using img_path
    '''
    im1 = Image.open(img_path)
    # use  cubic spline interpolation in a 4x4 environment filter options to resize the image
    try:
        im4 = im1.resize((width, height), Image.BICUBIC)
        im4.save(img_path.replace('.jpg','_good_shape.jpg'))
    except:
        #print "Unable to resize file: {0}".format(img_path)
        return 0
    return 1


def generate_pics(dataset_path,family,df_hw,number_pics,width,height):
    '''
    This script is used to generate pictures using the Keras utility ImageDataGenerator.
    
    Args:
        * dataset_path: dataset path
        * family: flower category
        * df_hw: dataframe with height width, channel and family name for each picture
        * number_pics: number of pictures to generate
        * width: desired width of the output picture
        * height: desired height of the output picture
        
    Return:
        none
    
    '''
    
    # Generate temporary folder and subfolder where the input pics to generate 
    # fake pictures will be located.
    pictures_path = dataset_path + family
    if not os.path.exists(pictures_path + '/' + 'temp'):
        os.mkdir(pictures_path + '/' + 'temp')
        os.mkdir(pictures_path + '/' + 'temp/pics')
    
    # Get pictures paths:
    picture_files = df_hw[(df_hw['family']==family)&
                          (df_hw['height']>=height)&
                          (df_hw['width']>=width)]['name'].unique()
    
    # Select a number_pics of pictures using a random sample:
    if len(picture_files) < number_pics:
        print """Warning: There are insufficient pictures with optimum size to generate augmented pics.
                 Process will repeat pictures to generate augmented data"""
        pics_diff = number_pics - len(picture_files)
        selected_pics = random.sample(picture_files,len(picture_files)) + random.sample(picture_files,pics_diff)
    else:
        selected_pics = random.sample(picture_files,number_pics)
    
    # Get wnid for the flower specie:
    wnid = selected_pics[0].split("_")[0]
    
    # Copy those files to the destination folder:
    for pic in selected_pics:
        shutil.copyfile(join(pictures_path, pic), pictures_path + '/' + 'temp/pics/' + pic)
        
    # Generate augmented pics. Docs: https://keras.io/preprocessing/image/
    datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.01,
        height_shift_range=0.1,
        shear_range=0.05,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest')

    # the .flow_from_directory() command below generates batches of randomly transformed images
    # and saves the results to save_to_dir directory
    for _ in datagen.flow_from_directory(pictures_path + '/' + 'temp', 
                                             target_size=(width,height), 
                                             batch_size=number_pics,
                                             classes=None, 
                                             class_mode=None, 
                                             shuffle=True,
                                             save_to_dir=pictures_path,  
                                             save_prefix='Generated',     
                                             save_format='jpg'):   
            break  # otherwise the generator would loop indefinitely
        
    # Rename generated files:
    for pic in [join(pictures_path, f) for f in os.listdir(pictures_path) if 'Generated' in f]:
        shutil.move( pic, pic.replace('Generated',wnid + '_fake'))
        
    # Remove temporary folder:
    shutil.rmtree(pictures_path + '/' + 'temp')

   
    
def sizes_distribution(dataset_path):
    '''
    This function generates a dataframes, one stores height and width and  
    the number of channels available in RGB model for all pictures
    
    Args:
        * dataset_path: dataset path
    
    Return:
        * df_hw: dataframe with height width, channel and family name for each picture
    
    '''
    # Dataframe to store results
    df_hw = pandas.DataFrame(columns=['name','height','width','channels'])
    idx = 0
    
    # Get list with all directories in the Dataset
    dir_list = [x[1] for x in walk(dataset_path)][0]
    for dir in dir_list:
        current_dir = dataset_path + "/" + dir
        picture_names = [f for f in listdir(current_dir) if isfile(join(current_dir, f))]
        for picture in picture_names:
            try: 
                img = ndimage.imread(current_dir + "/" + picture,mode='RGB').shape
            except:
                print 'Unable to open file {0}'.format(current_dir + "/" + picture)
                os.remove(current_dir + "/" + picture)
                continue
            df_hw.loc[idx] = [picture,img[0],img[1],img[2]]
            idx += 1
              
    # generate family column
    df_hw['family'] = df_hw['name'].apply(lambda x: x.split("_")[0])
    
    return df_hw


def generate_sets(dataset_path,output_path,sizes,selected_classes): 
    '''
    This function generates the train and test structure directories required by 
    function data_augmentation in models.py
    
    Args:
        * dataset_path: dataset path with images
        * output_path: train and test dataset path
        * sizes: list with the train/test/validation sizes distributions, i.e: [0.6,0.2,0.2]
        * selected_classes: list with selected flowers categories used for modeling
    
    Return:
        None
    ''' 
    
    assert (np.sum(sizes) != 1.),"Total sizes must sum 1!, i.e. [.8,.1,.1]"
    
    try:
        os.mkdir(output_path)
        os.mkdir(output_path + '/' + 'train')
        os.mkdir(output_path + '/' + 'test')
        os.mkdir(output_path + '/' + 'validation')
    except:
        pass
    
    # Get list with all directories in the Dataset
    dir_list = [x[1] for x in walk(dataset_path)][0]
    for dir in dir_list:
        
        # Just process those flowers categories included in selected_classes
        if dir not in selected_classes: continue
        
        # Create folder in output directory
        current_dir = dataset_path + "/" + dir
        os.mkdir(current_dir.replace(dataset_path,output_path+"/train"))
        os.mkdir(current_dir.replace(dataset_path,output_path+"/test"))
        os.mkdir(current_dir.replace(dataset_path,output_path+"/validation"))
        
        # Get picture names and shuffle:
        picture_names = [f for f in listdir(current_dir) if isfile(join(current_dir, f))]
        random.shuffle(picture_names)
        train_size = int(len(picture_names)*sizes[0])
        test_size = int(len(picture_names)*sizes[1])
        
        # Copy files to the corresponding output directory
        for picture in picture_names[:train_size]:
            shutil.copyfile(current_dir + "/" + picture, current_dir.replace(dataset_path,output_path+"/train") + "/" + picture)
        for picture in picture_names[train_size:train_size + test_size]:
            shutil.copyfile(current_dir + "/" + picture, current_dir.replace(dataset_path,output_path+"/test") + "/" + picture)
        for picture in picture_names[train_size + test_size:]:
            shutil.copyfile(current_dir + "/" + picture, current_dir.replace(dataset_path,output_path+"/validation") + "/" + picture)        


if __name__ == '__main__':
    print "Dataset Wrangling module loaded..."
    
    
    
    


    