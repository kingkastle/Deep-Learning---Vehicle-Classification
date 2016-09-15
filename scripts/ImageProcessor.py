import os, shutil
import random
from os import listdir
from os.path import isfile, join
import Image
from scipy import ndimage
from keras.preprocessing.image import ImageDataGenerator


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
        print "Unable to resize file: {0}".format(img_path)
        return 0
    return 1


def generate_pics(pictures_path,number_pics,width,height):
    '''
    This script is used to generate pictures using the Keras utility ImageDataGenerator.
    
    Args:
        * pictures_path: path to folder where a particular flowers specie is located
        * number_pics: number of pictures to generate
        * width: desired width of the output picture
        * height: desired height of the output picture
        
    Return:
        none
    
    '''
    
    # Generate temporary folder and subfolder where the input pics to generate 
    # fake pictures will be located.
    if not os.path.exists(pictures_path + '/' + 'temp'):
        os.mkdir(pictures_path + '/' + 'temp')
        os.mkdir(pictures_path + '/' + 'temp/pics')
    
    # Get pictures paths:
    picture_files = [f for f in listdir(pictures_path) 
                     if isfile(join(pictures_path, f))
                     if ndimage.imread(join(pictures_path,f),mode='RGB').shape[0]>height]
    if len(picture_files) < number_pics:
        print "Warning: there are insufficient files with optimum size to generate augmented pics"
        number_pics = len(picture_files)
    
    # Select a number_pics of pictures using a random sample:
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
    for pic in [join(pictures_path, f) for f in listdir(pictures_path) if 'Generated' in f]:
        shutil.move( pic, pic.replace('Generated',wnid + '_fake'))
        
    # Remove temporary folder:
    shutil.rmtree(pictures_path + '/' + 'temp')
    
pictures_path = '/home/rafaelcastillo/MLND/Project5/DeepLearning/Dataset2/n12914923'
number_pics = 4
width = 250
height = 250
generate_pics(pictures_path,number_pics,width,height)
    