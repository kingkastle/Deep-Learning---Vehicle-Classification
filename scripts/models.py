'''
Created on Aug 30, 2016

@author: rafaelcastillo

This script includes the modelling part

ref: https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
ref: https://blog.keras.io/how-convolutional-neural-networks-see-the-world.html
'''
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, AveragePooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from scipy import ndimage
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from datetime import datetime
import pandas, os
import numpy as np
import configuration, Dataset_wrangling
from sklearn.metrics import accuracy_score

def data_augmentation(path_dataset,pic_dims):
    """
    This function generates batches of input data for models from the directories.
    
    Args:
        * dataset_path: dataset path
        * pic_dims: list with width and height of pictures
        
    Returns:
        * train and test objects with the input images
    """
    
    # augmentation configuration used for training
    train_datagen = ImageDataGenerator(
            rescale=1/255.,
            rotation_range=30,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest')

    # augmentation configuration used for testing
    test_datagen = ImageDataGenerator(rescale=1./255)

    # reading images from the specified directory and generating batches of augmented data
    train_generator = train_datagen.flow_from_directory(
            '{0}/train'.format(path_dataset),
            color_mode="grayscale",
            target_size=(pic_dims[0], pic_dims[1]),
            batch_size=32,
            class_mode='binary')
    
    # reading images from the specified directory and generating batches of augmented data
    validation_generator = test_datagen.flow_from_directory(
            '{0}/test'.format(path_dataset),
            color_mode="grayscale",
            target_size=(pic_dims[0], pic_dims[1]),
            batch_size=32,
            class_mode='binary')
    
    return train_generator, validation_generator


def learning_curves(model_path,model_name,optimizer,history,show_plots):
    """
    Display and save learning curves.
    
    Args:
        * model_path: path to models directory
        * model_name: name of trained and validated model
        * show_plots: whether to show plots or not while executing
    """
    
    # accuracy
    plt.figure()
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('accuracy of the model')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['validation set','training set'], loc='lower right')
    plt.savefig(model_path + "/" + model_name + '_acc.png')
    if show_plots: plt.show()

    # loss
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('loss of the model')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['validation set','training set'], loc='upper right')
    plt.savefig(model_path + "/" + model_name + '_loss.png')
    if show_plots: plt.show()
    

def build_model(optimizer,pic_dims,layers):
    """
    Builds model with desired hyperparameters.
    
    Args: 
        * optimizer: optimizer used in the model
        * pic_dims: list with width and height of pictures
        * layers: number of conv layers included in the net
    
    Returns:
        * model: defined model with the selected optimizer
    """
    
    # Define the first conv layer
    model = Sequential()
    model.add(Convolution2D(32, 3, 3, activation='relu', input_shape=(1, pic_dims[0], pic_dims[1]), name='conv1'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    #Include as many convs layers as defined by parameter layer
    num_layers = 1
    while (num_layers < layers):
        num_layers += 1
        model.add(Convolution2D(64, 3, 3, activation='relu', name='conv{0}'.format(num_layers)))
        model.add(MaxPooling2D(pool_size=(2, 2)))

    # Define on top a fully connected net
    model.add(Flatten())
    model.add(Dense(150))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    
    return model

def run_model(optimizer,nb_epoch,model_path,path_dataset,pic_dims,layers):
    """
    This function builds the model as well as validate its over the validation set.
    Model weights are saved in a local file as well as training time and accuracy over validation sets.
    
    Args:
        * optimizer: optimizer used to calculate deep net loss
        * nb_epoch: number of epochs used during training process
        * model_path: path to models directory
        * path_dataset: path to input images
        * pic_dims: list with width and height of pictures
        * layers: number of conv layers included in the net
        
    Return:
        * model: trained model
        * time_to_train: time (seconds) required to train model
        * acc: model accuracy over validation set
        * model_name: name of trained and validated model
    """
    
    # Generate connectors to augmented train/test data:
    train_generator, validation_generator = data_augmentation(path_dataset,pic_dims)
    model = build_model(optimizer,pic_dims,layers)
    
    #Start calculating train processing time:
    startTime = datetime.now()
    
    # Train model:
    history = model.fit_generator(
                    train_generator,
                    samples_per_epoch=300,
                    nb_epoch=nb_epoch,
                    validation_data=validation_generator,
                    nb_val_samples=100,
                    verbose=1)
    
    # End training processing:
    time_to_train = datetime.now() - startTime
    
    # Generate learning plots:
    model_name = "Layers_{0}_".format(layers) + str(optimizer)
    learning_curves(model_path,model_name,optimizer,history,False)
    model.save_weights(model_path + "/" + model_name +'.h5')
    
    # Calculate accuracy over validation set:
    model_parameters = [model_path,optimizer,model_name,pic_dims]
    acc = validate_model(model_parameters,path_dataset)
    
    return model,model_name,time_to_train,acc

def validate_model(model_parameters,path_dataset):
    '''
    
    This script test a model over the validation dataset (using augmented data)
    
    Args:
        * model_parameters: List that includes:[model_path, optimizer, model_name, [height, width], layers]
        * path_dataset: path to dataset (where validation data is included)
        
    Returns:
        * Accuracy score
    '''
    
    model_path, optimizer, model_name, pic_dims = model_parameters
    
    # Load model
    model = build_model(optimizer,pic_dims,layers)
    model.load_weights(model_path + "/" + model_name + '.h5')
    
    # Load test data and labels
    test_datagen = ImageDataGenerator(rescale=1./255)
    i = 0
    for test_data in test_datagen.flow_from_directory(
            '{0}/validation'.format(configuration.dataset_train_test),
            target_size=(150, 150),
            batch_size=200,
            color_mode="grayscale",
            class_mode='binary',
            shuffle=False):
        i += 1
        if i > 2:
            break  # otherwise the generator would loop indefinitely
    X = test_data[0]
    y_true = test_data[1]
    
    # Predict on test data
    y_pred = model.predict(X, batch_size=1, verbose=0)
    
    # Round predictions to 1s and 0s
    y_pred = np.around(y_pred)
    
    return accuracy_score(y_true, y_pred)

def visualize_filters(path_to_image,layer_name,filters):
    '''
    This function is used to visualize two filters for a layer in a neural network
    
    * Args:
        * path_to_image: Path to image
        * layer_name: name of the layer to visualize
        * filters: list of length = 2 to visualize
        
    * Returns:
        Generates a visualization
    '''
    # Get layer index from model.layers
    layer_index = [i for i,x in enumerate(model.layers) if x.name == layer_name][0]
    
    # Function to get the layer output
    get_layer_output = K.function([model.layers[0].input, K.learning_phase()],
                                  [model.layers[layer_index].output])
    
    # Resize picture
    Dataset_wrangling.resize_pic(path_to_image,150,150)
    X = ndimage.imread(path_to_image.replace('.jpg','_good_shape.jpg'),flatten=True)
    y = np.expand_dims(X, axis=0)
    os.remove(path_to_image.replace('.jpg','_good_shape.jpg'))
    
    # Get layer output for the image:
    layer_output = get_layer_output([np.expand_dims(y, axis=0), 0])[0]
    
    filter_1 = filters[0]
    filter_2 = filters[1]
    
    # Generate visualization:
    fig = plt.figure()

    ax1 = plt.subplot(131)
    ax2 = plt.subplot(132)
    ax3 = plt.subplot(133)

    ax1.imshow(mpimg.imread(path_to_image))
    ax2.imshow(layer_output[0,filter_1,:,:],cmap='hot',alpha=1)
    ax3.imshow(layer_output[0,filter_2,:,:],cmap='hot',alpha=1)
    ax1.set_title('Original')
    ax2.set_title('Filter: {0}'.format(filter_1))
    ax3.set_title('Filter: {0}'.format(filter_2))

    fig.suptitle('Image representation in Layer {0}'.format(layer_name), fontsize=14, fontweight='bold')

    plt.tight_layout()
    



if __name__ == '__main__':
    print "Models module loaded..."
    
    results = pandas.DataFrame(columns=['model_name','time_to_train','accuracy'])
    dataset_path = '/home/rafaelcastillo/MLND/Project5/DeepLearning/Dataset_train_test'
    model_path = '/home/rafaelcastillo/MLND/Project5/DeepLearning/models'
    optimizer_list = ['adam', 'Adagrad']
    pic_dims = [150,150]
    nb_epocs = 100
    for layers in [1,2,3,4]:
        for optimizer in optimizer_list:
            model,model_name,time_to_train,acc = run_model(optimizer, nb_epocs, model_path, dataset_path,pic_dims,layers)
            results.loc[results.shape[0]+1,:] = [model_name,time_to_train,acc]
    results.to_csv(dataset_path + "/" + 'Net_results.csv',sep=",",index=False)

