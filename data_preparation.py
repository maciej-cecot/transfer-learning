import pickle
import os
import numpy as np
import random
import matplotlib.pyplot as plt
from keras.preprocessing.image import array_to_img
import tensorflow as tf
import tensorflow.python.platform
from tensorflow.python.platform import gfile


def load_data(file):
    with open(file, 'rb') as data_file:
        data = pickle.load(data_file, encoding='latin1')
    x = data['data']
    x = np.transpose(np.reshape(x,(-1,32,32,3), order='F'),axes=(0,2,1,3))
    y = np.asarray(data['labels'], dtype='uint8')
    return x, y


def create_path_list(array, folder_path):
    """
    Creates images from given arrays and returns list of paths for all images.
    """
    for directory in ['images/train','images/test']:
        if not os.path.exists(directory):
            os.makedirs(directory)
    full_path = []
    for i in range((len(array))):
        img = array_to_img(array[i])
        path = folder_path+str(i)+'.jpg'
        img.save(path)
        full_path.append(path)
    return full_path


def visualize_images(x_train, y_train, n_class):
    ind=[]
    for classes in range(n_class):
        indices = random.sample([i for i, x in enumerate(y_train) if x == classes], 10)
        ind.append(indices)
    f, axarr = plt.subplots(n_class,n_class, figsize=(10,10))
    for x in ind:
        for y in x:
            axarr[ind.index(x),x.index(y)].imshow(x_train[y])
            axarr[ind.index(x),x.index(y)].axis('off')       
    plt.show() 


def extract_features(image_paths, model_path, verbose=True):
    # creating graph
    with gfile.FastGFile(model_path, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')
    feature_dimension = 2048
    features = np.empty((len(image_paths), feature_dimension))
    #extracting features using given layer
    with tf.Session() as sess:
        tensor = sess.graph.get_tensor_by_name('pool_3:0')
        for i, image_path in enumerate(image_paths):
            if verbose and i%50==0:
                print('Processing %s...' % (image_path))
 
            if not gfile.Exists(image_path):
                tf.logging.fatal('File does not exist %s', image)
 
            image_data = gfile.FastGFile(image_path, 'rb').read()
            feature = sess.run(tensor, {
                'DecodeJpeg/contents:0': image_data
            })
            features[i, :] = np.squeeze(feature)
    return features 
