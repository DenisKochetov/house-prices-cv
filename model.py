import keras 
import os
import glob
from PIL import Image
import numpy as np
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16
from catboost import CatBoostRegressor


# load the model
# model = VGG16() 

# save model as pickle
# model.save('model.h5')



model = keras.models.load_model('model.h5')
catboost = CatBoostRegressor()      # parameters not required.
catboost.load_model('catboost')


def tile_images(files_list, size=128):
  houses = []
  tile_size = (size, size)
  for house in files_list:
    img_list = []
    tiled_image = np.zeros((tile_size[0] * 2, tile_size[1] * 2, 3), dtype='uint8')
    for i, location in enumerate(house):
      image = Image.open(location)
      image = image.convert('RGB')
      img_list.append(image.resize(tile_size))
    tiled_image = np.vstack([np.hstack([img_list[0], img_list[1]]), 
                             np.hstack([img_list[2], img_list[3]])])
    # convert to PIL from numpy
    houses.append(Image.fromarray(tiled_image))
  return houses



def load_image(filename):
    # load the image
    img = load_img(filename, target_size=(224, 224))
    # convert to array
    img = img_to_array(img)
    # reshape into a single sample with 3 channels
    img = img.reshape(1, 224, 224, 3)
    # center pixel data
    image =preprocess_input(img)
    return img

def predict_labels(model, image):
    # predict the probability across all output classes
    yhat = model.predict(image)
    # convert the probabilities to class labels
    label = decode_predictions(yhat)
    print(type(label))
    # retrieve the most likely result, e.g. highest probability
    # label = label[0][0]
    # print the classification
    return label


def tile_images(files_list):
    print(files_list[0])
    img_01 = Image.open(files_list[0])

    img_02 = Image.open(files_list[1])
    img_03 = Image.open(files_list[2])
    img_04 = Image.open(files_list[3])
    
    img_01_size = img_01.size
    img_02_size = img_02.size
    img_03_size = img_02.size
    img_02_size = img_02.size
    
    print('img 1 size: ', img_01_size)
    print('img 2 size: ', img_02_size)
    print('img 3 size: ', img_03_size)
    print('img 4 size: ', img_03_size)
    
    new_im = Image.new('RGB', (2*img_01_size[0],2*img_01_size[1]), (250,250,250))
    
    new_im.paste(img_01, (0,0))
    new_im.paste(img_02, (img_01_size[0],0))
    new_im.paste(img_03, (0,img_01_size[1]))
    new_im.paste(img_04, (img_01_size[0],img_01_size[1]))
    
    new_im.save(files_list[3], "PNG")