from data_function import Data_fun

import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow_addons.metrics import CohenKappa
from keras.preprocessing.image import ImageDataGenerator

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3,4,5,6,7"

class Predict_fun():
    
    def __init__(self,model_path,image_size=(224,224),batch_size=32):        
        self.model_path=model_path
        self.size_delta=image_size
        self.batch_size=batch_size
        try:
            self.model=load_model(self.model_path)
        except Exception as e:
            print(e)
    
    def predict_image(self,image_path):
        image=cv2.imread(image_path)
        obj=Data_fun()
        image=obj.preprocess(image)
        plt.imshow(image)
        image=tf.expand_dims(image,0)
        
        model=self.model
        prediction=model.predict(image)
        pred=np.argmax(prediction,axis=-1)
        return pred[0]
    
    def predict_generator(self,input_files):
        self.data=pd.DataFrame()
        self.df_copy=None
        for file in input_files:
            try:                
                file_path=file[0]     #file path
                x_col=file[1]         #image name
                y_col=file[2]         #image labels
                prefix=file[3]        #image paths
                suffix=file[4]        #image format
            except:
                print("\033[92m {}\033[00m" .format("Input Format:[01,02,03,04,05]\n\t01.File Location\n\t02.Column Name Containing FileName\n\t03.Column Name Containing Labels\n\t04.Image Location\n\t05.Image Format Eg('.png','.jpg')"))
                return
            data=pd.read_csv(file_path)
            data=data[[x_col]]
            data[y_col]=''
            data.columns=['id_code','diagnosis']
            self.df_copy=data.copy()
            if (suffix!=''):
                data['id_code'] = data['id_code']+suffix
            data['id_code'] = prefix+data['id_code']
            self.data=self.data.append(data,ignore_index = True)
        obj=Data_fun()
        datagen = ImageDataGenerator(preprocessing_function=obj.preprocess,rescale=1./255)
        gen = datagen.flow_from_dataframe(dataframe=self.data, 
                                                x_col="id_code", 
                                                y_col='diagnosis',
                                                class_mode=None,
                                                target_size=self.size_delta,
                                                batch_size=self.batch_size,
                                                shuffle=False,
                                                )
        model=self.model
        prediction=model.predict(gen)
        self.df_copy['diagnosis']=np.argmax(prediction, axis=1)
        return self.df_copy