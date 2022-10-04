import os
import warnings
import numpy as np
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix
from tensorflow.keras.optimizers import Adam
from tensorflow_addons.metrics import CohenKappa
from keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras.layers import Dense, Activation,Dropout,Conv2D,GlobalAveragePooling2D,AveragePooling2D,Flatten,Input

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3,4,5,6,7"

# Distributed Learning
strategy = tf.distribute.MirroredStrategy()

class Model_fun():
    def __init__(self,train_data,val_data,class_weights):
        self.train_gen = train_data
        self.val_gen = val_data
        self.class_weights = class_weights
        self.history = None
        
    def callbacks(self,m_name):
        mcp = keras.callbacks.ModelCheckpoint(f"Model_weights/{m_name}_Model.h5",
                                      monitor="val_cohen_kappa",
                                      save_best_only=True, 
                                      verbose=1,
                                      mode='max')
        es = EarlyStopping(monitor='val_cohen_kappa', 
                                       mode='max', 
                                       verbose=1, 
                                       patience=8)
        rlr = ReduceLROnPlateau(monitor='val_cohen_kappa',
                                       factor=0.5, 
                                       mode='max',
                                       patience=4, 
                                       min_lr=0.00001, 
                                       verbose=1)
        callback=[mcp,es,rlr]
        return callback
    
    def resnet50(self,image_size):
        with strategy.scope():
            backbone = tf.keras.applications.ResNet50(weights='imagenet',include_top=False,
                                                             input_shape=(image_size,image_size,3))
            model = Sequential()
            model.add(backbone)
            model.add(GlobalAveragePooling2D())
            model.add(Dropout(0.5))
            model.add(Dense(5, activation='softmax'))
            model.compile(
                loss='categorical_crossentropy',
                optimizer=Adam(learning_rate=0.00005),
                metrics=['accuracy',CohenKappa(num_classes=5)]
            )

        return model
    
    def densenet121(self,image_size):
        with strategy.scope():
            backbone = tf.keras.applications.DenseNet121(weights='imagenet',include_top=False,
                                                             input_shape=(image_size,image_size,3))
            model = Sequential()
            model.add(backbone)
            model.add(GlobalAveragePooling2D())
            model.add(Dropout(0.5))
            model.add(Dense(5, activation='softmax'))
            model.compile(
                loss='categorical_crossentropy',
                optimizer=Adam(learning_rate=0.00005),
                metrics=['accuracy',CohenKappa(num_classes=5)]
            )

        return model
    
    def densenet169(self,image_size):
        with strategy.scope():
            backbone = tf.keras.applications.DenseNet169(weights='imagenet',include_top=False,
                                                             input_shape=(image_size,image_size,3))
            model = Sequential()
            model.add(backbone)
            model.add(GlobalAveragePooling2D())
            model.add(Dropout(0.5))
            model.add(Dense(5, activation='softmax'))

            model.compile(
                loss='categorical_crossentropy',
                optimizer=Adam(learning_rate=0.00005),
                metrics=['accuracy',CohenKappa(num_classes=5)]
            )

        return model
    
    def efficientnetb0(self,image_size):
        with strategy.scope():
            backbone = tf.keras.applications.EfficientNetB0(weights='imagenet',include_top=False,
                                                             input_shape=(image_size,image_size,3))        
            model = Sequential()
            model.add(backbone)
            model.add(GlobalAveragePooling2D())
            model.add(Dropout(0.5))
            model.add(Dense(5, activation='softmax'))
            model.compile(
                loss='categorical_crossentropy',
                optimizer=Adam(learning_rate=0.00005),
                metrics=['accuracy',CohenKappa(num_classes=5)]
            )

        return model
        
    
    def train_model(self,m_name='resnet50',image_size=224,batch_size=32):
        steps_per_epoch = self.train_gen.n // batch_size
        validation_steps = self.val_gen.n // batch_size
        if m_name=='densenet121':
            model=self.densenet121(image_size)
        elif m_name=='densenet169':
            model=self.densenet169(image_size)
        elif m_name=='efficientnetb0':
            model=self.efficientnetb0(image_size)
        else:
            model=self.resnet50(image_size)
        self.history=model.fit(self.train_gen,
                  steps_per_epoch = steps_per_epoch,
                  epochs= 100,
                  verbose=1,
                  validation_data=self.val_gen,
                  validation_steps=validation_steps,
                  class_weight=self.class_weights,
                  callbacks=self.callbacks(m_name)
                 )
        return model
    
# m=Model_fun(train,valid,class_weights)
# model_name='densenet169'
# model=m.train_model(model_name)
# model.save(f"Model_weights/{model_name}_Model.h5")
# model.load_weights(f"Model_weights/{model_name}_Model_checkpoint.h5")
