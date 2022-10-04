import os
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import load_model
from tensorflow_addons.metrics import CohenKappa
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3,4,5,6,7"

strategy = tf.distribute.MirroredStrategy()

class Ensemble_fun():
    def __init__(self,image_size=224):
        self.image_size=image_size
        
    def load_models(self,model_paths):
        self.models=list()
        index=0
        for m in model_paths:
            model=load_model(m)
            model._name="model"+str(index)
            self.models.append(model)
            index+=1
            
    def ensemble(self,):
        with strategy.scope():
            ensemble_models = self.models
            ensemble_input = tf.keras.Input(shape=(self.image_size, self.image_size, 3))
            ensemble_outputs = [ensemble_model(ensemble_input) for ensemble_model in ensemble_models]
            ensemble_output = tf.keras.layers.Average()(ensemble_outputs)
            ensemble_model = tf.keras.Model(inputs=ensemble_input, outputs=ensemble_output)
            ensemble_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00005), 
                          loss='categorical_crossentropy', 
                          metrics=['accuracy',CohenKappa(num_classes=5)])
            ensemble_model._name="EnsembleModel"
        return ensemble_model