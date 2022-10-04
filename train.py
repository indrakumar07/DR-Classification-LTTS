from data_function import Data_fun
from model_function import Model_fun
from prediction_function import Predict_fun
from ensemble import Ensemble_fun

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import load_model
from tensorflow_addons.metrics import CohenKappa

#Loading Data
def main():
    data_obj=Data_fun()
    data_obj.load_data([["Aptos/train.csv",'id_code','diagnosis','Aptos/train_images/','.png']])
    train_datagen,validation_datagen=data_obj.getdata()

    #Calculating ClassWeights
    class_weights = class_weight.compute_class_weight(class_weight = 'balanced',
                                                     classes = np.unique(train_datagen.classes),
                                                    y = train_datagen.classes)
    class_weights = dict(zip(np.unique(train_datagen.classes), class_weights))
    print('Class Weights:',class_weights,sep='\n\t')

#Augmentation 
def augmentation(MAX_COUNT):
    train_datagen=data_obj.augment(MAX_COUNT)

#Training Model
def train(model_name):
    model_obj=Model_fun(train_datagen,validation_datagen,class_weights)
    # model_name='densenet169'
    model=model_obj.train_model(model_name)

    plt.plot(model_obj.history.history['accuracy'])
    plt.plot(model_obj.history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(f'Outputs/{model_name}_accuracy.png')

    #  "Loss"
    plt.plot(model_obj.history.history['loss'])
    plt.plot(model_obj.history.history['val_loss'])
    plt.title('model Loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(f'Outputs/{model_name}_loss.png')

    #  "Kappa"
    plt.plot(model_obj.history.history['cohen_kappa'])
    plt.plot(model_obj.history.history['val_cohen_kappa'])
    plt.title('model cohen_kappa')
    plt.ylabel('cohen_kappa')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(f'Outputs/{model_name}_cohenkappa.png')

def ensemble(model_list):
    ens_obj=Ensemble_fun()
    ens_obj.load_models([model_list])
    ens_model=ens_obj.ensemble()
    ens_model.save("Model_weights/ensemble_model.h5")
    
#evaluate Model
def evaluate(model_name,model_path):
    model = load_model(model_path)
    model.evaluate(validation_datagen)
    
    #Confusion_matrix 
    np.unique(validation_datagen.classes, return_counts=True)
    prediction=model.predict(validation_datagen)
    y_pred = np.argmax(prediction, axis=1)
    y_true = validation_datagen.classes    
    cm = confusion_matrix(y_true, y_pred)
    figure=sns.heatmap(cm, annot=True, fmt="d")
    figure.figure.savefig(f'Outputs/{model_name}_confusion_matrix.png')
    
def predict(model_path,file):
    labels={0:'No-DR',1:'Mild-DR',2:'Moderate-DR',3:'Severe-DR',4:'Proliferative-DR'}
    pre_obj=Predict_fun(model_path)
    if ('.png' or '.jpg') in file:
        pre=pre_obj.predict_image(file)    
        return labels[pre]
    else:
        df=pre_obj.predict_generator(file)
        return df
    
Dense121="Model_weights/densenet121_Model.h5"
Dense169="Model_weights/densenet169_Model.h5"
Res50="Model_weights/resnet50_Model.h5"
Ens="Model_weights/ensemble_model.h5"
model_list=[Dense121,Dense169,Res50]

# main()
# augmentation(700)
# train()
# ensemble(model_list)
# evaluate('Ensemble',Ens)
file=input("\033[31m {}\033[00m".format('Enter image or csv location:'))
res=predict(Ens,file)
if (type(res)==type('')):
    print("\033[92m {}\033[33m" .format('Given Image is of class '),res)
    print('\033[00m')
else:
    res.to_csv('submission.csv',index=False)
    print("\033[92m {}\033[00m" .format('submission.csv Saved Successfully'))