#import Required modules

import os
import cv2
import time
import random
import shutil
import numpy as np
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator

# Data function 
# Functions - Read dataset, Preprocess dataset, Augment dataset, Visualize dataset
class Data_fun():
    def __init__(self,size_delta=(224,224),batch_size=32):
        self.size_delta = size_delta
        self.batch_size = batch_size        
    
    def load_data(self,input_files):
        self.data=pd.DataFrame()
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
            data=data[[x_col,y_col]]
            data.columns=['id_code','diagnosis']
            if (suffix!=''):
                data['id_code'] = data['id_code']+suffix
            data['id_code'] = prefix+data['id_code']
            data['diagnosis'] = data['diagnosis'].map(str)
            self.data=self.data.append(data,ignore_index = True)      
        self.train_data,self.test_data = train_test_split(self.data,train_size = .75,shuffle=True, random_state=321)
        
    def brightness(self,image, brightness_delta=None):
        if not brightness_delta:
            brightness_delta = random.uniform(0.1,0.4)
        image = tf.image.adjust_brightness(image, brightness_delta)
        return image
    
    def contrast(self,image,contrast_delta=None):
        if not contrast_delta:
            contrast_delta = random.randint(1,4) #2
        image = tf.image.adjust_contrast(image,contrast_delta)
        return image
    
    def hue(self,image,hue_delta=None):
        if not hue_delta:
            hue_delta = random.uniform(0.1,0.4)  #0.2
        image = tf.image.adjust_hue(image, hue_delta)
        return image
    
    def saturation(self,image,saturation_delta=None):
        if not saturation_delta:
            saturation_delta = random.uniform(0.1,0.4)  #0.5
        image = tf.image.adjust_saturation(image, saturation_delta)
        return image
    
    def resize(self,image):
        image = cv2.resize(image, self.size_delta)
        return image
    
    def addweight(self,image):
        image = cv2.addWeighted (image,4, cv2.GaussianBlur(image, (0,0) ,10), -4, 128)
        return image
    
    def preprocess(self,image):
        image = self.resize(image)
        image = self.addweight(image)
        functions = [self.brightness,self.contrast,self.hue,self.saturation]
        image = random.choice(functions)(image)
        return image
    
    def getdata(self,):
        train_datagen = ImageDataGenerator(preprocessing_function=self.preprocess,rescale=1./255)
        train_gen = train_datagen.flow_from_dataframe(dataframe=self.train_data,
                                        # directory=train_data_dir,
                                        x_col="id_code", 
                                        y_col="diagnosis",
                                        class_mode="categorical",
                                        target_size=(self.size_delta), 
                                        batch_size=self.batch_size,
                                        # subset='training'
                                        )
        val_gen = train_datagen.flow_from_dataframe(dataframe=self.test_data,
                                                # directory=train_data_dir, 
                                                x_col="id_code", 
                                                y_col="diagnosis",
                                                class_mode="categorical",
                                                target_size=(self.size_delta), 
                                                batch_size=self.batch_size,
                                                # subset='validation'
                                                )
        return train_gen,val_gen
        
    def barchart(self,batch=''):
        if(batch=='train'):
            self.train_data['diagnosis'].value_counts().sort_index().plot(kind="bar",figsize=(10,5),rot=0)
        elif(batch=='test'):
            self.test_data['diagnosis'].value_counts().sort_index().plot(kind="bar",figsize=(10,5),rot=0)
        else:
            self.data['diagnosis'].value_counts().sort_index().plot(kind="bar",figsize=(10,5),rot=0)
        plt.title("Label Distribution",weight='bold',fontsize=15)
        plt.xticks(fontsize=13)
        plt.yticks(fontsize=13)
        plt.xlabel("Label", fontsize=15)
        plt.ylabel("Frequency", fontsize=15)
        plt.savefig(f'Outputs/{batch}dataset_barchart.png')
    
    def piechart(self,batch='dataset'):
        if(batch=='train'):
            chat_data = self.train_data.diagnosis.value_counts()
        elif(batch=='test'):
            chat_data = self.test_data.diagnosis.value_counts()
        else:
            chat_data = self.data.diagnosis.value_counts()
        plt.pie(chat_data, autopct='%1.1f%%', shadow=True, labels=["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"])
        plt.title('Per class sample Percentage');
        plt.show()
        plt.savefig(f'Outputs/{batch}_piechart.png')
        
    def visualize(self,method='Normal'):
        fig, ax = plt.subplots(1, 5, figsize=(15, 6))
        for i in range(5):
            sample = self.data[self.data['diagnosis'] == str(i)].sample(1)
            image_name = sample['id_code'].item()
            if(method=='preprocess'):
                X = self.preprocess(cv2.imread(f"{image_name}"))
            elif(method=='brightness'):
                X = self.brightness(cv2.imread(f"{image_name}"))
            elif(method=='contrast'):
                X = self.contrast(cv2.imread(f"{image_name}"))
            elif(method=='hue'):
                X = self.hue(cv2.imread(f"{image_name}"))
            elif(method=='saturation'):
                X = self.saturation(cv2.imread(f"{image_name}"))
            else:
                X = cv2.imread(f"{image_name}")
            ax[i].set_title(f"Image: {image_name.split('/')[2]}\n Label = {sample['diagnosis'].item()}", 
                            weight='bold', fontsize=10)
            ax[i].axis('off')
            ax[i].imshow(X)
        plt.savefig(f'Outputs/{method}.png')
            
    def trim (self,df, max_size, min_size, column):
        df=df.copy()
        sample_list=[] 
        groups=df.groupby(column)
        for label in df[column].unique():        
            group=groups.get_group(label)
            sample_count=len(group)         
            if sample_count> max_size :
                samples=group.sample(max_size, replace=False, weights=None, random_state=123, axis=0).reset_index(drop=True)
                sample_list.append(samples)
            elif sample_count>= min_size:
                sample_list.append(group)
        df=pd.concat(sample_list, axis=0).reset_index(drop=True)
        balance=list(df[column].value_counts())
        # print (balance)
        return df
    
    def balance(self,train_df,max_samples, min_samples, column, working_dir, image_size):
        train_df=train_df.copy()
        train_df=self.trim (train_df, max_samples, min_samples, column)    
        # make directories to store augmented images
        aug_dir=os.path.join(working_dir, 'augmented_dataset')
        if os.path.isdir(aug_dir):
            shutil.rmtree(aug_dir)
        os.mkdir(aug_dir)
        for label in train_df['diagnosis'].unique():    
            dir_path=os.path.join(aug_dir,label)    
            os.mkdir(dir_path)
        # create and store the augmented images  
        total=0
        gen=ImageDataGenerator(horizontal_flip=True,  rotation_range=20, width_shift_range=.2,
                                      height_shift_range=.2, zoom_range=.2, brightness_range=[0.7,1.3])
        groups=train_df.groupby('diagnosis') # group by class
        for label in train_df['diagnosis'].unique():  # for every class               
            group=groups.get_group(label)  # a dataframe holding only rows with the specified label 
            sample_count=len(group)   # determine how many samples there are in this class  
            if sample_count< max_samples: # if the class has less than target number of images
                aug_img_count=0
                delta=max_samples-sample_count  # number of augmented images to create
                target_dir=os.path.join(aug_dir, label)  # define where to write the images    
                aug_gen=gen.flow_from_dataframe( group,  x_col='id_code', y_col=None, target_size=image_size,
                                                class_mode=None, batch_size=1, shuffle=False, 
                                                save_to_dir=target_dir, save_prefix='aug-', color_mode='rgb',
                                                save_format='jpg')
                while aug_img_count<delta:
                    images=next(aug_gen)            
                    aug_img_count += len(images)
                total +=aug_img_count
        print('Total Augmented images created= ', total)
        # create aug_df and merge with train_df to create composite training set ndf
        if total>0:
            aug_fpaths=[]
            aug_labels=[]
            classlist=os.listdir(aug_dir)
            for klass in classlist:
                classpath=os.path.join(aug_dir, klass)     
                flist=os.listdir(classpath)    
                for f in flist:        
                    fpath=os.path.join(classpath,f)         
                    aug_fpaths.append(fpath)
                    aug_labels.append(klass)
            Fseries=pd.Series(aug_fpaths, name='id_code')
            Lseries=pd.Series(aug_labels, name='diagnosis')
            aug_df=pd.concat([Fseries, Lseries], axis=1)
            ndf=pd.concat([train_df,aug_df], axis=0).reset_index(drop=True)
        else:
            ndf=train_df
        print (list(ndf['diagnosis'].value_counts()) )
        return ndf
    
    def augment(self,max_samples):
        try:
            os.system("rm -r augmented_dataset")
        except:
            pass
        time.sleep(2)
        print('Before Augmentation :')
        print(self.train_data['diagnosis'].value_counts())
        min_samples=0
        column='diagnosis'
        working_dir = ''
        img_size=self.size_delta
        df=self.balance(self.train_data,max_samples, min_samples, column, working_dir, img_size)
        print('After Augmentation :')
        print(df['diagnosis'].value_counts())
        self.train_data = df