# Tulio Castro Silva


#importar as bibliotecas
import tensorflow as tf
from keras import layers, models
import keras.callbacks as tfc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder,StandardScaler
import os
from sklearn.model_selection import train_test_split
import seaborn as sn


PC = True
STEP = 5
epocas = 10
otimizador = "Adam"

label = {"PARAMLOG_B-12":0,"PARAMLOG_B-13":1,"PARAMLOG_B-14":2,"PARAMLOG_B-15":3,"PARAMLOG_B-16":4,"PARAMLOG_B-17":5,"PARAMLOG_B-18":6,"PARAMLOG_L-69_OPTION-10020":7,"PARAMLOG_N-79_OPTION-1":8}
if PC:
    path = "D:/GitHub/LMP/"
else:
    path = "C:/Users/tulio/OneDrive/Documentos/GitHub/LMP/"

checkpoint_filepath = path + "/ckp/"

def split_sqeuence(sequence,n_steps):
    x, y = [] , []
    for i in range(len(sequence)):
        len_step = i + n_steps
        if len_step > len(sequence) - 1:
            break
        y.append(label[sequence[len_step][13]])
        new = []
        for element in sequence[i:len_step]:
            new.append(element[0:13])
        x.append(new)
    return x,y

def create_df(path,step):
    arr = np.empty([0,14])
    scaler =  StandardScaler()
    for root, dirs, files in os.walk(path + "/ACIDENTES"):
        for arq in files :
            a = pd.read_csv(path + "ACIDENTES/{}".format(arq),sep=";",decimal=",")
            a.iloc[:,0:13] = pd.DataFrame(scaler.fit_transform(a.iloc[:,0:13])).to_numpy()
            arr = np.concatenate((arr,a))
    return split_sqeuence(arr,step)


def Create_train_test_split(df):
    X_train, X_test, y_train, y_test = train_test_split(df[0],df[1], test_size=0.2, random_state=1)
    X_train, X_val , y_train, y_val = train_test_split(X_train,y_train,test_size=0.25,random_state=1)
    return X_train, y_test, X_val, y_train,y_train, y_val


def create_model():
    model = models.Sequential()
    model.add(layers.Dense(13,activation="relu",input_shape=(STEP,13)))
    model.add(layers.Dense(26,activation="relu"))
    model.add(layers.Dense(22,activation="relu"))
    model.add(layers.Dense(18,activation="relu"))
    model.add(layers.Dense(9,activation="softmax"))
    model.compile(loss='sparse_categorical_crossentropy', optimizer=otimizador, metrics=['accuracy'])
    return model


df = create_df(path, STEP)
X_train, X_test, X_val, y_train,y_train, y_val = Create_train_test_split(df)


model = create_model()

callback = tfc.ModelCheckpoint(checkpoint_filepath+"best.h5",save_best_only=True)
early_stopping_callback = tfc.EarlyStopping(patience=5,restore_best_weights=True)     

X_train = np.asarray(X_train).astype('float32')
y_train = np.asarray(y_train)
X_val = np.asarray(X_val).astype('float32')
y_val = np.asarray(y_val)




history = model.fit(X_train, y_train, 
                    epochs=epocas, 
                    validation_data=(X_val, y_val), 
                    callbacks=[early_stopping_callback, callback],
                    verbose=1)
















