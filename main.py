# Tulio Castro Silva


#importar as bibliotecas
import tensorflow as tf
from keras import layers, models, callbacks
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder,StandardScaler
from tkinter.filedialog import askopenfile
import seaborn as sn
from random import randint
import os

path = "C:/Users/tulio/OneDrive/Documentos/GitHub/LMP/"
epocas = 500
otimizador = "Adam"

def create_df(path):
    df = pd.read_csv(path + "ACIDENTES/PARAMLOG_B-12.csv",sep=";",decimal=",")
    array = df.to_numpy()
    """
    for root, dirs, files in os.walk(path + "/ACIDENTES"):
        for arq in files :
            df.concat(pd.read_csv(path + "/ACIDENTES/{}".format(arq)))
    
    return df
    """
    print(array)

def split_sqeuence(sequence,n_steps):
    x, y = list(), list()
    for i in range(len(sequence)):
        len_step = i + n_steps
        if len_step > len(sequence) - 1:
            break
        x.append(sequence[i,len_step])
        y.append(sequence[len_step])
    return x,y


print(create_df(path))
"""
#Crir o dataframe
data_frame = pd.read_csv(path,sep=";",decimal=",")

labels = data_frame["LOG_ACIDENTE"].unique()                             #Colhendo os diferentes labels do data frame
encoder = LabelEncoder()                                                 #Criando o codificador
encoder.fit(labels)                                                      #Ajustando o codificador
data_frame["N_LOG"] = encoder.transform(data_frame["LOG_ACIDENTE"])      #Criando uma nova coluna no data frame

#Ajustando o data frame
scaler = StandardScaler()                                                                       #Criando o scaler
data_frame.iloc[:,0:13] = pd.DataFrame(scaler.fit_transform(data_frame.iloc[:,0:13]))           #Normalizando os valores


#Criando os valores de treino, teste e validacao 

train = data_frame.sample(frac=0.8)         #Separando 80% do data frame para o treino
data_frame = data_frame.drop(train.index)   #Removendo os valores de treino do data frame
valid = data_frame.sample(frac=0.5)         #Separando 50% dos valores para a validacao
data_frame = data_frame.drop(valid.index)   #Removendo os valores de validacao do data frame
test = data_frame.sample(frac=1)            #Separando o resto dos valores para o teste

X_train = train.iloc[:,0:13]                 
y_train = train.iloc[:,14]
X_valid = valid.iloc[:,0:13]
y_valid = valid.iloc[:,14]
X_test = test.iloc[:,0:13]
y_test = test.iloc[:,14]

print(X_train)
print(y_train)


#Criando o modelo

tf.random.set_seed(seed)

model = models.Sequential()

model.add(layers.Dense(13,activation="relu",input_dim = 13))

model.add(layers.Dense(26,activation="relu"))
model.add(layers.Dense(22,activation="relu"))
model.add(layers.Dense(18,activation="relu"))

model.add(layers.Dense(9,activation="softmax"))

#Otimizador
model.compile(loss='sparse_categorical_crossentropy', optimizer=otimizador, metrics=['accuracy'])

#model.summary()

#Callback
checkpoint_filepath = "D:/GitHub/Redes-Neurais/codigos/1_Trabalho/best.h5"

model_checkpoint_callback = callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)


#Treinamento
history = model.fit(X_train, y_train, 
                    epochs=epocas, 
                    batch_size=20, 
                    validation_data=(X_valid, y_valid), 
                    shuffle=True, 
                    callbacks=[model_checkpoint_callback],
                    verbose=1)

y_pred = model.predict(X_test)


#Avaliando a rede
model.load_weights(checkpoint_filepath)

model.evaluate(X_train, y_train)
model.evaluate(X_valid, y_valid)
model.evaluate(X_test, y_test)

#Criando os graficos
fig ,(ax1,ax2) = plt.subplots(2)

#Grafico de loss e val_loss
ax1.plot(history.history["loss"],label = "Loss")
ax1.plot(history.history["val_loss"],label = "Valid")
ax1.legend(loc = "upper right")

#Grafico de acuracia 
ax2.plot(history.history["accuracy"],label = "train")
ax2.plot(history.history["val_accuracy"],label = "Valid")
ax2.legend(loc = "upper right")


#Heatmap da matriz de confusao
y_pred_labels = [i.argmax() for i in y_pred]

cm = tf.math.confusion_matrix(labels=y_test, predictions=y_pred_labels)

plt.figure(figsize = (10,7))
sn.heatmap(cm, annot=True, fmt='d')
plt.ylabel('Truth')
plt.xlabel('Predicted')

plt.show()






"""