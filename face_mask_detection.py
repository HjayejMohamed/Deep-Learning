print("[INFO] SETUP...")

from tensorflow.keras.layers import Dense ,Flatten,AveragePooling2D,Conv2D
from tensorflow.keras.preprocessing.image import img_to_array,load_img
from sklearn.model_selection import train_test_split
from tensorflow import keras
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt


print("[INFO] debut...")
DATADIR=r"C:\Users\mohamed\Desktop\Face-Mask-Detection-master\Face-Mask-Detection-master\dataset"
CATEGORIES=["with_mask","without_mask"]

data=[]
labels=[]

print("[INFO] loading data...")
data=[]
labels=[]

#Data Collection....
for category in CATEGORIES:
    path=os.path.join(DATADIR,category)
    for img in os.listdir(path):
        img_path=os.path.join(path,img)
        image=load_img(img_path,target_size=(224,224))
        image=img_to_array(image)
        image=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
        data.append(image)
        labels.append(CATEGORIES.index(category))


#Scaling...
print("[INFO]scaling...")
data=np.array(data)
data=data / 255
labels=np.array(labels)

#Splitting...
print("[INFO] spliting...")
(x_train,x_test,y_train,y_test)=train_test_split(data,labels,test_size=.20,random_state=42)

#Model Archi...
cnn=keras.Sequential([
		Conv2D(32,kernel_size=(3,3),activation='relu',input_shape=(224,224,1)),
		AveragePooling2D((2,2),strides=2),
		Conv2D(64,kernel_size=(3,3),activation='relu'),
		AveragePooling2D((2,2),2),
		Flatten(),
		Dense(150,activation='relu'),
		Dense(2,activation='sigmoid')
	])


#Complation....
print("[INFO]compile... ")
cnn.compile(
		optimizer='adam',
		loss='sparse_categorical_crossentropy',
		metrics=['accuracy']
	)

#Fitting..
print("[INFO]fiting ")
cnn.fit(x_train,y_train,epochs=2)

#Predictions...
print("[INFO] predictions..")
predictions=cnn.predict(x_test)

#Affichage de Quelques Predictions.. 
print(predictions[:10])
predx=[np.argmax(i) for i in predictions]
print(predx[:10])
print(y_test[:10])

#Saving...
print("[INFO] saving mask detector model...")
cnn.save("my_mask_detector01.model", save_format="h5")

print("[INFO]fin... ")


