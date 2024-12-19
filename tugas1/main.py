import os
import numpy as np
import cv2 as cv
import pickle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

TrainPath = r'G:\\Coding\\python\\UTS KB\\tugas1\\Train'

pick = open('model.sav', 'rb')
model = pickle.load(pick)
pick.close()
print("Model loaded")

pickle_in = open("image.pickle", "rb")
data = pickle.load(pickle_in)


# Separate features and labels
features = []
labels = []

for feature, label in data:
    features.append(feature)
    labels.append(label)



# Convert features and labels to numpy arrays
features = np.array(features)
labels = np.array(labels)


xtrain, xtest, ytrain, ytest = train_test_split(features, labels, test_size=0.1)


categories = os.listdir(TrainPath)
predict = model.predict(xtest)
accuracy = model.score(xtest, ytest)
print("Model accuracy: ", accuracy)
print("Prediction: ", categories[predict[0]])

mySign = xtest[0].reshape(100, 100, 3)
plt.imshow(mySign, cmap='gray')
plt.show()
