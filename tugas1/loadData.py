import os
import numpy as np
import cv2 as cv
import pickle


# Load and preprocess images
TrainPath = r'G:\\Coding\\python\\UTS KB\\tugas1\\Train'
categories = os.listdir(TrainPath)


data = []

for category in categories:
    label = categories.index(category)
    path = os.path.join(TrainPath, category)
    for img in os.listdir(path):
        imgPath = os.path.join(path, img)
        try:
            imgArray = cv.imread(imgPath)
            imgResized = cv.resize(imgArray, (100, 100))
            image = np.array(imgResized).flatten()
            data.append([image, label])
            print("success load image from ", imgPath)
        except Exception as e:
            print("error load image from ", imgPath)
            pass


# save data
pickle_out = open("image.pickle", "wb")
pickle.dump(data, pickle_out)
pickle_out.close()
print("data saved to image.pickle")