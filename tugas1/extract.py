import pickle

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

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


# Split data into training and testing sets
xtrain, xtest, ytrain, ytest = train_test_split(features, labels, test_size=0.1)


# Train the model
model = SVC(C=1, kernel='poly', gamma='auto')
model.fit(xtrain, ytrain)

print("Model training completed")
