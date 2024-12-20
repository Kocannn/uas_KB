import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Modul 1: Preprocessing Data
def preprocess_images(image_folder, img_size=(128, 128)):

    data = []
    labels = []
    classes = os.listdir(image_folder)
    for label, class_name in enumerate(classes): 
        class_folder = os.path.join(image_folder, class_name)
        for img_file in os.listdir(class_folder):
            img_path = os.path.join(class_folder, img_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, img_size)
            data.append(img.flatten())
            labels.append(label)
    return np.array(data), np.array(labels), classes

# Modul 2: Membagi Data
def split_data(features, labels, test_size=0.3, random_state=42):
    return train_test_split(features, labels, test_size=test_size, random_state=random_state)

# Modul 3: Training Model
def train_svm(X_train, y_train, kernel='rbf', C=1.0):
    svm_model = SVC(kernel=kernel, C=C, random_state=42)
    svm_model.fit(X_train, y_train)
    return svm_model

def train_knn(X_train, y_train, n_neighbors=5):
    knn_model = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn_model.fit(X_train, y_train)
    return knn_model

def train_random_forest(X_train, y_train, n_estimators=100):
    rf_model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    rf_model.fit(X_train, y_train)
    return rf_model

# Modul 4: Evaluasi Model
def evaluate_model(model, X_test, y_test, class_names):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")
    print("\nClassification Report:")

    # Dapatkan label unik dari y_test dan y_pred
    unique_labels = np.unique(np.concatenate((y_test, y_pred)))

    # Filter class_names untuk mencocokkan label unik
    filtered_class_names = [class_names[i] for i in unique_labels]

    print(classification_report(y_test, y_pred, target_names=filtered_class_names)) # Use filtered_class_names

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=filtered_class_names, yticklabels=filtered_class_names) #Use filtered_class_names for confusion matrix labels
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.show()
    return accuracy

# Modul 5: Visualisasi Akurasi
def plot_accuracies(accuracies, model_names):
    plt.bar(model_names, accuracies, color=['blue', 'orange', 'green'])
    plt.xlabel('Models')
    plt.ylabel('Accuracy')
    plt.title('Model Comparison')
    plt.ylim(0, 1)
    for i, acc in enumerate(accuracies):
        plt.text(i, acc + 0.01, f"{acc:.2f}", ha='center')
    plt.show()

# Main Program
if __name__ == "__main__":

    image_folder = r"G:\Coding\python\UTS KB\tugas1\Train"  # Pastikan folder "ttd" ada di lokasi yang benar

    # Step 1: Preprocess data
    print("Preprocessing images...")
    features, labels, class_names = preprocess_images(image_folder)

    # Step 2: Split data
    print("Splitting data into training and validation sets...")
    X_train, X_test, y_train, y_test = split_data(features, labels)

    # Step 3: Train models
    print("Training models...")
    svm_model = train_svm(X_train, y_train)
    knn_model = train_knn(X_train, y_train)
    rf_model = train_random_forest(X_train, y_train)

    # Step 4: Evaluate models
    # print("Evaluating SVM model...")
    # svm_accuracy = evaluate_model(svm_model, X_test, y_test, class_names)

    # print("Evaluating K-NN model...")
    # knn_accuracy = evaluate_model(knn_model, X_test, y_test, class_names)

    print("Evaluating Random Forest model...")
    rf_accuracy = evaluate_model(rf_model, X_test, y_test, class_names)

    # Step 5: Visualize accuracies
    accuracies = [rf_accuracy]
    model_names = ['SVM', 'K-NN', 'Random Forest']
    plot_accuracies(accuracies, model_names)