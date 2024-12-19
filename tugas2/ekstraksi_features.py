import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score

# Load the dataset
file_path = r'G:\Coding\python\UTS KB\tugas2\dataset_features.csv'
data = pd.read_csv(file_path)

# Separate features and target variable
X = data.drop(columns=['Class'])
y = data['Class']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=20, )

# Initialize the KNN classifier
knn = KNeighborsClassifier(n_neighbors=15)

# Train the KNN model
knn.fit(X_train, y_train)

# Make predictions with KNN
y_pred_knn = knn.predict(X_test)

# Evaluate the KNN model
accuracy_knn = accuracy_score(y_test, y_pred_knn)
report_knn = classification_report(y_test, y_pred_knn)
conf_matrix_knn = confusion_matrix(y_test, y_pred_knn)

print(f'KNN Accuracy: {accuracy_knn}')
print('KNN Classification Report:')
print(report_knn)

# Initialize the SVM classifier
svm = SVC(kernel='sigmoid', random_state=44)

# Train the SVM model
svm.fit(X_train, y_train)

# Make predictions with SVM
y_pred_svm = svm.predict(X_test)

# Evaluate the SVM model
accuracy_svm = accuracy_score(y_test, y_pred_svm)
report_svm = classification_report(y_test, y_pred_svm)
conf_matrix_svm = confusion_matrix(y_test, y_pred_svm)

print(f'SVM Accuracy: {accuracy_svm}')
print('SVM Classification Report:')
print(report_svm)

# Perform cross-validation for KNN
cv_scores_knn = cross_val_score(knn, X, y, cv=5)
print(f'KNN Cross-Validation Scores: {cv_scores_knn}')
print(f'KNN Mean Cross-Validation Score: {cv_scores_knn.mean()}')

# Perform cross-validation for SVM
cv_scores_svm = cross_val_score(svm, X, y, cv=5)
print(f'SVM Cross-Validation Scores: {cv_scores_svm}')
print(f'SVM Mean Cross-Validation Score: {cv_scores_svm.mean()}')

# Plot confusion matrix for KNN
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix_knn, annot=True, fmt='d', cmap='Blues')
plt.title('KNN Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Plot confusion matrix for SVM
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix_svm, annot=True, fmt='d', cmap='Blues')
plt.title('SVM Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Plot accuracy comparison
plt.figure(figsize=(10, 7))
plt.bar(['KNN', 'SVM'], [accuracy_knn, accuracy_svm])
plt.ylim(0, 1)
plt.title('Model Accuracy Comparison')
plt.show()