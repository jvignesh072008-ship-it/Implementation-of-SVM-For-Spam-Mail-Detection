# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Data Loading and Preprocessing: Import the spam dataset, select relevant columns (label and message), and convert categorical labels ("ham" as 0 and "spam" as 1) into numerical values for model training.

2.Text Vectorization: Convert the raw text messages into numerical feature vectors using a CountVectorizer that removes English stop words, enabling the model to interpret text input as numeric data.

3.Data Splitting: Divide the dataset into training and testing subsets (80% training, 20% testing) to allow unbiased evaluation of model performance on unseen data.

4.Model Training: Initialize and train a Support Vector Machine (SVM) with a linear kernel on the training data to learn a decision boundary that separates spam from ham messages.

5.Prediction: Use the trained SVM model to predict labels for both training and testing datasets to assess both fitted and generalization performance.

6.Visualization: Plot the distribution of predicted labels for training and testing sets to observe class balance, and plot a confusion matrix for the test set to visualize true positives, false positives, true negatives, and false negatives.

7.Performance Evaluation: Calculate and print key metrics—accuracy, precision, recall, and F1-score—for both training and testing predictions to quantify model effectiveness in spam detection.


## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: VIGNESH J
RegisterNumber: 25014705
*/

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv('spam.csv', encoding='latin1')
df = df[['v1', 'v2']]
df.columns = ['label', 'message']
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Text features
vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['message'])
y = df['label']

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train SVM
svm = SVC(kernel='linear')
svm.fit(X_train, y_train)

# Predict
train_pred = svm.predict(X_train)
test_pred = svm.predict(X_test)

# Plot training data
plt.figure(figsize=(6,4))
sns.histplot(train_pred, kde=True, color='blue')
plt.title('Training Set Predictions Distribution:Ham vs Spam')
plt.xlabel('Message Length')
plt.ylabel('Count')
plt.show()

# Plot testing data
plt.figure(figsize=(6,4))
sns.histplot(test_pred, kde=True, color='orange')
plt.title('Testing Set Predictions Distribution:Ham vs Spam')
plt.xlabel('Message Length')
plt.ylabel('Count')
plt.show()

# Confusion matrix for test data
cm = confusion_matrix(y_test, test_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
plt.title('Confusion Matrix for Test Data')
plt.show()
from sklearn.metrics import precision_score, recall_score, f1_score

# Metrics calculation for training set predictions
train_accuracy = accuracy_score(y_train, train_pred)
train_precision = precision_score(y_train, train_pred)
train_recall = recall_score(y_train, train_pred)
train_f1 = f1_score(y_train, train_pred)

# Metrics calculation for testing set predictions
test_accuracy = accuracy_score(y_test, test_pred)
test_precision = precision_score(y_test, test_pred)
test_recall = recall_score(y_test, test_pred)
test_f1 = f1_score(y_test, test_pred)

print("Training Metrics:")
print(f"Accuracy: {train_accuracy:.4f}")
print(f"Precision: {train_precision:.4f}")
print(f"Recall: {train_recall:.4f}")
print(f"F1 Score: {train_f1:.4f}")

print("\nTesting Metrics:")
print(f"Accuracy: {test_accuracy:.4f}")
print(f"Precision: {test_precision:.4f}")
print(f"Recall: {test_recall:.4f}")
print(f"F1 Score: {test_f1:.4f}")

# Message length distribution plot
df['message_length'] = df['message'].apply(len)
plt.figure(figsize=(8,5))
sns.histplot(data=df, x='message_length', hue='label', bins=50, kde=True, palette=['green', 'red'])
plt.title('Message Length Distribution: Ham vs Spam')
plt.xlabel('Message Length')
plt.ylabel('Count')
plt.show()

```

## Output:
![SVM For Spam Mail Detection](sam.png)
<img width="1059" height="483" alt="Screenshot 2025-10-06 205637" src="https://github.com/user-attachments/assets/bbc46aa5-50cd-4d0d-bd2f-f6847475e946" />
<img width="1007" height="488" alt="Screenshot 2025-10-06 205649" src="https://github.com/user-attachments/assets/7395f763-9dc2-4fb0-8fd8-1d0b362ba737" />
<img width="1110" height="706" alt="Screenshot 2025-10-06 205659" src="https://github.com/user-attachments/assets/94dce306-05b7-4b4f-b6f5-adfeb0cdd231" />
<img width="1191" height="585" alt="Screenshot 2025-10-06 205709" src="https://github.com/user-attachments/assets/e137759f-c2f6-4e60-8f7b-1482158b8324" />






## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
