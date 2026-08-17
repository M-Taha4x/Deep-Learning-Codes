from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import numpy as np

# Let's say these are the final outputs from your ASL test set
# 0 = 'A', 1 = 'B'
true_labels = np.array([0, 1, 0, 0, 1, 1])
predictions = np.array([0, 1, 1, 0, 1, 0]) 

acc = accuracy_score(true_labels, predictions)
# average='macro' calculates the metric for each class and averages them
prec = precision_score(true_labels, predictions, average='macro')
rec = recall_score(true_labels, predictions, average='macro')

print(f"Accuracy: {acc:.2f}")
print(f"Precision: {prec:.2f}")
print(f"Recall: {rec:.2f}")

# Generating the Confusion Matrix
cm = confusion_matrix(true_labels, predictions)
print("Confusion Matrix:\n", cm)