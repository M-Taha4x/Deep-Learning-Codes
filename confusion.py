import numpy as np

def confusion_matrix_manual(y_actual, y_pred):
    TP = np.sum((y_actual == 1) & (y_pred == 1))
    TN = np.sum((y_actual == 0) & (y_pred == 0))
    FP = np.sum((y_actual == 0) & (y_pred == 1))
    FN = np.sum((y_actual == 1) & (y_pred == 0))
    return TP, TN, FP, FN

def evaluate(y_actual, y_pred):
    TP, TN, FP, FN = confusion_matrix_manual(y_actual, y_pred)
    
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN,
        'accuracy': accuracy, 'precision': precision,
        'recall': recall, 'f1': f1
    }

# Test: simulating our rare-disease example
# 1000 patients, 1 actually has the disease, model predicts "no disease" for everyone
y_actual = np.array([0]*999 + [1])       # 999 healthy, 1 sick
y_pred   = np.array([0]*1000)             # model always predicts "healthy"

results = evaluate(y_actual, y_pred)
print(results)