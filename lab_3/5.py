import numpy as np

def accuracy_score(y, y_pred):
    correct = np.sum(y == y_pred)
    total = len(y)
    return correct / total

def confusion_matrix(y, y_pred):
    tp = np.sum((y == 1) & (y_pred == 1))
    tn = np.sum((y == 0) & (y_pred == 0))
    fp = np.sum((y == 0) & (y_pred == 1))
    fn = np.sum((y == 1) & (y_pred == 0))
    return np.array([[tn, fp], [fn, tp]])

def precision_score(y, y_pred):
    cm = confusion_matrix(y, y_pred)
    tp = cm[1, 1]
    fp = cm[0, 1]
    if tp + fp == 0:
        return 0.0  
    return tp / (tp + fp)

def recall_score(y, y_pred):
    cm = confusion_matrix(y, y_pred)
    tp = cm[1, 1]
    fn = cm[1, 0]
    if tp + fn == 0:
        return 0.0
    return tp / (tp + fn)

def f1_score(y, y_pred):
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)

if __name__ == "__main__":
    y = np.array([1, 0, 1, 1, 0, 1, 0])
    y_pred = np.array([1, 0, 0, 1, 0, 1, 1])
    print("Accuracy:", accuracy_score(y, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y, y_pred))
    print("Precision:", precision_score(y, y_pred))
    print("Recall:", recall_score(y, y_pred))
    print("F1-score:", f1_score(y, y_pred))