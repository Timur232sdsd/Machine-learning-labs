import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from collections import defaultdict

def tree_vote(forest, data):
    predictions = []
    for tree in forest:
        predictions.append(tree.predict(data))
    predictions_per_object = list(zip(*predictions))
    voted_predictions = []
    for obj in predictions_per_object:
        voted_predictions.append(max(set(obj), key=obj.count))
    return voted_predictions

class RandomForestWithOOB:
    def __init__(self, n_estimators=100, random_state=None):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.estimators_ = []
        self.estimators_samples_ = []
        
    def fit(self, X, y):
        np.random.seed(self.random_state)
        n_samples = X.shape[0]
        
        self.estimators_ = []
        self.estimators_samples_ = []
        
        for i in range(self.n_estimators):
            bootstrap_indices = np.random.choice(n_samples, n_samples, replace=True)
            from sklearn.tree import DecisionTreeClassifier
            tree = DecisionTreeClassifier(random_state=self.random_state + i if self.random_state else None)
            tree.fit(X[bootstrap_indices], y[bootstrap_indices])
            
            self.estimators_.append(tree)
            self.estimators_samples_.append(bootstrap_indices)
    
    def oob_score(self, X, y):
        n_samples = X.shape[0]
        oob_predictions = defaultdict(list)
        
        for i, tree in enumerate(self.estimators_):
            bootstrap_indices = self.estimators_samples_[i]
            oob_indices = np.setdiff1d(np.arange(n_samples), bootstrap_indices)
            
            if len(oob_indices) > 0:
                predictions = tree.predict(X[oob_indices])
                for idx, pred in zip(oob_indices, predictions):
                    oob_predictions[idx].append(pred)
        
        correct_predictions = 0
        valid_samples = 0
        
        for idx in range(n_samples):
            if idx in oob_predictions and len(oob_predictions[idx]) > 0:
                votes = oob_predictions[idx]
                final_pred = max(set(votes), key=votes.count)
                if final_pred == y[idx]:
                    correct_predictions += 1
                valid_samples += 1
        
        return correct_predictions / valid_samples if valid_samples > 0 else 0.0

def plot_decision_boundary(forest, X, y, title, ax, oob_score=None):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = tree_vote(forest, grid_points)
    Z = np.array(Z).reshape(xx.shape)
    
    ax.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdYlBu)
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, edgecolors='k', s=20)
    ax.set_title(title)
    ax.set_xlabel('Признак 1')
    ax.set_ylabel('Признак 2')
    
    if oob_score is not None:
        ax.text(0.05, 0.95, f'OOB точность: {oob_score:.3f}', 
                transform=ax.transAxes, bbox=dict(boxstyle="round", facecolor='white'))

# 1. Создаем датасет и визуализируем
X, y = make_classification(n_samples=1000, n_features=2, n_redundant=0, 
                          n_informative=2, random_state=42)

fig, axes = plt.subplots(2, 2, figsize=(15, 12))
axes = axes.ravel()

for i, n_trees in enumerate([1, 3, 10, 50]):
    rf_oob = RandomForestWithOOB(n_estimators=n_trees, random_state=42)
    rf_oob.fit(X, y)
    oob_accuracy = rf_oob.oob_score(X, y)
    
    plot_decision_boundary(rf_oob.estimators_, X, y, 
                          f'Случайный лес ({n_trees} дерево(ьев))', 
                          axes[i], oob_accuracy)

plt.tight_layout()
plt.show()

# 2. Анализ OOB точности
print("OOB точность для разного количества деревьев:")
for n_trees in [1, 3, 10, 50, 100]:
    rf_oob = RandomForestWithOOB(n_estimators=n_trees, random_state=42)
    rf_oob.fit(X, y)
    oob_accuracy = rf_oob.oob_score(X, y)
    print(f"Деревьев: {n_trees:3d} | OOB точность: {oob_accuracy:.4f}")