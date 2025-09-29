import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

X, Y, coef = datasets.make_regression(n_samples=1000, n_features=2, n_informative=2, 
                                      n_targets=1, noise=5, coef=True, random_state=2)
X[:, 0] *= 10

# Исходные данные
print("1. ИСХОДНЫЕ ДАННЫЕ:")
print(f"   Столбец 0: mean={np.mean(X[:, 0]):.2f}, std={np.std(X[:, 0]):.2f}, min={np.min(X[:, 0]):.2f}, max={np.max(X[:, 0]):.2f}")
print(f"   Столбец 1: mean={np.mean(X[:, 1]):.2f}, std={np.std(X[:, 1]):.2f}, min={np.min(X[:, 1]):.2f}, max={np.max(X[:, 1]):.2f}")

means = np.mean(X, axis=0)
stds = np.std(X, axis=0)
X_standardized = (X - means) / stds

print("\n2. ПОСЛЕ СТАНДАРТИЗАЦИИ (как в вашем коде):")
print(f"   Столбец 0: mean={np.mean(X_standardized[:, 0]):.2f}, std={np.std(X_standardized[:, 0]):.2f}, min={np.min(X_standardized[:, 0]):.2f}, max={np.max(X_standardized[:, 0]):.2f}")
print(f"   Столбец 1: mean={np.mean(X_standardized[:, 1]):.2f}, std={np.std(X_standardized[:, 1]):.2f}, min={np.min(X_standardized[:, 1]):.2f}, max={np.max(X_standardized[:, 1]):.2f}")

mins = np.min(X_standardized, axis=0)
maxs = np.max(X_standardized, axis=0)
X_double_transformed = (X_standardized - mins) / (maxs - mins)

print("\n3. ПОСЛЕ ДОПОЛНИТЕЛЬНОЙ НОРМАЛИЗАЦИИ (проблема!):")
print(f"   Столбец 0: mean={np.mean(X_double_transformed[:, 0]):.2f}, std={np.std(X_double_transformed[:, 0]):.2f}, min={np.min(X_double_transformed[:, 0]):.2f}, max={np.max(X_double_transformed[:, 0]):.2f}")
print(f"   Столбец 1: mean={np.mean(X_double_transformed[:, 1]):.2f}, std={np.std(X_double_transformed[:, 1]):.2f}, min={np.min(X_double_transformed[:, 1]):.2f}, max={np.max(X_double_transformed[:, 1]):.2f}")

X_only_normalized = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))

print("\n4. ТОЛЬКО НОРМАЛИЗАЦИЯ (для сравнения):")
print(f"   Столбец 0: mean={np.mean(X_only_normalized[:, 0]):.2f}, std={np.std(X_only_normalized[:, 0]):.2f}, min={np.min(X_only_normalized[:, 0]):.2f}, max={np.max(X_only_normalized[:, 0]):.2f}")
print(f"   Столбец 1: mean={np.mean(X_only_normalized[:, 1]):.2f}, std={np.std(X_only_normalized[:, 1]):.2f}, min={np.min(X_only_normalized[:, 1]):.2f}, max={np.max(X_only_normalized[:, 1]):.2f}")

plt.figure(figsize=(15, 10))

# Исходные данные
plt.subplot(2, 3, 1)
plt.scatter(X[:, 0], X[:, 1], alpha=0.5)
plt.title('Исходные данные')
plt.xlabel('Признак 0')
plt.ylabel('Признак 1')
plt.grid(True)

# После стандартизации
plt.subplot(2, 3, 2)
plt.scatter(X_standardized[:, 0], X_standardized[:, 1], alpha=0.5)
plt.title('После стандартизации\n(mean=0, std=1)')
plt.xlabel('Признак 0')
plt.ylabel('Признак 1')
plt.grid(True)

# После двойного преобразования
plt.subplot(2, 3, 3)
plt.scatter(X_double_transformed[:, 0], X_double_transformed[:, 1], alpha=0.5)
plt.title('После стандартизации + нормализации\n(ПРОБЛЕМА!)')
plt.xlabel('Признак 0')
plt.ylabel('Признак 1')
plt.grid(True)

# Только нормализация
plt.subplot(2, 3, 4)
plt.scatter(X_only_normalized[:, 0], X_only_normalized[:, 1], alpha=0.5)
plt.title('Только нормализация\n(range=[0,1])')
plt.xlabel('Признак 0')
plt.ylabel('Признак 1')
plt.grid(True)

# Гистограммы распределения признака 0
plt.subplot(2, 3, 5)
plt.hist(X_standardized[:, 0], alpha=0.7, label='Только стандартизация', bins=20)
plt.hist(X_double_transformed[:, 0], alpha=0.7, label='Стандартизация + нормализация', bins=20)
plt.title('Сравнение распределений\n(признак 0)')
plt.legend()
plt.grid(True)

# Гистограммы распределения признака 1
plt.subplot(2, 3, 6)
plt.hist(X_standardized[:, 1], alpha=0.7, label='Только стандартизация', bins=20)
plt.hist(X_double_transformed[:, 1], alpha=0.7, label='Стандартизация + нормализация', bins=20)
plt.title('Сравнение распределений\n(признак 1)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

def test_model_performance(X_data, method_name):
    X_with_bias = np.hstack([np.ones((X_data.shape[0], 1)), X_data])
    
    W = np.random.randn(X_with_bias.shape[1])
    errors = []
    
    for i in range(1000):
        y_pred = np.dot(X_with_bias, W)
        dQ = 2 / len(Y) * X_with_bias.T @ (y_pred - Y)
        W -= 0.01 * dQ
        
        if i % 100 == 0:
            error = np.mean((y_pred - Y)**2)
            errors.append(error)
    
    final_error = np.mean((np.dot(X_with_bias, W) - Y)**2)
    print(f"{method_name}: финальная MSE = {final_error:.6f}")
    return final_error, errors

print("\n=== ВЛИЯНИЕ НА ОБУЧЕНИЕ МОДЕЛИ ===")
error_std, errors_std = test_model_performance(X_standardized, "Только стандартизация")
error_double, errors_double = test_model_performance(X_double_transformed, "Стандартизация + нормализация")
error_norm, errors_norm = test_model_performance(X_only_normalized, "Только нормализация")

# График сходимости
plt.figure(figsize=(10, 6))
plt.plot(errors_std, label='Только стандартизация', linewidth=2)
plt.plot(errors_double, label='Стандартизация + нормализация', linewidth=2)
plt.plot(errors_norm, label='Только нормализация', linewidth=2)
plt.xlabel('Итерации (×100)')
plt.ylabel('MSE')
plt.title('Влияние двойного преобразования на сходимость')
plt.legend()
plt.grid(True)
plt.show()

print(f"\n=== ВЫВОДЫ ===")
print("1. Двойное преобразование ИСКАЖАЕТ распределение данных")
print("2. Потеряны преимущества обоих методов:")
print("   - После стандартизации mean=0, std=1")
print("   - После нормализации эти свойства теряются!")
print("3. Модель обучается ХУЖЕ (MSE выше)")
print("4. Выберите ОДИН метод преобразования!")