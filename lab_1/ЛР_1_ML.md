# РОССИЙСКИЙ УНИВЕРСИТЕТ ДРУЖБЫ НАРОДОВ

**Факультет физико-математических и естественных наук**

**Кафедра теории вероятностей и кибербезопасности**

---

## **Отчет по лабораторной работе № 1**

**Дисциплина:** Компьютерные науки и технологии программирования

**Студент:** Каримов Тимур Ринатович

**Группа:** НММбд-02-24

**Преподаватель:** Бегишев В.О.

**МОСКВА 2025 г.**

---

## **Цель работы:**

**Теоретическое и практическое освоение фундаментальных основ машинного обучения на примере задачи линейной регрессии.**

---

## **Выполнение работы**

### **Задание 1:** 
**Формулировка задачи**: 1Подберите скорость обучения (eta) и количество итераций

В первую очередь мне необходимо было определить эталонные веса и MSE. Выше в теории было вычисление MSE для МНК, и я использовал функцию MSE для подбора вручную eta и кол-ва итераций. Так же еще проанализировал через график функцию ошибок.

```Python
n = X.shape[0]

  

eta = 0.099

n_iter = 10000

  

W = np.array([1, 0.5])

print(f'Number of objects = {n} \

       \nLearning rate = {eta} \

       \nInitial weights = {W} \n')

  

for i in range(n_iter):

    y_pred = np.dot(X, W)

    err = calc_mse(y, y_pred)

    # for k in range(W.shape[0]):

    #     W[k] -= eta * (1/n * 2 * X[:, k] @ (y_pred - y))

    W -= eta * (1/n * 2 * np.dot(X.T, y_pred - y))

  

    if i % 10 == 0:

        eta /= 1.1

        print(f'Iteration #{i}: W_new = {W}, MSE = {round(err, 2)}')

```

Вот результаты:

![[Pasted image 20250918224559.png]]
Заметим, что MSE для МНК (в предпоследней строке) почти совпадает с MSE для градиентного спуска. 
### **Задание 2:  
Формулировка задачи: 2: В этом коде мы избавляемся от итераций по весам, но тут есть ошибка, исправьте ее. 

Исходный код с ошибкой: 

```Python
n = X.shape[0]

  

eta = 1e-2

n_iter = 100

  

W = np.array([1, 0.5])

print(f'Number of objects = {n} \

       \nLearning rate = {eta} \

       \nInitial weights = {W} \n')

  

for i in range(n_iter):

    y_pred = np.dot(X, W)

    err = calc_mse(y, y_pred)

  

    W -= eta * (1/n * 2 * np.dot(X, y_pred - y))

  

    if i % 10 == 0:

        print(f'Iteration #{i}: W_new = {W}, MSE = {round(err,2)}')
```


Ошибка была в этой строке   `W -= eta * (1/n * 2 * np.dot(X, y_pred - y))`. Ошибка в коде заключается в неправильном использовании матричного умножения при обновлении весов. Вместо транспонирования матрицы признаков `X` используется исходная матрица, что приводит к несоответствию размерностей и некорректному вычислению градиента.

**Рабочий код** 
```Python
n = X.shape[0]

  

eta = 1e-2

n_iter = 100

  

W = np.array([1, 0.5])

print(f'Number of objects = {n} \

       \nLearning rate = {eta} \

       \nInitial weights = {W} \n')

  

for i in range(n_iter):

    y_pred = np.dot(X, W)

    err = calc_mse(y, y_pred)

  

    W -= eta * (1/n * 2 * np.dot(X.T, y_pred - y))

  

    if i % 10 == 0:

        print(f'Iteration #{i}: W_new = {W}, MSE = {round(err,2)}')
```

### **Задание 3:** 


**Формулировка задачи 3:**. Вместо того, чтобы задавать количество итераций, задайте другое условие остановки алгоритма - когда веса перестают изменяться меньше определенного порога $\epsilon$ 

Реализация: 
```Python
n = X.shape[0]

eta = 1e-2

epsilon = 1e-6

max_iter = 10000

  

W = np.array([1.0, 0.5])

print(f'Number of objects = {n} \nLearning rate = {eta} \nInitial weights = {W} \n')

  

for i in range(max_iter):

    W_old = W.copy()  

    y_pred = np.dot(X, W)

    gradient = (2/n) * np.dot(X.T, y_pred - y)

    W -= eta * gradient

  

    weight_change = np.linalg.norm(W - W_old)

    if weight_change < epsilon:

        print(f'Iteration #{i}: W = {W}, изменение весов {weight_change:.6f} < {epsilon}')

        break

  

    if i % 10 == 0:

        err = calc_mse(y, y_pred)

        print(f'Iteration #{i}: W = {W}, MSE = {err:.2f}, изменение весов = {weight_change:.6f}')

else:

    print(f'Достигнуто максимальное количество итераций {max_iter}')
  ```

Я добавил переменную *epsilon*, затем в функцию градиента и сохранял значения весов, чтобы у меня отслеживалось отклонение при каждой итерации. При достижении отклонения меньше epsilon цикл обрывался. 

![[Pasted image 20250918233201.png]] 
