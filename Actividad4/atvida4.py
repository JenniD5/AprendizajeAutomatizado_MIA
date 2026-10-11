import numpy as np
# Datos OR
X = np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1]
])
y = np.array([0,1,1,1])
# Pesos iniciales
w = np.zeros(2)
b = 0
learning_rate = 0.1
# Entrenamiento
for epoch in range(20):
    for i in range(len(X)):
        z = np.dot(X[i], w) + b
        y_pred = 1 if z >= 0 else 0
        error = y[i] - y_pred
        w += learning_rate * error * X[i]
        b += learning_rate * error
print("Pesos:", w)
print("Bias:", b)
# Prueba
for x in X:
    z = np.dot(x, w) + b
    pred = 1 if z >= 0 else 0
    print(f"{x} -> {pred}")