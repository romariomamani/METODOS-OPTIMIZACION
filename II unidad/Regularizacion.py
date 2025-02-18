import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso

x = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)  # Convertimos x a matriz columna
y = np.array([2, 2.8, 3.6, 4.5, 5.1])

model_lin = LinearRegression()
model_lin.fit(x, y)
theta0_lin = model_lin.intercept_
theta1_lin = model_lin.coef_[0]

model_ridge = Ridge(alpha=1)  # λ = 1
model_ridge.fit(x, y)
theta0_ridge = model_ridge.intercept_
theta1_ridge = model_ridge.coef_[0]

model_lasso = Lasso(alpha=1)  # λ = 1
model_lasso.fit(x, y)
theta0_lasso = model_lasso.intercept_
theta1_lasso = model_lasso.coef_[0]

print(f"Regresión Lineal:   θ0 = {theta0_lin:.3f}, θ1 = {theta1_lin:.3f}")
print(f"Ridge (L2):         θ0 = {theta0_ridge:.3f}, θ1 = {theta1_ridge:.3f}")
print(f"Lasso (L1):         θ0 = {theta0_lasso:.3f}, θ1 = {theta1_lasso:.3f}")

x_range = np.linspace(0, 6, 100).reshape(-1, 1)
y_pred_lin = model_lin.predict(x_range)
y_pred_ridge = model_ridge.predict(x_range)
y_pred_lasso = model_lasso.predict(x_range)

plt.scatter(x, y, color="black", label="Datos reales")
plt.plot(x_range, y_pred_lin, label="Regresión Lineal", linestyle="dashed")
plt.plot(x_range, y_pred_ridge, label="Ridge (L2)", linestyle="dotted")
plt.plot(x_range, y_pred_lasso, label="Lasso (L1)", linestyle="dashdot")
plt.legend()
plt.xlabel("x")
plt.ylabel("y")
plt.title("Comparación de Métodos de Regularización")
plt.show()

