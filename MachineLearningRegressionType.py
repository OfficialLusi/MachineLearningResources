import numpy as np

# Synthetic data generation
np.random.seed(0)
X = np.sort(5 * np.random.rand(80, 1), axis=0)
y = np.sin(X).ravel() + np.random.normal(0, 0.2, X.shape[0])

# Models
# 1. Linear regression
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(X, y)
y_pred_lin = lin_reg.predict(X)

# 2. Polynomial regression (4th grade)
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=4)
X_poly = poly.fit_transform(X)
poly_reg = LinearRegression()
poly_reg.fit(X_poly, y)
y_pred_poly = poly_reg.predict(X_poly)

# 3. Decision tree regression
from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor(max_depth=4)
tree_reg.fit(X, y)
y_pred_tree = tree_reg.predict(X)

# Plotting the results
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))

plt.scatter(X, y, color='gray', label='Dati reali', alpha=0.5)

# Linear
plt.plot(X, y_pred_lin, label='Lineare', linewidth=2)

# Polynomial
plt.plot(X, y_pred_poly, label='Polinomiale (grado 4)', linewidth=2)

# Decision Tree
plt.plot(X, y_pred_tree, label='Albero decisionale', linewidth=2)

plt.title('Confronto modelli di regressione')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
