import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib

X, y = make_regression(n_samples=500, n_features=1, noise=15, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=37)

X_b = np.c_[np.ones((X_train.shape[0], 1)), X_train]
w_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y_train)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Analytical Intercept: {w_best[0]}")
print(f"sklearn Intercept:    {model.intercept_}")
print(f"MSE: {mse:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}")

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.scatter(X_test, y_test, alpha=0.5, label='Actual Data')
plt.plot(X_test, y_pred, color='red', label='Regression Line')
plt.title("Regression Line")
plt.legend()

plt.subplot(1, 3, 2)
residuals = y_test - y_pred
plt.scatter(y_pred, residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.title("Residual Analysis")
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")

plt.subplot(1, 3, 3)
plt.hist(residuals, bins=20, color='skyblue', edgecolor='black')
plt.title("Error Distribution (Histogram)")
plt.xlabel("Residual Value")

plt.tight_layout()
plt.show()

joblib.dump(model, 'model.joblib')
print("Model saved successfully as 'model.joblib'")
