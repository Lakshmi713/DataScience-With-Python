import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
Y = np.array([42, 48, 65, 70, 78])

model = LinearRegression()
model.fit(X, Y)

# Predictions
Y_pred = model.predict(X)

# Plot actual points
plt.scatter(X, Y)

# Plot regression line
plt.plot(X, Y_pred)

plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.title("Regression Line")
plt.show()
