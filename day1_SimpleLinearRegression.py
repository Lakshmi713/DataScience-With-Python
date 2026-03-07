import numpy as np
from sklearn.linear_model import LinearRegression

# Independent variable (Study Hours)
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)

# Dependent variable (Marks)
Y = np.array([42, 48, 65, 70, 78])

# Create model
model = LinearRegression()

# Train model
model.fit(X, Y)

# Get slope and intercept
print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)

# Predict marks for 6 hours study
prediction = model.predict([[6]])
print("Predicted Marks for 6 hours:", prediction[0])
