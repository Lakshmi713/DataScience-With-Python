import numpy as np
from sklearn.linear_model import LinearRegression

# 4 Input Features:
# [Study Hours, Attendance, Previous Marks, Sleep Hours]

X = np.array([
    [2, 75, 60, 6],
    [3, 80, 65, 7],
    [4, 85, 70, 6],
    [5, 90, 75, 8],
    [6, 95, 80, 7]
])

# Output: Final Exam Marks
Y = np.array([55, 60, 68, 75, 85])

# Create model
model = LinearRegression()

# Train model
model.fit(X, Y)

# Print coefficients (slopes for each factor)
print("Coefficients:", model.coef_)

# Print intercept
print("Intercept:", model.intercept_)

# Predict for new student:
# 7 hours study, 92% attendance, 78 previous marks, 7 hours sleep
prediction = model.predict([[7, 92, 78, 7]])

print("Predicted Final Marks:", prediction[0])






