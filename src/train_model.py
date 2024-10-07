import numpy as np
import pickle

# Load preprocessed data
def load_data(file_path):
    try:
        data = np.loadtxt(file_path, delimiter=',', skiprows=1)
        return data
    except ValueError as e:
        print(f"Error loading data from {file_path}: {e}")
        return None

X = load_data('../data/X_preprocessed.csv')
y = load_data('../data/y_preprocessed.csv')

# Ensure data is loaded correctly
if X is None or y is None:
    raise RuntimeError("Failed to load data. Exiting.")

# Split data into training and testing sets
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Function to train Linear Regression using the Normal Equation
def train_linear_regression(X, y):
    # Adding bias term
    X_b = np.c_[np.ones(X.shape[0]), X]  
    # Normal Equation: theta = (X_b^T * X_b)^(-1) * X_b^T * y
    theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
    return theta

# Model 1: Simple Linear Regression
theta1 = train_linear_regression(X_train[:, 1:2], y_train)
pickle.dump(theta1, open('../models/regression_model1.pkl', 'wb'))

# Model 2: Multiple Linear Regression
theta2 = train_linear_regression(X_train, y_train)
pickle.dump(theta2, open('../models/regression_model2.pkl', 'wb'))

# Model 3: Polynomial Regression (using quadratic features)
X_train_poly = np.c_[X_train, X_train[:, 1]**2, X_train[:, 2]**2]
theta3 = train_linear_regression(X_train_poly, y_train)
pickle.dump(theta3, open('../models/regression_model3.pkl', 'wb'))
