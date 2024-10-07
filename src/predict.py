import numpy as np
import pickle

# Function to predict using model
def predict(X, theta):
    X = np.c_[np.ones(X.shape[0]), X]  # Adding bias term
    return X.dot(theta)

# Function to calculate metrics
def calculate_metrics(y_true, y_pred):
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_res = np.sum((y_true - y_pred) ** 2)
    r2_score = 1 - (ss_res / ss_tot)
    return mse, rmse, r2_score

# Load model and data
model_path = '../models/regression_model_final.pkl'
data_path = '../data/X_preprocessed.csv'
metrics_output_path = '../results/train_metrics.txt'
predictions_output_path = '../results/train_predictions.csv'

theta_final = pickle.load(open(model_path, 'rb'))
X = np.loadtxt(data_path, delimiter=',', skiprows=1)

# Predict and calculate metrics
y_pred = predict(X, theta_final)
np.savetxt(predictions_output_path, y_pred, delimiter=',')

y_true = np.loadtxt('../data/y_preprocessed.csv', delimiter=',', skiprows=1)
mse, rmse, r2_score = calculate_metrics(y_true, y_pred)

# Save metrics
with open(metrics_output_path, 'w') as f:
    f.write('Regression Metrics:\n')
    f.write(f'Mean Squared Error (MSE): {mse:.2f}\n')
    f.write(f'Root Mean Squared Error (RMSE): {rmse:.2f}\n')
    f.write(f'R-squared (R²) Score: {r2_score:.2f}\n')
