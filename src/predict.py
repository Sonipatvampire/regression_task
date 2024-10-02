import pandas as pd
import numpy as np
import pickle

class LinearRegression:
    def __init__(self):
        self.coefficients = None

    def fit(self, X, y):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]  # add bias term
        self.coefficients = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y

    def predict(self, X):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]  # add bias term
        return X_b @ self.coefficients

def load_model(file_path):
    with open(file_path, 'rb') as f:
        model = pickle.load(f)
    return model

def evaluate_model(model, X, y):
    predictions = model.predict(X)
    mse = np.mean((y - predictions) ** 2)
    rmse = np.sqrt(mse)
    r2 = 1 - (np.sum((y - predictions) ** 2) / np.sum((y - np.mean(y)) ** 2))
    
    return mse, rmse, r2

def save_metrics(metrics, file_path):
    with open(file_path, 'w') as f:
        f.write("Regression Metrics:\n")
        f.write(f"Mean Squared Error (MSE): {metrics[0]:.2f}\n")
        f.write(f"Root Mean Squared Error (RMSE): {metrics[1]:.2f}\n")
        f.write(f"R-squared (R²) Score: {metrics[2]:.2f}\n")

if __name__ == "__main__":
    import argparse
    from data_preprocessing import load_data, preprocess_data

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--metrics_output_path", required=True)
    parser.add_argument("--predictions_output_path", required=True)
    
    args = parser.parse_args()
    
    # Load data and model
    data = pd.read_csv(args.data_path)
    X, y = preprocess_data(data)  # Ensure preprocess_data is accessible
    model = load_model(args.model_path)
    
    # Evaluate
    metrics = evaluate_model(model, X, y)
    save_metrics(metrics, args.metrics_output_path)
    
    # Save predictions
    predictions = model.predict(X)
    pd.DataFrame(predictions).to_csv(args.predictions_output_path, header=False, index=False)
