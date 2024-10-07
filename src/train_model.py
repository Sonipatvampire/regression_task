import numpy as np
import pandas as pd

def linear_regression(X, y, learning_rate, epochs):
    """Train a linear regression model using gradient descent."""
    m, n = X.shape  # m: number of samples, n: number of features
    weights = np.zeros(n)  # Initialize weights
    bias = 0  # Initialize bias

    for epoch in range(epochs):
        # Make predictions
        y_pred = X.dot(weights) + bias
        
        # Compute the gradients
        error = y_pred - y
        weight_gradient = (1 / m) * (X.T.dot(error))
        bias_gradient = (1 / m) * np.sum(error)

        # Update weights and bias
        weights -= learning_rate * weight_gradient
        bias -= learning_rate * bias_gradient

        # Optional: Print loss for every 100 epochs
        if epoch % 100 == 0:
            loss = (1 / (2 * m)) * np.sum(error ** 2)
            print(f"Epoch {epoch}, Loss: {loss}")

    return weights, bias

def train_model(df, feature_names, label_name, learning_rate, epochs):
    """Train the model by feeding it data."""
    features = df[feature_names].values
    label = df[label_name].values

    # Train the linear regression model
    trained_weights, trained_bias = linear_regression(features, label, learning_rate, epochs)

    return trained_weights, trained_bias

def run_experiment(df, feature_names, label_name, learning_rate, epochs):
    """Run the training experiment."""
    print(f'INFO: starting training experiment with features={feature_names} and label={label_name}\n')

    trained_weights, trained_bias = train_model(df, feature_names, label_name, learning_rate, epochs)

    print('\nSUCCESS: training experiment complete\n')
    print(f'Trained weights: {trained_weights}\nTrained bias: {trained_bias}')

# Example usage
if __name__ == "__main__":
    # Load your data
    data_file_path = '/Users/kunalsingh/Documents/Kunal_Singh_A1/regression_task/data/training_data.csv'
    df = pd.read_csv(data_file_path)

    # Define feature names and label name
    feature_names = ['Year', 'ENGINE SIZE', 'CYLINDERS']
    label_name = 'FUEL CONSUMPTION'

    # Set parameters
    learning_rate = 0.01
    epochs = 1000

    # Run the experiment
    run_experiment(df, feature_names, label_name, learning_rate, epochs)
