import pandas as pd
import numpy as np

def format_currency(x):
    """Format numerical value as currency."""
    return "{:.2f}".format(x)

def build_batch(df, batch_size):
    """Build a random batch of data from the DataFrame."""
    batch = df.sample(n=batch_size).copy()
    batch.set_index(np.arange(batch_size), inplace=True)
    return batch

def predict_fuel_consumption(model, df, features, label, batch_size=50):
    """Make predictions on a batch of data."""
    batch = build_batch(df, batch_size)

    # Assuming model is a tuple of (trained_weights, trained_bias)
    trained_weights, trained_bias = model

    # Define a prediction function
    def predict(features, weights, bias):
        return features.dot(weights) + bias

    # Make predictions using the prediction function
    predicted_values = predict(batch.loc[:, features].values, trained_weights, trained_bias)

    data = {"PREDICTED_CONSUMPTION": [], "OBSERVED_CONSUMPTION": [], "L1_LOSS": [],
            features[0]: [], features[1]: [], features[2]: []}
    
    for i in range(batch_size):
        predicted = predicted_values[i]
        observed = batch.at[i, label]
        data["PREDICTED_CONSUMPTION"].append(format_currency(predicted))
        data["OBSERVED_CONSUMPTION"].append(format_currency(observed))
        data["L1_LOSS"].append(format_currency(abs(observed - predicted)))
        data[features[0]].append(batch.at[i, features[0]])
        data[features[1]].append(batch.at[i, features[1]])
        data[features[2]].append(batch.at[i, features[2]])

    output_df = pd.DataFrame(data)
    return output_df

def show_predictions(output):
    """Display the prediction results."""
    header = "-" * 80
    banner = header + "\n" + "|" + "PREDICTIONS".center(78) + "|" + "\n" + header
    print(banner)
    print(output)
    return

def save_metrics_and_predictions(y_true, y_pred, metrics, predictions_file='train_predictions.csv', metrics_file='train_metrics.txt'):
    """Save metrics and predictions to files."""
    
    # Save predictions to CSV
    predictions_df = pd.DataFrame({
        'Predicted Consumption': y_pred,
        'Observed Consumption': y_true,
        'L1 Loss': np.abs(y_true - y_pred)
    })
    
    predictions_df.to_csv(predictions_file, index=False)

    # Save metrics to text file
    with open(metrics_file, 'w') as f:
        for metric, value in metrics.items():
            f.write(f"{metric}: {value}\n")

# Code - Make predictions

def run_prediction(model, df, features, label):
    """Run the prediction process and save metrics and predictions."""
    # Make predictions using the updated prediction function
    output = predict_fuel_consumption(model, df, features, label)

    # Extract true values and predicted values
    y_true = df[label].values
    y_pred = output['PREDICTED_CONSUMPTION'].apply(lambda x: float(x.replace('$', '').replace(',', ''))).values

    # Calculate metrics
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot)

    # Create a dictionary of metrics
    metrics = {
        'Mean Squared Error (MSE)': mse,
        'Root Mean Squared Error (RMSE)': rmse,
        'R-squared (R²)': r2
    }

    # Save metrics and predictions
    save_metrics_and_predictions(y_true, y_pred, metrics)

    # Display predictions
    show_predictions(output)

# Update the feature and label names
features = ['ENGINE SIZE', 'CYLINDERS', 'COEMISSIONS']
label = 'FUEL CONSUMPTION'

# Assuming 'model_2' is the trained model from the training process
run_prediction(model_2, training_df, features, label)
