import pandas as pd
import numpy as np
import os  # Import os module to interact with the operating system

def preprocess_data(df):
    # Encode categorical variables using One-Hot Encoding
    categorical_columns = ['MAKE', 'MODEL', 'VEHICLE CLASS', 'TRANSMISSION', 'FUEL']
    df_encoded = pd.get_dummies(df, columns=categorical_columns)

    # Check for available numerical columns
    numerical_columns = ['ENGINE SIZE', 'CYLINDERS', 'COEMISSIONS']
    
    # Print available columns for debugging
    print("Available columns in the DataFrame:", df_encoded.columns.tolist())

    # Filter numerical columns to only those that exist in the DataFrame
    numerical_columns = [col for col in numerical_columns if col in df_encoded.columns]
    
    # Check if any numerical columns were found
    if not numerical_columns:
        print("No numerical columns found for normalization.")
    else:
        # Normalize numerical features
        df_encoded[numerical_columns] = (df_encoded[numerical_columns] - df_encoded[numerical_columns].mean()) / df_encoded[numerical_columns].std()

    # Prepare features and target
    X = df_encoded.drop(columns=['FUEL CONSUMPTION', 'Year'], errors='ignore')  # Use errors='ignore' to prevent errors if columns are missing
    y = df_encoded['FUEL CONSUMPTION'] if 'FUEL CONSUMPTION' in df_encoded.columns else None

    if y is None:
        print("Target column 'FUEL CONSUMPTION' is missing. Cannot proceed.")
        return None, None

    return X, y

# Load and preprocess data
data_path = 'regression_task/data/training_data.csv'
df = pd.read_csv(data_path)
X, y = preprocess_data(df)

# Check if the data was processed successfully before saving
if X is not None and y is not None:
    # Ensure the output directory exists
    output_dir = '../data'
    os.makedirs(output_dir, exist_ok=True)  # Create the directory if it does not exist

    # Save preprocessed data for training
    X.to_csv(os.path.join(output_dir, 'X_preprocessed.csv'), index=False)
    y.to_csv(os.path.join(output_dir, 'y_preprocessed.csv'), index=False)
else:
    print("Preprocessing failed, no data to save.")


