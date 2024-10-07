import pandas as pd

def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

def encode(data):
    # Identify categorical columns and apply one-hot encoding
    categorical_columns = data.select_dtypes(include=['object']).columns
    data = pd.get_dummies(data, columns=categorical_columns, drop_first=True)
    return data

def preprocess_data(data):
    # Encode categorical variables
    data = encode(data)

    # Handle missing values (fill with mean for numeric columns)
    data.fillna(data.mean(), inplace=True)

    # Select relevant features for regression and drop unnecessary columns
    X = data[['Year', 'ENGINE SIZE', 'CYLINDERS']]  # Add more features if needed
    y = data['FUEL CONSUMPTION']
    
    return X, y

if __name__ == "__main__":
    data = load_data('/Users/kunalsingh/Documents/Kunal_Singh_A1/regression_task/data/training_data.csv')
    X, y = preprocess_data(data)

