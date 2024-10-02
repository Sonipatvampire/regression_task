#@title Code - Load dependencies

# General
import pandas as pd

# File path
fueltrain_dataset = '/Users/kunalsingh/Documents/Kunal_Singh_A1/regression_task/data/training_data.csv'

# Read the CSV into a DataFrame
df = pd.read_csv(fueltrain_dataset)

# Strip whitespace from column names
df.columns = df.columns.str.strip()

# Select the columns for training
training_df = df[['VEHICLE CLASS', 'ENGINE SIZE', 'CYLINDERS', 'TRANSMISSION', 'FUEL', 'FUEL CONSUMPTION', 'COEMISSIONS']]

print('Read dataset completed successfully.')
print('Total number of rows: {0}\n\n'.format(len(training_df.index)))
