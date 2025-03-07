# -*- coding: utf-8 -*-
"""Assignment#2.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1RmQ13zSnrbDwEkunXpsckSMZaXHOBmgl
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from IPython.display import display



from google.colab import drive
drive.mount('/content/drive')

df=pd.read_csv('/content/drive/MyDrive/Dataset/dataset.csv')

df.head()

!pip install missingno

# Drop the unnamed column (if it exists)
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

# Transpose the DataFrame
df_transposed = df.T

# Select only the first 4 columns
df_transposed_selected = df_transposed.iloc[:, :5]

# Display the transposed DataFrame with only 4 columns
display(df_transposed_selected)

# Dimensions of the training data
num_rows, num_columns = df.shape
print(f"({num_rows}, {num_columns})")

# Explore columns in the dataset
column_names = df.columns
print(column_names)

# Description of the dataset (transposed)
description_transposed = df.describe().T
print(description_transposed.to_string(header=True))

# Check data types
data_types = df.dtypes
print(data_types)

df_transposed = df.head().T
df_transposed.iloc[:, :5]

df_transposed = df.tail().T
df_transposed.iloc[:, :5]

# Check for any null or missing values
missing_values = df.isnull().any().any()

# Print the result
print(missing_values)

missingValuesPerColumn = df.isnull().sum()
print(missingValuesPerColumn)

# print(data['mobile_subscriptions'].dtype)

col_list = [c for c in df.columns if df[c].dtype == 'object']
col_list

#Unique values in 'women_parliament_seats_rate'
distinct_values = df['women_parliament_seats_rate'].unique()
print(distinct_values)

# Make a copy of data
train_data_copy = df.copy()

# Compare Actual and Encoded labels
from sklearn.preprocessing import LabelEncoder

# Initialize the LabelEncoder
label_encoder = LabelEncoder()

# A dictionary to store the encodings for each column
encodings = {}

# List of columns you want to encode
columns_to_encode = ['national_income', 'mobile_subscriptions', 'improved_sanitation', 'women_parliament_seats_rate']

# Loop over the columns and encode them
for column in columns_to_encode:
    # Fit the label encoder and transform the data
    train_data_copy[column] = label_encoder.fit_transform(train_data_copy[column])

    # Store the actual and encoded labels for comparison
    encodings[column] = {
        'labels': list(label_encoder.classes_),
        'encoded': list(range(len(label_encoder.classes_)))
    }

# Print the actual and encoded labels for each column
for column, encoding in encodings.items():
    print(f"column: {column}")
    print(encoding['labels'])
    print(encoding['encoded'])

# Verify the changes
print(train_data_copy.dtypes)

import pandas as pd

# Define a function to convert "per 100" to "per 1000"
def convert_to_per_thousand(string):
    if string == '-1':
        return -1
    try:
        parts = string.split(' per ')
        numerator_str = parts[0]
        if 'per 1000' in string:
            # Extract the numeric part and return as integer
            return int(numerator_str)
        else:
            # If "per 100", convert to "per 1000" directly
            numeric_part = int(numerator_str)
            return numeric_part * 10
    except:
        return None  # Return None if conversion fails

# Replace 'unknown' with '-1'
train_data_copy['internet_users'] = train_data_copy['internet_users'].replace('unknown', '-1')

# Convert "per 100" to "per 1000" and extract numeric part
train_data_copy['internet_users'] = train_data_copy['internet_users'].apply(convert_to_per_thousand)

# Check unique values in 'internet_users' after conversion
print("\nUnique values in 'internet_users' after conversion:")
print(train_data_copy['internet_users'].unique())

# Fill missing values with the mean of non-missing values
mean_internet_users = train_data_copy['internet_users'].mean()
train_data_copy['internet_users'].fillna(mean_internet_users, inplace=True)

# Check missing values in each column of the DataFrame after filling
missing_values_after = train_data_copy.isnull().sum()
print("Missing values after filling:")
print(missing_values_after)

# Define columns with missing values
columns_with_missing = ['agricultural_land', 'forest_area', 'armed_forces_total',
                        'urban_pop_minor_cities', 'urban_pop_major_cities',
                        'inflation_annual', 'inflation_monthly', 'inflation_weekly',
                        'secure_internet_servers_total']

# Fill missing values with the mean of each column
for column in columns_with_missing:
    mean_value = train_data_copy[column].mean()
    train_data_copy[column].fillna(mean_value, inplace=True)

# Check missing values after filling
missing_values_after = train_data_copy.isnull().sum()
print("Missing values after filling:")
print(missing_values_after)

# plot joint plots for 'life_expectancy'
columns = ['surface_area','agricultural_land', 'forest_area', 'armed_forces_total',
           'urban_pop_minor_cities', 'urban_pop_major_cities',
           'national_income','inflation_annual', 'mobile_subscriptions','internet_users','secure_internet_servers_total','improved_sanitation','women_parliament_seats_rate']

# Loop through each column and plot jointplot with 'life_expectancy'
for column in columns:
    sns.jointplot(x=column, y='life_expectancy', data=train_data_copy, kind='reg')
    plt.show()

# Compute the correlation matrix
correlation_matrix = train_data_copy.corr()

# Filter the correlation matrix for values greater than or equal to +0.5 and less than or equal to -0.4
filtered_correlation = correlation_matrix[
    (correlation_matrix >= 0.5) | (correlation_matrix <= -0.4)
]

# Create a mask to hide the upper triangle of the heatmap
mask = filtered_correlation.isnull()

# Plot the correlation heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(filtered_correlation, annot=True, cmap='viridis', mask=mask, fmt=".2f")
plt.title('Correlation Heatmap (>= +0.5 or <= -0.4)')
plt.show()

# Plot the distribution of the target variable (life expectancy) with KDE
plt.figure(figsize=(15, 6))
sns.distplot(train_data_copy['life_expectancy'], kde=True, color='blue')
plt.title('Distribution of Life Expectancy')
plt.xlabel('Life Expectancy')
plt.ylabel('Density')
plt.show()

# Drop the specified columns from the DataFrame
train_data_copy.drop(['agricultural_land', 'forest_area', 'inflation_monthly', 'inflation_weekly'], axis=1, inplace=True)

# Check the shape of the DataFrame
print(train_data_copy.shape)

from sklearn.preprocessing import StandardScaler

# Columns not to be standardized
cols_to_exclude = ['national_income', 'mobile_subscriptions', 'life_expectancy',
                   'improved_sanitation', 'women_parliament_seats_rate']

# Standardize the data excluding specified columns
scaler = StandardScaler()
train_data_copy[train_data_copy.columns.difference(cols_to_exclude)] = scaler.fit_transform(train_data_copy[train_data_copy.columns.difference(cols_to_exclude)])
train_data_copy.head().transpose()

from sklearn.model_selection import train_test_split

# Assuming your target variable is 'life_expectancy'
X = train_data_copy.drop(columns=['life_expectancy'])  # Features
y = train_data_copy['life_expectancy']  # Target variable

# Split the data into training and testing sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Verify the shapes of the resulting sets
print("Training Set Dimensions:", X_train.shape)
print("Validation Set Dimensions:", X_test.shape)

# Train Random Forest Regressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
randomf = RandomForestRegressor(n_estimators=200, random_state=42)
randomf.fit(X_train, y_train)

# Measure mean absolute error for training and validation sets
print('Mean Absolute Error for Training Set:', mean_absolute_error(y_train, randomf.predict(X_train)))
print('Mean Absolute Error for Test Set:', mean_absolute_error(y_test, randomf.predict(X_test)))

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Train xgboost regressor
xgb = XGBRegressor(n_estimators=200, random_state=42)
xgb.fit(X_train, y_train)
y_pred = xgb.predict(X_test)

# Evaluate performance
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Measure mean absolute error for training and validation sets
mae_train = mean_absolute_error(y_train, xgb.predict(X_train))
mae_test = mean_absolute_error(y_test, xgb.predict(X_test))
print('Mean Absolute Error for Training Set:', mae_train)
print('Mean Absolute Error for Test Set:', mae_test)

from sklearn.ensemble import GradientBoostingRegressor

# Instantiate the Gradient Boosting Regressor
gbr = GradientBoostingRegressor(n_estimators=200, random_state=42)

# Train the Gradient Boosting Regressor model
gbr.fit(X_train, y_train)

y_pred_gbr = gbr.predict(X_test)

mae_gbr = mean_absolute_error(y_test, y_pred_gbr)

r2_gbr = r2_score(y_test, y_pred_gbr)

print('Mean Absolute Error for Training Set (Gradient Boosting):', mean_absolute_error(y_train, gbr.predict(X_train)))
print('Mean Absolute Error for Test Set (Gradient Boosting):', mean_absolute_error(y_test, gbr.predict(X_test)))

# important features for random forest regressor
for name, importance in zip(X.columns, xgb.feature_importances_):
    print('feature:', name, "=", importance)

import numpy as np
# Sort the features and importances in descending order
sorted_indices = np.argsort(xgb.feature_importances_)[::-1]
sorted_features = [train_data_copy.columns[i] for i in sorted_indices]
sorted_importances = [xgb.feature_importances_[i] for i in sorted_indices]

plt.figure(figsize=(6, 6))
# Create horizontal bar graph in descending order
plt.barh(sorted_features, sorted_importances)
plt.xlabel('Importance')
plt.ylabel('Features')
plt.title('Feature Importances')
plt.gca().invert_yaxis()  # Invert y-axis to show the most important feature at the top
plt.show()