''' Here's a Python code for analyzing traffic patterns:

Importing Libraries
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Load traffic data
traffic_data = pd.read_csv('traffic_data.csv')


# Data Preprocessing

# Convert date column to datetime format
traffic_data['date'] = pd.to_datetime(traffic_data['date'])

# Extract hour, day, and month from date column
traffic_data['hour'] = traffic_data['date'].dt.hour
traffic_data['day'] = traffic_data['date'].dt.dayofweek
traffic_data['month'] = traffic_data['date'].dt.month

# Drop date column
traffic_data.drop('date', axis=1, inplace=True)

''' Exploratory Data Analysis

Plot traffic volume over time
'''
plt.figure(figsize=(10, 6))
plt.plot(traffic_data['traffic_volume'])
plt.title('Traffic Volume Over Time')
plt.xlabel('Time')
plt.ylabel('Traffic Volume')
plt.show()

# Plot traffic volume by hour of day
plt.figure(figsize=(10, 6))
plt.bar(traffic_data['hour'], traffic_data['traffic_volume'])
plt.title('Traffic Volume by Hour of Day')
plt.xlabel('Hour of Day')
plt.ylabel('Traffic Volume')
plt.show()

# Plot traffic volume by day of week
plt.figure(figsize=(10, 6))
plt.bar(traffic_data['day'], traffic_data['traffic_volume'])
plt.title('Traffic Volume by Day of Week')
plt.xlabel('Day of Week')
plt.ylabel('Traffic Volume')
plt.show()

'''
Building Traffic Prediction Model

Define features and target variable '''

X = traffic_data.drop('traffic_volume', axis=1)
y = traffic_data['traffic_volume']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train random forest regressor model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions on testing data
y_pred = rf_model.predict(X_test)

# Evaluate model performance
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse:.2f}')






