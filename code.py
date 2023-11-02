import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from tensorflow import keras

# Load earthquake data (assuming you have a CSV file)
data = pd.read_csv('database.csv')

# Data Preprocessing
data = data.dropna()  # Remove rows with missing values

# Select features and target variable
X = data[['Latitude', 'Longitude', 'Depth']]
y = data['Magnitude']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create a simple neural network model
model = keras.Sequential([
    keras.layers.Dense(32, activation='relu', input_shape=(3,)),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(1)  # Output layer
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=1)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model's performance
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

#visualization
# Import necessary libraries for data visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Create subplots for a row-wise arrangement
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Visualize the distribution of earthquake magnitudes in your dataset
sns.histplot(data['Magnitude'], kde=True, ax=axes[0])
axes[0].set_title("Distribution of Earthquake Magnitudes")
axes[0].set_xlabel("Magnitude")
axes[0].set_ylabel("Frequency")

# Visualize the relationship between magnitude and depth
sns.scatterplot(x='Depth', y='Magnitude', data=data, ax=axes[1])
axes[1].set_title("Magnitude vs. Depth")
axes[1].set_xlabel("Depth")
axes[1].set_ylabel("Magnitude")

# Visualize the model predictions vs. actual values
axes[2].scatter(y_test, y_pred, alpha=0.5)
axes[2].set_title("Model Predictions vs. Actual Magnitudes")
axes[2].set_xlabel("Actual Magnitudes")
axes[2].set_ylabel("Predicted Magnitudes")

# Adjust layout to prevent overlap
plt.tight_layout()

# Show the plots
plt.show()

