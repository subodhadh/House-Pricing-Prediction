import numpy as np
import pandas as pd

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from keras.models import Sequential
from keras.layers import Dense


#loading data
data=fetch_california_housing()
X=data.data
y=data.target


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


model = Sequential()

# Input layer + hidden layers
model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dense(64, activation='relu'))

# Output layer (regression → 1 neuron, linear activation)
model.add(Dense(1))


model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)



model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.1
)



loss, mae = model.evaluate(X_test, y_test)
print("Test MAE:", mae)




# Get feature names
feature_names = list(data.feature_names)

# Take input from user
user_input = []
print("Enter the following details to predict house price:")

for feature in feature_names:
    value = float(input(f"{feature}: "))
    user_input.append(value)

# Convert to numpy array and scale
user_input = np.array(user_input).reshape(1, -1)
user_input_scaled = scaler.transform(user_input)

# Make prediction
predicted_price = model.predict(user_input_scaled)
print(f"\nPredicted Median House Price: ${predicted_price[0][0]*100000:.2f}")
