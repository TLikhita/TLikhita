import 'pandas as pd'

data = pd.read_csv('car_prices.csv')  # Replace with your actual file
data.isnull().sum()  # Check for missing values
data.fillna(data.mean(), inplace=True)  # Example of filling with mean
data = pd.get_dummies(data, columns=['make', 'fuel_type'], drop_first=True)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
features = data.drop('price', axis=1)
scaled_features = scaler.fit_transform(features)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(scaled_features, data['price'], test_size=0.2, random_state=42)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)  # Output layer for regression
])
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2)
test_loss, test_mae = model.evaluate(X_test, y_test)
print(f'Test MAE: {test_mae}')
from sklearn.metrics import mean_squared_error, r2_score

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'MSE: {mse}, R²: {r2}')
import matplotlib.pyplot as plt

plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='validation')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
