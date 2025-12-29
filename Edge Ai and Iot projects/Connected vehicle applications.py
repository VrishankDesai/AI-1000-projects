import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
 
# Simulate driving telemetry: speed (km/h), acceleration (m/s^2), brake pressure (psi), steering angle (degrees)
np.random.seed(42)
n_samples = 1000
 
speed = np.random.normal(60, 10, n_samples)
acceleration = np.random.normal(1.5, 0.7, n_samples)
brake_pressure = np.random.normal(5, 3, n_samples)
steering_angle = np.random.normal(0, 15, n_samples)  # larger deviation may indicate sharp turns
 
# Label aggressive behavior: high accel or brake pressure or erratic steering
aggressive = ((acceleration > 3) | (brake_pressure > 10) | (np.abs(steering_angle) > 25)).astype(int)
 
# Feature matrix and labels
X = np.stack([speed, acceleration, brake_pressure, steering_angle], axis=1)
y = aggressive
 
# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 
# Build classifier model
model = models.Sequential([
    layers.Input(shape=(4,)),
    layers.Dense(32, activation='relu'),
    layers.Dense(16, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Output: aggressive or not
])
 
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=15, batch_size=32, verbose=0)
 
# Evaluate
loss, acc = model.evaluate(X_test, y_test)
print(f"✅ Connected Vehicle Model Accuracy: {acc:.4f}")
 
# Predict sample driver behaviors
preds = (model.predict(X_test[:5]) > 0.5).astype(int).flatten()
for i in range(5):
    print(f"Vehicle {i+1}: Driving Behavior = {'Aggressive' if preds[i] else 'Normal'} (Actual: {'Aggressive' if y_test[i] else 'Normal'})")