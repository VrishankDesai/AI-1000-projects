import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
 
# Simulate WiFi RSSI values from 4 access points (in dBm, typically -90 to -30)
np.random.seed(42)
n_samples = 1000
ap1 = np.random.normal(-60, 5, n_samples)
ap2 = np.random.normal(-70, 5, n_samples)
ap3 = np.random.normal(-65, 5, n_samples)
ap4 = np.random.normal(-75, 5, n_samples)
 
# Simulate x, y coordinates (in meters) as labels — for example, a 20x20 meter indoor area
x_coords = np.random.uniform(0, 20, n_samples)
y_coords = np.random.uniform(0, 20, n_samples)
 
# Feature matrix (RSSI from access points)
X = np.stack([ap1, ap2, ap3, ap4], axis=1)
y = np.stack([x_coords, y_coords], axis=1)  # target: 2D location
 
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 
# Build regression model for coordinate prediction
model = models.Sequential([
    layers.Input(shape=(4,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(2)  # x and y coordinates
])
 
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)
 
# Evaluate
loss, mae = model.evaluate(X_test, y_test)
print(f"✅ Indoor Positioning MAE: {mae:.2f} meters")
 
# Predict sample positions
preds = model.predict(X_test[:5])
for i, pred in enumerate(preds):
    print(f"Sample {i+1}: Predicted (x, y) = ({pred[0]:.2f}, {pred[1]:.2f}) | Actual = ({y_test[i][0]:.2f}, {y_test[i][1]:.2f})")
 
# Optional: Visualize prediction vs actual
plt.figure(figsize=(6, 6))
plt.scatter(y_test[:50, 0], y_test[:50, 1], label='Actual', alpha=0.6)
plt.scatter(preds[:50, 0], preds[:50, 1], label='Predicted', alpha=0.6)
plt.title("Indoor Positioning System: Actual vs Predicted")
plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.legend()
plt.grid(True)
plt.show()