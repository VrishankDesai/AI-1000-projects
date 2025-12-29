import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
 
# Simulate embeddings from a face recognition system (e.g., FaceNet) — 128D vectors
np.random.seed(42)
n_users = 5
samples_per_user = 100
n_samples = n_users * samples_per_user
embedding_dim = 128
 
# Generate embeddings and labels
X = np.random.normal(loc=0, scale=1, size=(n_samples, embedding_dim))
y = np.repeat([f"user_{i}" for i in range(n_users)], samples_per_user)
 
# Encode user labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
 
# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
 
# Build face classifier model (input = 128D face embedding)
model = models.Sequential([
    layers.Input(shape=(embedding_dim,)),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(n_users, activation='softmax')  # Predict which user
])
 
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
 
# Evaluate
loss, acc = model.evaluate(X_test, y_test)
print(f"✅ Smart Mirror Face Recognition Accuracy: {acc:.4f}")
 
# Predict and personalize
preds = np.argmax(model.predict(X_test[:5]), axis=1)
for i, p in enumerate(preds):
    user = encoder.classes_[p]
    print(f"🪞 Detected: {user} → Displaying: {user}'s dashboard (weather, agenda, health stats)")