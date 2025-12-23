# Install if not already: pip install tensorflow
 
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
 
# Load MobileNetV2 without top layers (pre-trained on ImageNet)
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
base_model.trainable = False  # Freeze base model weights
 
# Add custom classification head
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(64, activation='relu')(x)
predictions = Dense(2, activation='softmax')(x)  # Binary classification example
 
# Final model
model = Model(inputs=base_model.input, outputs=predictions)
 
# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
 
# Data generators (assume data organized in 'data/train' and 'data/val' with subfolders for each class)
train_datagen = ImageDataGenerator(rescale=1./255, horizontal_flip=True, rotation_range=20)
val_datagen = ImageDataGenerator(rescale=1./255)
 
train_generator = train_datagen.flow_from_directory(
    'data/train', target_size=(128, 128), batch_size=32, class_mode='categorical')
 
val_generator = val_datagen.flow_from_directory(
    'data/val', target_size=(128, 128), batch_size=32, class_mode='categorical')
 
# Train the model
history = model.fit(train_generator, validation_data=val_generator, epochs=5)
 
# Plot accuracy
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title("Transfer Learning Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()