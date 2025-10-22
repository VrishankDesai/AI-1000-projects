import numpy as np
import tensorflow as tf #Used for training deep learning models/neural networks
from tensorflow.keras import datasets, layers, models #keras makes code cleaner datasets like mnist, layers like conv2D and maxpooling2d, models like sequential
from tensorflow.keras.utils import to_categorical #Used for one hot encoding of labels-convert to binary no's for comp to understand
import matplotlib.pyplot as plt

#Load mnist dataset-contains train images and test images
(train_images, train_labels), (test_images,test_labels)= datasets.mnist.load_data()

#Preprocess the data-makes training faster and efficient
train_images = train_images / 255.0 #Normalize pixel values to be between 0 and 1 which is why it is divided by 255 to scale values between 0 and 1
test_images = test_images / 255.0

#Reshape images to (28, 28, 1) as they are grayscale images-mnist images are greyscale which have only 1 channel
#28,28,1 represents height, width and colour channels respectively
train_images = train_images.reshape((train_images.shape[0],28,28,1))
test_images = test_images.reshape((test_images.shape[0],28,28,1))

#Convert labels to one hot encoded format
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

#Build the CNN model-builds model step by step in form of following steps
model = models.Sequential()

#First convoltional layer con2d detects edges,corners,curves of 2d image
model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1))) #32 is number of convolutional filters, (3,3) is the size of the pixels,activation function sets nregative pixel values to 0
model.add(layers.MaxPooling2D((2,2))) #maxpooling reduces size while keeping imp info, (2,2) is size of pooling window(2x2 matrix)

#Second convolutional layer-64 means more complex images
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))

#Third convolutional layer-shows digits in final output
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))

#Flatten the output to 1D and feed it into dense layers,converts 3d output to 1d vector
model.add(layers.Flatten()) #takes 3d output from last pooling ;layer and flattens it to 1d vector
model.add(layers.Dense(64, activation='relu')) #Dense is fully connected layer with 64 neurons,also activatin fn activates fn-turns negative output to zero

#Output layer with 10 neurons for 10 classes
model.add(layers.Dense(10, activation='softmax')) #Dense layer means connected to every neuron, there are 10 digits(0-9) in mnist,softmax converts output to probability which equals to 1

#Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#Optimizer decides how model updates its data during training adam is most efficient optimizer,loss calculates how prob is far from truth,metrics calculates performance of model

#Train the model
model.fit(train_images, train_labels, epochs=5, batch_size=64, validation_split=0.2)#Epoch means no of times model sees entire dataset,batch size means no of images model sees before model is updated,validation split means 20% of training data is used to check data(accuracy)


#Evaluate the model on test data
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Test accuracy: {test_acc*100:.2f}%")

#Make predictions on test images
predictions = model.predict(test_images)
print(f"Prediction for first test image: {np.argmax(predictions[4])}") #Picks digit with highest prob for 5th test image

plt.imshow(test_images[4].reshape(28,28), cmap='gray')
plt.title(f"Predicted Label: {np.argmax(predictions[4])}")
plt.show()