import numpy as np

#Sigmoid function activation and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x)) 

def sigmoid_derivative(x):
    return x * (1 - x)

#MSE loss function 
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

#Neural Network class
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
        #Weights initialization
        self.weights_input_hidden = np.random.rand(input_size, hidden_size)
        self.weights_hidden_output = np.random.rand(hidden_size, output_size)
        self.bias_hidden = np.random.rand(1, hidden_size)
        self.bias_output = np.random.rand(1, output_size)

    def forward(self, X):
        #Forward propagation
        self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = sigmoid(self.hidden_input)
        self.output_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.output = sigmoid(self.output_input)
        return self.output

    def backward(self, X, y, output, learning_rate):
        #Backward propagation
        output_error = y - output
        output_delta = output_error * sigmoid_derivative(output)
        hidden_error = np.dot(output_delta, self.weights_hidden_output.T)
        hidden_delta = hidden_error * sigmoid_derivative(self.hidden_output)

        #Weights and biases update
        self.weights_hidden_output += np.dot(self.hidden_output.T, output_delta) * learning_rate
        self.bias_output += np.sum(output_delta, axis=0, keepdims=True) * learning_rate
        self.weights_input_hidden += np.dot(X.T, hidden_delta) * learning_rate
        self.bias_hidden += np.sum(hidden_delta, axis=0, keepdims=True) * learning_rate
    #Train the neural network
    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output, learning_rate)
            if epoch % 100 == 0:
                loss = mean_squared_error(y, output)
                print(f'Epoch {epoch}, Loss: {loss}')
                
#XOR dataset
X = np.array([[0, 0],
              [0, 1],
                [1, 0],
                [1, 1]])
y = np.array([[0],
              [1],
                [1],
                [0]])
nn = NeuralNetwork(input_size=2, hidden_size=2, output_size=1)
nn.train(X, y, epochs=10000, learning_rate=0.1)
print("\nTest the trained neural network")
for i in range(len(X)):
    print(f"Input: {X[i]}, Predicted Output: {nn.forward(X[i])}, Actual Output: {y[i]}")