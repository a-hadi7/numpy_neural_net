import numpy as np

class NeuralNetwork:
    def __init__(self, input_size=784, hidden_size=128, output_size=10):
        """
        Initializes the architecture of our Multi-Layer Perceptron.
        784 inputs (pixels) -> 128 hidden neurons -> 10 outputs (digits 0-9)
        """
        # We use 'He Initialization' for the weights to help the network learn faster
        self.W1 = np.random.randn(hidden_size, input_size) * np.sqrt(2. / input_size)
        self.b1 = np.zeros((hidden_size, 1))
        self.W2 = np.random.randn(output_size, hidden_size) * np.sqrt(2. / hidden_size)
        self.b2 = np.zeros((output_size, 1))

    def relu(self, Z):
        """Activation Function: Turns all negative numbers to 0."""
        return np.maximum(0, Z)

    def relu_derivative(self, Z):
        """Calculus derivative of ReLU, used for training."""
        return Z > 0

    def softmax(self, Z):
        """Converts the final raw numbers into percentages/probabilities (0 to 1)."""
        # We subtract the max to prevent math overflow errors
        expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True))
        return expZ / np.sum(expZ, axis=0, keepdims=True)

    def forward_propagation(self, X):
        """
        THE GUESS: The data flows forward. We multiply the pixels by the weights, 
        add the bias, and pass it through the activation functions to get a prediction.
        """
        Z1 = np.dot(self.W1, X) + self.b1
        A1 = self.relu(Z1)
        Z2 = np.dot(self.W2, A1) + self.b2
        A2 = self.softmax(Z2)
        return Z1, A1, Z2, A2

    def backward_propagation(self, X, Y, Z1, A1, Z2, A2):
        """
        THE LEARNING: We calculate how wrong the guess was (loss), and use calculus 
        (the chain rule) to trace back exactly which weights need to be adjusted.
        """
        m = X.shape[1] # Number of images we are looking at
        
        # Calculate the error of the output layer
        dZ2 = A2 - Y
        dW2 = (1 / m) * np.dot(dZ2, A1.T)
        db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)
        
        # Calculate the error of the hidden layer
        dZ1 = np.dot(self.W2.T, dZ2) * self.relu_derivative(Z1)
        dW1 = (1 / m) * np.dot(dZ1, X.T)
        db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)
        
        return dW1, db1, dW2, db2

    def update_parameters(self, dW1, db1, dW2, db2, learning_rate):
        """
        THE TWEAK: We take a tiny step down the gradient to improve the weights 
        for the next round of guessing.
        """
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2