import numpy as np
import matplotlib.pyplot as plt
from src.utils import load_data, one_hot_encode
from src.network import NeuralNetwork

def get_predictions(A2):
    """Finds the highest probability in the output array to make a final guess."""
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    """Compares the AI's guesses to the actual answer key."""
    return np.sum(predictions == Y) / Y.size

def main():
    # 1. Load and prepare the data
    X, Y = load_data()

    # The data needs to be shaped (features, number of examples) for our math to work
    X = X.T
    Y_one_hot = one_hot_encode(Y).T

    # 2. Set up the Neural Network
    nn = NeuralNetwork(input_size=784, hidden_size=128, output_size=10)
    
    # Hyperparameters (You can tweak these later to see how they affect learning)
    iterations = 500
    learning_rate = 0.1
    
    accuracies = []

    print("\nStarting the training loop...")
    
    # 3. The Training Loop (The actual machine learning)
    for i in range(iterations):
        # Step A: Forward pass (Make a guess)
        Z1, A1, Z2, A2 = nn.forward_propagation(X)
        
        # Step B: Backward pass (Calculate the errors using calculus)
        dW1, db1, dW2, db2 = nn.backward_propagation(X, Y_one_hot, Z1, A1, Z2, A2)
        
        # Step C: Update parameters (Tweak the brain slightly)
        nn.update_parameters(dW1, db1, dW2, db2, learning_rate)
        
        # Print progress every 50 iterations so we can watch it learn
        if i % 50 == 0:
            predictions = get_predictions(A2)
            accuracy = get_accuracy(predictions, Y)
            accuracies.append(accuracy)
            print(f"Iteration: {i:03} | Accuracy: {accuracy * 100:.2f}%")

    print("Training complete!\n")

    # 4. Draw the Graph
    plt.plot(range(0, iterations, 50), accuracies, marker='o', color='blue', linewidth=2)
    plt.title("Neural Network Learning Progress (From Scratch)")
    plt.xlabel("Iterations")
    plt.ylabel("Accuracy")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

if __name__ == "__main__":
    main()