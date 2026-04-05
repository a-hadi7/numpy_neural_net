import numpy as np
from sklearn.datasets import fetch_openml

def load_data():
    """
    Downloads the MNIST dataset and formats it for our neural network.
    """
    print("Downloading MNIST dataset (this might take a minute or two)...")
    
    # Fetch the 70,000 images of handwritten digits (0-9)
    # as_frame=False ensures we get pure NumPy arrays, not Pandas dataframes
    mnist = fetch_openml('mnist_784', version=1, cache=True, as_frame=False)
    
    # X contains the image pixels, y contains the correct answers (labels)
    X = mnist.data
    y = mnist.target.astype(int)
    
    # NORMALIZE THE DATA:
    # Pixel values range from 0 to 255. Neural networks learn much faster
    # when numbers are small, so we divide by 255 to make them range from 0 to 1.
    X = X / 255.0
    
    print("Dataset loaded successfully!")
    return X, y

def one_hot_encode(y, num_classes=10):
    """
    Converts a single number label into an array. 
    Example: The number 3 becomes [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    Our neural network has 10 output neurons, so it needs answers in this format.
    """
    one_hot = np.zeros((y.size, num_classes))
    one_hot[np.arange(y.size), y] = 1
    return one_hot