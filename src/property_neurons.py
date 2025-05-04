"""
property_neurons.py
Neural network classifiers for number properties (prime, perfect, fibonacci, etc.)
"""
import numpy as np
from models.neuron import Neuron
from number_analysis import is_prime, is_perfect_number, is_fibonacci

def generate_prime_data(start, end):
    x = np.arange(start, end)
    y = np.array([1 if is_prime(n) else 0 for n in x])
    return x, y

def train_prime_neuron(train_range=(0, 1000), learning_rate=0.01, epochs=20000):
    x, y = generate_prime_data(*train_range)
    neuron = Neuron(inputSize=1)
    losses = neuron.train(x, y, learningRate=learning_rate, epochs=epochs)
    return neuron, losses

def test_prime_neuron(neuron, test_range=(0, 100), verbose=True):
    from sklearn.metrics import confusion_matrix
    x, y = generate_prime_data(*test_range)
    preds = neuron.predict(x)
    preds_binary = (preds > 0.5).astype(int)
    accuracy = np.mean(preds_binary == y)
    cm = confusion_matrix(y, preds_binary)
    if verbose:
        for n, pred, label in zip(x, preds_binary, y):
            print(f"Number: {n:5d}  Predicted: {'Prime' if pred else 'Not Prime'}  Actual: {'Prime' if label else 'Not Prime'}")
        print(f"Prime neuron accuracy: {accuracy:.2%}")
        print(f"Confusion Matrix:\n{cm}")
    return accuracy, cm

# Similar pattern can be used for perfect number, fibonacci, etc.
# Example for perfect number:
def generate_perfect_data(start, end):
    x = np.arange(start, end)
    y = np.array([1 if is_perfect_number(n) else 0 for n in x])
    return x, y

def train_perfect_neuron(train_range=(0, 10000), learning_rate=0.01, epochs=20000):
    x, y = generate_perfect_data(*train_range)
    neuron = Neuron(inputSize=1)
    losses = neuron.train(x, y, learningRate=learning_rate, epochs=epochs)
    return neuron, losses

def test_perfect_neuron(neuron, test_range=(0, 1000), verbose=True):
    from sklearn.metrics import confusion_matrix
    x, y = generate_perfect_data(*test_range)
    preds = neuron.predict(x)
    preds_binary = (preds > 0.5).astype(int)
    accuracy = np.mean(preds_binary == y)
    cm = confusion_matrix(y, preds_binary)
    if verbose:
        for n, pred, label in zip(x, preds_binary, y):
            print(f"Number: {n:5d}  Predicted: {'Perfect' if pred else 'Not Perfect'}  Actual: {'Perfect' if label else 'Not Perfect'}")
        print(f"Perfect neuron accuracy: {accuracy:.2%}")
        print(f"Confusion Matrix:\n{cm}")
    return accuracy, cm

# Example for fibonacci:
def generate_fibonacci_data(start, end):
    x = np.arange(start, end)
    y = np.array([1 if is_fibonacci(n) else 0 for n in x])
    return x, y

def train_fibonacci_neuron(train_range=(0, 10000), learning_rate=0.01, epochs=20000):
    x, y = generate_fibonacci_data(*train_range)
    neuron = Neuron(inputSize=1)
    losses = neuron.train(x, y, learningRate=learning_rate, epochs=epochs)
    return neuron, losses

def test_fibonacci_neuron(neuron, test_range=(0, 1000), verbose=True):
    from sklearn.metrics import confusion_matrix
    x, y = generate_fibonacci_data(*test_range)
    preds = neuron.predict(x)
    preds_binary = (preds > 0.5).astype(int)
    accuracy = np.mean(preds_binary == y)
    cm = confusion_matrix(y, preds_binary)
    if verbose:
        for n, pred, label in zip(x, preds_binary, y):
            print(f"Number: {n:5d}  Predicted: {'Fibonacci' if pred else 'Not Fibonacci'}  Actual: {'Fibonacci' if label else 'Not Fibonacci'}")
        print(f"Fibonacci neuron accuracy: {accuracy:.2%}")
        print(f"Confusion Matrix:\n{cm}")
    return accuracy, cm
