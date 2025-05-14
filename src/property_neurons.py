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

# New property generators/train/test
from number_analysis import (
    is_palindrome, is_armstrong, is_harshad, is_square_free, is_abundant, is_deficient, is_happy, is_triangular, is_catalan
)

def generate_palindrome_data(start, end):
    x = np.arange(start, end)
    y = np.array([1 if is_palindrome(n) else 0 for n in x])
    return x, y

def train_palindrome_neuron(train_range=(0, 10000), learning_rate=0.01, epochs=20000):
    x, y = generate_palindrome_data(*train_range)
    neuron = Neuron(inputSize=1)
    losses = neuron.train(x, y, learningRate=learning_rate, epochs=epochs)
    return neuron, losses

def test_palindrome_neuron(neuron, test_range=(0, 1000), verbose=True):
    from sklearn.metrics import confusion_matrix
    x, y = generate_palindrome_data(*test_range)
    preds = neuron.predict(x)
    preds_binary = (preds > 0.5).astype(int)
    accuracy = np.mean(preds_binary == y)
    cm = confusion_matrix(y, preds_binary)
    if verbose:
        for n, pred, label in zip(x, preds_binary, y):
            print(f"Number: {n:5d}  Predicted: {'Palindrome' if pred else 'Not Palindrome'}  Actual: {'Palindrome' if label else 'Not Palindrome'}")
        print(f"Palindrome neuron accuracy: {accuracy:.2%}")
        print(f"Confusion Matrix:\n{cm}")
    return accuracy, cm

def generate_armstrong_data(start, end):
    x = np.arange(start, end)
    y = np.array([1 if is_armstrong(n) else 0 for n in x])
    return x, y

def train_armstrong_neuron(train_range=(0, 10000), learning_rate=0.01, epochs=20000):
    x, y = generate_armstrong_data(*train_range)
    neuron = Neuron(inputSize=1)
    losses = neuron.train(x, y, learningRate=learning_rate, epochs=epochs)
    return neuron, losses

def test_armstrong_neuron(neuron, test_range=(0, 1000), verbose=True):
    from sklearn.metrics import confusion_matrix
    x, y = generate_armstrong_data(*test_range)
    preds = neuron.predict(x)
    preds_binary = (preds > 0.5).astype(int)
    accuracy = np.mean(preds_binary == y)
    cm = confusion_matrix(y, preds_binary)
    if verbose:
        for n, pred, label in zip(x, preds_binary, y):
            print(f"Number: {n:5d}  Predicted: {'Armstrong' if pred else 'Not Armstrong'}  Actual: {'Armstrong' if label else 'Not Armstrong'}")
        print(f"Armstrong neuron accuracy: {accuracy:.2%}")
        print(f"Confusion Matrix:\n{cm}")
    return accuracy, cm

def generate_harshad_data(start, end):
    x = np.arange(start, end)
    y = np.array([1 if is_harshad(n) else 0 for n in x])
    return x, y

def train_harshad_neuron(train_range=(0, 10000), learning_rate=0.01, epochs=20000):
    x, y = generate_harshad_data(*train_range)
    neuron = Neuron(inputSize=1)
    losses = neuron.train(x, y, learningRate=learning_rate, epochs=epochs)
    return neuron, losses

def test_harshad_neuron(neuron, test_range=(0, 1000), verbose=True):
    from sklearn.metrics import confusion_matrix
    x, y = generate_harshad_data(*test_range)
    preds = neuron.predict(x)
    preds_binary = (preds > 0.5).astype(int)
    accuracy = np.mean(preds_binary == y)
    cm = confusion_matrix(y, preds_binary)
    if verbose:
        for n, pred, label in zip(x, preds_binary, y):
            print(f"Number: {n:5d}  Predicted: {'Harshad' if pred else 'Not Harshad'}  Actual: {'Harshad' if label else 'Not Harshad'}")
        print(f"Harshad neuron accuracy: {accuracy:.2%}")
        print(f"Confusion Matrix:\n{cm}")
    return accuracy, cm

def generate_square_free_data(start, end):
    x = np.arange(start, end)
    y = np.array([1 if is_square_free(n) else 0 for n in x])
    return x, y

def train_square_free_neuron(train_range=(0, 10000), learning_rate=0.01, epochs=20000):
    x, y = generate_square_free_data(*train_range)
    neuron = Neuron(inputSize=1)
    losses = neuron.train(x, y, learningRate=learning_rate, epochs=epochs)
    return neuron, losses

def test_square_free_neuron(neuron, test_range=(0, 1000), verbose=True):
    from sklearn.metrics import confusion_matrix
    x, y = generate_square_free_data(*test_range)
    preds = neuron.predict(x)
    preds_binary = (preds > 0.5).astype(int)
    accuracy = np.mean(preds_binary == y)
    cm = confusion_matrix(y, preds_binary)
    if verbose:
        for n, pred, label in zip(x, preds_binary, y):
            print(f"Number: {n:5d}  Predicted: {'Square-Free' if pred else 'Not Square-Free'}  Actual: {'Square-Free' if label else 'Not Square-Free'}")
        print(f"Square-Free neuron accuracy: {accuracy:.2%}")
        print(f"Confusion Matrix:\n{cm}")
    return accuracy, cm

def generate_abundant_data(start, end):
    x = np.arange(start, end)
    y = np.array([1 if is_abundant(n) else 0 for n in x])
    return x, y

def train_abundant_neuron(train_range=(0, 10000), learning_rate=0.01, epochs=20000):
    x, y = generate_abundant_data(*train_range)
    neuron = Neuron(inputSize=1)
    losses = neuron.train(x, y, learningRate=learning_rate, epochs=epochs)
    return neuron, losses

def test_abundant_neuron(neuron, test_range=(0, 1000), verbose=True):
    from sklearn.metrics import confusion_matrix
    x, y = generate_abundant_data(*test_range)
    preds = neuron.predict(x)
    preds_binary = (preds > 0.5).astype(int)
    accuracy = np.mean(preds_binary == y)
    cm = confusion_matrix(y, preds_binary)
    if verbose:
        for n, pred, label in zip(x, preds_binary, y):
            print(f"Number: {n:5d}  Predicted: {'Abundant' if pred else 'Not Abundant'}  Actual: {'Abundant' if label else 'Not Abundant'}")
        print(f"Abundant neuron accuracy: {accuracy:.2%}")
        print(f"Confusion Matrix:\n{cm}")
    return accuracy, cm

def generate_deficient_data(start, end):
    x = np.arange(start, end)
    y = np.array([1 if is_deficient(n) else 0 for n in x])
    return x, y

def train_deficient_neuron(train_range=(0, 10000), learning_rate=0.01, epochs=20000):
    x, y = generate_deficient_data(*train_range)
    neuron = Neuron(inputSize=1)
    losses = neuron.train(x, y, learningRate=learning_rate, epochs=epochs)
    return neuron, losses

def test_deficient_neuron(neuron, test_range=(0, 1000), verbose=True):
    from sklearn.metrics import confusion_matrix
    x, y = generate_deficient_data(*test_range)
    preds = neuron.predict(x)
    preds_binary = (preds > 0.5).astype(int)
    accuracy = np.mean(preds_binary == y)
    cm = confusion_matrix(y, preds_binary)
    if verbose:
        for n, pred, label in zip(x, preds_binary, y):
            print(f"Number: {n:5d}  Predicted: {'Deficient' if pred else 'Not Deficient'}  Actual: {'Deficient' if label else 'Not Deficient'}")
        print(f"Deficient neuron accuracy: {accuracy:.2%}")
        print(f"Confusion Matrix:\n{cm}")
    return accuracy, cm

def generate_happy_data(start, end):
    x = np.arange(start, end)
    y = np.array([1 if is_happy(n) else 0 for n in x])
    return x, y

def train_happy_neuron(train_range=(0, 10000), learning_rate=0.01, epochs=20000):
    x, y = generate_happy_data(*train_range)
    neuron = Neuron(inputSize=1)
    losses = neuron.train(x, y, learningRate=learning_rate, epochs=epochs)
    return neuron, losses

def test_happy_neuron(neuron, test_range=(0, 1000), verbose=True):
    from sklearn.metrics import confusion_matrix
    x, y = generate_happy_data(*test_range)
    preds = neuron.predict(x)
    preds_binary = (preds > 0.5).astype(int)
    accuracy = np.mean(preds_binary == y)
    cm = confusion_matrix(y, preds_binary)
    if verbose:
        for n, pred, label in zip(x, preds_binary, y):
            print(f"Number: {n:5d}  Predicted: {'Happy' if pred else 'Not Happy'}  Actual: {'Happy' if label else 'Not Happy'}")
        print(f"Happy neuron accuracy: {accuracy:.2%}")
        print(f"Confusion Matrix:\n{cm}")
    return accuracy, cm

def generate_triangular_data(start, end):
    x = np.arange(start, end)
    y = np.array([1 if is_triangular(n) else 0 for n in x])
    return x, y

def train_triangular_neuron(train_range=(0, 10000), learning_rate=0.01, epochs=20000):
    x, y = generate_triangular_data(*train_range)
    neuron = Neuron(inputSize=1)
    losses = neuron.train(x, y, learningRate=learning_rate, epochs=epochs)
    return neuron, losses

def test_triangular_neuron(neuron, test_range=(0, 1000), verbose=True):
    from sklearn.metrics import confusion_matrix
    x, y = generate_triangular_data(*test_range)
    preds = neuron.predict(x)
    preds_binary = (preds > 0.5).astype(int)
    accuracy = np.mean(preds_binary == y)
    cm = confusion_matrix(y, preds_binary)
    if verbose:
        for n, pred, label in zip(x, preds_binary, y):
            print(f"Number: {n:5d}  Predicted: {'Triangular' if pred else 'Not Triangular'}  Actual: {'Triangular' if label else 'Not Triangular'}")
        print(f"Triangular neuron accuracy: {accuracy:.2%}")
        print(f"Confusion Matrix:\n{cm}")
    return accuracy, cm

def generate_catalan_data(start, end):
    x = np.arange(start, end)
    y = np.array([1 if is_catalan(n) else 0 for n in x])
    return x, y

def train_catalan_neuron(train_range=(0, 10000), learning_rate=0.01, epochs=20000):
    x, y = generate_catalan_data(*train_range)
    neuron = Neuron(inputSize=1)
    losses = neuron.train(x, y, learningRate=learning_rate, epochs=epochs)
    return neuron, losses

def test_catalan_neuron(neuron, test_range=(0, 1000), verbose=True):
    from sklearn.metrics import confusion_matrix
    x, y = generate_catalan_data(*test_range)
    preds = neuron.predict(x)
    preds_binary = (preds > 0.5).astype(int)
    accuracy = np.mean(preds_binary == y)
    cm = confusion_matrix(y, preds_binary)
    if verbose:
        for n, pred, label in zip(x, preds_binary, y):
            print(f"Number: {n:5d}  Predicted: {'Catalan' if pred else 'Not Catalan'}  Actual: {'Catalan' if label else 'Not Catalan'}")
        print(f"Catalan neuron accuracy: {accuracy:.2%}")
        print(f"Confusion Matrix:\n{cm}")
    return accuracy, cm
