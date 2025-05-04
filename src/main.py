#!/usr/bin/env python3
"""
Main script for the odd/even neural network classifier.
This refactored version uses modular components from separate files.
"""

import os
import argparse
import numpy as np

# Import our custom modules
from models.neuron import Neuron
from utils.dataUtils import generateTrainingData, generateTestData, saveModelParams, loadModelParams
from visualization.plots import plotTrainingLoss, plotNetworkAnalysis, displayTestResults
from utils.logging_utils import write_log
from utils.plot_utils import plot_training_loss, plot_confusion_matrix
from number_analysis import is_prime, get_multiples, get_divisors, is_perfect_number, is_perfect_square, is_perfect_cube, is_fibonacci
from property_neurons import (
    train_prime_neuron, test_prime_neuron,
    train_perfect_neuron, test_perfect_neuron,
    train_fibonacci_neuron, test_fibonacci_neuron
)


def train_model(train_range=(0, 100), learning_rate=0.01, epochs=20000, save_model=True):
    """
    Train the neural network model for odd/even classification.
    """
    numbers, labels = generateTrainingData(train_range[0], train_range[1])
    print(f"Training on {len(numbers)} numbers from {train_range[0]} to {train_range[1]}")
    neuron = Neuron(inputSize=1)
    losses = neuron.train(numbers, labels, learningRate=learning_rate, epochs=epochs)
    if save_model:
        os.makedirs('models', exist_ok=True)
        saveModelParams(neuron, 'models/odd_even_classifier.npz')
        print("Model parameters saved to 'models/odd_even_classifier.npz'")
    return neuron, losses


def test_model(neuron, test_count=100, max_value=1000000, include_negatives=True, analyze_numbers=False):
    """
    Test the trained odd/even neuron on various numbers.
    """
    test_numbers = generateTestData(test_count, max_value)
    if include_negatives:
        negatives = -np.random.randint(1, max_value, size=test_count // 2)
        test_numbers = np.concatenate([test_numbers, negatives])
    edge_cases = np.array([0, 1, max_value, -1, -max_value])
    test_numbers = np.concatenate([test_numbers, edge_cases])
    np.random.shuffle(test_numbers)
    predictions = neuron.predict(test_numbers)
    accuracy = displayTestResults(test_numbers, predictions)
    if analyze_numbers:
        print("\nNumber Analysis Results:")
        for n in test_numbers:
            print(f"\nNumber: {n}")
            print(f"  Prime: {is_prime(n)}")
            print(f"  Divisors: {get_divisors(n)}")
            print(f"  Multiples (first 5): {get_multiples(n, 5)}")
            print(f"  Perfect number: {is_perfect_number(n)}")
            print(f"  Perfect square: {is_perfect_square(n)}")
            print(f"  Perfect cube: {is_perfect_cube(n)}")
            print(f"  Fibonacci: {is_fibonacci(n)}")
    return accuracy


def visualizeResults(neuron, losses):
  """
    Create visualizations for the model training and performance.

    Args:
        neuron: Trained neuron model
        losses: Training loss history
    """
  # Create output directory if it doesn't exist
  os.makedirs('plots', exist_ok=True)

  # Plot training loss
  plotTrainingLoss(losses, 'plots/training_loss.png')

  # Plot network analysis
  plotNetworkAnalysis(neuron, 'plots/network_analysis.png')


def main():
    """
    Main function to run all neural network classifiers for number properties.
    """
    import importlib.util
    if importlib.util.find_spec('sklearn') is None:
        print("WARNING: scikit-learn is required for confusion matrix support. Please install it with 'pip install scikit-learn'.")
    parser = argparse.ArgumentParser(description='Train and test neural networks for number properties')
    parser.add_argument('--epochs', type=int, default=20000, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--test-count', type=int, default=1000, help='Number of test examples for each property')
    args = parser.parse_args()

    # --- Odd/Even Neuron ---
    print("\n--- Training and Testing Odd/Even Neuron ---")
    odd_even_neuron, odd_even_losses = train_model(train_range=(0, 100), learning_rate=args.lr, epochs=args.epochs)
    plot_training_loss(odd_even_losses, 'plots/odd_even_training_loss.png', title='Odd/Even Training Loss')
    plotNetworkAnalysis(odd_even_neuron, 'plots/odd_even_network_analysis.png')
    # Test and log
    import io
    odd_even_log = io.StringIO()
    import sys
    sys_stdout = sys.stdout
    sys.stdout = odd_even_log
    odd_even_accuracy = test_model(odd_even_neuron, test_count=args.test_count)
    sys.stdout = sys_stdout
    write_log('loggedOutputs/odd_even_results.txt', odd_even_log.getvalue())

    # --- Prime Neuron ---
    print("\n--- Training and Testing Prime Neuron ---")
    prime_neuron, prime_losses = train_prime_neuron(train_range=(0, 10000), learning_rate=args.lr, epochs=args.epochs)
    plot_training_loss(prime_losses, 'plots/prime_training_loss.png', title='Prime Training Loss')
    plotNetworkAnalysis(prime_neuron, 'plots/prime_network_analysis.png')
    # Test and log with multiple batches
    import numpy as np
    from sklearn.metrics import confusion_matrix
    prime_accuracies = []
    prime_cms = []
    prime_log = io.StringIO()
    sys.stdout = prime_log
    for batch in range(5):
        acc, cm = test_prime_neuron(prime_neuron, test_range=(0, 10000), verbose=False)
        prime_accuracies.append(acc)
        prime_cms.append(cm)
        print(f"Batch {batch+1}: accuracy={acc:.2%}, confusion matrix=\n{cm}\n")
    sys.stdout = sys_stdout
    mean_acc = np.mean(prime_accuracies)
    sum_cm = np.sum(prime_cms, axis=0)
    print(f"Prime neuron mean accuracy: {mean_acc:.2%}\nAggregate confusion matrix:\n{sum_cm}")
    write_log('loggedOutputs/prime_results.txt', prime_log.getvalue())
    plot_confusion_matrix(sum_cm, 'plots/prime_confusion_matrix.png', labels=['Not Prime', 'Prime'])
    # Save model
    os.makedirs('models', exist_ok=True)
    saveModelParams(prime_neuron, 'models/prime_classifier.npz')

    # --- Perfect Number Neuron ---
    print("\n--- Training and Testing Perfect Number Neuron ---")
    perfect_neuron, perfect_losses = train_perfect_neuron(train_range=(0, 10000), learning_rate=args.lr, epochs=args.epochs)
    plot_training_loss(perfect_losses, 'plots/perfect_training_loss.png', title='Perfect Number Training Loss')
    plotNetworkAnalysis(perfect_neuron, 'plots/perfect_network_analysis.png')
    # Save model
    os.makedirs('models', exist_ok=True)
    saveModelParams(perfect_neuron, 'models/perfect_classifier.npz')
    # Test and log
    perfect_log = io.StringIO()
    sys.stdout = perfect_log
    perfect_accuracy, perfect_cm = test_perfect_neuron(perfect_neuron, test_range=(0, args.test_count))
    sys.stdout = sys_stdout
    write_log('loggedOutputs/perfect_results.txt', perfect_log.getvalue())
    plot_confusion_matrix(perfect_cm, 'plots/perfect_confusion_matrix.png', labels=['Not Perfect', 'Perfect'])

    # --- Fibonacci Neuron ---
    print("\n--- Training and Testing Fibonacci Neuron ---")
    fibonacci_neuron, fibonacci_losses = train_fibonacci_neuron(train_range=(0, 1000), learning_rate=args.lr, epochs=args.epochs)
    plot_training_loss(fibonacci_losses, 'plots/fibonacci_training_loss.png', title='Fibonacci Training Loss')
    plotNetworkAnalysis(fibonacci_neuron, 'plots/fibonacci_network_analysis.png')
    # Save model
    os.makedirs('models', exist_ok=True)
    saveModelParams(fibonacci_neuron, 'models/fibonacci_classifier.npz')
    # Test and log
    fibonacci_log = io.StringIO()
    sys.stdout = fibonacci_log
    fibonacci_accuracy, fibonacci_cm = test_fibonacci_neuron(fibonacci_neuron, test_range=(0, args.test_count))
    sys.stdout = sys_stdout
    write_log('loggedOutputs/fibonacci_results.txt', fibonacci_log.getvalue())
    plot_confusion_matrix(fibonacci_cm, 'plots/fibonacci_confusion_matrix.png', labels=['Not Fibonacci', 'Fibonacci'])

    # --- Summary ---
    print("\n==================== SUMMARY ====================")
    print(f"Prime neuron mean accuracy: {mean_acc:.2%}\nAggregate confusion matrix:\n{sum_cm}")
    print(f"\nPerfect neuron accuracy: {perfect_accuracy:.2%}\nConfusion matrix:\n{perfect_cm}")
    print(f"\nFibonacci neuron accuracy: {fibonacci_accuracy:.2%}\nConfusion matrix:\n{fibonacci_cm}")
    print("================================================\n")


if __name__ == "__main__":
  main()
