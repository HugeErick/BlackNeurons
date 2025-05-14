#!/usr/bin/env python3
"""
Main script for the odd/even neural network classifier.
This refactored version uses modular components from separate files.
"""

import os
import argparse
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
import matplotlib.pyplot as plt

# Import our custom modules
from models.neuron import Neuron
from utils.dataUtils import generateTrainingData, generateTestData, saveModelParams, loadModelParams
from visualization.plots import plotTrainingLoss, plotNetworkAnalysis, displayTestResults
from utils.logging_utils import write_log
from utils.plot_utils import plot_training_loss, plot_confusion_matrix
from number_analysis import is_prime, get_multiples, get_divisors, is_perfect_number, is_perfect_square, is_perfect_cube, is_fibonacci
from property_neurons import (
    train_prime_neuron, test_prime_neuron, generate_prime_data,
    train_perfect_neuron, test_perfect_neuron, generate_perfect_data,
    train_fibonacci_neuron, test_fibonacci_neuron, generate_fibonacci_data,
    train_palindrome_neuron, test_palindrome_neuron, generate_palindrome_data,
    train_armstrong_neuron, test_armstrong_neuron, generate_armstrong_data,
    train_harshad_neuron, test_harshad_neuron, generate_harshad_data,
    train_square_free_neuron, test_square_free_neuron, generate_square_free_data,
    train_abundant_neuron, test_abundant_neuron, generate_abundant_data,
    train_deficient_neuron, test_deficient_neuron, generate_deficient_data,
    train_happy_neuron, test_happy_neuron, generate_happy_data,
    train_triangular_neuron, test_triangular_neuron, generate_triangular_data,
    train_catalan_neuron, test_catalan_neuron, generate_catalan_data
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


def plot_metrics_bar(metrics_dict, save_path, title):
    import matplotlib.pyplot as plt
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(6, 4))
    names = list(metrics_dict.keys())
    values = list(metrics_dict.values())
    plt.bar(names, values, color=['#4CAF50', '#2196F3', '#FFC107'])
    plt.ylim(0, 1)
    plt.title(title)
    plt.ylabel('Score')
    for i, v in enumerate(values):
        plt.text(i, v + 0.02, f"{v:.2f}", ha='center')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


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
    import matplotlib.pyplot as plt
    import os
    from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, classification_report
    import numpy as np
    if importlib.util.find_spec('sklearn') is None:
        print("WARNING: scikit-learn is required for confusion matrix support. Please install it with 'pip install scikit-learn'.")
    parser = argparse.ArgumentParser(description='Train and test neural networks for number properties')
    parser.add_argument('--epochs', type=int, default=20000, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--test_count', type=int, default=1000, help='Number of test samples')
    parser.add_argument('--analyze', action='store_true', help='Analyze numbers in test set')

    args = parser.parse_args()
    os.makedirs('plots', exist_ok=True)

    # List of all properties
    properties = [
        # name, train_fn, test_fn, data_fn, label0, label1
        ("Odd/Even", train_model, test_model, None, "Even", "Odd"),
        ("Prime", train_prime_neuron, test_prime_neuron, generate_prime_data, "Not Prime", "Prime"),
        ("Perfect", train_perfect_neuron, test_perfect_neuron, generate_perfect_data, "Not Perfect", "Perfect"),
        ("Fibonacci", train_fibonacci_neuron, test_fibonacci_neuron, generate_fibonacci_data, "Not Fibonacci", "Fibonacci"),
        ("Palindrome", train_palindrome_neuron, test_palindrome_neuron, generate_palindrome_data, "Not Palindrome", "Palindrome"),
        ("Armstrong", train_armstrong_neuron, test_armstrong_neuron, generate_armstrong_data, "Not Armstrong", "Armstrong"),
        ("Harshad", train_harshad_neuron, test_harshad_neuron, generate_harshad_data, "Not Harshad", "Harshad"),
        ("Square-Free", train_square_free_neuron, test_square_free_neuron, generate_square_free_data, "Not Square-Free", "Square-Free"),
        ("Abundant", train_abundant_neuron, test_abundant_neuron, generate_abundant_data, "Not Abundant", "Abundant"),
        ("Deficient", train_deficient_neuron, test_deficient_neuron, generate_deficient_data, "Not Deficient", "Deficient"),
        ("Happy", train_happy_neuron, test_happy_neuron, generate_happy_data, "Not Happy", "Happy"),
        ("Triangular", train_triangular_neuron, test_triangular_neuron, generate_triangular_data, "Not Triangular", "Triangular"),
        ("Catalan", train_catalan_neuron, test_catalan_neuron, generate_catalan_data, "Not Catalan", "Catalan"),
    ]

    summary = []
    for prop in properties:
        name, train_fn, test_fn, data_fn, label0, label1 = prop
        print(f"\n--- Training and Testing {name} Neuron ---")
        if name == "Odd/Even":
            neuron, losses = train_fn(train_range=(0, 100), learning_rate=args.lr, epochs=args.epochs)
            accuracy = test_fn(neuron, test_count=args.test_count)
            # Odd/Even: metrics are not binary classification, skip for now
            continue
        neuron, losses = train_fn(train_range=(0, 10000), learning_rate=args.lr, epochs=args.epochs)
        x, y = data_fn(0, args.test_count)
        preds = neuron.predict(x)
        # Use threshold 0.1 for rare classes, 0.5 for others
        if name in ["Prime", "Perfect", "Fibonacci", "Armstrong", "Abundant", "Deficient", "Happy", "Triangular", "Catalan"]:
            threshold = 0.1
        else:
            threshold = 0.5
        preds_binary = (preds > threshold).astype(int)
        accuracy = np.mean(preds_binary == y)
        cm = confusion_matrix(y, preds_binary)
        p = precision_score(y, preds_binary, zero_division=0)
        r = recall_score(y, preds_binary, zero_division=0)
        f1 = f1_score(y, preds_binary, zero_division=0)
        print(f"{name} Precision: {p:.3f}, Recall: {r:.3f}, F1: {f1:.3f}")
        print(f"Confusion Matrix:\n{cm}")
        print(classification_report(y, preds_binary, zero_division=0, target_names=[label0, label1]))
        # Save metrics bar plot
        plot_metrics_bar({'precision': p, 'recall': r, 'f1': f1}, f'plots/{name.lower().replace("/", "_")}_metrics.png', f'{name} Metrics')
        # Save confusion matrix heatmap
        plt.figure(figsize=(4, 3))
        plt.imshow(cm, cmap='Blues', interpolation='nearest')
        plt.title(f'{name} Confusion Matrix')
        plt.colorbar()
        tick_marks = np.arange(2)
        plt.xticks(tick_marks, [label0, label1])
        plt.yticks(tick_marks, [label0, label1])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        for i in range(2):
            for j in range(2):
                plt.text(j, i, cm[i, j], ha='center', va='center', color='black')
        plt.tight_layout()
        plt.savefig(f'plots/{name.lower().replace("/", "_")}_confusion.png')
        plt.close()
        # ROC Curve
        from sklearn.metrics import roc_curve, auc, precision_recall_curve
        if len(np.unique(y)) == 2 and np.any(preds != preds[0]):  # Only plot ROC/PR if both classes present
            fpr, tpr, _ = roc_curve(y, preds)
            roc_auc = auc(fpr, tpr)
            plt.figure(figsize=(5, 4))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'{name} ROC Curve')
            plt.legend(loc="lower right")
            plt.tight_layout()
            plt.savefig(f'plots/{name.lower().replace("/", "_")}_roc.png')
            plt.close()
            # Precision-Recall Curve
            precision, recall, _ = precision_recall_curve(y, preds)
            plt.figure(figsize=(5, 4))
            plt.plot(recall, precision, color='purple', lw=2)
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title(f'{name} Precision-Recall Curve')
            plt.tight_layout()
            plt.savefig(f'plots/{name.lower().replace("/", "_")}_pr.png')
            plt.close()
        # Learning curve (training loss vs. epoch)
        if losses is not None:
            plt.figure(figsize=(5, 4))
            plt.plot(losses, color='green', lw=2)
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title(f'{name} Learning Curve')
            plt.tight_layout()
            plt.savefig(f'plots/{name.lower().replace("/", "_")}_learning_curve.png')
            plt.close()
        summary.append([name, accuracy, p, r, f1])

    # Print summary table
    print("\n==================== SUMMARY TABLE ====================")
    print(f"{'Property':<15} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>6}")
    for row in summary:
        print(f"{row[0]:<15} {row[1]:9.3f} {row[2]:10.3f} {row[3]:8.3f} {row[4]:6.3f}")
    print("=====================================================")
    parser.add_argument('--predict', nargs='*', type=int, help='Predict properties for the given numbers (space-separated)')
    args = parser.parse_args()

    # If --predict is used, perform predictions and exit
    if args.predict is not None and len(args.predict) > 0:
        numbers = np.array(args.predict)
        print(f"\nPredictions for input numbers: {numbers.tolist()}")
        # Load all trained models (assuming they exist)
        odd_even_neuron = Neuron(inputSize=1)
        prime_neuron = Neuron(inputSize=1)
        perfect_neuron = Neuron(inputSize=1)
        fibonacci_neuron = Neuron(inputSize=1)
        loadModelParams(odd_even_neuron, 'models/odd_even_classifier.npz')
        loadModelParams(prime_neuron, 'models/prime_classifier.npz')
        loadModelParams(perfect_neuron, 'models/perfect_classifier.npz')
        loadModelParams(fibonacci_neuron, 'models/fibonacci_classifier.npz')
        # Predict
        odd_even_preds = odd_even_neuron.predict(numbers)
        prime_preds = prime_neuron.predict(numbers)
        perfect_preds = perfect_neuron.predict(numbers)
        fibonacci_preds = fibonacci_neuron.predict(numbers)
        # Print results as table
        print("\nNumber | Odd/Even | Prime | Perfect | Fibonacci")
        print("-"*50)
        for i, n in enumerate(numbers):
            oe = 'Odd' if odd_even_preds[i] == 1 else 'Even'
            pr = 'Prime' if prime_preds[i] == 1 else 'Not Prime'
            pf = 'Perfect' if perfect_preds[i] == 1 else 'Not Perfect'
            fib = 'Fibonacci' if fibonacci_preds[i] == 1 else 'Not Fibonacci'
            print(f"{n:>6} | {oe:^8} | {pr:^11} | {pf:^12} | {fib:^12}")
        print()
        # Compute and print metrics for each model
        from sklearn.metrics import precision_score, recall_score, f1_score
        x_odd_even, y_odd_even = generateTrainingData(0, 100)
        x_prime, y_prime = generate_prime_data(0, 100)
        x_perfect, y_perfect = generate_perfect_data(0, 100)
        x_fib, y_fib = generate_fibonacci_data(0, 100)
        odd_even_preds = odd_even_neuron.predict(x_odd_even)
        prime_preds = prime_neuron.predict(x_prime)
        perfect_preds = perfect_neuron.predict(x_perfect)
        fibonacci_preds = fibonacci_neuron.predict(x_fib)
        p_odd_even = precision_score(y_odd_even, odd_even_preds)
        r_odd_even = recall_score(y_odd_even, odd_even_preds)
        f1_odd_even = f1_score(y_odd_even, odd_even_preds)
        p_prime = precision_score(y_prime, prime_preds)
        r_prime = recall_score(y_prime, prime_preds)
        f1_prime = f1_score(y_prime, prime_preds)
        p_perfect = precision_score(y_perfect, perfect_preds)
        r_perfect = recall_score(y_perfect, perfect_preds)
        f1_perfect = f1_score(y_perfect, perfect_preds)
        p_fib = precision_score(y_fib, fibonacci_preds)
        r_fib = recall_score(y_fib, fibonacci_preds)
        f1_fib = f1_score(y_fib, fibonacci_preds)
        print(f"Odd/Even Precision: {p_odd_even:.3f}, Recall: {r_odd_even:.3f}, F1: {f1_odd_even:.3f}")
        print(f"Prime Precision: {p_prime:.3f}, Recall: {r_prime:.3f}, F1: {f1_prime:.3f}")
        print(f"Perfect Precision: {p_perfect:.3f}, Recall: {r_perfect:.3f}, F1: {f1_perfect:.3f}")
        print(f"Fibonacci Precision: {p_fib:.3f}, Recall: {r_fib:.3f}, F1: {f1_fib:.3f}")
        plot_metrics_bar({'precision': p_odd_even, 'recall': r_odd_even, 'f1': f1_odd_even}, 'plots/odd_even_metrics.png', 'Odd/Even Metrics')
        plot_metrics_bar({'precision': p_prime, 'recall': r_prime, 'f1': f1_prime}, 'plots/prime_metrics.png', 'Prime Metrics')
        plot_metrics_bar({'precision': p_perfect, 'recall': r_perfect, 'f1': f1_perfect}, 'plots/perfect_metrics.png', 'Perfect Metrics')
        plot_metrics_bar({'precision': p_fib, 'recall': r_fib, 'f1': f1_fib}, 'plots/fibonacci_metrics.png', 'Fibonacci Metrics')
        return

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
    # Compute and print metrics for prime
    from sklearn.metrics import precision_score, recall_score, f1_score
    x_prime, y_prime = generate_prime_data(0, args.test_count)
    # Use a lower threshold for rare classes
    prime_probs = prime_neuron.forward(prime_neuron.preprocessInput(x_prime))
    preds_prime = (prime_probs > 0.1).astype(int)
    print(f"Prime: predicted positives: {np.sum(preds_prime)}, total samples: {len(preds_prime)}")
    print(f"Prime: output probability range: min={prime_probs.min():.3f}, max={prime_probs.max():.3f}")
    p_prime = precision_score(y_prime, preds_prime, zero_division=0)
    r_prime = recall_score(y_prime, preds_prime, zero_division=0)
    f1_prime = f1_score(y_prime, preds_prime, zero_division=0)
    print(classification_report(y_prime, preds_prime, zero_division=0, target_names=["Not Prime", "Prime"]))
    if np.sum(preds_prime) == 0:
        print("WARNING: No positive predictions for prime. Consider improving model or dataset.")
    print(f"Prime Precision: {p_prime:.3f}, Recall: {r_prime:.3f}, F1: {f1_prime:.3f}")
    plot_metrics_bar({'precision': p_prime, 'recall': r_prime, 'f1': f1_prime}, 'plots/prime_metrics.png', 'Prime Metrics')
    print(f"\nPerfect neuron accuracy: {perfect_accuracy:.2%}\nConfusion matrix:\n{perfect_cm}")
    # Compute and print metrics for perfect
    x_perfect, y_perfect = generate_perfect_data(0, args.test_count)
    perfect_probs = perfect_neuron.forward(perfect_neuron.preprocessInput(x_perfect))
    preds_perfect = (perfect_probs > 0.1).astype(int)
    print(f"Perfect: predicted positives: {np.sum(preds_perfect)}, total samples: {len(preds_perfect)}")
    print(f"Perfect: output probability range: min={perfect_probs.min():.3f}, max={perfect_probs.max():.3f}")
    p_perfect = precision_score(y_perfect, preds_perfect, zero_division=0)
    r_perfect = recall_score(y_perfect, preds_perfect, zero_division=0)
    f1_perfect = f1_score(y_perfect, preds_perfect, zero_division=0)
    print(classification_report(y_perfect, preds_perfect, zero_division=0, target_names=["Not Perfect", "Perfect"]))
    if np.sum(preds_perfect) == 0:
        print("WARNING: No positive predictions for perfect. Consider improving model or dataset.")
    print(f"Perfect Precision: {p_perfect:.3f}, Recall: {r_perfect:.3f}, F1: {f1_perfect:.3f}")
    plot_metrics_bar({'precision': p_perfect, 'recall': r_perfect, 'f1': f1_perfect}, 'plots/perfect_metrics.png', 'Perfect Metrics')
    print(f"\nFibonacci neuron accuracy: {fibonacci_accuracy:.2%}\nConfusion matrix:\n{fibonacci_cm}")
    # Compute and print metrics for fibonacci
    x_fib, y_fib = generate_fibonacci_data(0, args.test_count)
    fib_probs = fibonacci_neuron.forward(fibonacci_neuron.preprocessInput(x_fib))
    preds_fib = (fib_probs > 0.1).astype(int)
    print(f"Fibonacci: predicted positives: {np.sum(preds_fib)}, total samples: {len(preds_fib)}")
    print(f"Fibonacci: output probability range: min={fib_probs.min():.3f}, max={fib_probs.max():.3f}")
    p_fib = precision_score(y_fib, preds_fib, zero_division=0)
    r_fib = recall_score(y_fib, preds_fib, zero_division=0)
    f1_fib = f1_score(y_fib, preds_fib, zero_division=0)
    print(classification_report(y_fib, preds_fib, zero_division=0, target_names=["Not Fibonacci", "Fibonacci"]))
    if np.sum(preds_fib) == 0:
        print("WARNING: No positive predictions for Fibonacci. Consider improving model or dataset.")
    print(f"Fibonacci Precision: {p_fib:.3f}, Recall: {r_fib:.3f}, F1: {f1_fib:.3f}")
    plot_metrics_bar({'precision': p_fib, 'recall': r_fib, 'f1': f1_fib}, 'plots/fibonacci_metrics.png', 'Fibonacci Metrics')
    print("================================================\n")


if __name__ == "__main__":
  main()
