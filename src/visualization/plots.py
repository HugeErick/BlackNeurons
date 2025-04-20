"""
Visualization utilities for the neural network.
"""

import os
import numpy as np
import matplotlib.pyplot as plt


def plotTrainingLoss(losses, savePath='training_loss.png'):
    """
    Plot the training loss curve.
    
    Args:
        losses: List of loss values during training
        savePath: Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    # Skip first 100 epochs for better visualization if there are enough epochs
    startIdx = min(100, len(losses) // 10) if len(losses) > 100 else 0
    plt.plot(losses[startIdx:])
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(savePath) if os.path.dirname(savePath) else '.', exist_ok=True)
    plt.savefig(savePath)
    plt.close()
    
    print(f"Training loss plot saved to '{savePath}'")


def plotNetworkAnalysis(neuron, savePath='network_analysis.png'):
    """
    Visualize the network's decision boundary and parameters.
    
    Args:
        neuron: Trained neuron instance
        savePath: Path to save the plot
    """
    plt.figure(figsize=(12, 6))
    
    # Generate a range of numbers to visualize
    vizNumbers = np.arange(0, 50)
    vizPredictions = neuron.forward(neuron.preprocessInput(vizNumbers))
    
    plt.subplot(1, 2, 1)
    # Flatten the predictions to make them compatible with bar plot
    plt.bar(vizNumbers, vizPredictions.flatten())
    plt.axhline(y=0.5, color='r', linestyle='--')
    plt.title('Network Output for Numbers 0-49')
    plt.xlabel('Number')
    plt.ylabel('Network Output (>0.5 = Odd)')
    
    # Plot weights and bias separately to avoid array dimension issues
    plt.subplot(1, 2, 2)
    plt.bar(['Weight'], [float(neuron.weights[0])])
    plt.bar(['Bias'], [float(neuron.bias)])
    plt.title('Network Parameters')
    plt.ylabel('Value')
    
    plt.tight_layout()
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(savePath) if os.path.dirname(savePath) else '.', exist_ok=True)
    plt.savefig(savePath)
    plt.close()
    
    print(f"Network analysis visualizations saved to '{savePath}'")


def displayTestResults(testNumbers, predictions):
    """
    Display the test results in a formatted way.
    
    Args:
        testNumbers: List of numbers to test
        predictions: Predictions from the neural network
    
    Returns:
        accuracy: The accuracy of the predictions
    """
    print("\nTesting the neural network:")
    for num, pred in zip(testNumbers, predictions):
        expected = "odd" if num % 2 else "even"
        predicted = "odd" if pred == 1 else "even"
        correct = "✓" if expected == predicted else "✗"
        print(f"Number: {num}, Predicted: {predicted}, Actual: {expected} {correct}")
    
    # Calculate accuracy on test set
    correctPredictions = predictions.flatten() == (np.array(testNumbers) % 2)
    accuracy = np.mean(correctPredictions)
    print(f"\nAccuracy on test set: {accuracy * 100:.2f}%")
    
    return accuracy


if __name__ == "__main__":
    # Example usage if this file is run directly
    from src.models.neuron import Neuron
    
    # Create a sample neuron and some fake data for demonstration
    neuron = Neuron()
    neuron.weights = np.array([0.5])
    neuron.bias = 0.1
    
    # Generate fake loss data
    fakeLosses = [np.exp(-i/100) for i in range(1000)]
    
    # Plot the fake loss data
    plotTrainingLoss(fakeLosses, 'demo_loss.png')
    
    # Plot network analysis
    plotNetworkAnalysis(neuron, 'demo_analysis.png')
