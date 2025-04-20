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


def trainModel(trainRange=(0, 100), learningRate=0.01, epochs=20000, saveModel=True):
  """
    Train the neural network model.

    Args:
        trainRange: Tuple of (start, end) for training data range
        learningRate: Learning rate for training
        epochs: Number of training epochs
        saveModel: Whether to save the model parameters

    Returns:
        neuron: Trained neuron model
        losses: Training loss history
    """
  # Generate training data
  numbers, labels = generateTrainingData(trainRange[0], trainRange[1])
  print(f"Training on {len(numbers)} numbers from {trainRange[0]} to {trainRange[1]}")

  # Create and train the neuron
  neuron = Neuron(inputSize=1)
  losses = neuron.train(numbers, labels, learningRate=learningRate, epochs=epochs)

  # Save model parameters if requested
  if saveModel:
    os.makedirs('models', exist_ok=True)
    saveModelParams(neuron, 'models/odd_even_classifier.npz')

    return neuron, losses


def testModel(neuron, testCount=100, maxValue=1000000, includeNegatives=True):
  """
    Test the trained model on various numbers.

    Args:
        neuron: Trained neuron model
        testCount: Number of test examples
        maxValue: Maximum value for test numbers
        includeNegatives: Whether to test negative numbers (default: True)

    Returns:
        accuracy: Test accuracy
    """
  # Generate test data
  testNumbers = generateTestData(testCount, maxValue)

  # Add the negative nums if requested
  if includeNegatives:
    negatives = -np.random.randint(1, maxValue, size=testCount // 2)
    testNumbers = np.concatenate([testNumbers, negatives])

  # Add edge cases (0, 1, maxValue, etc.)
  edgeCases = np.array([0, 1, maxValue, -1, -maxValue])
  testNumbers = np.concatenate([testNumbers, edgeCases])

  # Shuffle the test numbers
  np.random.shuffle(testNumbers)

  # Make predictions
  predictions = neuron.predict(testNumbers)

  # Display results and get accuracy
  accuracy = displayTestResults(testNumbers, predictions)

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
    Main function to run the neural network training and testing.
    """
  # Parse command line arguments
  parser = argparse.ArgumentParser(description='Train and test a neural network for odd/even classification')
  parser.add_argument('--train', action='store_true', help='Train a new model')
  parser.add_argument('--test', action='store_true', help='Test the model')
  parser.add_argument('--visualize', action='store_true', help='Create visualizations')
  parser.add_argument('--epochs', type=int, default=20000, help='Number of training epochs')
  parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
  parser.add_argument('--test-count', type=int, default=15, help='Number of test examples')

  args = parser.parse_args()

  # If no specific actions are requested, do everything
  if not (args.train or args.test or args.visualize):
    args.train = args.test = args.visualize = True

    # Train or load the model
    if args.train:
      print("Training a new neural network model...")
      neuron, losses = trainModel(trainRange=(0, 100), 
                                  learningRate=args.lr, 
                                  epochs=args.epochs)
    else:
      # Load existing model
      print("Loading existing model...")
      neuron = Neuron(inputSize=1)
      neuron = loadModelParams(neuron, 'models/odd_even_classifier.npz')
      losses = []  # No losses if we're just loading the model

    # Test the model if requested
    if args.test:
      print("\nTesting the neural network...")
      testModel(neuron, testCount=args.test_count)

    # Create visualizations if requested and we have losses
    if args.visualize and len(losses) > 0:
      print("\nCreating visualizations...")
      visualizeResults(neuron, losses)


if __name__ == "__main__":
  main()
