"""
Utility functions for data preparation and processing.
"""

import numpy as np


def generateTrainingData(start=0, end=100):
  """
    Generate training data for odd/even classification.

    Args:
        start: Start of the range (inclusive)
        end: End of the range (exclusive)

    Returns:
        numbers: Array of numbers
        labels: Array of labels (0 for even, 1 for odd)
    """
  numbers = np.arange(start, end + 1)
  labels = numbers % 2
  return numbers, labels


def generateTestData(count=10, maxValue=10000, includeEdgeCases=True):
  """
    Generate diverse test data for odd/even classification.

    Args:
        count: Number of test samples
        maxValue: Maximum value for random numbers

    Returns:
        testNumbers: Array of test numbers
    """
  # Include some specific numbers for testing
  edgeCases = [0, 1, 2, -1, -2, maxValue, -maxValue]

  # Generate random numbers for the rest
  randomPositives = np.random.randint(1, maxValue, size=count)
  randomNegatives = -np.random.randint(1, maxValue, size=count)

  # Combine specific and random numbers
  testNumbers = np.concatenate([edgeCases, randomPositives, randomNegatives])

  # Shuffle the test numbers
  np.random.shuffle(testNumbers)

  return testNumbers[:count * 3]


def saveModelParams(neuron, filepath='model_params.npz'):
  """
    Save the model parameters to a file.

    Args:
        neuron: Trained neuron instance
        filepath: Path to save the parameters
    """
  np.savez(filepath, weights=neuron.weights, bias=neuron.bias)
  print(f"Model parameters saved to '{filepath}'")


def loadModelParams(neuron, filepath='model_params.npz'):
  """
    Load model parameters from a file.

    Args:
        neuron: Neuron instance to load parameters into
        filepath: Path to load parameters from

    Returns:
        neuron: Neuron instance with loaded parameters
    """
  try:
    params = np.load(filepath)
    neuron.weights = params['weights']
    neuron.bias = params['bias']
    print(f"Model parameters loaded from '{filepath}'")
  except FileNotFoundError:
    print(f"Warning: Parameter file '{filepath}' not found. Using random initialization.")
  except Exception as e:
    print(f"Error loading parameters: {str(e)}")
  return neuron


if __name__ == "__main__":
  # Example usage if this file is run directly
  trainNumbers, trainLabels = generateTrainingData(0, 20)
  print("Training data sample:")
  for num, label in zip(trainNumbers[:5], trainLabels[:5]):
    print(f"Number: {num}, Label: {label} ({'odd' if label else 'even'})")

  testNumbers = generateTestData(5, 100)
  print("\nTest data sample:")
  for num in testNumbers:
    print(f"Number: {num}, Expected: {'odd' if num % 2 else 'even'}")
