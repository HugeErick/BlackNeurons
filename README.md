# BlackNeurons

Sample of some neural networks to understand the structure of themselfsand the implementation was using NumPy to determine several things on aa number

## Project Structure

```
BlackNeurons/
├── src/
│   └── main.py      # Main implementation of the neural network
├── requirements.txt # Project dependencies
└── README.md        # This file
```

## Description

This project implements a simple neural network from scratch using only NumPy. The neural network consists of a single neuron that learns to classify numbers as odd or even.

## Features

- Pure NumPy implementation without TensorFlow or other deep learning frameworks
- Single neuron with sigmoid activation function
- Training visualization with matplotlib
- Simple preprocessing of input data

## Requirements

- Python 3.6+
- NumPy
- Matplotlib

## Setup and Installation

1. Create a virtual environment:
```bash
python -m venv venv
```

2. Activate the virtual environment:
```bash
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

2. Run the main script:
   ```sh
   python src/main.py
   ```

3. Outputs and plots will be saved in `loggedOutputs/` and `plots/` directories.

## Evaluation Improvements
- **Larger Ranges:** Neurons (especially for prime, perfect, and Fibonacci) are now trained and tested on a much larger range (up to 10,000 by default).
- **Multiple Batches:** For each property, 5 randomized test batches are run and results are aggregated (mean accuracy, summed confusion matrix) for a more robust evaluation.
- **Class Imbalance:** Some properties (especially primes) are rare among large numbers, making them difficult for a simple neuron to learn. This is reflected in the confusion matrix and accuracy. For best results, consider exploring class balancing or more advanced architectures.

## Interpreting Results
- **Logs:** Detailed results for each neuron (including all test batches) are saved in `loggedOutputs/{property}_results.txt`.
- **Plots:** Training loss curves, network analysis, and confusion matrices are saved in `plots/`.
- **Prime Neuron:** Due to class imbalance, the prime neuron may have high accuracy but poor recall for actual primes. Check the confusion matrix for true/false positives/negatives.

## Customization
- You can adjust the training/testing range and number of batches in `src/main.py`.
- All code follows Python snake_case conventions.
