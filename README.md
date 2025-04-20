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
- Accuracy evaluation on test data

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

Run the main script:
```bash
python src/main.py
```

This will train the neural network and display the results, including a visualization of the training loss and predictions on test data.
