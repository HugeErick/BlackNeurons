"""
Neural Network model implementation using NumPy to determine if a number is odd or even.
This implementation uses a single neuron with a sigmoid activation function.
"""

import numpy as np


class Neuron:
    def __init__(self, inputSize=1):
        """
        Initialize a single neuron with carefully initialized weights and bias.
        
        Args:
            inputSize: Number of input features (default is 1 for a single number)
        """
        # Initialize weights with small random values to prevent saturation
        """
        Smaller initial weights help prevent the sigmoid from saturating
        early in training, which can cause vanishing gradients.
        """
        self.weights = np.random.randn(inputSize) * 0.1
        
        # Initialize bias to zero for odd/even classification
        """
        Starting with a zero bias is often good practice as it allows
        the model to learn the appropriate bias during training.
        """
        self.bias = 0.0
        
    def sigmoid(self, x):
        """Sigmoid activation function"""
        """
        The sigmoid function is a non-linear activation function that maps the input values
        to a range between 0 and 1. This is useful for binary classification problems
        where the output of the neuron should be a probability between 0 and 1.
        """
        return 1 / (1 + np.exp(-x))
    
    def sigmoidDerivative(self, x):
        """Derivative of sigmoid function"""
        return x * (1 - x)
    
    def preprocessInput(self, x):
        """
        Preprocess the input number to extract useful features.
        For odd/even classification, we only need the least significant bit.
        
        Args:
            x: Input number or array of numbers
        
        Returns:
            Features for the neural network that emphasize odd/even property
        """
        # Convert to numpy array if not already
        x = np.array(x, dtype=float)
        
        # For odd/even classification, the key feature is the least significant bit (x % 2)
        # But we'll use a better representation that helps the network learn
        # We'll use both the original number (normalized) and its modulo 2 value
        xNormalized = x / np.max(np.abs(x) + 1)  # Better normalization
        xMod2 = x % 2  # This directly captures the odd/even property
        
        # Return a feature that emphasizes the odd/even property
        return xMod2.reshape(-1, 1)
    
    def forward(self, x):
        """
        Forward pass through the neuron.
        
        Args:
            x: Input features
            
        Returns:
            Output of the neuron (probability between 0 and 1)
        """
        # Calculate weighted sum plus bias
        # Reshape input if needed to handle both single values and arrays
        if len(x.shape) == 1:
            # For a batch of inputs, we need to reshape to column vector for proper dot product
            z = np.dot(x.reshape(-1, 1), self.weights.reshape(1, -1)) + self.bias
            return self.sigmoid(z.flatten())  # Flatten to match input shape
        else:
            # If already properly shaped
            z = np.dot(x, self.weights.reshape(1, -1)) + self.bias
            return self.sigmoid(z)
    
    def train(self, X, y, learningRate=0.1, epochs=10000):
        """
        Train the neuron using gradient descent.
        
        Args:
            X: Training inputs
            y: Target outputs (0 for even, 1 for odd)
            learningRate: Learning rate for gradient descent
            epochs: Number of training iterations
        
        Returns:
            List of loss values during training
        """

        """
        Preprocess the input data i.e normalize the input 
        i.e scale the input to a range between 0 and 1
        """
        X = self.preprocessInput(X)

        """
        Convert y to a numpy array and reshape it to a column vector
        else y will be a single value
        """
        y = np.array(y).reshape(-1, 1) if isinstance(y, (list, np.ndarray)) else y
        
        """
        Initialize a list to store the loss values
        """
        losses = []
        
        """
        Iterate over the number of epochs
        epochs in spanish is epocas I guess
        """
        for epoch in range(epochs):
            # Forward pass
            output = self.forward(X)
            
            """
            Ensure output and y have compatible shapes
            """
            if len(output.shape) == 1 and len(y.shape) > 1:
                output = output.reshape(-1, 1)
            elif len(y.shape) == 1 and len(output.shape) > 1:
                y = y.reshape(-1, 1)
            
            # Calculate loss
            loss = np.mean(np.square(y - output))
            losses.append(loss)
            
            # Backward pass
            error = y - output
            dOutput = error * self.sigmoidDerivative(output)
            
            # Ensure proper shapes for the update
            if len(X.shape) == 1:
                XReshaped = X.reshape(-1, 1)
                dOutputReshaped = dOutput.reshape(-1, 1) if len(dOutput.shape) == 1 else dOutput
                self.weights += learningRate * np.dot(XReshaped.T, dOutputReshaped).flatten()
            else:
                self.weights += learningRate * np.dot(X.T, dOutput).flatten()
            
            self.bias += learningRate * np.sum(dOutput)
            
            # Print progress every 1000 epochs
            if epoch % 1000 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.6f}")
                
        return losses
    
    def predict(self, x):
        """
        Predict whether a number is odd (1) or even (0).
        
        Args:
            x: Input number or array of numbers
            
        Returns:
            Predictions (1 for odd, 0 for even)
        """
        x = self.preprocessInput(x)
        predictions = self.forward(x)
        # Round to get binary output (0 or 1)
        return np.round(predictions)


if __name__ == "__main__":
    # Example usage if this file is run directly
    neuron = Neuron(inputSize=1)
    
    # Simple test
    testNumbers = [2, 3, 4, 5]
    processed = neuron.preprocessInput(testNumbers)
    print(f"Input numbers: {testNumbers}")
    print(f"Preprocessed input: {processed.flatten()}")
