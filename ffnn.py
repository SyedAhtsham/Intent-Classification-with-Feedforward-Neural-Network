import numpy as np
import numpy.typing as npt

from model.model_utils import softmax, relu, relu_prime
from typing import Tuple


class NeuralNetwork(object):
    def __init__(
        self, 
        input_size: int,
        hidden_size: int, 
        num_classes: int,
        seed: int = 1
    ):
        """
        Initialize neural network's weights and biases.
        """
        ############################# STUDENT SOLUTION ####################
        # YOUR CODE HERE
        #     TODO:
        #         1) Set a seed so that your model is reproducible
        #         2) Initialize weight matrices and biases with uniform
        #         distribution in the range (-1, 1).

        np.random.seed(seed)

        # Initialize weights and biases with uniform distribution in the range (-1, 1)
        self.W1 = np.random.uniform(-1, 1, (input_size, hidden_size))

        self.b1 = np.random.uniform(-1, 1, (hidden_size, 1))

        self.W2 = np.random.uniform(-1, 1, (num_classes, hidden_size))
        self.b2 = np.random.uniform(-1, 1, (num_classes, 1))

        # print(self.W1)

        ###################################################################

    def forward(self, X: npt.ArrayLike) -> npt.ArrayLike:
        """
        Forward pass with X as input matrix, returning the model prediction
        Y_hat.
        """
        ######################### STUDENT SOLUTION #########################
        # YOUR CODE HERE
        #     TODO:
        #         1) Perform only a forward pass with X as input.

        # Input layer to hidden layer
        # print(self.W1.shape)
        # print(self.b1.shape)
        # print("Weight1 shape:", self.W1.shape)
        # print(np.array(self.W1).T.shape)

        z1 = np.dot(np.array(self.W1).T, X) + self.b1
        a1 = relu(z1)
        z2 = np.dot(self.W2, a1) + self.b2
        a2 = softmax(z2)

        # print(np.dot(a1, np.array(self.W2).T).shape)
        # print(self.b2.shape)
        # print(a2)
        # print(a2.shape)
        # print(a2)




        return a2
        #####################################################################



    def predict(self, X: npt.ArrayLike) -> npt.ArrayLike:
        """
        Create a prediction matrix with `self.forward()`
        """
        ######################### STUDENT SOLUTION ###########################
        # YOUR CODE HERE
        #     TODO:
        #         1) Create a prediction matrix of the intent data using
        #         `self.forward()` function. The shape of prediction matrix
        #         should be similar to label matrix produced with
        #         `labels_matrix()`
        # Perform forward pass to obtain probability distribution

        predictions = self.forward(X)
        argmax_indices = np.argmax(predictions, axis=0)

        prediction_matrix = [np.eye(7)[index] for index in argmax_indices]

        return np.array(prediction_matrix).T
        ######################################################################

    def backward(
        self, 
        X: npt.ArrayLike, 
        Y: npt.ArrayLike
    ) -> Tuple[npt.ArrayLike, npt.ArrayLike, npt.ArrayLike, npt.ArrayLike]:
        """
        Backpropagation algorithm.
        """
        ########################## STUDENT SOLUTION ###########################
        # YOUR CODE HERE
        #     TODO:
        #         1) Perform forward pass, then backpropagation
        #         to get gradient for weight matrices and biases
        #         2) Return the gradient ftrainor weight matrices and biases
        # Forward pass
        z1 = np.dot(np.array(self.W1).T, X) + self.b1
        a1 = relu(z1)
        z2 = np.dot(self.W2, a1) + self.b2

        a2 = softmax(z2)

        # Calculate loss
        loss = compute_loss(a2, Y)

        # Backward pass

        # Output layer error term
        delta2 = a2 - Y

        # Hidden layer error term
        delta1 = np.dot(self.W2.T, delta2) * relu_prime(z1)

        # Compute gradients
        dW2 = np.dot(delta2, a1.T)
        db2 = np.sum(delta2, axis=1, keepdims=True)

        dW1 = np.dot(delta1, X.T)
        db1 = np.sum(delta1, axis=1, keepdims=True)



        return dW1, db1, dW2, db2
        #######################################################################


def compute_loss(pred: npt.ArrayLike, truth: npt.ArrayLike) -> float:
    """
    Compute the cross entropy loss.
    """


    # Calculate cross-entropy loss
    loss =  pred - truth


    return loss
    ############################################################################