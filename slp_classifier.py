"""
SimpleSLPClassifier - Single Layer Perceptron for Classification
"""

from typing import Optional, Tuple
import numpy as np
from numpy.typing import NDArray
from activations import softmax
from slp_base import BaseSLPEstimator


class SimpleSLPClassifier(BaseSLPEstimator):
    """
    Simple Single Layer Perceptron Classifier with one hidden layer.

    Compatible interface with sklearn.neural_network.MLPClassifier.
    """

    def __init__(
        self,
        hidden_layer_size: int = 100,
        activation: str = "relu",
        learning_rate: float = 0.01,
        max_iter: int = 200,
        random_state: Optional[int] = None,
    ) -> None:
        """
        Initialize the SLP classifier.

        Parameters:
        -----------
        hidden_layer_size : int
            Number of neurons in the hidden layer
        activation : str
            Activation function ('identity', 'logistic', 'tanh', 'relu'}, default='logistic')
        learning_rate : float
            Learning rate for gradient descent
        max_iter : int
            Maximum number of iterations
        random_state : int or None
            Random seed for reproducibility
        """
        super().__init__(
            hidden_layer_size, activation, learning_rate, max_iter, random_state
        )

        # Classifier-specific attributes
        self.classes_: Optional[NDArray[np.int_]] = None  # Unique class labels
        self.n_outputs_: Optional[int] = None  # Number of output neurons

    def _forward_propagation(self, X: NDArray[np.floating]) -> Tuple[
        NDArray[np.floating],
        NDArray[np.floating],
        NDArray[np.floating],
        NDArray[np.floating],
    ]:
        """
        Perform forward propagation.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Input data

        Returns:
        --------
        z1, a1, z2, y_pred : tuple of arrays
            Intermediate values for backpropagation
        """
        activation_fn, _ = self._get_activation_function()

        z1 = X @ self.W1_ + self.b1_
        a1 = activation_fn(z1)
        z2 = a1 @ self.W2_ + self.b2_
        # Softmax handles both binary (K=2) and multi-class uniformly
        y_pred = softmax(z2)

        return z1, a1, z2, y_pred

    def _backward_propagation(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.floating],
        z1: NDArray[np.floating],
        a1: NDArray[np.floating],
        z2: NDArray[np.floating],
        y_pred: NDArray[np.floating],
    ) -> Tuple[
        NDArray[np.floating],
        NDArray[np.floating],
        NDArray[np.floating],
        NDArray[np.floating],
    ]:
        """
        Perform backpropagation to compute gradients.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Input data
        y : array-like, shape (n_samples, n_outputs)
            One-hot encoded target
        z1, a1, z2, y_pred : arrays
            Values from forward propagation

        Returns:
        --------
        dW1, db1, dW2, db2 : tuple of arrays
            Gradients for weights and biases
        """
        n_samples = X.shape[0]
        _, activation_derivative_fn = self._get_activation_function()

        # Output layer gradient (softmax + cross-entropy simplifies to y_pred - y)
        dz2 = (y_pred - y) / n_samples
        dW2 = a1.T @ dz2
        db2 = np.sum(dz2, axis=0)

        # Hidden layer gradient (chain rule through activation)
        dz1 = (dz2 @ self.W2_.T) * activation_derivative_fn(z1)
        dW1 = X.T @ dz1
        db1 = np.sum(dz1, axis=0)

        return dW1, db1, dW2, db2

    def _compute_loss(
        self, y_true: NDArray[np.floating], y_pred: NDArray[np.floating]
    ) -> float:
        """
        Compute cross-entropy loss.

        Parameters:
        -----------
        y_true : array-like, shape (n_samples, n_classes)
            One-hot encoded true labels
        y_pred : array-like, shape (n_samples, n_classes)
            Predicted probabilities

        Returns:
        --------
        loss : float
            Cross-entropy loss
        """
        # Clip to avoid log(0) which gives -inf and breaks the loss
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return float(-np.mean(np.sum(y_true * np.log(y_pred), axis=1)))

    def fit(
        self, X: NDArray[np.floating], y: NDArray[np.int_]
    ) -> "SimpleSLPClassifier":
        """
        Fit the SLP classifier to training data.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,)
            Target class labels

        Returns:
        --------
        self : object
            Fitted estimator
        """
        # Random seed is handled in _initialize_weights via default_rng (local generator)

        y = np.asarray(y)
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)

        # Ensure at least 2 output neurons even when y has a single class
        self.n_outputs_ = max(n_classes, 2)

        # One-hot encode y using class indices
        n_samples = X.shape[0]
        y_indices = np.searchsorted(self.classes_, y)
        y_onehot = np.zeros((n_samples, self.n_outputs_))
        y_onehot[np.arange(n_samples), y_indices] = 1

        n_features = X.shape[1]
        self._initialize_weights(n_features, self.n_outputs_)
        self.loss_curve_ = []

        for _ in range(self.max_iter):
            z1, a1, z2, y_pred = self._forward_propagation(X)

            loss = self._compute_loss(y_onehot, y_pred)
            self.loss_curve_.append(loss)

            dW1, db1, dW2, db2 = self._backward_propagation(
                X, y_onehot, z1, a1, z2, y_pred
            )

            # Gradient descent: W = W - learning_rate * gradient
            self.W1_ -= self.learning_rate * dW1
            self.b1_ -= self.learning_rate * db1
            self.W2_ -= self.learning_rate * dW2
            self.b2_ -= self.learning_rate * db2

        return self

    def predict_proba(self, X: NDArray[np.floating]) -> NDArray[np.floating]:
        """
        Predict class probabilities for X.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Samples

        Returns:
        --------
        proba : array-like, shape (n_samples, n_classes)
            Class probabilities
        """
        _, _, _, y_pred = self._forward_propagation(X)
        return y_pred

    def predict(self, X: NDArray[np.floating]) -> NDArray[np.int_]:
        """
        Predict class labels for X.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Samples

        Returns:
        --------
        y_pred : array-like, shape (n_samples,)
            Predicted class labels
        """
        proba = self.predict_proba(X)
        # argmax gives the column index of max proba; map back to the actual class label
        class_indices = np.argmax(proba, axis=1)
        return self.classes_[class_indices]

    def score(self, X: NDArray[np.floating], y: NDArray[np.int_]) -> float:
        """
        Return the mean accuracy on the given test data and labels.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Test samples
        y : array-like, shape (n_samples,)
            True labels

        Returns:
        --------
        score : float
            Mean accuracy
        """
        return float(np.mean(self.predict(X) == np.asarray(y)))
