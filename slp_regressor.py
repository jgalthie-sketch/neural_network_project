"""
SimpleSLPRegressor - Single Layer Perceptron for Regression
"""

from typing import Optional, Tuple
import numpy as np
from numpy.typing import NDArray
from slp_base import BaseSLPEstimator


class SimpleSLPRegressor(BaseSLPEstimator):
    """
    Simple Single Layer Perceptron Regressor with one hidden layer.

    Compatible interface with sklearn.neural_network.MLPRegressor.
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
        Initialize the SLP regressor.

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
        # Regression: identity output, y_pred = z2
        y_pred = z2

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
            Target values
        z1, a1, z2, y_pred : arrays
            Values from forward propagation

        Returns:
        --------
        dW1, db1, dW2, db2 : tuple of arrays
            Gradients for weights and biases
        """
        n_samples = X.shape[0]
        _, activation_derivative_fn = self._get_activation_function()

        # Output layer gradient (identity + MSE simplifies to y_pred - y)
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
        Compute mean squared error loss.

        Parameters:
        -----------
        y_true : array-like
            True values
        y_pred : array-like
            Predicted values

        Returns:
        --------
        loss : float
            MSE loss
        """
        return float(np.mean((y_true - y_pred) ** 2))

    def fit(
        self, X: NDArray[np.floating], y: NDArray[np.floating]
    ) -> "SimpleSLPRegressor":
        """
        Fit the SLP regressor to training data.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,) or (n_samples, n_outputs)
            Target values

        Returns:
        --------
        self : object
            Fitted estimator
        """
        # Ensure y is 2D (n_samples, n_outputs) to match y_pred shape
        y = np.asarray(y, dtype=float)
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        n_features = X.shape[1]
        n_outputs = y.shape[1]

        self._initialize_weights(n_features, n_outputs)
        self.loss_curve_ = []

        for _ in range(self.max_iter):
            z1, a1, z2, y_pred = self._forward_propagation(X)

            loss = self._compute_loss(y, y_pred)
            self.loss_curve_.append(loss)

            dW1, db1, dW2, db2 = self._backward_propagation(
                X, y, z1, a1, z2, y_pred
            )

            # Gradient descent: W = W - learning_rate * gradient
            self.W1_ -= self.learning_rate * dW1
            self.b1_ -= self.learning_rate * db1
            self.W2_ -= self.learning_rate * dW2
            self.b2_ -= self.learning_rate * db2

        return self

    def predict(self, X: NDArray[np.floating]) -> NDArray[np.floating]:
        """
        Predict using the trained model.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Samples

        Returns:
        --------
        y_pred : array-like, shape (n_samples,) or (n_samples, n_outputs)
            Predicted values
        """
        _, _, _, y_pred = self._forward_propagation(X)

        # Return 1D array if single output (sklearn convention)
        if y_pred.shape[1] == 1:
            return y_pred.ravel()
        return y_pred

    def score(self, X: NDArray[np.floating], y: NDArray[np.floating]) -> float:
        """
        Return the R² score on the given test data.

        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Test samples
        y : array-like, shape (n_samples,) or (n_samples, n_outputs)
            True values

        Returns:
        --------
        score : float
            R² score
        """
        y = np.asarray(y, dtype=float)
        y_pred = self.predict(X)

        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return float(1 - ss_res / ss_tot)
