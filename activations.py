"""
Activation Functions and Their Derivatives
"""

import numpy as np
from numpy.typing import NDArray


def relu(z: NDArray[np.floating]) -> NDArray[np.floating]:
    """
    ReLU activation: f(z) = max(0, z), applied element-wise.

    Parameters:
    -----------
    z : array-like
        Input values

    Returns:
    --------
    array-like
        Activated values
    """
    # np.maximum: element-wise comparison (not np.max which reduces to scalar)
    return np.maximum(0, z)


def relu_derivative(z: NDArray[np.floating]) -> NDArray[np.floating]:
    """
    Derivative of ReLU: 1 where z > 0, else 0.

    Parameters:
    -----------
    z : array-like
        Input values (pre-activation)

    Returns:
    --------
    array-like
        Gradient values
    """
    # Boolean array (True/False) cast to float (1.0/0.0) matches the derivative
    return (z > 0).astype(float)


def tanh(z: NDArray[np.floating]) -> NDArray[np.floating]:
    """
    Hyperbolic tangent: f(z) = (e^z - e^-z) / (e^z + e^-z), in [-1, 1].

    Parameters:
    -----------
    z : array-like
        Input values

    Returns:
    --------
    array-like
        Activated values
    """
    return np.tanh(z)


def tanh_derivative(z: NDArray[np.floating]) -> NDArray[np.floating]:
    """
    Derivative of tanh: 1 - tanh(z)^2.

    Parameters:
    -----------
    z : array-like
        Input values (pre-activation)

    Returns:
    --------
    array-like
        Gradient values
    """
    return 1 - np.tanh(z) ** 2


def logistic(z: NDArray[np.floating]) -> NDArray[np.floating]:
    """
    Logistic (sigmoid) activation: f(z) = 1 / (1 + exp(-z)), in (0, 1).

    Parameters:
    -----------
    z : array-like
        Input values

    Returns:
    --------
    array-like
        Activated values
    """
    # Clip to avoid overflow in exp for very negative z
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))


def logistic_derivative(z: NDArray[np.floating]) -> NDArray[np.floating]:
    """
    Derivative of sigmoid: sigma(z) * (1 - sigma(z)).

    Parameters:
    -----------
    z : array-like
        Input values (pre-activation)

    Returns:
    --------
    array-like
        Gradient values
    """
    sig = logistic(z)
    return sig * (1 - sig)


def softmax(z: NDArray[np.floating]) -> NDArray[np.floating]:
    """
    Softmax activation for output layer: converts scores to probabilities
    that sum to 1 along each row.

    Parameters:
    -----------
    z : array-like, shape (n_samples, n_classes)
        Input values

    Returns:
    --------
    array-like, shape (n_samples, n_classes)
        Probabilities that sum to 1 for each sample
    """
    # Subtract row-wise max for numerical stability (invariant property of softmax)
    z_shifted = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z_shifted)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)