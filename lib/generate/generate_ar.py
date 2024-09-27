
import numpy as np


def simulate_ar(phi, sigma, length, seed=None, shift=100):
    """
    Simulate a time series using an AR model with given coefficients and noise.

    Parameters:
    phi (array-like): Coefficients of the AR model.
    sigma (float): Standard deviation of the noise.
    length (int): Length of the time series to generate.
    seed (int, optional): Seed for the random number generator.
    shift (int, optional): Shift for the time series.

    Returns:
    np.ndarray: Simulated time series.
    """
    if seed is not None:
        np.random.seed(seed)
    order = len(phi)
    shifted_length = length + shift
    y = np.zeros(shifted_length)
    epsilon = np.random.normal(0, sigma, shifted_length)
    y[:order] = np.random.normal(0, sigma, order)
    for t in range(order, shifted_length):
        y[t] = np.dot(phi, y[ t -order:t][::-1]) + epsilon[t]
    return y[-length:]


def generate_stationary_ar_coefficients(degree, seed):
    """
    Generate stationary AR coefficients.

    Parameters:
    degree (int): Number of coefficients.
    seed (int, optional): Seed for the random number generator.

    Returns:
    np.ndarray: Array of stationary AR coefficients.
    """
    if seed is not None:
        np.random.seed(seed)
    while True:
        coefficients = np.random.uniform(-1, 1, degree)
        coefficients = coefficients / np.sum(np.abs(coefficients))
        if check_stationary(coefficients):
            return coefficients


def check_stationary(coefficients):
    """ Check if the AR coefficients are stationary. """
    if len(coefficients) == 0:
        return False
    p = np.poly1d([1] + list(-coefficients))
    roots = np.roots(p)
    return np.all(np.abs(roots) < 1)