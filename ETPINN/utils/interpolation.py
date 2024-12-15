# -*- coding: utf-8 -*-
"""
Created on Mon May 15 16:38:23 2023

@author: chen096
"""

import numpy as np
from scipy.spatial import distance as ssd

def idw(points, values, targets, power=2, tol=1e-14, npOnce=int(1E7)):
    """
    Inverse Distance Weighting (IDW) interpolation.
    
    This function interpolates values at target points based on weighted averages
    of known data points, where weights are inversely proportional to the distance
    raised to the specified power.
    
    Parameters:
        points : array-like, shape (n_samples, n_features)
            Coordinates of known data points.
        values : array-like, shape (n_samples, n_targets)
            Values at each data point.
        targets : array-like, shape (n_targets, n_features)
            Coordinates of points where interpolation is desired.
        power : float, optional (default=2)
            The power parameter for inverse distance weighting.
            Larger values give more weight to nearer points.
        tol : float, optional (default=1e-14)
            Tolerance for distances below which points are considered identical.
        npOnce : int, optional (default=1E7)
            Maximum number of computations to process at once for memory efficiency.
    
    Returns:
        interpolated_values : ndarray, shape (n_targets, n_targets)
            Interpolated values at the target points.
    """
    # Convert inputs to NumPy arrays
    points = np.array(points)
    values = np.array(values)
    targets = np.array(targets)

    # Number of points and targets
    npoint = points.shape[0]
    ntar = targets.shape[0]
    
    # Batch size for processing targets
    ndeal = npOnce // npoint + 1

    # Start index for processing targets
    ist = 0

    # Array to store the interpolated values
    interpolated_values = np.zeros((0, values.shape[1]), dtype=values.dtype)

    # Process in batches to handle large inputs
    for i in range(ntar // ndeal + 1):
        # Batch start and end indices
        ist = i * ndeal
        ind = min((i + 1) * ndeal, ntar)

        # Compute distances between target batch and all known points
        distances = ssd.cdist(targets[ist:ind], points)
        distances = np.maximum(distances, tol)  # Avoid division by zero

        # Compute weights based on distances
        weights = 1.0 / distances**power
        weights /= weights.sum(axis=1)[:, None]  # Normalize weights

        # Interpolate values using weighted averages
        interpolated_values = np.concatenate(
            (interpolated_values, np.dot(weights, values)),
            axis=0
        )

    # Ensure the output has the correct number of interpolated values
    assert interpolated_values.shape[0] == targets.shape[0]
    
    return interpolated_values


def idw_weights(points, targets, power=2, tol=1e-14):
    """
    Compute weights for Inverse Distance Weighting (IDW).
    
    This function computes weights for each target point based on the distances
    to known data points. The weights can be used for custom interpolations.
    
    Parameters:
        points : array-like, shape (n_samples, n_features)
            Coordinates of known data points.
        targets : array-like, shape (n_targets, n_features)
            Coordinates of points where interpolation is desired.
        power : float, optional (default=2)
            The power parameter for inverse distance weighting.
            Larger values give more weight to nearer points.
        tol : float, optional (default=1e-14)
            Tolerance for distances below which points are considered identical.
    
    Returns:
        weights : ndarray, shape (n_targets, n_samples)
            Normalized weights for each target point.
    """
    # Convert inputs to NumPy arrays
    points = np.array(points)
    targets = np.array(targets)

    # Compute distances between targets and points
    distances = ssd.cdist(targets, points)
    distances = np.maximum(distances, tol)  # Avoid division by zero

    # Compute weights based on distances
    weights = 1.0 / distances**power
    weights /= weights.sum(axis=1)[:, None]  # Normalize weights
    
    return weights
