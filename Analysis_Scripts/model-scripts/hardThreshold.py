import pandas as pd
import numpy as np

def hard_threshold(components, sparsity=0.2):
    '''
    Set all but the top (sparsity*100)% values to zero in each PC.
    Args:
        components (np.ndarray): The PCA components matrix (V).
        sparsity (float): The fraction of values to keep (0 < sparsity < 1).
    Returns:
        np.ndarray: The modified components matrix with hard thresholding applied.
    '''
    threshold = np.percentile(
        np.abs(components), 
        100 * (1 - sparsity),  # Percentile cutoff (e.g., 80th for 20% sparsity)
        axis=1, 
        keepdims=True
    )
    return components * (np.abs(components) >= threshold)