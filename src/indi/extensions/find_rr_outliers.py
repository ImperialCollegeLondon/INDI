import numpy as np


def check_mostly_identical_mad(data):
    """Check if median absolute deviation (MAD) is zero.

    Args:
        data (np.ndarray): 1D array of values to assess.

    Returns:
        tuple[bool, float, float]:
            Flag indicating whether MAD is zero, the MAD value, and the median.
    """
    median = np.nanmedian(data)
    mad = np.nanmedian(np.abs(data - median))

    is_mostly_identical = mad == 0

    return is_mostly_identical, mad, median


def detect_outliers_percentage(data, threshold_percent=50):
    """Detect outliers using percentage difference from the median.

    This method is suited for datasets where most values are identical.

    Args:
        data (np.ndarray): 1D array of values to evaluate.
        threshold_percent (float): Percentage difference threshold (e.g., 50 for 50%).

    Returns:
        tuple[np.ndarray, np.ndarray]:
            Boolean mask marking outliers and the array of outlier values.
    """
    median = np.nanmedian(data)

    if median == 0:
        # Avoid division by zero - use absolute difference instead
        outlier_mask = np.abs(data - median) > threshold_percent
    else:
        # Calculate percentage difference
        percent_diff = np.abs((data - median) / median) * 100
        outlier_mask = percent_diff > threshold_percent

    return outlier_mask, data[outlier_mask]


def detect_outliers_mad(data, threshold=3.5):
    """Detect outliers using the modified Z-score (MAD-based) method.

    Args:
        data (np.ndarray): 1D array of values to evaluate.
        threshold (float): Modified Z-score threshold; 3.5 is recommended.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            Boolean mask marking outliers and the array of outlier values.
    """
    median = np.nanmedian(data)
    mad = np.nanmedian(np.abs(data - median))

    # Modified z-score
    modified_z_scores = 0.6745 * (data - median) / mad if mad != 0 else np.zeros_like(data)

    outlier_mask = np.abs(modified_z_scores) > threshold

    return outlier_mask, data[outlier_mask]


def detect_outliers_iqr(data, threshold=1.5):
    """Detect outliers using the interquartile range (IQR) method.

    Args:
        data (np.ndarray): 1D array of values to evaluate.
        threshold (float): IQR multiplier; 1.5 is standard, 3.0 is conservative.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            Boolean mask marking outliers and the array of outlier values.
    """
    q1 = np.nanpercentile(data, 25)
    q3 = np.nanpercentile(data, 75)
    iqr = q3 - q1

    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr

    outlier_mask = (data < lower_bound) | (data > upper_bound)

    return outlier_mask, data[outlier_mask]
