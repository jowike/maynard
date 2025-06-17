import numpy as np
import pandas as pd
import warnings
from typing import Union, Tuple, Optional, Dict, List


def load_data(
    ds: Union[str, pd.DataFrame],
    Spec: dict,
    sample: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, list]:
    """
    Loads and prepares a vintage dataset based on the model specification.

    This function reads time series data (from a file or DataFrame), sorts the series
    to match the model spec, applies transformations, and optionally filters it
    to a defined sample period.

    Parameters
    ----------
    ds : Union[str, pd.DataFrame]
        Either a file path (Excel format expected) or a preloaded DataFrame.
    Spec : dict
        Model specification dictionary, including keys like 'seriesid', 'transformation', etc.
    sample : float, optional
        Optional cutoff date to trim the dataset to the desired sample period.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, list]
        - X: Transformed dataset (T × N).
        - Time: Corresponding time index array (length T).
        - Z: Raw untransformed dataset (T × N), aligned with X.
        - header: List of variable names from the original dataset.
    """

    print("Loading data...")

    Z, Time, Mnem = read_data(ds)

    # Sort data based on model specification
    Z = sort_data(Z, Mnem, Spec)

    # Transform data based on model specification
    X, Time, Z, header = transform_data(Z, Time, Spec)

    # Drop data not in estimation sample
    if sample is not None:
        X, Time, Z = drop_data(X, Time, Z, sample)

    return X, Time, Z, header


def read_data(ds: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Extracts raw time series data from a formatted Excel-like DataFrame.

    The input DataFrame is expected to have the following structure:
    - First row: Variable names (series IDs), starting from the second column.
    - First column (excluding the first row): Time index values.
    - Remaining cells: Time series observations for each variable.

    Parameters
    ----------
    ds : pd.DataFrame
        DataFrame read from an Excel file or similar source.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, List[str]]
        - Z: Data matrix with raw observations (without headers).
        - Time: Array of time index values.
        - Mnem: List of series identifiers (column names).
    """
    Mnem = ds.iloc[0, 1:].tolist()
    Time = ds.iloc[1:, 0].to_numpy()
    Z = ds.iloc[:, 1:].to_numpy()
    return Z, Time, Mnem


def sort_data(Z: np.ndarray, Mnem: List[str], Spec: Dict[str, List[str]]) -> np.ndarray:
    """
    Filters and reorders series to match the model specification.

    This function removes any variables not listed in the model spec (`Spec["seriesid"]`)
    and reorders the remaining columns in `Z` to follow the same order as in the spec.

    Parameters
    ----------
    Z : np.ndarray
        The raw data matrix, shape (T, N).
    Mnem : list of str
        List of series IDs (column identifiers) corresponding to the columns in `Z`.
    Spec : dict
        Model specification dictionary containing the expected order in 'seriesid'.

    Returns
    -------
    np.ndarray
        The filtered and sorted data matrix with columns aligned to the spec.
    """
    in_spec = np.isin(Mnem, Spec["seriesid"])
    Mnem = [mnem for mnem, keep in zip(Mnem, in_spec) if keep]
    Z = Z[:, in_spec]

    # Sort series by ordering of Spec
    N = len(Spec["seriesid"])
    permutation = [Mnem.index(spec_id) for spec_id in Spec["seriesid"]]

    Mnem = [Mnem[i] for i in permutation]
    Z = Z[:, permutation]

    return Z


def transform_data(
    Z: np.ndarray, Time: np.ndarray, Spec: Dict[str, List]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Applies specified transformations to make time series stationary.

    This function transforms each series in the dataset based on the methods listed in `Spec`.
    It supports common transformations like differences, percent changes, and logs — typically used
    to prepare data for models such as Dynamic Factor Models.

    Parameters
    ----------
    Z : np.ndarray
        Raw observed data array of shape (T+1, N). The first row is assumed to be the header.
    Time : np.ndarray
        Array of observation dates (T+1,), aligned with the rows of `Z`.
    Spec : dict
        Model specification containing metadata for each series. Must include:
            - 'seriesid': List of variable names.
            - 'seriesname': List of human-readable names.
            - 'transformation': List of transformation codes (e.g., 'lin', 'chg', 'pch').
            - 'frequency': List of frequencies ('m' for monthly, 'q' for quarterly).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]
        - X: Transformed (typically stationary) data of shape (T, N).
        - Time: Trimmed array of time values, excluding initial rows dropped due to lagging.
        - Z: Raw data (numeric), with early rows removed to match `X`.
        - header: List of original variable names (from the first row of `Z`).
    """
    header = Z[0, :]
    Z = np.float64(Z[1:, :])

    T, N = Z.shape

    X = np.full((T, N), np.nan)

    for i in range(N):
        formula = Spec["transformation"][i]
        freq = Spec["frequency"][i]
        step = 1 if freq == "m" else 3
        t1 = step
        n = step / 12

        assert header[i] == Spec["seriesid"][i]
        series = Spec["seriesname"][i]

        # Apply transformations based on formula
        if formula == "lin":  # Levels (No Transformation)
            X[:, i] = Z[:, i]
        elif formula == "chg":  # Change (Difference)
            X[(t1 - 1 + step) : T, i] = (
                Z[(t1 - 1 + step) : T, i] - Z[(t1 - 1) : (T - t1), i]
            )
        elif formula == "ch1":  # Year over Year Change (Difference)
            if T > 12:
                X[(12 + t1 - 1) : T, i] = (
                    Z[(12 + t1 - 1) : T, i] - Z[(t1 - 1) : (T - 12), i]
                )
        elif formula == "pch":  # Percent Change
            X[(t1 - 1 + step) : T, i] = 100 * (
                Z[(t1 - 1 + step) : T, i] / Z[(t1 - 1) : (T - t1), i] - 1
            )
        elif formula == "pc1":  # Year over Year Percent Change
            if T > 12:
                # Year over Year Percent Change, handle division by zero
                X[(12 + t1 - 1) : T, i] = 100 * (
                    Z[(12 + t1 - 1) : T, i] / Z[(t1 - 1) : (T - 12), i] - 1
                )
        elif formula == "pca":  # Percent Change (Annual Rate)
            X[(t1 - 1 + step) : T, i] = 100 * (
                (Z[(t1 - 1 + step) : T, i] / Z[(t1 - 1) : (T - step), i]) ** (1 / n) - 1
            )
        elif formula == "log":  # Natural Log
            X[:, i] = np.log(Z[:, i])
        else:
            warnings.warn(
                f"Transformation '{formula}' not found for {series}. Using untransformed data."
            )
            X[:, i] = Z[:, i]

    # Drop first quarter of observations since transformations cause missing values
    Time = Time[3:]
    Z = Z[3:, :]
    X = X[3:, :]

    return X, Time, Z, header


def drop_data(
    X: np.ndarray, Time: np.ndarray, Z: np.ndarray, sample: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Drops all observations that come before the specified sample start date.

    This function filters the input arrays to keep only the part of the time series
    that falls within or after the given sample start date.

    Parameters
    ----------
    X : np.ndarray
        Transformed data array.
    Time : np.ndarray
        Array of datetime-like observation dates.
    Z : np.ndarray
        Raw (untransformed) data array.
    sample : float
        Start date of the estimation sample (e.g., as a `datetime` or timestamp).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        Filtered versions of X, Time, and Z, containing only observations from the sample period onward.
    """
    idx_drop = Time < sample

    Time = Time[~idx_drop]
    X = X[~idx_drop, :]
    Z = Z[~idx_drop, :]

    return X, Time, Z
