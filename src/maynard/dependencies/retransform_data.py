import numpy as np
import warnings
import numpy as np
import warnings
from datetime import datetime


def retransform_data(
    X: np.ndarray,
    Z: np.ndarray,
    Time: np.ndarray,
    Spec: dict,
    header: list,
    cutoff_date: datetime,
) -> np.ndarray:
    """
    Converts stationary (transformed) data back to its original scale.

    This function takes transformed forecast data and reverts it to its original values
    using the transformation types specified in the model spec. It handles different
    transformations like levels, differences, percent changes, and logs.

    Parameters
    ----------
    X : np.ndarray
        The transformed (stationary) forecast data, shape (H, N), where H is the number of time points,
        and N is the number of series.
    Z : np.ndarray
        The original, untransformed historical data, shape (H, N).
    Time : np.ndarray
        Time index for the series, typically as datetime objects.
    Spec : dict
        Specification dictionary with metadata, including:
            - "transformation": list of transformation types for each series.
            - "frequency": list of data frequencies (e.g. 'm' for monthly).
            - "seriesid": list of series identifiers.
            - "seriesname": list of series names.
    header : list
        List of column names from the data. These must match the IDs in `Spec["seriesid"]`.
    cutoff_date : datetime
        The date when the forecast period starts. Data from this point onward is treated as predicted.

    Returns
    -------
    np.ndarray
        Data with original scale restored, shape (H, N). Includes historical values from `Z`
        and retransformed forecasts from `X`.

    Raises
    ------
    AssertionError
        If any value in `header` doesn't match the corresponding value in `Spec["seriesid"]`.
    Warning
        If a transformation type isn't recognized, the function leaves that series unchanged and shows a warning.

    Examples
    --------
    >>> import numpy as np
    >>> from datetime import datetime
    >>> X = np.random.randn(100, 3)
    >>> Z = np.random.randn(120, 3)
    >>> Time = np.array([datetime(2020, 1, 1) + np.timedelta64(i, 'M') for i in range(120)])
    >>> Spec = {
    ...     "transformation": ["lin", "chg", "log"],
    ...     "frequency": ["m", "m", "m"],
    ...     "seriesid": ["series1", "series2", "series3"],
    ...     "seriesname": ["Series 1", "Series 2", "Series 3"]
    ... }
    >>> header = ["series1", "series2", "series3"]
    >>> cutoff_date = datetime(2023, 1, 1)
    >>> V_final = retransform_data(X, Z, Time, Spec, header, cutoff_date)
    """

    # Filter indexes that denote predictions
    c_idx = np.where(Time >= cutoff_date)[0]

    # Get the number of periods of predictions
    T = c_idx.shape[0]
    # Get number of periods, series
    H = X.shape[0]
    N = X.shape[1]
    # Initialize T x N matrix filled with NaN's
    V = np.full((T, N), np.nan)

    for i in range(N):
        formula = Spec["transformation"][i]
        freq = Spec["frequency"][i]
        step = 1 if freq == "m" else 3
        t1 = step
        n = step / 12
        assert header[i] == Spec["seriesid"][i]
        series = Spec["seriesname"][i]

        # Apply inverse transformations based on formula
        if formula == "lin":  # Levels (No Transformation)
            V[:, i] = X[:, i][c_idx]
        elif formula == "chg":  # Change (Difference)
            V[0:T:step, i] = (
                np.cumsum(X[c_idx[0] : H : step, i]) + Z[c_idx[0] - step, i]
            )  # Assuming X[t1-step] is a last historical value
        elif formula == "ch1":
            V[0:T:step, i] = np.add(
                Z[c_idx[0] - 12 : H - 12 : step, i], X[c_idx[0] : H : step, i]
            )
        elif formula == "pch":  # Percent Change
            V[0:T:step, i] = (
                np.cumprod(1 + (X[c_idx[0] : H : step, i] / 100))
                * Z[c_idx[0] - step, i]
            )  # Assuming X[t1-step] is a last historical value
        elif formula == "pc1":  # Year over Year Percent Change
            V[0:T:step, i] = np.multiply(
                1 + (X[c_idx[0] : H : step, i] / 100),
                Z[c_idx[0] - 12 : H - 12 : step, i],
            )
        elif formula == "pca":  # Percent Change (Annual Rate)
            V[0:T:step, i] = (
                np.cumprod((1 + X[c_idx[0] : H : step, i] / 100) ** n)
                * Z[c_idx[0] - step, i]
            )
        elif formula == "log":  # Natural Log
            V[:, i] = np.exp(X[:, i][c_idx])
        else:
            warnings.warn(
                f"Transformation '{formula}' not found for {series}. Using untransformed data."
            )
            V[:, i] = X[:, i][c_idx]

        V_final = np.full((H, N), np.nan)
        for _ in range(N):
            V_final[:, _] = np.concatenate((Z[: c_idx[0], _], V[:, _]))
    return V_final


# Example usage retransform_data(X, Z, Time, Spec, header, datetime(2023, 1, 1))
