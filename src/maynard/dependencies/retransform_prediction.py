import numpy as np
import warnings
import pandas as pd
import warnings
from datetime import datetime


def retransform_(
    X: np.ndarray,
    Z: np.ndarray,
    Time: np.ndarray,
    Spec: dict,
    header: list,
    cutoff_date: datetime,
) -> np.ndarray:
    """
    Converts transformed predictions back to their original values.

    This function takes data that was previously transformed to be stationary (like changes, log returns, etc.)
    and brings it back to the original scale using the provided transformation specs. It supports multiple
    common transformation types such as levels, differences, percent changes, and logs.

    Parameters
    ----------
    X : np.ndarray
        The transformed forecast data (stationary), shape (H, N) — H is the number of time points,
        N is the number of series.
    Z : np.ndarray
        The original (untransformed) historical data, shape (H, N).
    Time : np.ndarray
        Time index for the series, typically as datetime objects.
    Spec : dict
        A dictionary with transformation settings. It should include:
            - "transformation": List of how each series was transformed.
            - "frequency": List of data frequencies, like 'm' for monthly.
            - "seriesid": IDs for each series.
            - "seriesname": Human-readable names for each series.
    header : list
        List of column names, expected to match the series IDs.
    cutoff_date : datetime
        The date where the forecast period starts — everything from this date onward is considered predicted.

    Returns
    -------
    np.ndarray
        A NumPy array with shape (H, N) that includes both historical values and retransformed forecasts.

    Raises
    ------
    AssertionError
        If a header name doesn't match the corresponding series ID from `Spec`.
    Warning
        If the transformation type is unknown for any series, a warning is issued, and the function
        returns the series without any retransformation.

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
            # V[0:T:step, i] = np.cumsum(X[c_idx[0]:H:step, i]) + Z[c_idx[0] - step, i]  # Assuming X[t1-step] is a last historical value
            V[0:T:step, i] = np.add(
                X[c_idx[0] : H : step, i], Z[c_idx[0] - step : H - step : step, i]
            )
        elif formula == "ch1":
            V[0:T:step, i] = np.add(
                Z[c_idx[0] - 12 : H - 12 : step, i], X[c_idx[0] : H : step, i]
            )
        elif formula == "pch":  # Percent Change
            # V[0:T:step, i] = np.cumprod(1 + (X[c_idx[0]:H:step, i] / 100)) * Z[c_idx[0] - step, i]  # Assuming X[t1-step] is a last historical value
            V[0:T:step, i] = np.multiply(
                1 + (X[c_idx[0] : H : step, i] / 100),
                Z[c_idx[0] - step : H - step : step, i],
            )
        elif formula == "pc1":  # Year over Year Percent Change
            V[0:T:step, i] = np.multiply(
                1 + (X[c_idx[0] : H : step, i] / 100),
                Z[c_idx[0] - 12 : H - 12 : step, i],
            )
        elif formula == "pca":  # Percent Change (Annual Rate)
            # V[0:T:step, i] = np.cumprod((1 + X[c_idx[0]:H:step, i] / 100) ** n) * Z[c_idx[0] - step, i]
            V[0:T:step, i] = ((1 + X[c_idx[0] : H : step, i] / 100) ** n) * Z[
                c_idx[0] - step : H - step : step, i
            ]
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
