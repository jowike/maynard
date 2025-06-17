import pandas as pd


def shift_to_fill_trailing_nans(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Fixes trailing NaNs in time series by shifting data forward.

    For each column that ends with missing values, this function shifts the non-missing
    part of the series forward so that the last valid value lines up with the end.
    It’s a simple way to handle the ragged edge problem in time series panels.

    Parameters
    ----------
    dataframe : pd.DataFrame
        A DataFrame with time series data (each column is a separate variable).

    Returns
    -------
    pd.DataFrame
        A new DataFrame with trailing NaNs minimized by shifting series forward.
    """

    # Iterate over each column in the DataFrame
    for column in dataframe.columns:
        series = dataframe[column]

        # Check if the column has missing values at the end
        if series.iloc[-1:].isna().all():
            # Find the index of the last non-NaN value
            last_valid_index = series.last_valid_index()

            # If a valid non-NaN index exists, create a lagged version of the series
            if last_valid_index is not None:
                lag_amount = len(series) - series.index.get_loc(last_valid_index) - 1
                dataframe[column] = series.shift(lag_amount)

    return dataframe
