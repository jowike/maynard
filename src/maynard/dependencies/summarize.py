import numpy as np
import pandas as pd
from typing import Dict


def summarize(X: np.ndarray, Time: np.ndarray, Spec: Dict[str, list]) -> None:
    """
    Prints a summary of the data going into the nowcasting process.

    For each time series, this function shows the number of observations, units, frequency,
    average, standard deviation, minimum and maximum values — along with the dates they occurred.
    It gives a quick look at what the input data looks like before modeling.

    Parameters
    ----------
    X : np.ndarray
        A 2D array with transformed data, shape (T, N), where T is the number of time steps
        and N is the number of time series.
    Time : np.ndarray
        An array of timestamps (datetime-like), one for each row in `X`.
    Spec : dict
        A dictionary with metadata about each series. Should include:
            - 'seriesname': List of full names for each series.
            - 'seriesid': Unique IDs for each series.
            - 'frequency': Data frequency for each series (e.g. 'm' for monthly).
            - 'units': Description of the original measurement units.
            - 'transformation': Transformation applied to each series.

    Returns
    -------
    None
        The summary table is printed to the console.
    """

    print("\n\n\n")
    print("Table: Data Summary \n")

    T, N = X.shape
    print(f"N = {N:4d} data series")
    print(f"T = {T:4d} observations from {Time[0]:%Y-%m-%d} to {Time[-1]:%Y-%m-%d}")

    print(
        f'{"Data Series":30s} | {"Observations":17s} {"Units":12s} {"Frequency":10s} {"Mean":8s} {"Std. Dev.":8s} {"Min":10s} {"Max":10s}'
    )
    print("-" * 130)

    for i in range(N):
        # time indexes for which there are observed values for series i
        t_obs = ~np.isnan(X[:, i])

        data_series = Spec["seriesname"][i]
        if len(data_series) > 30:
            data_series = f"{data_series[:27]}..."

        series_id = Spec["seriesid"][i]
        if len(series_id) > 28:
            series_id = f"{series_id[:25]}..."
        series_id = f"[{series_id}]"

        num_obs = np.sum(t_obs)
        freq = Spec["frequency"][i]

        if freq == "m":
            format_date = "%b %Y"
            frequency = "Monthly"
        elif freq == "q":
            format_date = "Q%q %Y"
            frequency = "Quarterly"

        units = Spec["units"][i]
        transform = Spec["transformation"][i]

        # display transformed units
        if "Index" in units:
            units_transformed = "Index"
        elif transform == "chg":
            if "%" in units:
                units_transformed = "Ppt. change"
            else:
                units_transformed = "Level change"
        elif transform == "pch" and freq == "m":
            units_transformed = "MoM%"
        elif transform == "pca" and freq == "q":
            units_transformed = "QoQ% AR"
        else:
            units_transformed = (
                f"{units} ({transform})"
                if len(f"{units} ({transform})") <= 12
                else f"{units[:6]} ({transform})"
            )

        if num_obs:
            t_obs_start = np.where(t_obs)[0][0]
            t_obs_end = np.where(t_obs)[0][-1]
            obs_date_start = pd.to_datetime(Time[t_obs_start]).strftime(format_date)
            obs_date_end = pd.to_datetime(Time[t_obs_end]).strftime(format_date)

            y = X[t_obs, i]
            d = Time[t_obs]
            mean_series = np.nanmean(y)
            stdv_series = np.nanstd(y)
            min_series, t_min = np.min(y), np.argmin(y)
            max_series, t_max = np.max(y), np.argmax(y)
            min_date = pd.to_datetime(d[t_min]).strftime(format_date)
            max_date = pd.to_datetime(d[t_max]).strftime(format_date)
        else:
            obs_date_start, obs_date_end, min_date, max_date = [""] * 4
            mean_series, stdv_series, min_series, max_series = [np.nan] * 4

        date_range = f"{obs_date_start}-{obs_date_end}"

        print(
            f"{data_series:30s} | {num_obs:17d} {units_transformed:12s} {frequency:10s} {mean_series:8.1f} {stdv_series:8.1f} {min_series:10.1f} {max_series:10.1f}"
        )
        print(
            f'{series_id:30s} | {date_range:17s} {"":12s} {"":10s} {"":8s} {"":8s} {min_date:8s} {max_date:8s}'
        )

    print("\n\n\n")


# Example usage:
# X = np.array(...)  # Your data array
# Time = np.array(...)  # Your time array
# Spec = {
#     'SeriesName': [...],
#     'SeriesID': [...],
#     'Frequency': [...],
#     'Units': [...],
#     'Transformation': [...]
# }
# vintage = '2024-08-08'
# summarize(X, Time, Spec, vintage)
