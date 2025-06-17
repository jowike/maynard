import os
import pandas as pd
import numpy as np
from typing import List, Tuple

from maynard.dependencies.tools import _align_dates, _convert_to_datetime

import warnings

warnings.filterwarnings("ignore")


def prepare_real_time_vintage_data(
    ds: pd.DataFrame,
    y_code: str,
    ref_date: str,
    series_code_col: str,
    ref_date_col: str,
    pub_date_col: str,
    series_val_col: str,
) -> pd.DataFrame:
    """
    Builds a real-time vintage snapshot as of a given reference date.

    This function reconstructs how the dataset would have looked at a specific point in time,
    using only data that was actually available then. For each predictor, it selects the most
    recent value published *before* the first official estimate of the target variable (`y_code`)
    at the given reference date.

    Parameters
    ----------
    ds : pd.DataFrame
        The full long-format dataset with time series values and associated reference/publication dates.
    y_code : str
        Series code for the target variable (e.g., "GDP").
    ref_date : str
        The reference date for building the vintage snapshot (e.g., "2020-03-01").
    series_code_col : str
        Name of the column containing the series codes (e.g., "CODE").
    ref_date_col : str
        Name of the column with reference dates (e.g., "REF_DATE").
    pub_date_col : str
        Name of the column with publication dates (e.g., "PUB_DATE").
    series_val_col : str
        Name of the column containing the observed values (e.g., "VALUE").

    Returns
    -------
    pd.DataFrame
        A long-format DataFrame with real-time vintage values for all series,
        aligned with what would have been available as of the reference date.
    """

    def _format_individual_series(
        df: pd.DataFrame,
        ref_date: pd.Timestamp,
        pub_date_limit: str,
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Prepares real-time series for each non-target variable up to a given publication date.

        For each explanatory variable, this function selects the latest data release
        available before the target publication date and reshapes it into a long-format
        series. This simulates what would have been known in real time.

        Parameters
        ----------
        df : pd.DataFrame
            Subset of the full dataset containing only non-target series.
        ref_date : pd.Timestamp
            The reference date for which the vintage dataset is being built.
        pub_date_limit : str
            The latest publication date allowed (i.e., the first release date for the target series).

        Returns
        -------
        Tuple[pd.DataFrame, List[str]]
            - A long-format DataFrame with one row per date and series value, limited to data
            that would have been published before `pub_date_limit`.
            - A list of series codes that still contain missing values after processing.
        """

        df_long = pd.DataFrame(
            columns=[series_code_col, ref_date_col, pub_date_col, series_val_col]
        )
        null_cols = []

        for variable_code in df[series_code_col].unique():
            series_long = df[df[series_code_col] == variable_code]

            series_pivot = series_long.pivot(
                index=pub_date_col, columns=ref_date_col, values=series_val_col
            )
            series_pivot = (
                series_pivot.reindex(
                    sorted(
                        pd.date_range(
                            min(series_pivot.columns),
                            max(series_pivot.columns),
                            freq="MS",
                        )
                    ),
                    axis=1,
                )
                .sort_index()
                .ffill()
            )
            series_min_index = series_pivot.index.min()
            if series_min_index is None:
                continue
            if series_min_index.strftime("%Y-%m-%d") >= pub_date_limit:
                print(
                    f"The observation for {variable_code} was unavailable before {pub_date_limit}."
                )
                series = pd.DataFrame(
                    {series_val_col: np.nan, pub_date_col: np.datetime64("NaT")},
                    index=series_pivot.columns,
                )
            else:
                pivot_limit = series_pivot[series_pivot.index < pub_date_limit]

                last_release_index = (
                    pivot_limit[
                        min(
                            max(pivot_limit.dropna(how="all", axis=1).columns), ref_date
                        )
                    ]
                    .dropna(how="all")
                    .last_valid_index()
                )
                if last_release_index is None:
                    continue
                last_release_dt = last_release_index.strftime("%Y-%m-%d")

                series = pivot_limit.loc[last_release_dt].to_frame()
                series = series.loc[
                    series.first_valid_index() : min(
                        series.last_valid_index(), ref_date
                    )
                ]
                series = series.reindex(
                    sorted(
                        pd.date_range(min(series.index), max(series.index), freq="MS")
                    )
                )
                series.columns = [series_val_col]
                series[pub_date_col] = last_release_dt

            series[series_code_col] = variable_code

            if series[series_val_col].isnull().any():
                null_cols.append(variable_code)

            df_long = pd.concat([df_long, series.reset_index()])

        return df_long, null_cols

    # Start of the main function
    ref_date = pd.to_datetime(ref_date)
    ds = _align_dates(dataframe=ds, date_colname=ref_date_col)

    y_long = ds[ds[series_code_col] == y_code]
    y_pivot = y_long.pivot(
        index=pub_date_col, columns=ref_date_col, values=series_val_col
    )
    y_pivot = (
        y_pivot.reindex(
            sorted(
                pd.date_range(min(y_pivot.columns), max(y_pivot.columns), freq="MS")
            ),
            axis=1,
        )
        .sort_index()
        .ffill()
    )
    first_valid_index = y_pivot[ref_date].first_valid_index()

    if first_valid_index is None:
        return

    y_first_est_release_dt = first_valid_index.strftime("%Y-%m-%d")
    X_df = ds[ds[series_code_col] != y_code]

    # Call the nested helper function for non-target series
    X_df_long, _ = _format_individual_series(
        df=X_df, ref_date=ref_date, pub_date_limit=y_first_est_release_dt
    )

    rt_vintage_dt = y_pivot[y_pivot.index < y_first_est_release_dt].index.max()
    y_series = y_pivot[y_pivot.columns[y_pivot.columns <= ref_date]].loc[rt_vintage_dt]
    y_series.loc[ref_date] = y_pivot.loc[y_first_est_release_dt, ref_date]

    y_long = (
        y_series.to_frame()
        .reset_index()
        .rename(columns={y_series.name: series_val_col})
    )
    y_long[series_code_col] = y_code
    y_long[pub_date_col] = np.where(
        y_long[ref_date_col] < ref_date, rt_vintage_dt, y_first_est_release_dt
    )

    df_long = pd.concat(
        [
            _convert_to_datetime(y_long, [ref_date_col, pub_date_col]),
            _convert_to_datetime(X_df_long, [ref_date_col, pub_date_col]),
        ]
    )

    return df_long
