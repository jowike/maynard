import mifs
import pandas as pd
from typing import Literal


def mtsfs(
    ds: pd.DataFrame, series_name: str, method: Literal["JMI", "JMIM", "MRMR"]
) -> pd.DataFrame:
    """
    Selects relevant predictors for a target series using mutual information.

    This function applies mutual information-based feature selection to choose the most
    informative predictors for a given target series from a multivariate time-indexed dataset.
    It’s model-agnostic and useful for filtering features before forecasting or regression.

    Parameters
    ----------
    ds : pd.DataFrame
        A time-indexed DataFrame that includes the target series and candidate predictors.
    series_name : str
        The name of the target series (must be one of the columns in `ds`).
    method : Literal["JMI", "JMIM", "MRMR"]
        The mutual information selection strategy to use:
            - "JMI": Joint Mutual Information
            - "JMIM": Joint Mutual Information Maximization
            - "MRMR": Minimum Redundancy Maximum Relevance

    Returns
    -------
    pd.DataFrame
        A DataFrame with the selected features and the original target series,
        indexed by time and aligned with the input.

    Notes
    -----
    - Uses `MutualInformationFeatureSelector` from the `mifs` package:
      https://github.com/danielhomola/mifs
    - Gracefully handles known edge cases like all-zero columns and logs any selection errors.
    """

    df = ds.sort_index()
    df.index = pd.to_datetime(df.index)

    X = df.drop(columns=[series_name])
    y = df[[series_name]]

    feat_selector = mifs.MutualInformationFeatureSelector(
        method=method, n_features="auto", categorical=False
    )

    # Find all relevant features
    try:
        feat_selector.fit(X.values, y.values.ravel())
    except ValueError as e:
        print(
            f"ERROR: Exception raised for {series_name}: {e}"
        )  # https://github.com/danielhomola/mifs/issues/15

    # Call transform() on X to filter it down to selected features
    X_support = pd.DataFrame(
        feat_selector.transform(X.values),
        columns=X.columns[feat_selector._support_mask],
        index=X.index,
    )
    to_write = pd.merge(X_support, y, left_index=True, right_index=True, how="right")

    return to_write
