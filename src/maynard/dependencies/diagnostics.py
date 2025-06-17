import pandas as pd
from typing import List

from statsmodels.tsa.stattools import adfuller
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import MaxAbsScaler

import warnings

warnings.filterwarnings("ignore")


def test_stationarity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identifies non-stationary series using the Augmented Dickey-Fuller test.

    Applies the ADF test to each column in the input DataFrame. Series with p-values
    above the default significance level (0.05) are flagged as non-stationary.

    Parameters
    ----------
    df : pd.DataFrame
        A DataFrame where each column is a separate time series.

    Returns
    -------
    pd.DataFrame
        A DataFrame listing the column names of non-stationary series.
    """

    def __adfuller_test(
        series: pd.Series, signif: float = 0.05, name: str = "", verbose: bool = False
    ):
        """
        Runs the Augmented Dickey-Fuller (ADF) test to check for stationarity.

        This function tests whether a given time series has a unit root (i.e., is non-stationary).
        If `verbose=True`, it prints a summary of the test results including the test statistic,
        p-value, number of lags, and critical values.

        Parameters
        ----------
        series : pd.Series
            The time series to test.
        signif : float, optional
            Significance level to evaluate the p-value against (default is 0.05).
        name : str, optional
            Name of the series (used in printed output).
        verbose : bool, optional
            If True, prints a detailed test report.

        Returns
        -------
        float
            The p-value from the ADF test.
        """

        r = adfuller(series, autolag="AIC")
        output = {
            "test_statistic": round(r[0], 4),
            "pvalue": round(r[1], 4),
            "n_lags": round(r[2], 4),
            "n_obs": r[3],
        }
        p_value = output["pvalue"]

        def adjust(val, length=6):
            return str(val).ljust(length)

        # Print Summary
        if verbose:
            print(f' Augmented Dickey-Fuller Test on "{name}"', "\n   ", "-" * 47)
            print(" Null Hypothesis: Data has unit root. Non-Stationary.")
            print(f" Significance Level = {signif}")
            print(f' Test Statistic = {output["test_statistic"]}')
            print(' No. Lags Chosen = {output["n_lags"]}')

            for key, val in r[4].items():
                print(f" Critical value {adjust(key)} = {round(val, 3)}")

            if p_value <= signif:
                print(f" => P-Value = {p_value}. Rejecting Null Hypothesis.")
                print(" => Series is Stationary.")
            else:
                print(
                    f" => P-Value = {p_value}. Weak evidence to reject the Null Hypothesis."
                )
                print(" => Series is Non-Stationary.")
        return p_value

    stat = []

    for _id in df.columns:
        series_df = df[[_id]]
        series = series_df.dropna().sort_index()

        test_pval = __adfuller_test(series=series, name=_id)
        if test_pval < 0.05:
            stat.append(_id)

    return list(set(df.columns) - set(stat))


def test_variance(data: pd.DataFrame) -> List[str]:
    """
    Finds low-variance features in a dataset.

    This function scales the input data using MaxAbsScaler, then calculates the variance
    for each feature. It flags features whose variance falls below the 5th percentile
    and returns their column names.

    Parameters
    ----------
    data : pd.DataFrame
        Input DataFrame with numeric features.

    Returns
    -------
    List[str]
        A list of column names identified as low-variance features.
    """
    transformer = MaxAbsScaler().fit(data)
    df_scaled = pd.DataFrame(transformer.transform(data), columns=data.columns)
    tsh = df_scaled.var().quantile(0.05)
    selector = VarianceThreshold()

    selector = VarianceThreshold(threshold=tsh)
    selector.fit(df_scaled)

    features = df_scaled.columns[selector.get_support(indices=True)]

    return list(set(data.columns) - set(features))
