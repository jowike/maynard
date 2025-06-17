import re
import numpy as np
import pandas as pd
from typing import List, Dict

import warnings

warnings.filterwarnings("ignore")


def cast_spec_to_dict(df: pd.DataFrame) -> Dict[str, list]:
    """
    Turns a spec DataFrame into a dictionary.

    This function takes a DataFrame with model specification info and converts it into
    a dictionary format. It makes sure column names are lowercase and checks for all
    required fields like 'seriesid', 'frequency', and 'transformation'. If any are missing,
    it raises an error.

    Parameters
    ----------
    df : pd.DataFrame
        A DataFrame with metadata for each series (e.g., from an Excel spec sheet).

    Returns
    -------
    dict
        A dictionary where keys are column names and values are lists of values for each series.

    Raises
    ------
    ValueError
        If any required column is missing from the input DataFrame.
    """
    raw_data = df.copy()
    # Convert all headers to lowercase for consistency
    raw_data.columns = raw_data.columns.str.lower()

    # Initialize spec dictionary
    spec = {}

    # Fields to extract from the Excel file
    field_names = [
        "seriesid",
        "seriesname",
        "frequency",
        "transformation",
        "units",
        "category",
        "model",
    ]
    # Extract required fields and ensure they exist
    for field in field_names:
        if field not in raw_data.columns:
            raise ValueError(f"{field} column missing from model specification.")
        spec[field] = raw_data[field].tolist()

    return spec


def suggest_transformation(unit: str) -> str:
    """
    Suggests a default transformation code based on a unit description.

    Uses common patterns in unit descriptions to recommend a typical transformation
    often used in macroeconomic data preprocessing.

    Parameters
    ----------
    unit : str
        Text describing the unit of a time series (e.g., "Billions of Chained 2017 Dollars").

    Returns
    -------
    str
        Suggested transformation code:
            - 'pc1' → year-over-year percent change
            - 'ch1' → year-over-year level change
            - 'lin' → no transformation
    """
    if re.search(
        r"(Billions|Millions|Thousands)\s+of\s+Chained\s+\d{4}\s+Dollars", unit
    ):
        return "pc1"

    if re.search(r"Index", unit):
        return "ch1"  # Wskaźniki, np. CPI, używamy zmiany

    if re.search(r"Percent", unit):
        return "lin"  # Jeśli jednostka to Percent, nie stosujemy transformacji

    if re.search(r"Percent\s+Change\s+at\s+Annual\s+Rate", unit):
        return "lin"  # Jeśli to już procentowa zmiana roczna, nie wymagamy dalszej transformacji

    if re.search(r"Level", unit):
        return "pc1"  # Poziomy (np. liczba osób, PKB) — stosujemy procentową zmianę roczną (pca)

    if re.search(r"Ratio", unit):
        return "lin"  # Wskaźniki proporcji — nie wymagają transformacji

    if re.search(r"Rate", unit):
        return "lin"  # Stopy procentowe (np. procentowa stopa bezrobocia) — pozostawiamy bez zmian

    if re.search(
        r"(Dollars|Euro|Yen|Pounds|Rupees|Franc|Pesos)\s+per\s+[A-Za-z]+", unit
    ):
        return "pc1"  # Ceny jednostkowe (np. "Dollars per Gallon") — zmiana procentowa

    if re.search(r"Thousands\s+of\s+[A-Za-z]+", unit):
        return "pc1"  # Jednostki liczbowe w tysiącach — logarytmizacja

    return "ch1"


def _convert_to_datetime(df: pd.DataFrame, colnames: List[str]) -> pd.DataFrame:
    """
    Converts selected columns in a DataFrame to datetime format.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame.
    colnames : List[str]
        List of column names that should be converted to datetime.

    Returns
    -------
    pd.DataFrame
        The DataFrame with the specified columns converted to datetime.
    """
    for col in colnames:
        df[col] = pd.to_datetime(df[col])
    return df


def _align_dates(dataframe: pd.DataFrame, date_colname: str) -> pd.DataFrame:
    """
    Shifts all dates in a column to the first day of their month.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Input DataFrame containing a date column.
    date_colname : str
        Name of the column with dates to align.

    Returns
    -------
    pd.DataFrame
        A copy of the DataFrame with dates adjusted to the 1st of each month.
    """
    df = dataframe.reset_index(drop=True).reset_index()
    df[date_colname] = df[date_colname].apply(lambda x: x.replace(day=1))
    return df.drop(columns=["index"])


def _error(actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    """
    Calculates basic forecast error.

    Parameters
    ----------
    actual : np.ndarray
        The true (observed) values.
    predicted : np.ndarray
        The predicted or forecasted values.

    Returns
    -------
    np.ndarray
        Element-wise difference: actual minus predicted.
    """
    return actual - predicted


def _percentage_error(actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    """
    Calculates element-wise percentage error.

    Note
    ----
    The result is in decimal form (not multiplied by 100).

    Parameters
    ----------
    actual : np.ndarray
        The true (observed) values.
    predicted : np.ndarray
        The predicted or forecasted values.

    Returns
    -------
    np.ndarray
        Element-wise percentage error: (actual - predicted) / actual.
    """
    EPSILON = 1e-10
    return _error(actual, predicted) / (actual + EPSILON)


def mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    Calculates Mean Absolute Percentage Error (MAPE).

    Note
    ----
    The result is in decimal form (not multiplied by 100).

    Parameters
    ----------
    actual : np.ndarray
        The true (observed) values.
    predicted : np.ndarray
        The predicted or forecasted values.

    Returns
    -------
    float
        The mean absolute percentage error.
    """
    return np.mean(np.abs(_percentage_error(actual, predicted)))


def mse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    Calculates Mean Squared Error (MSE).

    Parameters
    ----------
    actual : np.ndarray
        The true (observed) values.
    predicted : np.ndarray
        The predicted or forecasted values.

    Returns
    -------
    float
        The mean of the squared errors.
    """
    return np.mean(np.square(_error(actual, predicted)))


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    Calculates Root Mean Squared Error (RMSE).

    Parameters
    ----------
    actual : np.ndarray
        The true (observed) values.
    predicted : np.ndarray
        The predicted or forecasted values.

    Returns
    -------
    float
        The square root of the mean squared error.
    """
    return np.sqrt(mse(actual, predicted))
