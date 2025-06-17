import pandas as pd
import numpy as np
from typing import Dict


def load_spec(specfile: str) -> Dict[str, list]:
    """
    Loads model spec from an Excel file and returns it as a dictionary.

    The Excel file should contain a table with columns like: Model, SeriesID, SeriesName, Frequency,
    Transformation, Units, Category, and Region (column names are case-insensitive and may have spaces).

    The function keeps only rows where 'Model' is not 0 and builds a spec dictionary with key info
    needed for modeling. It also sorts series by frequency and maps transformation codes to readable names.

    Parameters
    ----------
    specfile : str
        Path to the Excel file containing the model specification.

    Returns
    -------
    dict
        A dictionary with metadata for each time series, including fields like 'seriesid', 'seriesname',
        'frequency', 'transformation', and more.
    """

    # Read the Excel file
    raw_data = pd.read_excel(specfile, sheet_name=None, header=None)
    raw_data = list(raw_data.values())[
        0
    ]  # Get the first sheet data if multiple sheets exist
    raw_data.columns = raw_data.iloc[0].str.replace(" ", "")
    raw_data = raw_data.drop(0)

    # Convert all headers to lowercase for consistency
    raw_data.columns = raw_data.columns.str.lower()

    # Find and drop series from Spec that are not in Model
    model_idx = raw_data["model"].astype(int) != 0
    raw_data = raw_data[model_idx]

    # Initialize spec dictionary
    spec = {}

    # Fields to extract from the Excel file
    field_names = [
        "model",
        "seriesid",
        "seriesname",
        "frequency",
        "units",
        "transformation",
        "category",
        "region",
    ]
    for field in field_names:
        if field in raw_data.columns:
            spec[field] = raw_data[field].tolist()
        else:
            raise ValueError(f"{field} column missing from model specification.")

    # Sort all fields of 'Spec' in order of decreasing frequency
    frequency_order = ["d", "w", "m", "q", "sa", "a"]
    permutation = []
    for freq in frequency_order:
        permutation.extend(np.where(np.array(spec["frequency"]) == freq)[0])

    for field in spec.keys():
        spec[field] = [spec[field][i] for i in permutation]

    # Transformations
    transformation_map = {
        "lin": "Levels (No Transformation)",
        "chg": "Change (Difference)",
        "ch1": "Year over Year Change (Difference)",
        "pch": "Percent Change",
        "pc1": "Year over Year Percent Change",
        "pca": "Percent Change (Annual Rate)",
        "cch": "Continuously Compounded Rate of Change",
        "cca": "Continuously Compounded Annual Rate of Change",
        "log": "Natural Log",
    }
    spec["unitstransformed"] = [
        transformation_map.get(trans, trans) for trans in spec["transformation"]
    ]

    # Summarize model specification
    print("Table 1: Model specification")
    try:
        tabular = pd.DataFrame(
            {
                "SeriesID": spec["seriesid"],
                "SeriesName": spec["seriesname"],
                "Units": spec["units"],
                "Transformation": spec["unitstransformed"],
            }
        )
        print(tabular)
    except Exception as e:
        print(f"Failed to display table: {e}")

    return spec


# Example usage:
# spec = load_spec('your_spec_file.xlsx')
