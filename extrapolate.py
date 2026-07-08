import pandas as pd


def df_columns_to_numpy_arrays(df: pd.DataFrame, columns_map: dict):
    return {key: df[column].to_numpy() for key, column in columns_map.items()}
