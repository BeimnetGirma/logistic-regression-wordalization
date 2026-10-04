import pandas as pd


def drop_outlier_rows(df, columns=None, k=3.0):
    """Drop rows where any of the given numeric columns falls outside
    [Q1 - k*IQR, Q3 + k*IQR]. Column-agnostic so it applies to any dataset.
    Returns (cleaned_df, dropped_counts) where dropped_counts maps each
    column to how many rows it flagged (columns can overlap, so this can
    exceed the total number of rows actually removed).
    """
    if columns is None:
        columns = df.select_dtypes(include="number").columns.tolist()

    keep_mask = pd.Series(True, index=df.index)
    dropped_counts = {}

    for col in columns:
        if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
            continue
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        lower, upper = q1 - k * iqr, q3 + k * iqr
        col_mask = df[col].between(lower, upper)
        dropped_counts[col] = int((~col_mask).sum())
        keep_mask &= col_mask

    return df[keep_mask].copy(), dropped_counts

