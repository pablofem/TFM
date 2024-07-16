import pandas as pd
import numpy as np


def apply_sin_cos_transform(column, column_name):
    radians = 2 * np.pi * column / column.max()
    return pd.DataFrame({
        'sin_'+column_name: np.sin(radians),
        'cos_'+column_name: np.cos(radians)
    })


def add_sin_cos_transforms(df, dt_columns):
    for col in dt_columns:
        if col == 'month':
            transformed_col = apply_sin_cos_transform(df['time'].dt.month, col)
        elif col == 'dayofweek':
            transformed_col = apply_sin_cos_transform(df['time'].dt.dayofweek, col)
        elif col == 'day':
            transformed_col = apply_sin_cos_transform(df['time'].dt.day, col)
        elif col == 'hour':
            transformed_col = apply_sin_cos_transform(df['time'].dt.hour, col)
        elif col == 'minute':
            transformed_col = apply_sin_cos_transform(df['time'].dt.minute, col)

        # Concatenating transformed columns to the original DataFrame
        df = pd.concat([df, transformed_col], axis=1)

    return df