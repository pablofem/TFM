import pandas as pd
import numpy as np
import datetime as dt
from tqdm import tqdm


def apply_sin_cos_transform(column, column_name, max_val):
    radians = 2 * np.pi * column / max_val
    return pd.DataFrame({
        'sin_'+column_name: np.sin(radians),
        'cos_'+column_name: np.cos(radians)
    })


def add_sin_cos_transforms(df, dt_columns):
    for col in dt_columns:
        if col == 'month':
            transformed_col = apply_sin_cos_transform(df['time'].dt.month, col, 12)
        elif col == 'dayofweek':
            transformed_col = apply_sin_cos_transform(df['time'].dt.dayofweek, col, 6)
        elif col == 'hour':
            transformed_col = apply_sin_cos_transform(df['time'].dt.hour, col, 23)

        # Concatenating transformed columns to the original DataFrame
        df = pd.concat([df, transformed_col], axis=1)

    return df


def create_hourly_features(df, target_date, execution_timestamp, hours, columns):

    df = df[df["time"] <= execution_timestamp]

    # Set the 'time' column as the index
    df.set_index('time', inplace=True)

    new_df = pd.DataFrame([{"time": target_date}])

    for column in columns:
        new_df[f'{column}_avg_{hours}h'] = df[column].rolling(f'{hours}H').mean().tail(1).values[0]

    return new_df


def create_weekly_features(df, target_date, weeks, columns):

    # Convert 'time' column to datetime if not already
    df['time'] = pd.to_datetime(df['time'])
    weekday = target_date.weekday()
    hour = target_date.hour

    # Filter the dataframe for the specified weekday and hour
    df_filtered = df[(df['time'].dt.weekday == weekday) & (df['time'].dt.hour == hour)]

    # Get the last N weeks
    latest_time = df_filtered['time'].max()
    earliest_time = latest_time - pd.Timedelta(weeks=weeks)

    df_filtered = df_filtered[df_filtered['time'] >= earliest_time]

    # Calculate averages
    avg_data = {f'avg_{col}_{weeks}w': df_filtered[col].mean() for col in columns}

    # Create the new dataframe with one row
    new_row = {'time': target_date}
    new_row.update(avg_data)

    new_df = pd.DataFrame([new_row])

    return new_df


def prepare_features_dataframe(df, start_timestamp, end_timestamp, execution_timestamp, weekly_info_length, hourly_info_length):
    dr = pd.date_range(start_timestamp, end_timestamp, freq="H")
    columns = df.columns.tolist()
    columns.remove("time")

    row_list = []
    for date in dr:
        date = date.to_pydatetime()
        weekly_df = create_weekly_features(df, date, weekly_info_length, columns)
        hourly_df = create_hourly_features(df, date, execution_timestamp, hourly_info_length, columns)
        row_df = pd.merge(weekly_df, hourly_df, how="inner", on="time")
        row_list.append(row_df)

    predictor_dataframe = pd.concat(row_list)
    return predictor_dataframe


def get_predictions_timestamps(date, offer_type):

    start_timestamp = date
    end_timestamp = date + dt.timedelta(hours=23)
    if offer_type == "market_offer":
        execution_timestamp = date - dt.timedelta(hours=12)
    elif offer_type == "first_session":
        execution_timestamp = date - dt.timedelta(hours=1)
    elif offer_type == "last_session":
        execution_timestamp = date + dt.timedelta(hours=12)
        start_timestamp = date + dt.timedelta(hours=13)

    return start_timestamp, end_timestamp, execution_timestamp


def prepare_predictor_dataframe(df, start_date, end_date, offer_type, weekly_info_length, hourly_info_length):
    dr = pd.date_range(start_date, end_date, freq="D")

    predictor_dataframe_list = []
    for date in tqdm(dr):
        date = date.to_pydatetime()
        start_timestamp, end_timestamp, execution_timestamp = get_predictions_timestamps(date, offer_type)
        predictor_dataframe = prepare_features_dataframe(df, start_timestamp, end_timestamp, execution_timestamp, weekly_info_length, hourly_info_length)
        predictor_dataframe_list.append(predictor_dataframe)

    train_predictor_dataframe = pd.concat(predictor_dataframe_list)
    return train_predictor_dataframe


def get_demand_df(interval_df, dataset):
    interval_df = interval_df[["time"]]
    dataset = dataset[["time", "total_load_actual"]]
    demand_df = pd.merge(interval_df, dataset)
    return demand_df
