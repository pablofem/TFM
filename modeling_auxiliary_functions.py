import pandas as pd
import numpy as np
import datetime as dt
from tqdm import tqdm
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.decomposition import PCA
import scipy.stats as stats
import matplotlib.pyplot as plt


def pivot_from_column_ref(df, index_col, new_columns_ref):
    """
    Transforms the dataframe to have separate columns
    for each city and variable combination.

    Parameters:
    df (pd.DataFrame): The original dataframe with
    columns 'time', 'city_name', and variables.

    Returns:
    pd.DataFrame: The transformed dataframe.
    """

    variables = df.columns.tolist()
    variables.remove(index_col)
    variables.remove(new_columns_ref)

    # Create a list to store the transformed dataframes for each variable
    transformed_dfs = []

    for var in variables:
        # Pivot the dataframe for the current variable
        pivot_df = df.pivot(
            index=index_col,
            columns=new_columns_ref,
            values=var)

        # Rename the columns to include the variable name
        pivot_df.columns = [f"{var}_{city.strip()}" for city in pivot_df.columns]

        # Add the pivoted dataframe to the list
        transformed_dfs.append(pivot_df)

    # Concatenate all the transformed dataframes along the columns
    final_df = pd.concat(transformed_dfs, axis=1)

    # Reset the index to make 'time' a column again
    final_df.reset_index(inplace=True)

    return final_df


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
    if offer_type == "daily_market":
        execution_timestamp = date - dt.timedelta(hours=12)
    elif offer_type == "third_session":
        execution_timestamp = date - dt.timedelta(hours=1)
    elif offer_type == "sixth_session":
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


def add_total_load(interval_dataset, basic_dataset):
    dataset = basic_dataset[["time", "total_load_actual"]]
    complete_df = pd.merge(interval_dataset, dataset, how="inner", on="time")
    complete_df = complete_df[~complete_df["total_load_actual"].isna()]
    return complete_df


def train_model(X_train, y_train, model_name):
    if model_name == "RFO":
        model = RandomForestRegressor(random_state=42)
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [2, 5, 8],
            'min_samples_split': [2, 4, 8],  # Note: changed from 1 to 2
        }
    elif model_name == "XGB":
        model = XGBRegressor(objective='reg:squarederror', random_state=42)
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [2, 5, 8],
            'learning_rate': [0.01, 0.1, 0.2],
        }
    elif model_name == "MLP":
        model = MLPRegressor(random_state=42)
        param_grid = {
            'hidden_layer_sizes': [(16,), (32,), (64,)],
            'activation': ['relu', 'tanh'],
            'max_iter': [100, 500, 1000]
        }
    elif model_name == "KNN":
        model = KNeighborsRegressor()
        param_grid = {
            'n_neighbors': [3, 5, 7],
            'weights': ['uniform', 'distance'],
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
        }
    else:
        raise ValueError("Model name must be 'RFO', 'XGB', 'MLP', or 'KNN'.")

    print(f"Training {model_name} model ...")
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1, scoring='neg_median_absolute_error', verbose=True)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    return best_model



def apply_pca(min_variance, X_train, X_test):
    """
    Applies PCA to the input dataframe X and returns a dataframe with
    columns: 'components', 'accumulated_variance'. Also returns the
    dataframe X reduced to only the components that give at least the min_variance.
    
    Parameters:
    X (pd.DataFrame): The input dataframe with predictor columns.
    min_variance (float): The minimum variance that must be explained by the selected components.
    
    Returns:
    pca_summary (pd.DataFrame): A dataframe with columns 'components', 'accumulated_variance'.
    X_reduced (pd.DataFrame): The input dataframe X reduced to the selected components.
    """
    pca_transformer = PCA()
    X_train = pca_transformer.fit_transform(X_train)
    X_test = pca_transformer.transform(X_test)
    
    # Calculate accumulated variance
    cumulative_variance = pca_transformer.explained_variance_ratio_.cumsum()
    
    # Determine the number of components required to reach min_variance
    num_components = (cumulative_variance >= min_variance).argmax() + 1
    
    # Create the summary dataframe
    pca_summary = pd.DataFrame({
        'components': range(1, len(cumulative_variance) + 1),
        'accumulated_variance': cumulative_variance
    })
    
    # Reduce the dataset to the selected components
    X_train = pd.DataFrame(X_train[:, :num_components])
    X_test = pd.DataFrame(X_test[:, :num_components])
    
    return pca_summary, num_components, X_train, X_test


def split_train_test_date(data, target_col, sep_date):
    data = data[~data[target_col].isna()]

    train_data = X_train = data[data["time"] < sep_date]
    test_data = data[data["time"] >= sep_date]

    X_train = train_data.drop([target_col, "time"], axis=1)
    y_train = train_data[target_col].values

    X_test = test_data.drop([target_col, "time"], axis=1)
    y_test = test_data[["time", target_col]]

    return X_train, y_train, X_test, y_test


def prep_results_df(X_test, y_test, models_dic, offer_type):
    tmp_df_list = []
    for model_name, model_predictor in models_dic.items():
        tmp_df = y_test.copy()
        tmp_df["prediction"] = model_predictor.predict(X_test).round(0)
        tmp_df["model"] = model_name
        tmp_df["offer_type"] = offer_type
        tmp_df_list.append(tmp_df)

    results_df = pd.concat(tmp_df_list)
    return results_df


def plot_feature_importance(X_train, model):

    # Get feature importances
    importances = model.feature_importances_

    # Create a DataFrame for feature importances
    feature_importances_df = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    print("================= ANÁLISIS DE IMPORTANCIA DE VARIABLES =================")
    # Plot feature importances
    plt.figure(figsize=(6, 3))
    plt.barh(feature_importances_df['Feature'], feature_importances_df['Importance'])
    plt.xlabel('Importance')
    plt.title(f'Feature Importances')
    plt.gca().invert_yaxis()
    plt.grid()
    plt.show()


def check_gaussian_residuals(results_df, model_name):

    data = results_df[results_df["model"] == model_name]
    data = data["total_load_actual"] - data["prediction"]
    # Shapiro-Wilk Test
    shapiro_test = stats.shapiro(data)

    # D'Agostino's K-squared Test
    k2_test = stats.normaltest(data)

    # Conclusion based on p-values
    alpha = 0.05

    print("================= ANÁLISIS DE RESIDUOS =================")

    if shapiro_test.pvalue > alpha and k2_test.pvalue > alpha:
        print(f"Los residuos forman una distribución Gaussiana.")
    else:
        print(f"Los residuos no forman una distribución Gaussiana.")

    # Kurtosis
    kurtosis = stats.kurtosis(data, fisher=True)

    if kurtosis > 0:
        print(f"La distribución es leptocúrtica (K={round(kurtosis,2)})")
    elif kurtosis < 0:
        print(f"La distribución es platicúrtica (K={round(kurtosis,2)})")
    else:
        print(f"La distribución es mesocúrtica (K={round(kurtosis,2)})")

    print("\n")
