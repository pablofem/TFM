import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.decomposition import PCA
import scipy.stats as stats
import matplotlib.pyplot as plt


def train_model(X_train, y_train, model_name):
    if model_name == "RFO":
        model = RandomForestRegressor(random_state=42)
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [2, 5, 8],
            'min_samples_split': [2, 4, 8],
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
