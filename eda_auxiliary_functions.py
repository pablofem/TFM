from statsmodels.tsa.seasonal import seasonal_decompose
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


def make_general_histograms(df, figsize=(20, 12), xlabel="", group_cols_N=3, columns=None):
    if columns is None:
        columns = df.columns

    num_columns = df[columns].select_dtypes(include='number').columns
    num_plots = len(num_columns)
    num_rows = (num_plots + group_cols_N - 1) // group_cols_N  # Calculate the number of rows needed for 3 columns

    fig, axes = plt.subplots(num_rows, group_cols_N, figsize=figsize)
    axes = axes.flatten()

    for i, col in enumerate(num_columns):
        ax = axes[i]
        sns.histplot(df[col], kde=True, ax=ax)

        # Calculate statistics
        mean = df[col].mean()
        median = df[col].median()
        p25 = df[col].quantile(0.25)
        p75 = df[col].quantile(0.75)

        # Plot vertical lines
        ax.axvline(mean, color='#03bef7', linestyle='--', label='Media')
        ax.axvline(median, color='#0498ff', linestyle='-', label='Mediana')
        ax.axvline(p25, color='#0460ff', linestyle='-', label='Percentil 25')
        ax.axvline(p75, color='#2104ff', linestyle='-', label='Percentil 75')

        # Add title and labels
        ax.set_title(f'Histograma de {col}', fontsize=10)
        ax.set_ylabel("Frecuencia")
        ax.set_xlabel(xlabel)

        # Add legend
        ax.legend(fontsize=8)

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


def plot_stacked_barchart(df, columns, frequency='M', figsize=(20, 6),
                          title="", ylabel="", xlabel=""):
    """
    Plots a stacked bar chart from the provided dataframe.

    Parameters:
    df (pd.DataFrame): DataFrame with columns 'timestamp' and other numeric columns.
    frequency (str): Frequency for resampling the timestamp column.
    """

    # Set 'timestamp' as the DataFrame index
    dx = df.set_index('time')
    dx = dx[columns]

    # Resample the data
    dx_resampled = dx.resample(frequency).sum()

    # Plotting the stacked bar chart
    ax = dx_resampled.plot(kind='bar', stacked=True, figsize=figsize, colormap='rainbow_r')

    # Setting labels and title
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_title(title, fontsize=14)
    plt.legend(title='Variables', bbox_to_anchor=(1.05, 1), loc='upper center')

    plt.tight_layout()
    plt.show()


def make_general_boxplots(df, figsize):
    # Get numeric columns
    numeric_columns = df.select_dtypes(include=['number']).columns

    # Set up the grid
    num_plots = len(numeric_columns)
    num_cols = 3
    num_rows = (num_plots + num_cols - 1) // num_cols

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows))
    axes = axes.flatten()

    # Loop through numeric columns and create box plots
    for i, column in enumerate(numeric_columns):
        sns.boxplot(x='city_name', y=column, data=df, ax=axes[i], palette='Set3')
        axes[i].set_title(column)
        axes[i].figure.set_size_inches(*figsize)

    # Remove empty subplots if any
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


def make_seasonal_decomposition(input_data, model='multiplicative',
                                period=365*24):
    decomposition= seasonal_decompose(input_data, model=model, period=period)  # 'additive' 'multiplicative'
    plt.rc("figure", figsize=(16, 6))
    plt.rc("font", size=10)
    decomposition.plot()
    plt.show()


def make_categories_pie_chart(df):
    # Select all text columns
    text_columns = df.select_dtypes(include='object')

    # Plot pie charts for each text column
    for col in text_columns.columns:
        plt.figure(figsize=(12, 3))
        value_counts = text_columns[col].value_counts(normalize=True) * 100
        labels = [f'{label}: {pct:.1f}%' for label, pct in zip(value_counts.index, value_counts.values)]
        fontsize = 6 if len(value_counts) > 20 else 10
        plt.pie(value_counts, labels=labels, autopct='%1.1f%%', startangle=140)
        plt.title(f'Distribución de categorías en {col}')
        plt.legend(labels, loc="best", fontsize=fontsize)
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        plt.show()


def count_outliers(df):
    """
    This function takes a pandas DataFrame and returns a new DataFrame with the count of outliers 
    for each numerical column.
    
    An outlier is defined as a data point that is below Q1 - 1.5 * IQR or above Q3 + 1.5 * IQR.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame
    
    Returns:
    pd.DataFrame: A DataFrame with the count of outliers for each numerical column
    """
    
    outlier_counts = {}
    
    # Iterate through each numerical column
    for column in df.select_dtypes(include=['number']).columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = ((df[column] < lower_bound) | (df[column] > upper_bound)).sum()
        outlier_counts[column] = outliers
    
    outliers_df = pd.DataFrame(list(outlier_counts.items()), columns=['column', 'outlier_count'])
    outliers_df = outliers_df.sort_values("outlier_count", ascending=False)
    return outliers_df


def replace_outliers(df):
    # Create a copy of the DataFrame to avoid modifying the original one
    df_copy = df.copy()
    
    # Select numerical columns
    numerical_cols = df_copy.select_dtypes(include=['number']).columns
    
    for col in numerical_cols:
        # Calculate Q1 (25th percentile) and Q3 (75th percentile)
        Q1 = df_copy[col].quantile(0.25)
        Q3 = df_copy[col].quantile(0.75)
        
        # Calculate IQR (Interquartile Range)
        IQR = Q3 - Q1
        
        # Define lower and upper bounds for outliers
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Identify outliers
        outliers = (df_copy[col] < lower_bound) | (df_copy[col] > upper_bound)
        
        # Replace outliers with the median value
        median_value = df_copy[col].mean()
        df_copy.loc[outliers, col] = median_value
    
    return df_copy
