from statsmodels.tsa.seasonal import seasonal_decompose
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


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