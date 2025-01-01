import os
import sys
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

###########################################################################################################
#                                SUMMARY FUNCTIONS                                                           #                                            
###########################################################################################################

# Define numerical & categorical columns
def print_feature_types(df):
    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = df.select_dtypes(include=['object']).columns.tolist()

    # Print columns
    print(f'We have {len(numeric_features)} numerical features : {numeric_features}')
    print(f'\nWe have {len(categorical_features)} categorical features : {categorical_features}')


# categorical columns
def print_categories(df):
    columns = df.select_dtypes(include=['object']).columns
    for col in columns:
        print(f"Categories in \033[1m'{col}'\033[0m there are \033[1m{df[col].nunique()}\033[0m categories: \033[1m{df[col].unique()}\033[0m")


#Helper function for Average, Median and Standard deviation 
def calculate_key_metrics(df, numeric_columns=None, handle_missing='drop'):
    """
    Calculate key metrics for specified numeric columns in the DataFrame.
    
    Parameters:
    - df (pandas.DataFrame): The DataFrame containing the data.
    - numeric_columns (list): List of numeric column names to summarize. If None, all numeric columns are considered.
    - handle_missing (str): 'drop' to drop missing values, 'fill' to fill them with the column mean, or 'ignore'.
    
    Returns:
    - dict: Dictionary containing the calculated key metrics.
    """
    if numeric_columns is None:
        # Automatically select numeric columns if none are specified
        numeric_columns = df.select_dtypes(include='number').columns.tolist()

    # Handle missing values based on the chosen method
    if handle_missing == 'drop':
        df = df.dropna(subset=numeric_columns)
    elif handle_missing == 'fill':
        df[numeric_columns] = df[numeric_columns].apply(lambda x: x.fillna(x.mean()))

    # Initialize the key metrics dictionary
    key_metrics = {}

    # Calculate the key metrics for each numeric column dynamically
    for col in numeric_columns:
        if col in df.columns:
            key_metrics.update({
                f"Average {col}": df[col].mean(),
                f"Median {col}": df[col].median(),
                f"Std Dev {col}": df[col].std()
            })

    return key_metrics


# Helper function for count and percentages calculations
def calculate_status_distribution(df, column_name=None):
    """
    Calculate the count and percentage distribution of values in a binary categorical column.
    
    Parameters:
    - df (pandas.DataFrame): The DataFrame containing the data.
    - column_name (str): The name of the column to analyze (e.g., 'Churn').
    
    Returns:
    - dict: A dictionary with the counts and percentages for each category.
    """
    # Ensure the column exists in the DataFrame
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in the DataFrame.")
    
    # Ensure the DataFrame is not empty
    if df.empty:
        raise ValueError("The DataFrame is empty.")
    
    # Total customers
    total_customers = len(df)

    # Count values and calculate percentages
    value_counts = df[column_name].value_counts()
    percentages = (value_counts / total_customers * 100).round(2)
    
    # Create a results DataFrame
    results = pd.DataFrame({
        column_name: value_counts.index,
        "Counts": value_counts.values,
        "Percentages": percentages.values
    })
    
    return results


###########################################################################################################
#                                PLOT FUNCTIONS                                                           #                                            
###########################################################################################################

# Helper function for pie chart
def plot_pie_chart(data, column, labels, colors, ax):
    value_counts = data[column].value_counts()
    ax.pie(
        value_counts,
        labels=labels,
        colors=colors,
        autopct='%1.1f%%',
        startangle=90,
        wedgeprops={'edgecolor': 'white'}
    )
    ax.set_title(f' Distribution of {column}')


# Helper function for KDE plot
def plot_kde(data, column, ax, color, title):
    sns.kdeplot(data[column], label=title, ax=ax, color=color, fill=True)
    ax.set_title(f'KDE Plot: {title}')


# Helper function to plot countplot
def plot_countplot(data, column, ax, color, title):
    sns.countplot(x=column, data=data, ax=ax, color=color)
    ax.set_title(title)


# Helper function to plot side-by-side bars using countplot
def count_bar_plot_with_legend(data, column, target_column, ax):
    """
    Plots a side-by-side bar chart for a given feature with the target variable (e.g., Churn) as the hue.
    
    Parameters:
    - data: DataFrame containing the dataset
    - column: Feature to plot against target_column
    - target_column: Target variable (e.g., Churn) for hue
    - ax: Axis object to plot on
    """
    sns.countplot(x=column, hue=target_column, data=data, ax=ax, palette='tab10', dodge=True)
    ax.set_title(f'{target_column} by {column}')
    ax.set_ylabel('Count')
    ax.set_xlabel(column)
    ax.legend(title=target_column, loc='upper right')

    # Rotate x-axis labels by 45 degrees
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')


def plot_heatmap(data, columns, ax, cmap='coolwarm'):
    correlation_matrix = data[columns].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap=cmap, fmt='.2f', ax=ax, vmin=-1, vmax=1, center=0, cbar_kws={"shrink": 0.8})
    ax.set_title('Correlation Heatmap')


def box_plot(data, value_columns, category_column):
    plt.figure(figsize=(12, 6))
    for i, col in enumerate(value_columns, 1):
        plt.subplot(1, len(value_columns), i)
        sns.boxplot(x=category_column, y=col, data=data, palette="Set2")
        plt.title(f'{col} by {category_column}')
