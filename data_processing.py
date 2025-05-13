"""
This module contains functions for cleaning and processing COVID-19 data.
It handles data loading, cleaning, and preparation for analysis.
"""

import pandas as pd
import numpy as np
from datetime import datetime

def load_data(file_path):
    """
    Load COVID-19 dataset from CSV file.
    
    Parameters:
    -----------
    file_path : str
        Path to the COVID-19 data file
        
    Returns:
    --------
    pandas.DataFrame
        Loaded COVID-19 dataset
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Data loaded successfully with {df.shape[0]} rows and {df.shape[1]} columns.")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def convert_date_column(df):
    """
    Convert 'date' column to datetime type.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        COVID-19 dataset
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with date column converted to datetime
    """
    df['date'] = pd.to_datetime(df['date'])
    return df

def filter_countries(df, countries=None):
    """
    Filter dataset for specific countries.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        COVID-19 dataset
    countries : list
        List of countries to include. If None, all countries are included.
        
    Returns:
    --------
    pandas.DataFrame
        Filtered dataset
    """
    if countries is not None:
        return df[df['location'].isin(countries)]
    return df

def calculate_derived_metrics(df):
    """
    Calculate additional metrics from the dataset.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        COVID-19 dataset
        
    Returns:
    --------
    pandas.DataFrame
        Dataset with additional calculated metrics
    """
    # Calculate death rate (mortality rate)
    # Use np.where to handle division by zero
    df['death_rate'] = np.where(
        df['total_cases'] > 0,
        df['total_deaths'] / df['total_cases'] * 100,
        np.nan
    )
    
    # Calculate cases per million (if not already in dataset)
    if 'total_cases_per_million' not in df.columns:
        df['total_cases_per_million'] = np.where(
            df['population'] > 0,
            df['total_cases'] / df['population'] * 1000000,
            np.nan
        )
    
    # Calculate deaths per million (if not already in dataset)
    if 'total_deaths_per_million' not in df.columns:
        df['total_deaths_per_million'] = np.where(
            df['population'] > 0,
            df['total_deaths'] / df['population'] * 1000000,
            np.nan
        )
    
    # Calculate vaccination rate
    if 'people_fully_vaccinated' in df.columns:
        df['vaccination_rate'] = np.where(
            df['population'] > 0,
            df['people_fully_vaccinated'] / df['population'] * 100,
            np.nan
        )
    
    return df

def handle_missing_values(df):
    """
    Handle missing values in the dataset.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        COVID-19 dataset
        
    Returns:
    --------
    pandas.DataFrame
        Dataset with handled missing values
    """
    # For critical columns, forward fill missing values within each country
    critical_columns = [
        'total_cases', 'total_deaths', 'new_cases', 'new_deaths',
        'total_cases_per_million', 'total_deaths_per_million'
    ]
    
    # Add vaccination columns if they exist
    vax_columns = [
        'total_vaccinations', 'people_vaccinated', 'people_fully_vaccinated',
        'total_boosters', 'new_vaccinations'
    ]
    
    for col in vax_columns:
        if col in df.columns:
            critical_columns.append(col)
    
    # Forward fill missing values for each country
    for country in df['location'].unique():
        country_mask = df['location'] == country
        df.loc[country_mask, critical_columns] = df.loc[country_mask, critical_columns].ffill()
    
    # For new_cases and new_deaths, replace remaining NaNs with 0
    if 'new_cases' in df.columns:
        df['new_cases'] = df['new_cases'].fillna(0)
    
    if 'new_deaths' in df.columns:
        df['new_deaths'] = df['new_deaths'].fillna(0)
    
    return df

def get_latest_data(df):
    """
    Extract the latest data point for each country.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        COVID-19 dataset
        
    Returns:
    --------
    pandas.DataFrame
        Dataset with the latest entry for each country
    """
    return df.sort_values('date').groupby('location').last().reset_index()

def filter_date_range(df, start_date=None, end_date=None):
    """
    Filter dataset for a specific date range.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        COVID-19 dataset
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str
        End date in 'YYYY-MM-DD' format
        
    Returns:
    --------
    pandas.DataFrame
        Filtered dataset
    """
    if 'date' not in df.columns:
        print("Date column not found in dataset.")
        return df
    
    # Ensure date column is datetime
    if not pd.api.types.is_datetime64_dtype(df['date']):
        df['date'] = pd.to_datetime(df['date'])
    
    if start_date:
        start_date = pd.to_datetime(start_date)
        df = df[df['date'] >= start_date]
    
    if end_date:
        end_date = pd.to_datetime(end_date)
        df = df[df['date'] <= end_date]
    
    return df

def get_top_countries_by_metric(df, metric, n=10, ascending=False):
    """
    Get top N countries by a specific metric.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        COVID-19 dataset
    metric : str
        Column name to sort by
    n : int
        Number of countries to return
    ascending : bool
        Sort in ascending order if True
        
    Returns:
    --------
    pandas.DataFrame
        Top N countries by the specified metric
    """
    if metric not in df.columns:
        print(f"Metric '{metric}' not found in dataset.")
        return None
    
    # Get the latest data for each country
    latest_data = get_latest_data(df)
    
    # Sort and return top N countries
    return latest_data.sort_values(metric, ascending=ascending).head(n)

def prepare_data_for_analysis(file_path, countries=None, start_date=None, end_date=None):
    """
    Main function to prepare data for analysis.
    
    Parameters:
    -----------
    file_path : str
        Path to the COVID-19 data file
    countries : list
        List of countries to include
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str
        End date in 'YYYY-MM-DD' format
        
    Returns:
    --------
    pandas.DataFrame
        Cleaned and prepared dataset
    """
    # Load data
    df = load_data(file_path)
    if df is None:
        return None
    
    # Process data
    df = convert_date_column(df)
    df = filter_countries(df, countries)
    df = filter_date_range(df, start_date, end_date)
    df = calculate_derived_metrics(df)
    df = handle_missing_values(df)
    
    print(f"Data prepared successfully. Shape: {df.shape}")
    return df
