"""
This module contains functions for exploratory data analysis and visualization of COVID-19 data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# Set plot style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("colorblind")

def plot_total_cases_over_time(df, countries=None, figsize=(12, 8)):
    """
    Plot total COVID-19 cases over time for selected countries.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        COVID-19 dataset
    countries : list
        List of countries to include. If None, top 5 countries by total cases are used.
    figsize : tuple
        Figure size
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure object containing the plot
    """
    # If no countries specified, use top 5 by total cases
    if countries is None:
        latest_data = df.sort_values('date').groupby('location').last().reset_index()
        top_countries = latest_data.sort_values('total_cases', ascending=False).head(5)['location'].tolist()
        countries = top_countries
    
    # Filter for the specified countries
    df_filtered = df[df['location'].isin(countries)]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot each country
    for country in countries:
        country_data = df_filtered[df_filtered['location'] == country]
        ax.plot(country_data['date'], country_data['total_cases'], label=country, linewidth=2)
    
    # Set labels and title
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Total Cases', fontsize=12)
    ax.set_title('COVID-19 Total Cases Over Time', fontsize=14, fontweight='bold')
    
    # Format y-axis to show numbers in millions/thousands
    ax.get_yaxis().set_major_formatter(
        plt.FuncFormatter(lambda x, loc: f"{int(x):,}")
    )
    
    # Add legend
    ax.legend(fontsize=10)
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Rotate x-tick labels for better readability
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    return fig

def plot_total_deaths_over_time(df, countries=None, figsize=(12, 8)):
    """
    Plot total COVID-19 deaths over time for selected countries.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        COVID-19 dataset
    countries : list
        List of countries to include. If None, top 5 countries by total deaths are used.
    figsize : tuple
        Figure size
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure object containing the plot
    """
    # If no countries specified, use top 5 by total deaths
    if countries is None:
        latest_data = df.sort_values('date').groupby('location').last().reset_index()
        top_countries = latest_data.sort_values('total_deaths', ascending=False).head(5)['location'].tolist()
        countries = top_countries
    
    # Filter for the specified countries
    df_filtered = df[df['location'].isin(countries)]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot each country
    for country in countries:
        country_data = df_filtered[df_filtered['location'] == country]
        ax.plot(country_data['date'], country_data['total_deaths'], label=country, linewidth=2)
    
    # Set labels and title
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Total Deaths', fontsize=12)
    ax.set_title('COVID-19 Total Deaths Over Time', fontsize=14, fontweight='bold')
    
    # Format y-axis to show numbers with commas
    ax.get_yaxis().set_major_formatter(
        plt.FuncFormatter(lambda x, loc: f"{int(x):,}")
    )
    
    # Add legend
    ax.legend(fontsize=10)
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Rotate x-tick labels for better readability
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    return fig

def plot_new_cases(df, countries=None, rolling_window=7, figsize=(12, 8)):
    """
    Plot daily new COVID-19 cases for selected countries with rolling average.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        COVID-19 dataset
    countries : list
        List of countries to include. If None, top 5 countries by total cases are used.
    rolling_window : int
        Window size for rolling average
    figsize : tuple
        Figure size
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure object containing the plot
    """
    # If no countries specified, use top 5 by total cases
    if countries is None:
        latest_data = df.sort_values('date').groupby('location').last().reset_index()
        top_countries = latest_data.sort_values('total_cases', ascending=False).head(5)['location'].tolist()
        countries = top_countries
    
    # Filter for the specified countries
    df_filtered = df[df['location'].isin(countries)]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot each country
    for country in countries:
        country_data = df_filtered[df_filtered['location'] == country]
        
        # Calculate rolling average
        rolling_avg = country_data['new_cases'].rolling(window=rolling_window).mean()
        
        # Plot rolling average
        ax.plot(country_data['date'], rolling_avg, label=f"{country} ({rolling_window}-day avg)", linewidth=2)
    
    # Set labels and title
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel(f'New Cases ({rolling_window}-day Rolling Average)', fontsize=12)
    ax.set_title('COVID-19 New Cases Over Time', fontsize=14, fontweight='bold')
    
    # Format y-axis to show numbers with commas
    ax.get_yaxis().set_major_formatter(
        plt.FuncFormatter(lambda x, loc: f"{int(x):,}")
    )
    
    # Add legend
    ax.legend(fontsize=10)
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Rotate x-tick labels for better readability
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    return fig

def plot_death_rate(df, countries=None, figsize=(12, 8)):
    """
    Plot COVID-19 death rate (deaths/cases) for selected countries.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        COVID-19 dataset
    countries : list
        List of countries to include. If None, top 5 countries by total deaths are used.
    figsize : tuple
        Figure size
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure object containing the plot
    """
    # If no countries specified, use top 5 by total deaths
    if countries is None:
        latest_data = df.sort_values('date').groupby('location').last().reset_index()
        top_countries = latest_data.sort_values('total_deaths', ascending=False).head(5)['location'].tolist()
        countries = top_countries
    
    # Filter for the specified countries
    df_filtered = df[df['location'].isin(countries)]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot each country
    for country in countries:
        country_data = df_filtered[df_filtered['location'] == country]
        # Calculate death rate if not already in dataset
        if 'death_rate' not in country_data.columns:
            country_data = country_data.copy()
            country_data['death_rate'] = np.where(
                country_data['total_cases'] > 0,
                (country_data['total_deaths'] / country_data['total_cases']) * 100,
                np.nan
            )
        
        ax.plot(country_data['date'], country_data['death_rate'], label=country, linewidth=2)
    
    # Set labels and title
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Death Rate (%)', fontsize=12)
    ax.set_title('COVID-19 Death Rate Over Time', fontsize=14, fontweight='bold')
    
    # Format y-axis to show percentage
    ax.get_yaxis().set_major_formatter(
        plt.FuncFormatter(lambda x, loc: f"{x:.1f}%")
    )
    
    # Add legend
    ax.legend(fontsize=10)
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Rotate x-tick labels for better readability
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    return fig

def plot_top_countries_bar(df, metric='total_cases', top_n=10, figsize=(12, 8)):
    """
    Plot bar chart of top countries by a specific metric.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        COVID-19 dataset
    metric : str
        Column name to sort by
    top_n : int
        Number of top countries to show
    figsize : tuple
        Figure size
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure object containing the plot
    """
    # Get latest data for each country
    latest_data = df.sort_values('date').groupby('location').last().reset_index()
    
    # Filter continental aggregates
    continents = ['World', 'Asia', 'Europe', 'North America', 'South America', 
                  'Africa', 'Oceania', 'European Union']
    latest_data = latest_data[~latest_data['location'].isin(continents)]
    
    # Sort and get top N countries
    top_countries = latest_data.sort_values(metric, ascending=False).head(top_n)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Define colors
    colors = sns.color_palette("viridis", top_n)
    
    # Create bar plot
    bars = ax.bar(top_countries['location'], top_countries[metric], color=colors)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}',
                ha='center', va='bottom', rotation=0, fontsize=9)
    
    # Set labels and title
    ax.set_xlabel('Country', fontsize=12)
    
    # Set appropriate y-label based on the metric
    if metric == 'total_cases':
        ylabel = 'Total Cases'
    elif metric == 'total_deaths':
        ylabel = 'Total Deaths'
    elif metric == 'total_cases_per_million':
        ylabel = 'Total Cases per Million'
    elif metric == 'total_deaths_per_million':
        ylabel = 'Total Deaths per Million'
    else:
        ylabel = metric.replace('_', ' ').title()
    
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(f'Top {top_n} Countries by {ylabel}', fontsize=14, fontweight='bold')
    
    # Format y-axis to show numbers with commas
    ax.get_yaxis().set_major_formatter(
        plt.FuncFormatter(lambda x, loc: f"{int(x):,}")
    )
    
    # Rotate x-tick labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    
    return fig

def plot_cases_vs_deaths_scatter(df, figsize=(14, 10)):
    """
    Create a scatter plot of total cases vs. total deaths by country.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        COVID-19 dataset
    figsize : tuple
        Figure size
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure object containing the plot
    """
    # Get latest data for each country
    latest_data = df.sort_values('date').groupby('location').last().reset_index()
    
    # Filter continental aggregates and countries with NaN values
    continents = ['World', 'Asia', 'Europe', 'North America', 'South America', 
                  'Africa', 'Oceania', 'European Union']
    data_filtered = latest_data[
        (~latest_data['location'].isin(continents)) & 
        (latest_data['total_cases'].notna()) & 
        (latest_data['total_deaths'].notna())
    ]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create scatter plot with population as size
    scatter = ax.scatter(
        data_filtered['total_cases'], 
        data_filtered['total_deaths'],
        s=data_filtered['population'] / 1e6,  # Adjust size based on population
        alpha=0.6,
        c=data_filtered['total_deaths'] / data_filtered['total_cases'],  # Color by death rate
        cmap='coolwarm'
    )
    
    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Death Rate (Deaths/Cases)', fontsize=10)
    
    # Add country labels to some points
    # Label top 10 countries by cases
    top_countries = data_filtered.sort_values('total_cases', ascending=False).head(10)
    for _, row in top_countries.iterrows():
        ax.annotate(
            row['location'],
            (row['total_cases'], row['total_deaths']),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=9
        )
    
    # Set labels and title
    ax.set_xlabel('Total Cases', fontsize=12)
    ax.set_ylabel('Total Deaths', fontsize=12)
    ax.set_title('COVID-19 Cases vs. Deaths by Country', fontsize=14, fontweight='bold')
    
    # Format axes to show numbers with commas
    ax.get_xaxis().set_major_formatter(
        plt.FuncFormatter(lambda x, loc: f"{int(x):,}")
    )
    ax.get_yaxis().set_major_formatter(
        plt.FuncFormatter(lambda x, loc: f"{int(x):,}")
    )
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Add a reference line for a specific death rate (e.g., 2%)
    x_range = ax.get_xlim()
    reference_death_rate = 0.02  # 2%
    ax.plot(x_range, [x * reference_death_rate for x in x_range], 'r--', alpha=0.5)
    ax.text(x_range[1] * 0.7, x_range[1] * reference_death_rate * 0.7, 
            f"{reference_death_rate*100:.1f}% Death Rate", 
            color='red', fontsize=10, alpha=0.7)
    
    plt.tight_layout()
    
    return fig

def plot_heatmap(df, countries=None, metric='new_cases_per_million', figsize=(14, 10)):
    """
    Create a heatmap showing a metric over time for selected countries.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        COVID-19 dataset
    countries : list
        List of countries to include. If None, top 10 countries by total cases are used.
    metric : str
        Metric to visualize on the heatmap
    figsize : tuple
        Figure size
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure object containing the plot
    """
    # If no countries specified, use top 10 by total cases
    if countries is None:
        latest_data = df.sort_values('date').groupby('location').last().reset_index()
        top_countries = latest_data.sort_values('total_cases', ascending=False).head(10)['location'].tolist()
        countries = top_countries
    
    # Filter for the specified countries
    df_filtered = df[df['location'].isin(countries)]
    
    # Pivot data for heatmap
    # Sample monthly data to reduce size
    df_monthly = df_filtered.copy()
    df_monthly['month_year'] = df_monthly['date'].dt.strftime('%Y-%m')
    monthly_data = df_monthly.groupby(['location', 'month_year'])[metric].mean().reset_index()
    
    # Pivot the data
    pivot_data = monthly_data.pivot(index='location', columns='month_year', values=metric)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    sns.heatmap(pivot_data, cmap='YlOrRd', linewidths=0.5, ax=ax, annot=False)
    
    # Set labels and title
    ax.set_title(f'{metric.replace("_", " ").title()} Heatmap by Country and Month', fontsize=14, fontweight='bold')
    ax.set_xlabel('Month-Year', fontsize=12)
    ax.set_ylabel('Country', fontsize=12)
    
    # Rotate x-tick labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    
    return fig

def analyze_and_visualize(df, countries=None):
    """
    Perform a complete exploratory data analysis with visualizations.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        COVID-19 dataset
    countries : list
        List of countries to include in the analysis
        
    Returns:
    --------
    dict
        Dictionary of figure objects
    """
    figures = {}
    
    # If no countries specified, use some notable ones
    if countries is None:
        countries = ['United States', 'India', 'Brazil', 'United Kingdom', 'France']
    
    # Generate all plots
    print("Generating total cases plot...")
    figures['total_cases'] = plot_total_cases_over_time(df, countries)
    
    print("Generating total deaths plot...")
    figures['total_deaths'] = plot_total_deaths_over_time(df, countries)
    
    print("Generating new cases plot...")
    figures['new_cases'] = plot_new_cases(df, countries)
    
    print("Generating death rate plot...")
    figures['death_rate'] = plot_death_rate(df, countries)
    
    print("Generating top countries by cases bar plot...")
    figures['top_cases'] = plot_top_countries_bar(df, metric='total_cases')
    
    print("Generating top countries by deaths bar plot...")
    figures['top_deaths'] = plot_top_countries_bar(df, metric='total_deaths')
    
    print("Generating top countries by cases per million bar plot...")
    if 'total_cases_per_million' in df.columns:
        figures['top_cases_per_million'] = plot_top_countries_bar(df, metric='total_cases_per_million')
    
    print("Generating cases vs deaths scatter plot...")
    figures['cases_vs_deaths'] = plot_cases_vs_deaths_scatter(df)
    
    print("Generating heatmap...")
    if 'new_cases_per_million' in df.columns:
        figures['heatmap'] = plot_heatmap(df, countries)
    
    return figures
