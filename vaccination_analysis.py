#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script analyzes vaccination rollouts across selected countries,
plotting cumulative vaccinations over time and comparing vaccination rates.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime

# Set style for visualizations
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("deep")

def load_processed_data():
    """Load the processed COVID-19 data."""
    try:
        return pd.read_csv('processed_covid_data.csv')
    except FileNotFoundError:
        print("Error: processed_covid_data.csv not found. Run data_cleaning.py first.")
        exit(1)

def analyze_vaccinations(df, countries):
    """
    Analyze vaccination progress for selected countries.
    
    Args:
        df: Processed COVID-19 DataFrame
        countries: List of countries to analyze
    """
    # Filter for selected countries
    df_vax = df[df['location'].isin(countries)].copy()
    
    # Ensure date is in datetime format
    df_vax['date'] = pd.to_datetime(df_vax['date'])
    
    # Create figure for line charts
    plt.figure(figsize=(14, 10))
    
    # Plot 1: Total vaccinations over time
    plt.subplot(2, 1, 1)
    
    for country in countries:
        country_data = df_vax[df_vax['location'] == country]
        
        # Some countries might have NaN values for vaccinations
        if 'total_vaccinations' in country_data.columns and not country_data['total_vaccinations'].isna().all():
            plt.plot(country_data['date'], country_data['total_vaccinations'], label=country, linewidth=2)
    
    plt.title('Total COVID-19 Vaccinations Over Time', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Total Vaccinations', fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    
    # Plot 2: Percentage of population vaccinated
    plt.subplot(2, 1, 2)
    
    for country in countries:
        country_data = df_vax[df_vax['location'] == country]
        
        # Check if percentage columns exist
        if 'people_vaccinated_per_hundred' in country_data.columns and not country_data['people_vaccinated_per_hundred'].isna().all():
            plt.plot(country_data['date'], country_data['people_vaccinated_per_hundred'], 
                     label=f"{country} (At least 1 dose)", linewidth=2)
        
        if 'people_fully_vaccinated_per_hundred' in country_data.columns and not country_data['people_fully_vaccinated_per_hundred'].isna().all():
            plt.plot(country_data['date'], country_data['people_fully_vaccinated_per_hundred'], 
                     label=f"{country} (Fully)", linestyle='--', linewidth=2)
    
    plt.title('Percentage of Population Vaccinated', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Percentage of Population', fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('visualizations/vaccination_progress_lines.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create pie charts for latest vaccination status
    plot_latest_vaccination_status(df_vax, countries)

def plot_latest_vaccination_status(df, countries):
    """
    Create pie charts showing the latest vaccination status for each country.
    
    Args:
        df: Processed COVID-19 DataFrame
        countries: List of countries to analyze
    """
    latest_data = []
    
    for country in countries:
        country_df = df[df['location'] == country]
        
        # Get the latest date with vaccination data
        if 'people_vaccinated_per_hundred' in country_df.columns:
            latest = country_df[country_df['people_vaccinated_per_hundred'].notna()].sort_values('date').tail(1)
            if not latest.empty:
                latest_data.append(latest)
    
    if not latest_data:
        print("No vaccination data available for the selected countries.")
        return
    
    latest_df = pd.concat(latest_data)
    
    # Create figure for pie charts
    fig, axes = plt.subplots(1, len(latest_df), figsize=(5*len(latest_df), 6))
    
    # If only one country, axes is not an array
    if len(latest_df) == 1:
        axes = [axes]
    
    for i, (_, row) in enumerate(latest_df.iterrows()):
        country = row['location']
        vaccinated = row.get('people_vaccinated_per_hundred', 0)
        
        if pd.isna(vaccinated):
            vaccinated = 0
            
        unvaccinated = 100 - vaccinated
        
        # Data for pie chart
        sizes = [vaccinated, unvaccinated]
        labels = ['Vaccinated (at least one dose)', 'Unvaccinated']
        colors = ['#4CAF50', '#E0E0E0']
        
        # Create pie chart
        axes[i].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        axes[i].set_title(f'Vaccination Status in {country}\n(as of {row["date"].strftime("%Y-%m-%d")})', fontsize=14)
    
    plt.tight_layout()
    plt.savefig('visualizations/vaccination_status_pies.png', dpi=300, bbox_inches='tight')
    plt.close()

def compare_vaccination_rates(df, countries):
    """
    Compare vaccination rates between countries using a bar chart.
    
    Args:
        df: Processed COVID-19 DataFrame
        countries: List of countries to analyze
    """
    # Get the latest data for each country
    latest_data = []
    
    for country in countries:
        country_df = df[df['location'] == country]
        
        # Find the latest date with vaccination data
        if 'people_fully_vaccinated_per_hundred' in country_df.columns:
            latest = country_df[country_df['people_fully_vaccinated_per_hundred'].notna()].sort_values('date').tail(1)
            if not latest.empty:
                latest_data.append(latest)
    
    if not latest_data:
        print("No full vaccination data available for the selected countries.")
        return
    
    latest_df = pd.concat(latest_data)
    
    # Sort by vaccination rate
    latest_df = latest_df.sort_values('people_fully_vaccinated_per_hundred', ascending=False)
    
    # Create bar chart
    plt.figure(figsize=(12, 8))
    
    # Plot fully vaccinated percentages
    bars = plt.bar(latest_df['location'], latest_df['people_fully_vaccinated_per_hundred'], 
                  color='#1976D2', alpha=0.7)
    
    # Add data labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                 f'{height:.1f}%', ha='center', va='bottom', fontsize=10)
    
    plt.title('Fully Vaccinated Population by Country (Latest Available Data)', fontsize=16)
    plt.xlabel('Country', fontsize=12)
    plt.ylabel('Percentage of Population Fully Vaccinated', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig('visualizations/vaccination_comparison_bars.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    # Create visualizations directory if it doesn't exist
    import os
    if not os.path.exists('visualizations'):
        os.makedirs('visualizations')
        
    # Load processed data
    covid_data = load_processed_data()
    
    # Define countries of interest
    countries_to_analyze = ['United States', 'India', 'Brazil', 'United Kingdom', 'Kenya']
    
    # Analyze vaccinations
    analyze_vaccinations(covid_data, countries_to_analyze)
    
    # Compare vaccination rates
    compare_vaccination_rates(covid_data, countries_to_analyze)
    
    print("Vaccination analysis complete. Visualizations saved to 'visualizations' directory.")
