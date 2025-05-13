"""
This script generates a world map visualization showing COVID-19 cases
and vaccination rates by country using Plotly Express.
"""

import pandas as pd
import plotly.express as px
import os

def load_processed_data():
    """Load the processed COVID-19 data."""
    try:
        return pd.read_csv('processed_covid_data.csv')
    except FileNotFoundError:
        print("Error: processed_covid_data.csv not found. Run data_cleaning.py first.")
        exit(1)

def create_latest_data_snapshot(df):
    """
    Create a dataframe with the latest data for each country.
    
    Args:
        df: Processed COVID-19 DataFrame
    
    Returns:
        DataFrame with latest data for each country
    """
    # Convert date to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Get the latest date in the dataset
    latest_date = df['date'].max()
    print(f"Creating maps using data from {latest_date.strftime('%Y-%m-%d')}")
    
    # Get all data from the latest date
    latest_df = df[df['date'] == latest_date].copy()
    
    # Keep only necessary columns
    cols_to_keep = ['iso_code', 'location', 'total_cases', 'total_deaths', 
                    'total_cases_per_million', 'total_deaths_per_million',
                    'people_vaccinated_per_hundred', 'people_fully_vaccinated_per_hundred']
    
    # Keep only columns that exist in the dataset
    cols_to_keep = [col for col in cols_to_keep if col in latest_df.columns]
    
    latest_df = latest_df[cols_to_keep]
    
    return latest_df

def create_cases_choropleth(df):
    """
    Create a choropleth map of total COVID-19 cases per million.
    
    Args:
        df: DataFrame with latest COVID data including iso_code and total_cases_per_million
    """
    # Drop rows with missing data
    map_df = df.dropna(subset=['iso_code', 'total_cases_per_million'])
    
    # Create the choropleth map
    fig = px.choropleth(map_df, 
                        locations="iso_code",
                        color="total_cases_per_million", 
                        hover_name="location",
                        color_continuous_scale=px.colors.sequential.Reds,
                        title="COVID-19 Cases per Million Population (Latest Data)",
                        labels={"total_cases_per_million": "Cases per Million"})
    
    fig.update_layout(
        geo=dict(
            showframe=False,
            showcoastlines=True,
            projection_type='equirectangular'
        ),
        width=1200,
        height=700
    )
    
    # Save the figure
    if not os.path.exists('visualizations'):
        os.makedirs('visualizations')
    
    fig.write_html("visualizations/covid_cases_map.html")
    
    # Also save as image for reports
    fig.write_image("visualizations/covid_cases_map.png", scale=2)
    
    print("Cases choropleth map created and saved.")

def create_deaths_choropleth(df):
    """
    Create a choropleth map of total COVID-19 deaths per million.
    
    Args:
        df: DataFrame with latest COVID data including iso_code and total_deaths_per_million
    """
    # Drop rows with missing data
    map_df = df.dropna(subset=['iso_code', 'total_deaths_per_million'])
    
    # Create the choropleth map
    fig = px.choropleth(map_df, 
                        locations="iso_code",
                        color="total_deaths_per_million", 
                        hover_name="location",
                        color_continuous_scale=px.colors.sequential.Purples,
                        title="COVID-19 Deaths per Million Population (Latest Data)",
                        labels={"total_deaths_per_million": "Deaths per Million"})
    
    fig.update_layout(
        geo=dict(
            showframe=False,
            showcoastlines=True,
            projection_type='equirectangular'
        ),
        width=1200,
        height=700
    )
    
    # Save the figure
    if not os.path.exists('visualizations'):
        os.makedirs('visualizations')
    
    fig.write_html("visualizations/covid_deaths_map.html")
    
    # Also save as image for reports
    fig.write_image("visualizations/covid_deaths_map.png", scale=2)
    
    print("Deaths choropleth map created and saved.")

def create_vaccination_choropleth(df):
    """
    Create a choropleth map of vaccination rates.
    
    Args:
        df: DataFrame with latest COVID data including iso_code and people_fully_vaccinated_per_hundred
    """
    # Check if vaccination data exists
    if 'people_fully_vaccinated_per_hundred' not in df.columns:
        print("Vaccination data not available in the dataset.")
        return
    
    # Drop rows with missing data
    map_df = df.dropna(subset=['iso_code', 'people_fully_vaccinated_per_hundred'])
    
    # Create the choropleth map
    fig = px.choropleth(map_df, 
                        locations="iso_code",
                        color="people_fully_vaccinated_per_hundred", 
                        hover_name="location",
                        color_continuous_scale=px.colors.sequential.Greens,
                        title="Percentage of Population Fully Vaccinated (Latest Data)",
                        labels={"people_fully_vaccinated_per_hundred": "% Fully Vaccinated"})
    
    fig.update_layout(
        geo=dict(
            showframe=False,
            showcoastlines=True,
            projection_type='equirectangular'
        ),
        width=1200,
        height=700
    )
    
    # Save the figure
    if not os.path.exists('visualizations'):
        os.makedirs('visualizations')
    
    fig.write_html("visualizations/vaccination_map.html")
    
    # Also save as image for reports
    fig.write_image("visualizations/vaccination_map.png", scale=2)
    
    print("Vaccination choropleth map created and saved.")

if __name__ == "__main__":
    # Load processed data
    covid_data = load_processed_data()
    
    # Create a snapshot of the latest data
    latest_data = create_latest_data_snapshot(covid_data)
    
    # Create choropleth maps
    create_cases_choropleth(latest_data)
    create_deaths_choropleth(latest_data)
    create_vaccination_choropleth(latest_data)
    
    print("All choropleth maps have been generated and saved to the 'visualizations' directory.")
