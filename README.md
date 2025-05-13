# COVID-19 Global Data Tracker

A comprehensive data analysis project to track global COVID-19 trends including cases, deaths, recoveries, and vaccinations across countries and time periods.

## Project Overview

This project analyzes real-world COVID-19 data to:
- Track the spread of COVID-19 across different countries and regions
- Compare case numbers, death rates, and vaccination progress
- Visualize trends with informative charts and maps
- Generate insights based on data analysis

## Project Structure

- `data_collection.py`: Script for downloading COVID-19 data
- `data_processing.py`: Functions for cleaning and preparing data
- `exploratory_analysis.py`: EDA functions and visualizations
- `vaccination_analysis.py`: Vaccination-specific analysis
- `map_visualizations.py`: Functions for creating choropleth maps
- `main_analysis.ipynb`: Main Jupyter notebook that ties everything together
- `requirements.txt`: Required Python packages

## Data Source

The primary data source is the **Our World in Data COVID-19 Dataset**, which provides comprehensive, up-to-date COVID-19 data.

## Setup Instructions

1. Clone this repository:
```bash
git clone https://github.com/yourusername/covid19-global-tracker.git
cd covid19-global-tracker
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

4. Run data collection script to download the latest data:
```bash
python data_collection.py
```

5. Open and run the main analysis notebook:
```bash
jupyter notebook main_analysis.ipynb
```

## Key Features

- **Data Collection**: Automated downloading of COVID-19 data
- **Exploratory Data Analysis**: Comprehensive analysis of trends
- **Interactive Visualizations**: Dynamic charts and maps
- **Vaccination Progress Tracking**: Analysis of global vaccination efforts
- **Geographical Visualization**: Choropleth maps showing global impact

## Requirements

See `requirements.txt` for a complete list of dependencies.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Our World in Data for providing the COVID-19 dataset
- Johns Hopkins University for their contribution to COVID-19 data collection
