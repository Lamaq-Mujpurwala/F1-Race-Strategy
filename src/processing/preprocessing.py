import pandas as pd
from typing import Dict, Any

def clean_data(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    """
    Performs initial cleaning of the raw lap data.
    """
    print("--- Starting Data Cleaning ---")
    df_cleaned = df.dropna(subset=['LapTimeinSeconds']).copy()
    print(f"Dropped {len(df) - len(df_cleaned)} rows with missing LapTimeinSeconds.")

    dry_compounds = ['SOFT', 'MEDIUM', 'HARD']
    original_rows = len(df_cleaned)
    df_cleaned = df_cleaned[df_cleaned['Compound'].isin(dry_compounds)]
    print(f"Dropped {original_rows - len(df_cleaned)} rows for non-dry compounds.")

    original_rows = len(df_cleaned)
    #race_median_times = df_cleaned.groupby(['Year', 'Track'])['LapTimeinSeconds'].transform('median')
    #utlier_threshold = params.get('outlier_lap_time_percentage', 2.0)
    #df_cleaned = df_cleaned[df_cleaned['LapTimeinSeconds'] < (race_median_times * outlier_threshold)]
    #print(f"Dropped {original_rows - len(df_cleaned)} outlier laps.")
    
    print("--- Data Cleaning Complete ---")
    return df_cleaned


def create_features_for_db(df: pd.DataFrame) -> pd.DataFrame:
    """
    Selects the final columns needed for the clean dataset that will be
    stored in PostgreSQL. No one-hot encoding is done here.
    """
    print("--- Selecting Features for Database ---")
    relevant_cols = [
        'LapTimeinSeconds',
        'TyreLife',
        'LapNumber',
        'Compound',
        'Track',
        'Year',
        'Driver',
        'AirTemp',
        'TrackTemp'
    ]
    # Drop rows with missing values in the selected columns, if any
    df_selected = df[relevant_cols].dropna().copy()
    
    print(f"Selected {len(relevant_cols)} relevant columns.")
    print(f"Final shape for DB: {df_selected.shape}")
    
    return df_selected