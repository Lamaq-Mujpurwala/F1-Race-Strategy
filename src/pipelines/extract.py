import fastf1 as f1
import pandas as pd
import numpy as np
import yaml
import os

params = yaml.safe_load(open("params.yaml"))['extract']

def fetch_races_data(start_year,end_year,output_path):
    races_in_year = [
        {"year": 2018 ,"races": 21},
        {"year": 2019 ,"races": 21},
        {"year": 2020 ,"races": 17},
        {"year": 2021 ,"races": 22},
        {"year": 2022 ,"races": 22},
        {"year": 2023 ,"races": 23},
        {"year": 2024 ,"races": 24},
    ]

    all_races = []

    for year in races_in_year:
        for race in range(year['races']):
            session = f1.get_session(year=year['year'], gp=race+1, identifier='R')
            session.load(laps=True, weather=True)

            laps = session.laps
            laps = laps[['Time','Driver','LapNumber','Compound','Stint', 'TyreLife', 'FreshTyre','LapTime']]
            laps['LapTimeinSeconds'] = laps['LapTime'].dt.total_seconds()
            laps.drop(['LapTime'], axis=1, inplace=True)
            laps['Track'] = session.event.Country
            laps['Year'] = session.event.year

            weather = session.weather_data
            weather = weather[['Time','AirTemp', 'TrackTemp', 'Rainfall']]

            laps = laps.sort_values('Time')
            weather = weather.sort_values('Time')

            # Merge using merge_asof
            laps_with_weather = pd.merge_asof(
                laps,
                weather,
                on='Time',
                direction='backward'
            )

            # Arrange columns: all except LapTimeinSeconds, then LapTimeinSeconds last
            cols = [col for col in laps_with_weather.columns if col != 'LapTimeinSeconds'] + ['LapTimeinSeconds']
            laps_with_weather = laps_with_weather[cols]

            all_races.append(laps_with_weather)

    # Concatenate all races into a single DataFrame
    all_data = pd.concat(all_races, ignore_index=True)
    os.makedirs(os.path.dirname(output_path),exist_ok=True)
    all_data.to_csv(output_path,index=False)

    print(f"Data Fetched from {start_year} to {end_year} File Saved to {output_path}")

if __name__=="__main__":
    fetch_races_data(params['start_year'],params['end_year'],params['output_path'])
