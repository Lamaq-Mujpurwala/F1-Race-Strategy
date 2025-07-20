import fastf1 as f1
import pandas as pd
import numpy as np
import yaml
import os
import json

def fetch_races_data(year, n_races, output_path):
    all_races = []
    for race in range(n_races):
        session = f1.get_session(year=year, gp=race+1, identifier='R')
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

    # Concatenate all races for the year into a single DataFrame
    all_data = pd.concat(all_races, ignore_index=True)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    all_data.to_csv(output_path, index=False)
    print(f"Data Fetched for year {year} with {n_races} races. File Saved to {output_path}")

def dataset_exists(output_path):
    return os.path.exists(output_path)

def fetch_new_races_current_year(year, output_path, state_file='processed_races.json', max_races=24):
    """
    Brute-force: Try to fetch up to max_races for the year, stop at the first race that does not exist or has not taken place yet.
    Update processed_races.json to be stateful.
    """
    # Load state
    if os.path.exists(state_file):
        with open(state_file, 'r') as f:
            try:
                state = json.load(f)
            except json.JSONDecodeError:
                state = {}
    else:
        state = {}
    processed = state.get(str(year), 0)

    print(f"Brute-force: Attempting to fetch up to {max_races} races for {year}. Already processed: {processed}")

    all_races = []
    new_processed = processed
    for race in range(processed, max_races):
        try:
            print(f"Fetching race {race+1} for {year}")
            session = f1.get_session(year=year, gp=race+1, identifier='R')
            session.load(laps=True, weather=True)
            laps = session.laps
            if laps is None or len(laps) == 0:
                print(f"No data for race {race+1} in {year}. Stopping.")
                break
            laps = laps[['Time','Driver','LapNumber','Compound','Stint', 'TyreLife', 'FreshTyre','LapTime']]
            laps['LapTimeinSeconds'] = laps['LapTime'].dt.total_seconds()
            laps.drop(['LapTime'], axis=1, inplace=True)
            laps['Track'] = session.event.Country
            laps['Year'] = session.event.year

            weather = session.weather_data
            weather = weather[['Time','AirTemp', 'TrackTemp', 'Rainfall']]

            laps = laps.sort_values('Time')
            weather = weather.sort_values('Time')

            laps_with_weather = pd.merge_asof(
                laps,
                weather,
                on='Time',
                direction='backward'
            )
            cols = [col for col in laps_with_weather.columns if col != 'LapTimeinSeconds'] + ['LapTimeinSeconds']
            laps_with_weather = laps_with_weather[cols]
            all_races.append(laps_with_weather)
            new_processed = race + 1
        except Exception as e:
            print(f"Error fetching race {race+1} for {year}: {e}. Stopping.")
            break

    if not all_races:
        print(f"No new races to process for {year}.")
        return

    # Save new races to file (append if file exists, else create new)
    if os.path.exists(output_path):
        existing = pd.read_csv(output_path)
        all_data = pd.concat([existing] + all_races, ignore_index=True)
    else:
        all_data = pd.concat(all_races, ignore_index=True)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    all_data.to_csv(output_path, index=False)
    print(f"Fetched and saved new races {processed+1} to {new_processed} for {year} to {output_path}")

    # Update state
    state[str(year)] = new_processed
    with open(state_file, 'w') as f:
        json.dump(state, f)

if __name__=="__main__":
    params = yaml.safe_load(open(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "params.yaml")))['extract']
    races_in_year = [
        {"year": 2018 ,"races": 21},
        #{"year": 2019 ,"races": 21},
        #{"year": 2020 ,"races": 17},
        #{"year": 2021 ,"races": 22},
        #{"year": 2022 ,"races": 22},
        #{"year": 2023 ,"races": 23},
        #{"year": 2024 ,"races": 24},
    ]
    raw_folder = params['output_path'] if 'output_path' in params else 'data/raw/'
    state_file = params.get('state_file', 'processed_races.json')
    for race_info in races_in_year:
        year = race_info['year']
        n_races = race_info['races']
        output_path = os.path.join(raw_folder, f"laps_{year}.csv")
        if dataset_exists(output_path):
            print(f"Dataset for year {year} already exists at {output_path}. Skipping...")
            continue
        fetch_races_data(year, n_races, output_path)

    fetch_new_races_current_year(2025, os.path.join(raw_folder, 'laps_2025.csv'), state_file=state_file, max_races=24)
