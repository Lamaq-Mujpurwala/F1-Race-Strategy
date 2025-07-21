import pandas as pd
import os
import glob
import yaml

# --- Configuration ---
params = yaml.safe_load(open("params.yaml"))['combine']

def combine_yearly_data(start_year,end_year,input_path,output_path,file_name):
    """
    Finds all 'laps_{year}.csv' files within the specified year range,
    combines them into a single pandas DataFrame, and saves the result.
    """
    print("--- Starting Data Combination ---")
    
    # Ensure the output directory exists
    os.makedirs(output_path, exist_ok=True)

    # Find all relevant CSV files using a glob pattern
    all_files = []
    for year in range(start_year, end_year + 1):
        file_path = os.path.join(input_path, f'laps_{year}.csv')
        if os.path.exists(file_path):
            all_files.append(file_path)
            print(f"Found: {file_path}")
        else:
            print(f"Warning: Could not find file for year {year} at {file_path}")

    if not all_files:
        print("Error: No data files found. Please check the params['input_path'] path and file names.")
        return

    # Read and concatenate all found files into a single DataFrame
    df_list = [pd.read_csv(file) for file in all_files]
    master_df = pd.concat(df_list, ignore_index=True)

    print(f"\nSuccessfully combined {len(all_files)} files.")
    print(f"Master DataFrame shape: {master_df.shape}")

    # Save the combined DataFrame
    output = os.path.join(output_path, file_name)
    master_df.to_csv(output, index=False)

    print(f"--- Master dataset saved to: {output} ---")

if __name__ == '__main__':
    # To run this script, navigate to your project root in the terminal
    # and execute: python scripts/combine_datasets.py
    input_path= params['input_path']
    output_path= params['output_path'] 
    file_name= params['file_name']
    start_year= params['start_year']
    end_year = params['end_year']

    combine_yearly_data(start_year,end_year,input_path,output_path,file_name)
