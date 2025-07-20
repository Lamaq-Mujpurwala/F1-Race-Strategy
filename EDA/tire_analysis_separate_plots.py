"""
Tire Analysis with LapTime in Seconds - Separate Plots
======================================================

This script analyzes tire performance with LapTime converted from time delta format 
to seconds and creates separate plots for better visualization.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import fastf1 as f1
import fastf1.plotting as f1plt

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")

def load_session_data(season,race,name='Bahrain',):
    """Load F1 session data"""
    print(f"Loading {name} 2022 race session...")
    bahrain_session = f1.get_session(season, race, 'R')
    bahrain_session.load(telemetry=True, laps=True, weather=True)
    return bahrain_session

def convert_laptime_to_seconds(laps):
    """Convert LapTime from time delta format to seconds"""
    print(f"Original LapTime format example: {laps['LapTime'].iloc[0]}")
    print(f"LapTime dtype: {laps['LapTime'].dtype}")
    
    # Convert LapTime from time delta to seconds
    laps['LapTime_Seconds'] = laps['LapTime'].dt.total_seconds()
    
    print(f"\nConverted LapTime_Seconds dtype: {laps['LapTime_Seconds'].dtype}")
    print(f"LapTime_Seconds range: {laps['LapTime_Seconds'].min():.3f} - {laps['LapTime_Seconds'].max():.3f} seconds")
    
    return laps

def select_tire_columns(laps):
    """Select relevant columns for tire analysis"""
    tire_laps = laps[['Driver', 'LapTime_Seconds', 'Compound', 'Stint', 'TyreLife', 'FreshTyre']]
    print("\nSelected columns for tire analysis:")
    print(tire_laps.head())
    return tire_laps

def plot_compound_boxplot(laps, race_name):
    """Create box plot comparing tire compounds"""
    print("\n=== Creating Box Plot: Lap Times by Tire Compound ===")
    
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=laps, x='Compound', y='LapTime_Seconds')
    plt.title(f'Lap Times by Tire Compound - {race_name}', fontsize=14, fontweight='bold')
    plt.ylabel('Lap Time (seconds)', fontsize=12)
    plt.xlabel('Tire Compound', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_compound_violin(laps, race_name):
    """Create violin plot comparing tire compounds"""
    print("\n=== Creating Violin Plot: Lap Time Distribution by Tire Compound ===")
    
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=laps, x='Compound', y='LapTime_Seconds')
    plt.title(f'Lap Time Distribution by Tire Compound - {race_name}', fontsize=14, fontweight='bold')
    plt.ylabel('Lap Time (seconds)', fontsize=12)
    plt.xlabel('Tire Compound', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_tire_degradation_scatter(laps, race_name):
    """Create scatter plot showing tire degradation"""
    print("\n=== Creating Scatter Plot: Tire Degradation Analysis ===")
    
    plt.figure(figsize=(12, 8))
    for compound in laps['Compound'].unique():
        compound_data = laps[laps['Compound'] == compound]
        plt.scatter(compound_data['TyreLife'], compound_data['LapTime_Seconds'], 
                   alpha=0.6, label=compound, s=30)
    
    plt.xlabel('Tire Life (laps)', fontsize=12)
    plt.ylabel('Lap Time (seconds)', fontsize=12)
    plt.title(f'Tire Degradation Analysis - {race_name}', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_tire_degradation_trends(laps, race_name):
    """Create line plot showing tire degradation trends with rolling average"""
    print("\n=== Creating Line Plot: Tire Degradation Trends ===")
    
    plt.figure(figsize=(12, 8))
    for compound in laps['Compound'].unique():
        compound_data = laps[laps['Compound'] == compound]
        # Calculate rolling average for smoother trend
        compound_data_sorted = compound_data.sort_values('TyreLife')
        rolling_avg = compound_data_sorted['LapTime_Seconds'].rolling(window=3, min_periods=1).mean()
        plt.plot(compound_data_sorted['TyreLife'], rolling_avg, label=compound, linewidth=3, marker='o', markersize=4)
    
    plt.xlabel('Tire Life (laps)', fontsize=12)
    plt.ylabel('Lap Time (seconds) - Rolling Average', fontsize=12)
    plt.title(f'Tire Degradation Trends - {race_name}', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_fresh_vs_used_tires(laps, race_name):
    """Create box plot comparing fresh vs used tires"""
    print("\n=== Creating Box Plot: Fresh vs Used Tires ===")
    
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=laps, x='FreshTyre', y='LapTime_Seconds')
    plt.title(f'Fresh vs Used Tires - {race_name}', fontsize=14, fontweight='bold')
    plt.xlabel('Fresh Tire', fontsize=12)
    plt.ylabel('Lap Time (seconds)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_driver_heatmap(driver_compound_performance, race_name):
    """Create heatmap of driver performance by compound"""
    print("\n=== Creating Heatmap: Driver Performance by Tire Compound ===")
    
    # Heatmap of driver performance by compound
    pivot_data = driver_compound_performance.pivot(index='Driver', columns='Compound', values='AvgLapTime')
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='RdYlGn_r', center=pivot_data.mean().mean())
    plt.title(f'Driver Performance by Tire Compound - {race_name} (Average Lap Time in Seconds)', fontsize=14, fontweight='bold')
    plt.xlabel('Tire Compound', fontsize=12)
    plt.ylabel('Driver', fontsize=12)
    plt.tight_layout()
    plt.show()

def plot_compound_performance_bar(laps, race_name):
    """Create bar plot showing average lap times by compound"""
    print("\n=== Creating Bar Plot: Average Lap Times by Compound ===")
    
    compound_avg = laps.groupby('Compound')['LapTime_Seconds'].mean().sort_values()
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(compound_avg.index, compound_avg.values, color='skyblue', edgecolor='navy', alpha=0.7)
    
    # Add value labels on bars
    for bar, value in zip(bars, compound_avg.values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{value:.3f}s', ha='center', va='bottom', fontweight='bold')
    
    plt.title(f'Average Lap Times by Tire Compound - {race_name}', fontsize=14, fontweight='bold')
    plt.ylabel('Average Lap Time (seconds)', fontsize=12)
    plt.xlabel('Tire Compound', fontsize=12)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()

def plot_stint_efficiency(laps, race_name):
    """Create scatter plot showing stint efficiency"""
    print("\n=== Creating Scatter Plot: Stint Efficiency ===")
    
    # Calculate average lap time per stint
    stint_efficiency = laps.groupby(['Driver', 'Stint', 'Compound'])['LapTime_Seconds'].mean().reset_index()
    stint_efficiency = stint_efficiency.rename(columns={'LapTime_Seconds': 'AvgLapTime'})
    
    plt.figure(figsize=(12, 8))
    for compound in stint_efficiency['Compound'].unique():
        compound_data = stint_efficiency[stint_efficiency['Compound'] == compound]
        plt.scatter(compound_data['Stint'], compound_data['AvgLapTime'], 
                   alpha=0.7, label=compound, s=50)
    
    plt.xlabel('Stint Number', fontsize=12)
    plt.ylabel('Average Lap Time (seconds)', fontsize=12)
    plt.title(f'Stint Efficiency by Compound - {race_name}', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def analyze_compound_performance(laps):
    """Analyze lap times by tire compound"""
    print("\n=== Tire Compound Performance Analysis ===")
    
    # Basic statistics by tire compound
    print("\nAverage lap times by tire compound:")
    compound_stats = laps.groupby('Compound')['LapTime_Seconds'].agg(['mean', 'std', 'min', 'max', 'count']).round(3)
    print(compound_stats)
    
    print("\nTire compound performance summary:")
    for compound in laps['Compound'].unique():
        compound_data = laps[laps['Compound'] == compound]
        print(f"{compound}: {compound_data['LapTime_Seconds'].mean():.3f}s ± {compound_data['LapTime_Seconds'].std():.3f}s (n={len(compound_data)})")
    
    return compound_stats

def analyze_driver_performance(laps):
    """Analyze driver performance by compound"""
    print("\n=== Driver Performance by Tire Compound ===")
    
    # Driver performance by compound
    driver_compound_performance = laps.groupby(['Driver', 'Compound'])['LapTime_Seconds'].agg(['mean', 'std', 'count']).round(3)
    driver_compound_performance = driver_compound_performance.reset_index()
    driver_compound_performance = driver_compound_performance.rename(columns={'mean': 'AvgLapTime', 'std': 'StdLapTime', 'count': 'LapCount'})
    
    print("Driver Performance by Tire Compound:")
    print(driver_compound_performance)
    
    # Find best driver for each compound
    print("\nBest Driver for Each Compound:")
    for compound in laps['Compound'].unique():
        compound_data = driver_compound_performance[driver_compound_performance['Compound'] == compound]
        if not compound_data.empty:
            best_driver = compound_data.loc[compound_data['AvgLapTime'].idxmin()]
            print(f"{compound}: {best_driver['Driver']} ({best_driver['AvgLapTime']:.3f}s)")
    
    return driver_compound_performance

def main():
    """Main function to run the complete tire analysis with separate plots"""
    print("=== F1 Tire Analysis with LapTime in Seconds - Separate Plots ===\n")
    
    # Define race parameters
    season = 2022
    race = 20
    race_name= 'Brazil'
    session = f1.get_session(season,race,'R')
    session.load(laps=True)
    laps = session.laps
    
    # Convert LapTime to seconds
    laps = convert_laptime_to_seconds(laps)
    
    # Select relevant columns for tire analysis
    tire_laps = select_tire_columns(laps)
    
    # Perform analyses
    compound_stats = analyze_compound_performance(tire_laps)
    driver_performance = analyze_driver_performance(tire_laps)
    
    # Create separate plots
    plot_compound_boxplot(tire_laps, race_name)
    plot_compound_violin(tire_laps, race_name)
    plot_compound_performance_bar(tire_laps, race_name)
    plot_tire_degradation_scatter(tire_laps, race_name)
    plot_tire_degradation_trends(tire_laps, race_name)
    plot_fresh_vs_used_tires(tire_laps, race_name)
    plot_stint_efficiency(tire_laps, race_name)
    plot_driver_heatmap(driver_performance, race_name)
    
    print("\n=== Analysis Complete ===")
    print("All plots have been generated separately for better visualization.")
    print("Key benefits of converting LapTime to seconds:")
    print("1. Easier Analysis: Numerical operations are much simpler with seconds")
    print("2. Better Visualization: Plots and charts are more intuitive")
    print("3. Statistical Analysis: Mean, standard deviation, and other statistics are more meaningful")
    print("4. Modeling: Machine learning models work better with numerical data")
    print("5. Comparison: Easier to compare lap times across different sessions and races")
    
    return tire_laps, compound_stats, driver_performance

if __name__ == "__main__":
    # Run the analysis
    tire_laps, compound_stats, driver_performance = main()
    
    # You can also access the converted data directly:
    print(f"\nConverted data shape: {tire_laps.shape}")
    print(f"Sample of converted data:")
    print(tire_laps.head()) 