import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from pathlib import Path

def calculate_price_metrics(df):
    """
    Calculate various price changes and returns from a price history DataFrame
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing price history with 'Date' and 'Close' columns
    
    Returns:
    --------
    dict
        Dictionary containing various price metrics
    """
    # Ensure date is datetime and sort
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    
    # Get latest date and price
    latest_date = df['Date'].max()
    latest_price = df[df['Date'] == latest_date]['Close'].iloc[0]
    
    # Calculate YTD metrics
    ytd_start = datetime(latest_date.year, 1, 1)
    ytd_price = df[df['Date'] >= ytd_start]['Close'].iloc[0]
    ytd_change = (latest_price / ytd_price - 1) * 100
    ytd_return = latest_price / ytd_price
    
    # Calculate other time periods
    periods = {
        '1M': 30,
        '3M': 90,
        '6M': 180,
        '12M': 365
    }
    
    metrics = {
        'YTD_Change': ytd_change,
        'YTD_Return': ytd_return
    }
    
    for period, days in periods.items():
        past_date = latest_date - timedelta(days=days)
        past_price = df[df['Date'] <= past_date]['Close'].iloc[-1]
        
        # Calculate percentage change
        change = (latest_price / past_price - 1) * 100
        metrics[f'{period}_Change'] = change
        
        # Calculate return (decimal)
        ret = latest_price / past_price
        metrics[f'{period}_Return'] = ret
    
    return metrics

def process_price_files(csv_dir):
    """
    Process all price CSV files in a directory and calculate price metrics
    
    Parameters:
    -----------
    csv_dir : str
        Directory containing price CSV files
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing ticker and price metrics
    """
    # Initialize results list
    results = []
    
    # Get all CSV files in directory
    csv_files = [f for f in os.listdir(csv_dir) if f.endswith('.csv')]
    
    for file in csv_files:
        try:
            # Extract ticker from filename
            ticker = file.replace('.csv', '')
            
            # Read price data
            df = pd.read_csv(os.path.join(csv_dir, file))
            
            # Calculate price metrics
            metrics = calculate_price_metrics(df)
            
            # Add ticker to results
            metrics['Ticker'] = ticker
            results.append(metrics)
            
        except Exception as e:
            print(f"Error processing {file}: {str(e)}")
    
    # Convert to DataFrame
    price_metrics_df = pd.DataFrame(results)
    
    # Reorder columns
    columns = ['Ticker', 
               'YTD_Change', 'YTD_Return',
               '1M_Change', '1M_Return',
               '3M_Change', '3M_Return',
               '6M_Change', '6M_Return',
               '12M_Change', '12M_Return']
    price_metrics_df = price_metrics_df[columns]
    
    return price_metrics_df

def main():
    # Directory containing price CSV files
    csv_dir = 'csv'  # Update this to your actual directory path
    
    # Process files and create DataFrame
    price_metrics_df = process_price_files(csv_dir)
    
    # Save results
    output_file = 'price_changes.csv'
    price_metrics_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")
    
    # Display sample of results
    print("\nSample of price metrics:")
    print(price_metrics_df.head())

if __name__ == "__main__":
    main() 