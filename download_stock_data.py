import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import time
import os
import json
import random

def download_stock_data():
    """
    Download and save 10-year historical data for all stocks in merged_data.csv
    using a more reliable approach with yfinance
    """
    # Create data directory if it doesn't exist
    if not os.path.exists('stock_data'):
        os.makedirs('stock_data')
    
    # Load tickers from merged_data.csv
    try:
        df = pd.read_csv('merged_data.csv')
        tickers = df['Ticker'].unique().tolist()
        print(f"Found {len(tickers)} unique tickers")
    except Exception as e:
        print(f"Error loading merged_data.csv: {e}")
        return
    
    # Set date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*10)  # 10 years of data
    
    # Download data for each ticker
    successful_downloads = 0
    failed_downloads = []
    
    for i, ticker in enumerate(tickers, 1):
        print(f"\nProcessing {ticker} ({i}/{len(tickers)})")
        
        # Skip if already downloaded
        if os.path.exists(f'stock_data/{ticker}_data.csv'):
            print(f"Data for {ticker} already exists. Skipping.")
            successful_downloads += 1
            continue
            
        try:
            # Download data with the working approach
            stock = yf.download(ticker, start=start_date, end=end_date, progress=False)
            
            if not stock.empty:
                # Save to CSV
                filename = f'stock_data/{ticker}_data.csv'
                stock.to_csv(filename)
                print(f"Successfully downloaded {ticker} data: {len(stock)} rows")
                successful_downloads += 1
            else:
                print(f"No data available for {ticker}")
                failed_downloads.append(ticker)
                
        except Exception as e:
            print(f"Error downloading {ticker}: {e}")
            failed_downloads.append(ticker)
        
        # Add delay between requests to prevent rate limiting
        time.sleep(1 + random.uniform(0, 0.5))
    
    # Save summary
    summary = {
        'total_tickers': len(tickers),
        'successful_downloads': successful_downloads,
        'failed_downloads': len(failed_downloads),
        'failed_tickers': failed_downloads,
        'download_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'start_date': start_date.strftime('%Y-%m-%d'),
        'end_date': end_date.strftime('%Y-%m-%d')
    }
    
    # Save summary to JSON
    with open('stock_data/download_summary.json', 'w') as f:
        json.dump(summary, f, indent=4)
    
    # Print summary
    print("\nDownload Summary:")
    print(f"Total tickers: {len(tickers)}")
    print(f"Successfully downloaded: {successful_downloads}")
    print(f"Failed downloads: {len(failed_downloads)}")
    if failed_downloads:
        print("\nFailed tickers:")
        print(", ".join(failed_downloads))
    
    return successful_downloads, failed_downloads

def load_stock_data(ticker):
    """
    Load historical data for a specific ticker
    """
    try:
        filename = f'stock_data/{ticker}_data.csv'
        data = pd.read_csv(filename, index_col=0, parse_dates=True)
        return data
    except Exception as e:
        print(f"Error loading data for {ticker}: {e}")
        return None

def analyze_stock_performance(ticker, start_date=None, end_date=None):
    """
    Analyze performance of a stock with optional date range
    """
    data = load_stock_data(ticker)
    if data is None:
        return None
    
    # Filter by date range if provided
    if start_date:
        data = data[data.index >= start_date]
    if end_date:
        data = data[data.index <= end_date]
    
    if len(data) < 2:
        print(f"Not enough data for {ticker} in the specified date range")
        return None
    
    # Calculate basic metrics
    first_price = data['Close'].iloc[0]
    last_price = data['Close'].iloc[-1]
    percent_change = ((last_price - first_price) / first_price) * 100
    
    # Calculate volatility (standard deviation of daily returns)
    daily_returns = data['Close'].pct_change().dropna()
    volatility = daily_returns.std() * 100
    
    # Calculate max drawdown
    rolling_max = data['Close'].cummax()
    drawdown = ((data['Close'] - rolling_max) / rolling_max) * 100
    max_drawdown = drawdown.min()
    
    return {
        'ticker': ticker,
        'start_date': data.index[0].strftime('%Y-%m-%d'),
        'end_date': data.index[-1].strftime('%Y-%m-%d'),
        'start_price': first_price,
        'end_price': last_price,
        'percent_change': percent_change,
        'annualized_return': (((last_price/first_price) ** (365/len(data))) - 1) * 100,
        'volatility': volatility,
        'max_drawdown': max_drawdown
    }

if __name__ == "__main__":
    print("Starting stock data download...")
    successful, failed = download_stock_data()
    print(f"\nDownload complete. Downloaded {successful} tickers successfully. {len(failed)} tickers failed.")
    print("Data saved in 'stock_data' directory.")