import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import time
import sys

def download_spy_data(retries=3, delay=5):
    """
    Download SPY data with retries and save to CSV
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*10)  # 10 years of data
    
    for attempt in range(retries):
        try:
            print(f"Attempt {attempt + 1} to download SPY data...")
            
            # Download data
            spy = yf.download('SPY', 
                            start=start_date, 
                            end=end_date,
                            progress=False)
            
            if spy.empty:
                print("Downloaded data is empty. Retrying...")
                time.sleep(delay)
                continue
            
            # Clean and format the data
            spy.index = pd.to_datetime(spy.index)
            spy = spy.sort_index()
            
            # Save data with proper date format
            spy.to_csv('spy_data.csv', date_format='%Y-%m-%d')
            
            print(f"Successfully downloaded and saved SPY data with {len(spy)} rows")
            print(f"Date range: {spy.index.min()} to {spy.index.max()}")
            print("\nFirst few rows of data:")
            print(spy.head())
            return True
            
        except Exception as e:
            print(f"Error on attempt {attempt + 1}: {str(e)}")
            if attempt < retries - 1:
                print(f"Waiting {delay} seconds before retrying...")
                time.sleep(delay)
            else:
                print("All attempts failed. Could not download SPY data.")
                return False

if __name__ == "__main__":
    print("Starting SPY data download...")
    success = download_spy_data()
    if not success:
        sys.exit(1)  # Exit with error code if download failed
    print("\nDownload complete. You can now run the main ESG app.") 