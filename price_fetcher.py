import yfinance as yf
import pandas as pd
import requests
from datetime import datetime, timedelta
import time
from alpha_vantage.timeseries import TimeSeries
import streamlit as st

def get_alpha_vantage_data(symbol, start_date, end_date):
    """Get stock data from Alpha Vantage"""
    try:
        # Get API key from Streamlit secrets
        api_key = st.secrets["api_keys"]["alpha_vantage"]
        ts = TimeSeries(key=api_key, output_format='pandas')
        data, _ = ts.get_daily_adjusted(symbol=symbol, outputsize='full')
        
        # Convert index to datetime if needed
        data.index = pd.to_datetime(data.index)
        
        # Filter date range
        mask = (data.index >= start_date) & (data.index <= end_date)
        data = data[mask]
        
        # Rename columns to match yfinance format
        data = data.rename(columns={
            '4. close': 'Close',
            '5. adjusted close': 'Adj Close',
            '1. open': 'Open',
            '2. high': 'High',
            '3. low': 'Low',
            '6. volume': 'Volume'
        })
        
        return data[['Close', 'Adj Close', 'Open', 'High', 'Low', 'Volume']]
    except Exception as e:
        st.warning(f"Alpha Vantage error for {symbol}: {str(e)}")
        return None

@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_stock_data(symbol, start_date, end_date, retry_count=3):
    """
    Get stock data with fallback options and caching
    """
    # Try yfinance first
    for attempt in range(retry_count):
        try:
            data = yf.download(symbol, start=start_date, end=end_date, progress=False)
            if not data.empty:
                return data
        except Exception as e:
            st.warning(f"yfinance attempt {attempt + 1} failed for {symbol}: {str(e)}")
        time.sleep(1)  # Wait between attempts
    
    # If yfinance fails, try Alpha Vantage
    st.info(f"Trying Alpha Vantage for {symbol}")
    av_data = get_alpha_vantage_data(symbol, start_date, end_date)
    if av_data is not None and not av_data.empty:
        return av_data
    
    return None

@st.cache_data(ttl=3600)
def get_multiple_stocks(symbols, start_date, end_date):
    """
    Get data for multiple stocks with progress tracking and caching
    """
    stock_data = {}
    failed_symbols = []
    
    progress_bar = st.progress(0)
    for i, symbol in enumerate(symbols):
        progress_bar.progress((i + 1) / len(symbols))
        st.write(f"Fetching data for {symbol}...")
        
        data = get_stock_data(symbol, start_date, end_date)
        if data is not None and not data.empty:
            stock_data[symbol] = data
        else:
            failed_symbols.append(symbol)
            st.warning(f"Failed to get data for {symbol}")
    
    progress_bar.empty()
    return stock_data, failed_symbols

@st.cache_data(ttl=3600)
def get_spy_data(start_date, end_date):
    """
    Get S&P500 data with retries and caching
    """
    spy_data = get_stock_data('SPY', start_date, end_date, retry_count=5)
    if spy_data is None or spy_data.empty:
        raise ValueError("Could not fetch S&P500 data from any source")
    return spy_data

def get_portfolio_data(stocks, start_date, end_date):
    """
    Get portfolio data with proper error handling and progress tracking
    """
    stock_data, failed_symbols = get_multiple_stocks(stocks, start_date, end_date)
    
    if failed_symbols:
        st.warning(f"Failed to fetch data for: {', '.join(failed_symbols)}")
    
    if not stock_data:
        return None
    
    # Convert to DataFrame with only Close prices
    portfolio_data = {}
    for symbol, data in stock_data.items():
        if isinstance(data.columns, pd.MultiIndex):
            portfolio_data[symbol] = data[('Close', symbol)]
        else:
            portfolio_data[symbol] = data['Close']
    
    return pd.DataFrame(portfolio_data)

if __name__ == "__main__":
    # Test the functionality
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*2)
    
    # Test with SPY
    print("Testing with SPY...")
    spy_data = get_spy_data(start_date, end_date)
    print("SPY data shape:", spy_data.shape)
    
    # Test with a few stocks
    test_symbols = ['AAPL', 'MSFT', 'GOOGL']
    print("\nTesting with sample stocks...")
    stock_data, failed = get_multiple_stocks(test_symbols, start_date, end_date)
    
    for symbol, data in stock_data.items():
        print(f"{symbol} data shape:", data.shape)
    if failed:
        print("Failed to fetch:", failed) 