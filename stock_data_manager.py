import pandas as pd
import os
from datetime import datetime
import streamlit as st

@st.cache_data
def load_stock_data(ticker):
    """Load historical data for a specific ticker from esg_stock_pred.csv"""
    try:
        if not os.path.exists('esg_stock_pred.csv'):
            st.error("esg_stock_pred.csv not found. Please ensure the file exists.")
            return None
            
        # Read the CSV file
        data = pd.read_csv('esg_stock_pred.csv')
        data['Date'] = pd.to_datetime(data['Date'])
        
        # Filter for the specific ticker
        ticker_data = data[data['Ticker'] == ticker]
        if ticker_data.empty:
            st.error(f"No data found for {ticker} in esg_stock_pred.csv")
            return None
            
        # Set Date as index and return Close prices
        ticker_data = ticker_data.set_index('Date')
        return ticker_data['Close']
    except Exception as e:
        st.error(f"Error loading data for {ticker}: {e}")
        return None

@st.cache_data
def get_portfolio_data(tickers, start_date=None, end_date=None):
    """Get portfolio data for multiple tickers from esg_stock_pred.csv"""
    try:
        if not os.path.exists('esg_stock_pred.csv'):
            st.error("esg_stock_pred.csv not found. Please ensure the file exists.")
            return None
            
        # Read the CSV file
        data = pd.read_csv('esg_stock_pred.csv')
        data['Date'] = pd.to_datetime(data['Date'])
        
        portfolio_data = {}
        for ticker in tickers:
            # Filter for the specific ticker
            ticker_data = data[data['Ticker'] == ticker]
            if not ticker_data.empty:
                # Filter date range if specified
                if start_date and end_date:
                    mask = (ticker_data['Date'] >= start_date) & (ticker_data['Date'] <= end_date)
                    ticker_data = ticker_data[mask]
                
                if not ticker_data.empty:
                    # Set Date as index and get Close prices
                    ticker_data = ticker_data.set_index('Date')
                    portfolio_data[ticker] = ticker_data['Close']
                else:
                    st.warning(f"No data available for {ticker} in specified date range")
        
        if not portfolio_data:
            return None
        
        # Combine all series into a DataFrame
        return pd.DataFrame(portfolio_data)
    except Exception as e:
        st.error(f"Error getting portfolio data: {e}")
        return None

def check_data_freshness():
    """Check if esg_stock_pred.csv exists"""
    if not os.path.exists('esg_stock_pred.csv'):
        st.error("esg_stock_pred.csv not found. Please ensure the file exists.")
        return False
    return True 