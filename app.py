import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import os

# Import LSTMPredictor class
from lstm_predictor import LSTMPredictor

@st.cache_data
def load_data():
    """Load all required data from CSV files"""
    try:
        # Load ESG stock data
        esg_data = pd.read_csv('esg_stock_pred.csv')
        esg_data['Date'] = pd.to_datetime(esg_data['Date'])
        
        # Load SPY data
        spy_data = pd.read_csv('spy_data.csv', index_col=0, parse_dates=True)['Close']
        
        # Load ESG categories
        esg_categories = pd.read_csv('merged_data.csv')
        esg_categories['ESG_Category'] = pd.qcut(esg_categories['totalEsg'], q=3, labels=['Low', 'Medium', 'High'])
        
        return esg_data, spy_data, esg_categories
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None, None

@st.cache_resource
def get_lstm_predictor():
    """Get cached LSTM predictor"""
    return LSTMPredictor(sequence_length=60)

@st.cache_data
def get_spy_predictions(spy_data):
    """Get cached SPY predictions"""
    predictor = get_lstm_predictor()
    predictor.train(spy_data.values, epochs=50)
    future_days = 756  # 3 years
    last_sequence = spy_data[-predictor.sequence_length:].values
    predictions = predictor.predict_future(last_sequence, future_days)
    future_dates = pd.date_range(start=spy_data.index[-1], periods=future_days+1)[1:]
    return predictions, future_dates

def get_portfolio_data(tickers, start_date, end_date, esg_data):
    """Get portfolio data for specified tickers and date range"""
    portfolio_data = {}
    for ticker in tickers:
        ticker_data = esg_data[esg_data['Ticker'] == ticker]
        if not ticker_data.empty:
            mask = (ticker_data['Date'] >= start_date) & (ticker_data['Date'] <= end_date)
            filtered_data = ticker_data[mask]
            if not filtered_data.empty:
                filtered_data = filtered_data.set_index('Date')
                portfolio_data[ticker] = filtered_data['Close']
    
    if not portfolio_data:
        return None
    return pd.DataFrame(portfolio_data)

def plot_portfolio_prediction(portfolio_returns, spy_data, port_pred=None, spy_pred=None, future_dates=None):
    """Plot portfolio and SPY predictions"""
    fig = go.Figure()
    
    # Historical data (10 years)
    if portfolio_returns is not None:
        fig.add_trace(go.Scatter(
            x=portfolio_returns.index,
            y=portfolio_returns/portfolio_returns.iloc[0],
            name='Portfolio Historical',
            line=dict(color='blue')
        ))
    
    if spy_data is not None:
        fig.add_trace(go.Scatter(
            x=spy_data.index,
            y=spy_data/spy_data.iloc[0],
            name='S&P500 Historical',
            line=dict(color='red')
        ))
    
    # Predictions (3 years)
    if port_pred is not None and future_dates is not None:
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=port_pred.flatten()/port_pred[0],
            name='Portfolio Prediction',
            line=dict(color='blue', dash='dash')
        ))
    
    if spy_pred is not None and future_dates is not None:
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=spy_pred.flatten()/spy_pred[0],
            name='S&P500 Prediction',
            line=dict(color='red', dash='dash')
        ))
    
    fig.update_layout(
        title='Portfolio vs S&P500 Performance',
        xaxis_title='Date',
        yaxis_title='Normalized Price',
        showlegend=True,
        height=400
    )
    return fig

def main():
    st.set_page_config(layout="wide")
    st.title('ESG Investment Strategy Analyzer')
    
    # Load all data
    esg_data, spy_data, esg_categories = load_data()
    if esg_data is None or spy_data is None or esg_categories is None:
        return
    
    # Create two columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("S&P500 Performance")
        # Get SPY predictions
        spy_pred, future_dates = get_spy_predictions(spy_data)
        
        # Plot historical and predicted SPY performance
        spy_fig = plot_portfolio_prediction(None, spy_data, None, spy_pred, future_dates)
        st.plotly_chart(spy_fig, use_container_width=True)
        
        # Display SPY returns
        spy_return = (spy_pred[-1]/spy_pred[0] - 1) * 100
        st.write(f"S&P500 Predicted 3-Year Return: {spy_return:.2f}%")
    
    with col2:
        st.subheader("Portfolio Builder")
        
        # Portfolio weights
        col_weights1, col_weights2, col_weights3 = st.columns(3)
        with col_weights1:
            high_esg = st.number_input('High ESG Weight (%)', min_value=0, max_value=100, value=33)
        with col_weights2:
            med_esg = st.number_input('Medium ESG Weight (%)', min_value=0, max_value=100, value=33)
        with col_weights3:
            low_esg = st.number_input('Low ESG Weight (%)', min_value=0, max_value=100, value=34)
        
        if high_esg + med_esg + low_esg != 100:
            st.warning('Portfolio weights must sum to 100%')
        else:
            # Get stocks for each ESG category
            high_esg_stocks = esg_categories[esg_categories['ESG_Category'] == 'High']['Ticker'].sample(3).tolist()
            med_esg_stocks = esg_categories[esg_categories['ESG_Category'] == 'Medium']['Ticker'].sample(3).tolist()
            low_esg_stocks = esg_categories[esg_categories['ESG_Category'] == 'Low']['Ticker'].sample(3).tolist()
            
            # Show selected stocks
            st.write("Selected Stocks:")
            stock_col1, stock_col2, stock_col3 = st.columns(3)
            with stock_col1:
                st.write("High ESG:", ", ".join(high_esg_stocks))
            with stock_col2:
                st.write("Medium ESG:", ", ".join(med_esg_stocks))
            with stock_col3:
                st.write("Low ESG:", ", ".join(low_esg_stocks))
            
            if st.button("Generate Prediction"):
                with st.spinner("Calculating portfolio prediction..."):
                    # Get portfolio data
                    all_stocks = high_esg_stocks + med_esg_stocks + low_esg_stocks
                    weights = ([high_esg/300] * 3 + [med_esg/300] * 3 + [low_esg/300] * 3)
                    
                    end_date = datetime.now()
                    start_date = end_date - timedelta(days=365*10)  # 10 years of data
                    
                    portfolio_df = get_portfolio_data(all_stocks, start_date, end_date, esg_data)
                    
                    if portfolio_df is not None:
                        # Calculate weighted returns
                        portfolio_returns = pd.Series(0, index=portfolio_df.index)
                        for i, stock in enumerate(portfolio_df.columns):
                            portfolio_returns += portfolio_df[stock] * weights[i]
                        
                        # Get cached predictor
                        predictor = get_lstm_predictor()
                        
                        # Train and predict
                        predictor.train(portfolio_returns.values, epochs=50)
                        future_days = 756  # 3 years
                        last_sequence = portfolio_returns[-predictor.sequence_length:].values
                        port_pred = predictor.predict_future(last_sequence, future_days)
                        
                        # Plot predictions
                        pred_fig = plot_portfolio_prediction(
                            portfolio_returns,
                            spy_data,
                            port_pred,
                            spy_pred,
                            future_dates
                        )
                        st.plotly_chart(pred_fig, use_container_width=True)
                        
                        # Calculate and display returns
                        port_return = (port_pred[-1]/port_pred[0] - 1) * 100
                        st.write(f"Portfolio Predicted 3-Year Return: {port_return:.2f}%")
                    else:
                        st.error("Could not create portfolio. Please try different stocks.")

if __name__ == '__main__':
    main()