import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import os
import json

# Import LSTMPredictor class
from lstm_predictor import LSTMPredictor

# Download and save SPY data if not exists
@st.cache_data
def get_spy_data():
    """Get SPY data with caching"""
    try:
        spy = pd.read_csv('spy_data.csv', index_col=0, parse_dates=True)
        if spy.empty:
            raise ValueError("SPY data file is empty")
        return spy['Close']
    except FileNotFoundError:
        st.error("SPY data file not found. Please run download_spy_data.py first.")
        return None
    except Exception as e:
        st.error(f"Error loading SPY data: {e}")
        return None

@st.cache_resource
def get_lstm_predictor():
    """Initialize and return LSTM predictor"""
    return LSTMPredictor(sequence_length=120)

def load_predictions():
    """Load saved predictions if they exist"""
    try:
        predictions = pd.read_csv('spy_predictions.csv')
        predictions['Date'] = pd.to_datetime(predictions['Date'])
        return predictions
    except FileNotFoundError:
        return None

# In esg_app.py, modify the portfolio prediction section:

@st.cache_resource
def get_portfolio_predictor():
    """Initialize and return LSTM predictor for portfolio"""
    return LSTMPredictor(sequence_length=60)  # Using 60 days for portfolio predictions

def generate_portfolio_prediction(portfolio_returns, spy_data):
    """Generate predictions using LSTM models"""
    try:
        # Get predictors
        port_predictor = get_portfolio_predictor()
        spy_predictor = get_lstm_predictor()
        
        # Train models
        port_history = port_predictor.train(portfolio_returns.values, epochs=50, batch_size=32)
        spy_history = spy_predictor.train(spy_data.values, epochs=50, batch_size=32)
        
        # Generate predictions
        future_days = 756  # 3 years
        port_pred = port_predictor.predict_future(
            portfolio_returns[-port_predictor.sequence_length:].values, 
            future_days
        )
        spy_pred = spy_predictor.predict_future(
            spy_data[-spy_predictor.sequence_length:].values,
            future_days
        )
        
        # Create future dates
        last_date = portfolio_returns.index[-1]
        future_dates = pd.date_range(start=last_date, periods=future_days+1)[1:]
        
        return {
            'port_pred': port_pred,
            'spy_pred': spy_pred,
            'future_dates': future_dates
        }
    except Exception as e:
        st.error(f"Error generating predictions: {str(e)}")
        return None
    
def plot_spy_returns_with_predictions(spy_data):
    """Plot SPY returns with predictions"""
    fig = go.Figure()
    
    if spy_data is not None and len(spy_data) > 0:
        # Plot historical data
        normalized_spy = spy_data/spy_data.values[0]
        fig.add_trace(
            go.Scatter(x=normalized_spy.index, 
                      y=normalized_spy.values,
                      name='Historical S&P500',
                      line=dict(color='blue'))
        )
        
        # Load and plot predictions if they exist
        predictions = load_predictions()
        if predictions is not None:
            # Normalize predictions to match historical data
            pred_normalized = predictions['Predicted_Close'] / predictions['Predicted_Close'].iloc[0]
            pred_normalized *= normalized_spy.values[-1]  # Scale to match last historical value
            
            fig.add_trace(
                go.Scatter(x=predictions['Date'],
                          y=pred_normalized,
                          name='3-Year Prediction',
                          line=dict(color='red', dash='dash'))
            )
        
        fig.update_layout(
            title='S&P500 Historical Performance and 3-Year Prediction',
            xaxis_title='Date',
            yaxis_title='Normalized Price',
            showlegend=True,
            height=400
        )
    else:
        fig.update_layout(
            title='Error: No S&P500 data available',
            height=400
        )
    
    return fig

def plot_portfolio_prediction(portfolio_returns, spy_data, port_pred, spy_pred, future_dates):
    fig = go.Figure()
    
    # Historical data
    if portfolio_returns is not None and len(portfolio_returns) > 0:
        fig.add_trace(
            go.Scatter(x=portfolio_returns.index, 
                      y=portfolio_returns/portfolio_returns.iloc[0],
                      name='Portfolio Historical',
                      line=dict(color='blue'))
        )
    
    if spy_data is not None and len(spy_data) > 0:
        fig.add_trace(
            go.Scatter(x=spy_data.index,
                      y=spy_data/spy_data.iloc[0],
                      name='S&P500 Historical',
                      line=dict(color='red'))
        )
    
    # Predictions
    if port_pred is not None:
        fig.add_trace(
            go.Scatter(x=future_dates,
                      y=port_pred.flatten()/port_pred[0],
                      name='Portfolio Prediction',
                      line=dict(color='blue', dash='dash'))
        )
    
    if spy_pred is not None:
        fig.add_trace(
            go.Scatter(x=future_dates,
                      y=spy_pred.flatten()/spy_pred[0],
                      name='S&P500 Prediction',
                      line=dict(color='red', dash='dash'))
        )
    
    fig.update_layout(
        title='Portfolio vs S&P500 Prediction',
        xaxis_title='Date',
        yaxis_title='Normalized Price',
        showlegend=True,
        height=400
    )
    
    return fig

@st.cache_data
def get_default_portfolio():
    """Get default portfolio data with caching"""
    try:
        # Load data
        df = pd.read_csv('merged_data.csv')
        df['ESG_Category'] = pd.qcut(df['totalEsg'], q=3, labels=['Low', 'Medium', 'High'])
        
        # Get default stocks (using seed for consistency)
        np.random.seed(42)  # For consistent default portfolio
        high_esg_stocks = df[df['ESG_Category'] == 'High']['Ticker'].sample(3).tolist()
        med_esg_stocks = df[df['ESG_Category'] == 'Medium']['Ticker'].sample(3).tolist()
        low_esg_stocks = df[df['ESG_Category'] == 'Low']['Ticker'].sample(3).tolist()
        
        # Default weights (33/33/34)
        all_stocks = high_esg_stocks + med_esg_stocks + low_esg_stocks
        weights = ([33/300] * 3 + [33/300] * 3 + [34/300] * 3)
        
        # Get historical data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365*2)
        
        portfolio_data = {}
        for stock in all_stocks:
            try:
                data = yf.download(stock, start=start_date, end=end_date)
                if not data.empty:
                    portfolio_data[stock] = data['Close']
            except Exception:
                continue
        
        if portfolio_data:
            portfolio_df = pd.DataFrame(portfolio_data)
            portfolio_df = portfolio_df.fillna(method='ffill').fillna(method='bfill')
            
            # Calculate weighted returns
            portfolio_returns = pd.Series(0, index=portfolio_df.index)
            for i, stock in enumerate(portfolio_data.keys()):
                portfolio_returns += portfolio_df[stock] * weights[i]
            
            return {
                'returns': portfolio_returns,
                'stocks': {
                    'high': high_esg_stocks,
                    'medium': med_esg_stocks,
                    'low': low_esg_stocks
                }
            }
    except Exception:
        return None

@st.cache_data
def get_default_predictions(portfolio_returns, spy_data):
    """Generate default predictions with caching"""
    try:
        sequence_length = 60
        X_port, y_port, scaler_port = prepare_data(portfolio_returns.values, sequence_length)
        X_spy, y_spy, scaler_spy = prepare_data(spy_data.values, sequence_length)
        
        # Create and train models
        port_model = create_lstm_model(sequence_length)
        spy_model = create_lstm_model(sequence_length)
        
        port_model.fit(X_port, y_port, epochs=50, batch_size=32, verbose=0)
        spy_model.fit(X_spy, y_spy, epochs=50, batch_size=32, verbose=0)
        
        # Predictions
        future_days = 756  # 3 years
        port_pred = predict_future(port_model, portfolio_returns[-sequence_length:].values, 
                                scaler_port, future_days)
        spy_pred = predict_future(spy_model, spy_data[-sequence_length:].values,
                               scaler_spy, future_days)
        
        # Create future dates
        last_date = portfolio_returns.index[-1]
        future_dates = pd.date_range(start=last_date, periods=future_days+1)[1:]
        
        return {
            'port_pred': port_pred,
            'spy_pred': spy_pred,
            'future_dates': future_dates
        }
    except Exception:
        return None

def display_performance_metrics(portfolio_returns, spy_data, port_pred, spy_pred):
    """Display historical and predicted performance metrics"""
    st.subheader('Performance Metrics')
    metrics_col1, metrics_col2 = st.columns(2)
    
    with metrics_col1:
        hist_port_return = (portfolio_returns[-1] / portfolio_returns[0] - 1) * 100
        hist_spy_return = (spy_data[-1] / spy_data[0] - 1) * 100
        st.write('Historical Returns:')
        st.write(f'Portfolio: {hist_port_return:.2f}%')
        st.write(f'S&P500: {hist_spy_return:.2f}%')
    
    with metrics_col2:
        pred_port_return = (port_pred[-1] / port_pred[0] - 1) * 100
        pred_spy_return = (spy_pred[-1] / spy_pred[0] - 1) * 100
        st.write('Predicted Returns (3 Years):')
        st.write(f'Portfolio: {pred_port_return:.2f}%')
        st.write(f'S&P500: {pred_spy_return:.2f}%')

def main():
    # Page config
    st.set_page_config(layout="wide")
    
    # Header
    st.title('ESG Investment Strategy Analyzer')
    st.markdown("""
    This tool helps analyze and predict portfolio performance based on ESG (Environmental, Social, and Governance) criteria.
    Compare your ESG-weighted portfolio against the S&P500 index.
    """)
    
    try:
        # Load data
        spy_data = get_spy_data()
        if spy_data is None:
            st.error("Failed to load S&P500 data. Please run download_spy_data.py first.")
            return
            
        # Create two columns for graphs
        col1, col2 = st.columns(2)
        
        with col1:
            # SPY returns plot with predictions
            spy_fig = plot_spy_returns_with_predictions(spy_data)
            st.plotly_chart(spy_fig, use_container_width=True)
            
            # Add button to update LSTM predictions
            if st.button("Update S&P500 Predictions"):
                with st.spinner("Training LSTM model and generating predictions..."):
                    predictor = get_lstm_predictor()
                    history = predictor.train(spy_data.values, epochs=100)
                    
                    # Generate 3-year predictions
                    last_sequence = spy_data[-predictor.sequence_length:].values
                    n_future = 756  # 3 years of trading days
                    predictions = predictor.predict_future(last_sequence, n_future)
                    
                    # Save predictions
                    future_dates = pd.date_range(start=spy_data.index[-1], periods=n_future+1)[1:]
                    pred_df = pd.DataFrame({
                        'Date': future_dates,
                        'Predicted_Close': predictions.flatten()
                    })
                    pred_df.to_csv('spy_predictions.csv', index=False)
                    
                    # Calculate returns
                    initial_price = spy_data[-1]
                    final_price = predictions[-1][0]
                    total_return = (final_price - initial_price) / initial_price * 100
                    annual_return = ((1 + total_return/100) ** (1/3) - 1) * 100
                    
                    st.success("Predictions updated successfully!")
                    st.write(f"Predicted 3-Year Return: {total_return:.2f}%")
                    st.write(f"Predicted Annual Return: {annual_return:.2f}%")
                    
                    # Refresh the plot
                    st.rerun()
        
        with col2:
            # Get default portfolio and predictions
            default_portfolio = get_default_portfolio()
            
            if default_portfolio is not None:
                # Show default portfolio stocks
                st.subheader("Default Portfolio")
                stock_col1, stock_col2, stock_col3 = st.columns(3)
                with stock_col1:
                    st.write("High ESG:", ", ".join(default_portfolio['stocks']['high']))
                with stock_col2:
                    st.write("Medium ESG:", ", ".join(default_portfolio['stocks']['medium']))
                with stock_col3:
                    st.write("Low ESG:", ", ".join(default_portfolio['stocks']['low']))
                
                # Get and show default predictions
                default_preds = get_default_predictions(default_portfolio['returns'], spy_data)
                if default_preds is not None:
                    pred_fig = plot_portfolio_prediction(
                        default_portfolio['returns'], 
                        spy_data,
                        default_preds['port_pred'],
                        default_preds['spy_pred'],
                        default_preds['future_dates']
                    )
                    st.plotly_chart(pred_fig, use_container_width=True)
            
            # Portfolio allocation inputs
            st.subheader("Customize Portfolio")
            col_weights1, col_weights2, col_weights3 = st.columns(3)
            
            with col_weights1:
                high_esg = st.number_input('High ESG Weight (%)', min_value=0, max_value=100, value=33)
            with col_weights2:
                med_esg = st.number_input('Medium ESG Weight (%)', min_value=0, max_value=100, value=33)
            with col_weights3:
                low_esg = st.number_input('Low ESG Weight (%)', min_value=0, max_value=100, value=34)
            
            total = high_esg + med_esg + low_esg
            if total != 100:
                st.warning('Portfolio weights must sum to 100%')
            else:
                # Get stocks and calculate portfolio
                high_esg_stocks = df[df['ESG_Category'] == 'High']['Ticker'].sample(3).tolist()
                med_esg_stocks = df[df['ESG_Category'] == 'Medium']['Ticker'].sample(3).tolist()
                low_esg_stocks = df[df['ESG_Category'] == 'Low']['Ticker'].sample(3).tolist()
                
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
                        # Calculate portfolio
                        portfolio_data = {}
                        all_stocks = high_esg_stocks + med_esg_stocks + low_esg_stocks
                        weights = ([high_esg/300] * 3 + [med_esg/300] * 3 + [low_esg/300] * 3)
                        
                        end_date = datetime.now()
                        start_date = end_date - timedelta(days=365*2)
                        
                        # Download and process stock data
                        for stock in all_stocks:
                            try:
                                data = yf.download(stock, start=start_date, end=end_date)
                                if not data.empty:
                                    # Handle MultiIndex structure correctly
                                    if isinstance(data.columns, pd.MultiIndex):
                                        close_prices = data[('Close', stock)]
                                    else:
                                        close_prices = data['Close']
                                    portfolio_data[stock] = close_prices
                            except Exception as e:
                                st.warning(f"Could not download data for {stock}: {str(e)}")
                                continue
                        
                        if portfolio_data:
                            # Calculate portfolio returns
                            portfolio_df = pd.DataFrame(portfolio_data)
                            portfolio_df = portfolio_df.fillna(method='ffill').fillna(method='bfill')
                            portfolio_returns = pd.Series(0, index=portfolio_df.index)
                            for i, stock in enumerate(portfolio_data.keys()):
                                portfolio_returns += portfolio_df[stock] * weights[i]
                            
                            # Generate predictions
                            predictions = generate_portfolio_prediction(portfolio_returns, spy_data)
                            
                            if predictions:
                                # Update prediction plot
                                pred_fig = plot_portfolio_prediction(
                                    portfolio_returns, 
                                    spy_data,
                                    predictions['port_pred'],
                                    predictions['spy_pred'],
                                    predictions['future_dates']
                                )
                                st.plotly_chart(pred_fig, use_container_width=True)
                                
                                # Display metrics
                                display_performance_metrics(
                                    portfolio_returns,
                                    spy_data,
                                    predictions['port_pred'],
                                    predictions['spy_pred']
                                )
                        else:
                            st.error("Could not create portfolio. Please try different stocks.")
                            
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        st.error("Please refresh the page and try again.")

if __name__ == '__main__':
    main()