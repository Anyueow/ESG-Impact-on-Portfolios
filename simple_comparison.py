import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# Set page config
st.set_page_config(layout="wide")

# Title
st.title("S&P 500 vs ESG Portfolio Returns Comparison")

# Read the data
@st.cache_data
def load_data():
    spy_data = pd.read_csv('spy_data.csv')
    print(spy_data.head())
    esg_data = pd.read_csv('esg_stock_pred.csv')
    print(esg_data.head())
    return spy_data, esg_data

spy_data, esg_data = load_data()

# Convert date columns to datetime
spy_data['Date'] = pd.to_datetime(spy_data['Date'])
esg_data['Date'] = pd.to_datetime(esg_data['Date'])

# Calculate S&P 500 returns
spy_data['Returns'] = spy_data['Close'].pct_change()

# Calculate ESG portfolio returns
# First, pivot the ESG data to get daily returns for each stock
esg_pivot = esg_data.pivot(index='Date', columns='Ticker', values='Close')
# Calculate daily returns for each stock
esg_returns = esg_pivot.pct_change()
# Calculate equal-weighted portfolio returns
esg_portfolio_returns = esg_returns.mean(axis=1)

# Create subplots
fig = make_subplots(rows=1, cols=2, 
                    subplot_titles=('S&P 500 Returns', 'ESG Portfolio Returns'),
                    shared_xaxes=True,
                    shared_yaxes=True)

# Add S&P 500 trace
fig.add_trace(
    go.Scatter(x=spy_data['Date'], y=spy_data['Returns'], name='S&P 500'),
    row=1, col=1
)

# Add ESG Portfolio trace
fig.add_trace(
    go.Scatter(x=esg_portfolio_returns.index, y=esg_portfolio_returns, name='ESG Portfolio'),
    row=1, col=2
)

# Update layout
fig.update_layout(
    height=600,
    showlegend=True,
    title_text="10-Year Returns Comparison",
    xaxis_title="Date",
    yaxis_title="Daily Returns",
    xaxis2_title="Date",
    yaxis2_title="Daily Returns"
)

# Display the plot
st.plotly_chart(fig, use_container_width=True)

# Calculate cumulative returns
spy_cumulative = (1 + spy_data['Returns']).cumprod()
esg_cumulative = (1 + esg_portfolio_returns).cumprod()

# Add some statistics
col1, col2 = st.columns(2)

with col1:
    st.subheader("S&P 500 Statistics")
    st.write(f"Total Return: {((spy_cumulative.iloc[-1] - 1) * 100):.2f}%")
    st.write(f"Annualized Return: {((spy_cumulative.iloc[-1]) ** (1/10) - 1) * 100:.2f}%")
    st.write(f"Daily Volatility: {(spy_data['Returns'].std() * 100):.2f}%")

with col2:
    st.subheader("ESG Portfolio Statistics")
    st.write(f"Total Return: {((esg_cumulative.iloc[-1] - 1) * 100):.2f}%")
    st.write(f"Annualized Return: {((esg_cumulative.iloc[-1]) ** (1/10) - 1) * 100:.2f}%")
    st.write(f"Daily Volatility: {(esg_portfolio_returns.std() * 100):.2f}%") 