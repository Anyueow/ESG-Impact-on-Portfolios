import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import statsmodels.api as sm

class ESGPortfolioAnalyzer:
    def __init__(self, esg_data_path, price_data_path):
        """
        Initialize the ESG Portfolio Analyzer
        
        Parameters:
        -----------
        esg_data_path : str
            Path to the ESG scores data file
        price_data_path : str
            Path to the price data file
        """
        self.esg_data = pd.read_csv(esg_data_path)
        self.price_data = pd.read_csv(price_data_path)
        self.portfolio_returns = None
        self.benchmark_returns = None
        
    def preprocess_data(self):
        """Preprocess and merge ESG and price data"""
        # Convert date columns to datetime
        self.price_data['Date'] = pd.to_datetime(self.price_data['Date'])
        
        # Ensure consistent ticker format
        self.esg_data['Ticker'] = self.esg_data['Ticker'].str.upper()
        
        # Calculate daily returns
        self.price_data = self.price_data.sort_values('Date')
        self.price_data['Returns'] = self.price_data.groupby('Ticker')['Close'].pct_change()
        
    def construct_portfolios(self, top_percentile=0.25, bottom_percentile=0.25):
        """
        Construct high and low ESG portfolios
        
        Parameters:
        -----------
        top_percentile : float
            Top percentile for high ESG portfolio
        bottom_percentile : float
            Bottom percentile for low ESG portfolio
        """
        # Calculate ESG score percentiles
        self.esg_data['ESG_Percentile'] = self.esg_data['ESG_Score'].rank(pct=True)
        
        # Identify high and low ESG stocks
        high_esg = self.esg_data[self.esg_data['ESG_Percentile'] >= (1 - top_percentile)]['Ticker']
        low_esg = self.esg_data[self.esg_data['ESG_Percentile'] <= bottom_percentile]['Ticker']
        
        # Calculate portfolio returns
        high_esg_returns = self.price_data[self.price_data['Ticker'].isin(high_esg)].groupby('Date')['Returns'].mean()
        low_esg_returns = self.price_data[self.price_data['Ticker'].isin(low_esg)].groupby('Date')['Returns'].mean()
        
        # Calculate long-short portfolio returns
        self.portfolio_returns = high_esg_returns - low_esg_returns
        
    def calculate_performance_metrics(self):
        """Calculate key performance metrics"""
        if self.portfolio_returns is None:
            raise ValueError("Portfolios must be constructed first")
            
        # Calculate cumulative returns
        cumulative_returns = (1 + self.portfolio_returns).cumprod()
        
        # Calculate annualized returns
        annualized_return = (1 + self.portfolio_returns.mean()) ** 252 - 1
        
        # Calculate Sharpe ratio
        risk_free_rate = 0.02  # Assuming 2% risk-free rate
        excess_returns = self.portfolio_returns - risk_free_rate/252
        sharpe_ratio = np.sqrt(252) * excess_returns.mean() / excess_returns.std()
        
        # Calculate alpha and beta
        benchmark_returns = self.price_data.groupby('Date')['Returns'].mean()
        X = sm.add_constant(benchmark_returns)
        model = sm.OLS(self.portfolio_returns, X).fit()
        alpha = model.params[0] * 252
        beta = model.params[1]
        
        return {
            'Cumulative Returns': cumulative_returns,
            'Annualized Return': annualized_return,
            'Sharpe Ratio': sharpe_ratio,
            'Alpha': alpha,
            'Beta': beta
        }
    
    def plot_performance(self, metrics):
        """Plot portfolio performance metrics"""
        plt.figure(figsize=(15, 10))
        
        # Plot cumulative returns
        plt.subplot(2, 2, 1)
        metrics['Cumulative Returns'].plot()
        plt.title('Cumulative Returns')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Return')
        
        # Plot rolling Sharpe ratio
        plt.subplot(2, 2, 2)
        rolling_sharpe = self.portfolio_returns.rolling(window=252).mean() / self.portfolio_returns.rolling(window=252).std() * np.sqrt(252)
        rolling_sharpe.plot()
        plt.title('Rolling Sharpe Ratio (1-year)')
        plt.xlabel('Date')
        plt.ylabel('Sharpe Ratio')
        
        plt.tight_layout()
        plt.show()
        
    def analyze_esg_impact(self):
        """Analyze the relationship between ESG scores and returns"""
        # Merge ESG scores with returns
        merged_data = pd.merge(
            self.esg_data[['Ticker', 'ESG_Score']],
            self.price_data.groupby('Ticker')['Returns'].mean().reset_index(),
            on='Ticker'
        )
        
        # Plot ESG score vs returns
        plt.figure(figsize=(10, 6))
        sns.regplot(x='ESG_Score', y='Returns', data=merged_data)
        plt.title('ESG Score vs Average Returns')
        plt.xlabel('ESG Score')
        plt.ylabel('Average Daily Return')
        plt.show()
        
        # Run regression analysis
        X = sm.add_constant(merged_data['ESG_Score'])
        y = merged_data['Returns']
        model = sm.OLS(y, X).fit()
        print(model.summary())

def main():
    # Initialize analyzer
    analyzer = ESGPortfolioAnalyzer(
        esg_data_path='sp500_esg_data.csv',
        price_data_path='sp500_price_data.csv'
    )
    
    # Process data and construct portfolios
    analyzer.preprocess_data()
    analyzer.construct_portfolios()
    
    # Calculate and display performance metrics
    metrics = analyzer.calculate_performance_metrics()
    print("\nPerformance Metrics:")
    for metric, value in metrics.items():
        if metric != 'Cumulative Returns':
            print(f"{metric}: {value:.4f}")
    
    # Plot performance
    analyzer.plot_performance(metrics)
    
    # Analyze ESG impact
    analyzer.analyze_esg_impact()

if __name__ == "__main__":
    main() 