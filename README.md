# ESG Investment Strategy Analyzer

This project analyzes and predicts portfolio performance based on ESG (Environmental, Social, and Governance) criteria, comparing ESG-weighted portfolios against the S&P500 index using LSTM-based predictions.

## Project Structure

```
├── download_spy_data.py    # Script to download and save S&P500 historical data
├── lstm_predictor.py       # LSTM model implementation for predictions
├── esg_app.py             # Main Streamlit application
├── requirements.txt        # Project dependencies
├── merged_data.csv        # ESG scores and company data
└── spy_data.csv           # Cached S&P500 historical data
```

## Methodology

### 1. Data Processing
- **ESG Categorization**: Companies are categorized into High, Medium, and Low ESG tiers based on their total ESG scores
- **Stock Data**: Historical price data is fetched using yfinance
- **S&P500 Benchmark**: Uses SPY ETF as market benchmark
- **Data Cleaning**: Handles missing data through forward and backward filling

### 2. Portfolio Construction
- **ESG Weighting**: Portfolios are constructed with customizable weights for High/Medium/Low ESG stocks
- **Stock Selection**: Random sampling of 3 stocks from each ESG category
- **Return Calculation**: Weighted average of individual stock returns
- **Default Portfolio**: Pre-configured 33/33/34 split between High/Medium/Low ESG stocks

### 3. LSTM Prediction Model
- **Architecture**:
  - Two LSTM layers (50 units each)
  - ReLU activation
  - Dense output layer
  - Adam optimizer
- **Training Parameters**:
  - Sequence length: 60 days for portfolio, 120 days for S&P500
  - Batch size: 32
  - Epochs: 50-100
- **Prediction Horizon**: 3 years (756 trading days)

### 4. Performance Metrics
- Historical returns
- Predicted returns
- Normalized price comparisons
- Annual return calculations

## Setup and Installation

1. Create a new conda environment:
```bash
conda create -n esg_env python=3.11
conda activate esg_env
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Download initial data:
```bash
python download_spy_data.py
```

## Running the Application

1. Start the Streamlit app:
```bash
streamlit run esg_app.py
```

2. Using the Interface:
   - Left panel shows S&P500 historical data and predictions
   - Right panel shows:
     - Default ESG portfolio performance
     - Portfolio customization options
     - Performance metrics

3. Features:
   - Adjust ESG weights (0-100%)
   - View selected stocks for each ESG category
   - Generate new predictions
   - Compare portfolio performance with S&P500
   - View both historical and predicted returns

## Data Requirements

- `merged_data.csv`: Should contain columns:
  - Ticker: Stock symbol
  - totalEsg: ESG score
  - Other ESG-related metrics

- `spy_data.csv`: Generated automatically by download_spy_data.py
  - Contains S&P500 historical data
  - Updated when running the download script

## Model Details

### LSTM Predictor
- Uses MinMaxScaler for data normalization
- Implements sequence-to-sequence prediction
- Handles both portfolio and S&P500 predictions
- Includes dropout for regularization
- Saves predictions for future use

### Prediction Process
1. Data preprocessing and scaling
2. Sequence generation
3. Model training
4. Future value prediction
5. Inverse scaling
6. Performance calculation

## Error Handling

- Handles missing stock data
- Validates portfolio weights
- Manages MultiIndex data structures
- Provides informative error messages
- Implements data validation checks

## Caching

- Uses Streamlit's caching for:
  - S&P500 data
  - LSTM predictors
  - Default portfolio
  - Predictions

## Future Improvements

1. Add more sophisticated portfolio optimization
2. Implement risk metrics (Sharpe ratio, volatility)
3. Add confidence intervals for predictions
4. Include sector-specific analysis
5. Add more technical indicators

## Notes

- Predictions are for educational purposes only
- Past performance doesn't guarantee future results
- Model should be regularly retrained with new data
- Consider market conditions when interpreting results

## Contributing

Feel free to submit issues and enhancement requests! 