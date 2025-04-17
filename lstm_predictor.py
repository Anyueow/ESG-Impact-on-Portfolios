import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
import plotly.graph_objects as go
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import yfinance as yf

class LSTMPredictor:
    def __init__(self, sequence_length=120):  # 6 months of trading days
        self.sequence_length = sequence_length
        self.scaler = MinMaxScaler()
        self.model = None
        
    def create_model(self, dropout_rate=0.2):
        # Define input layer
        inputs = Input(shape=(self.sequence_length, 1))
        
        # First LSTM layer
        x = LSTM(100, activation='tanh', return_sequences=True)(inputs)
        x = Dropout(dropout_rate)(x)
        
        # Second LSTM layer
        x = LSTM(50, activation='tanh', return_sequences=False)(x)
        x = Dropout(dropout_rate)(x)
        
        # Dense layers
        x = Dense(25, activation='relu')(x)
        outputs = Dense(1)(x)
        
        # Create model
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        
        self.model = model
        return model
    
    def prepare_data(self, data):
        # Scale the data
        scaled_data = self.scaler.fit_transform(data.reshape(-1, 1))
        
        X, y = [], []
        for i in range(len(scaled_data) - self.sequence_length):
            X.append(scaled_data[i:(i + self.sequence_length)])
            y.append(scaled_data[i + self.sequence_length])
            
        return np.array(X), np.array(y)
    
    def train(self, data, validation_split=0.2, epochs=100, batch_size=32):
        X, y = self.prepare_data(data)
        
        # Create and train model
        self.create_model()
        history = self.model.fit(
            X, y,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            verbose=0
        )
        return history
    
    def predict_future(self, last_sequence, n_future):
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
            
        predictions = []
        current_sequence = last_sequence.copy()
        
        for _ in range(n_future):
            # Scale the current sequence
            scaled_sequence = self.scaler.transform(current_sequence.reshape(-1, 1))
            
            # Reshape for prediction
            x = scaled_sequence.reshape(1, self.sequence_length, 1)
            
            # Make prediction
            pred = self.model.predict(x, verbose=0)
            
            # Inverse transform the prediction
            pred = self.scaler.inverse_transform(pred)[0][0]
            predictions.append(pred)
            
            # Update sequence
            current_sequence = np.append(current_sequence[1:], pred)
            
        return np.array(predictions).reshape(-1, 1)

def plot_predictions(historical_data, predictions, future_dates):
    fig = go.Figure()
    
    # Historical data
    fig.add_trace(
        go.Scatter(x=historical_data.index, 
                  y=historical_data.values,
                  name='Historical',
                  line=dict(color='blue'))
    )
    
    # Predictions
    fig.add_trace(
        go.Scatter(x=future_dates,
                  y=predictions.flatten(),
                  name='Prediction',
                  line=dict(color='red', dash='dash'))
    )
    
    fig.update_layout(
        title='S&P500 3-Year Prediction',
        xaxis_title='Date',
        yaxis_title='Price',
        showlegend=True
    )
    
    return fig

def plot_training_history(history):
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss During Training')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_history.png')
    plt.close()

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
                    # Handle MultiIndex structure correctly
                    if isinstance(data.columns, pd.MultiIndex):
                        close_prices = data[('Close', stock)]
                    else:
                        close_prices = data['Close']
                    portfolio_data[stock] = close_prices
            except Exception as e:
                st.warning(f"Could not download data for {stock}: {str(e)}")
                continue
    except Exception as e:
        st.warning(f"Could not load default portfolio: {str(e)}")
        return None

def main():
    # Load data
    print("Loading data...")
    spy_data = pd.read_csv('spy_data.csv', index_col=0, parse_dates=True)
    close_prices = spy_data['Close']
    
    # Initialize predictor
    predictor = LSTMPredictor(sequence_length=120)
    
    # Train model
    print("\nTraining model...")
    history = predictor.train(close_prices.values, epochs=100)
    
    # Plot training history
    plot_training_history(history)
    print("\nTraining history plot saved as 'training_history.png'")
    
    # Make 3-year prediction
    last_sequence = close_prices[-predictor.sequence_length:].values
    n_future = 756  # 3 years of trading days
    predictions = predictor.predict_future(last_sequence, n_future)
    
    # Create future dates
    last_date = close_prices.index[-1]
    future_dates = pd.date_range(start=last_date, periods=n_future+1)[1:]
    
    # Plot results
    fig = plot_predictions(close_prices, predictions, future_dates)
    
    # Save predictions
    pred_df = pd.DataFrame({
        'Date': future_dates,
        'Predicted_Close': predictions.flatten()
    })
    pred_df.to_csv('spy_predictions.csv', index=False)
    
    # Calculate returns
    initial_price = close_prices[-1]
    final_price = predictions[-1][0]
    total_return = (final_price - initial_price) / initial_price * 100
    annual_return = ((1 + total_return/100) ** (1/3) - 1) * 100
    
    print("\nPrediction Results:")
    print(f"Initial Price: ${initial_price:.2f}")
    print(f"Predicted Price (3 years): ${final_price:.2f}")
    print(f"Total Return: {total_return:.2f}%")
    print(f"Annualized Return: {annual_return:.2f}%")
    
    # Save plot
    fig.write_html("spy_prediction_plot.html")
    print("\nPrediction plot saved as 'spy_prediction_plot.html'")
    print("Predictions saved as 'spy_predictions.csv'")

if __name__ == "__main__":
    main() 