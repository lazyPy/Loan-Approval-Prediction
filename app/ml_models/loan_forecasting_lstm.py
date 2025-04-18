import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid Tkinter issues
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import os
import warnings
warnings.filterwarnings('ignore')

# Data loading and preprocessing
def load_data(file_path):
    """
    Load and preprocess the sales data
    """
    # Load data
    df = pd.read_csv(file_path)
    
    # Clean and convert Sales column (handle comma separators and convert to float)
    df['Sales'] = df['Sales'].astype(str).str.replace(',', '').astype(float)
    
    # Convert Date to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Set Date as index
    df.set_index('Date', inplace=True)
    
    return df

def resample_data(df, frequency):
    """
    Resample data to specified frequency
    frequency: 'W' for weekly, 'M' for monthly, 'Q' for quarterly
    """
    # Define resampling function based on frequency
    if frequency == 'W':
        # For weekly data - use end of week
        resampled_df = df.resample('W-FRI').sum()
    elif frequency == 'M':
        # For monthly data
        resampled_df = df.resample('M').sum()
    elif frequency == 'Q':
        # For quarterly data
        resampled_df = df.resample('Q').sum()
    else:
        raise ValueError("Frequency must be 'W', 'M', or 'Q'")
    
    # Fill any missing values using forward fill
    resampled_df.fillna(method='ffill', inplace=True)
    
    return resampled_df

def create_features(df):
    """
    Create time-based features
    """
    # Copy dataframe to avoid modifying original
    df_features = df.copy()
    
    # Extract date features
    df_features['dayofweek'] = df_features.index.dayofweek
    df_features['month'] = df_features.index.month
    df_features['quarter'] = df_features.index.quarter
    df_features['year'] = df_features.index.year
    
    # Create lag features (previous 1, 2, and 3 periods)
    for lag in range(1, 4):
        df_features[f'sales_lag_{lag}'] = df_features['Sales'].shift(lag)
    
    # Create rolling statistics (mean and std of last 3 periods)
    df_features['sales_rolling_mean_3'] = df_features['Sales'].rolling(window=3).mean()
    df_features['sales_rolling_std_3'] = df_features['Sales'].rolling(window=3).std()
    
    # Drop NaN values
    df_features.dropna(inplace=True)
    
    return df_features

def prepare_sequences(data, target_col, sequence_length):
    """
    Create sequences for LSTM model
    """
    X, y = [], []
    
    for i in range(len(data) - sequence_length):
        X.append(data[i:(i + sequence_length)])
        y.append(data[target_col][i + sequence_length])
    
    return np.array(X), np.array(y)

def train_test_split(X, y, train_size=0.8):
    """
    Split data into training and testing sets
    """
    train_size = int(len(X) * train_size)
    
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    return X_train, X_test, y_train, y_test

def scale_data(X_train, X_test, y_train, y_test):
    """
    Scale features using MinMaxScaler
    """
    # Initialize scalers
    X_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()
    
    # Reshape data for scaling
    n_samples_train = X_train.shape[0]
    n_samples_test = X_test.shape[0]
    n_features = X_train.shape[2]
    
    # Reshape to 2D array for scaling
    X_train_reshaped = X_train.reshape(n_samples_train * X_train.shape[1], n_features)
    X_test_reshaped = X_test.reshape(n_samples_test * X_test.shape[1], n_features)
    
    # Fit and transform
    X_train_scaled = X_scaler.fit_transform(X_train_reshaped)
    X_test_scaled = X_scaler.transform(X_test_reshaped)
    
    # Reshape back to 3D
    X_train_scaled = X_train_scaled.reshape(n_samples_train, X_train.shape[1], n_features)
    X_test_scaled = X_test_scaled.reshape(n_samples_test, X_test.shape[1], n_features)
    
    # Scale y values (reshape to column vector)
    y_train_reshaped = y_train.reshape(-1, 1)
    y_test_reshaped = y_test.reshape(-1, 1)
    
    y_train_scaled = y_scaler.fit_transform(y_train_reshaped)
    y_test_scaled = y_scaler.transform(y_test_reshaped)
    
    return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, y_scaler

def build_lstm_model(sequence_length, n_features, lstm_units=50):
    """
    Build and compile LSTM model
    """
    model = Sequential()
    
    # First LSTM layer with return sequences
    model.add(LSTM(units=lstm_units, return_sequences=True, 
                  input_shape=(sequence_length, n_features)))
    model.add(Dropout(0.2))
    
    # Second LSTM layer
    model.add(LSTM(units=lstm_units, return_sequences=True))
    model.add(Dropout(0.2))
    
    # Third LSTM layer
    model.add(LSTM(units=lstm_units))
    model.add(Dropout(0.2))
    
    # Output layer
    model.add(Dense(1))
    
    # Compile model
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    return model

def train_model(model, X_train, y_train, X_test, y_test, epochs=100, batch_size=32):
    """
    Train the LSTM model with early stopping
    """
    # Early stopping to prevent overfitting
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    # Train model
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping],
        verbose=1
    )
    
    return model, history

def evaluate_model(model, X_test, y_test, y_scaler):
    """
    Evaluate model using RMSE, MAE and MAPE
    """
    # Make predictions
    y_pred_scaled = model.predict(X_test)
    
    # Inverse transform predictions and actual values
    y_pred = y_scaler.inverse_transform(y_pred_scaled)
    y_actual = y_scaler.inverse_transform(y_test)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
    mae = mean_absolute_error(y_actual, y_pred)
    mape = mean_absolute_percentage_error(y_actual, y_pred) * 100
    
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"MAPE: {mape:.2f}%")
    
    return y_pred, y_actual, rmse, mae, mape

def plot_results(y_actual, y_pred, title='Actual vs Predicted Values'):
    """
    Plot actual vs predicted values
    """
    plt.figure(figsize=(12, 6))
    plt.plot(y_actual, label='Actual')
    plt.plot(y_pred, label='Predicted')
    plt.title(title)
    
    # Set appropriate x-axis label based on title
    if 'Weekly' in title:
        x_label = 'Week'
    elif 'Monthly' in title:
        x_label = 'Month'
    elif 'Quarterly' in title:
        x_label = 'Quarter'
    else:
        x_label = 'Data Point'
        
    plt.xlabel(x_label)
    plt.ylabel('Loan Amount')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_').lower()}.png")
    plt.show()

def plot_forecast(history_data, forecast_data, title='Forecast'):
    plt.figure(figsize=(12, 6))
    
    # Plot historical data
    history_len = len(history_data)
    forecast_len = len(forecast_data)
    total_len = history_len + forecast_len
    
    x_history = np.arange(history_len)
    x_forecast = np.arange(history_len, total_len)
    
    # Determine the appropriate x-axis label based on title
    x_label = 'Date'
    if 'Weekly' in title:
        x_label = 'Week'
        # Create week labels
        x_ticks = np.arange(total_len)
        x_tick_labels = [f"Week {i+1}" for i in range(total_len)]
    elif 'Monthly' in title:
        x_label = 'Month'
        # Create month labels
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        x_ticks = np.arange(total_len)
        x_tick_labels = [months[i % 12] for i in range(total_len)]
    elif 'Quarterly' in title:
        x_label = 'Quarter'
        # Create quarter labels
        x_ticks = np.arange(total_len)
        x_tick_labels = [f"Q{(i%4)+1}" for i in range(total_len)]
    else:
        x_ticks = np.arange(total_len)
        x_tick_labels = [f"Point {i+1}" for i in range(total_len)]
    
    plt.plot(x_history, history_data, 'b-', label='Historical Data')
    plt.plot(x_forecast, forecast_data, 'r--', label='Forecast')
    plt.title(title)
    
    # Set the custom tick positions and labels
    plt.xticks(x_ticks, x_tick_labels)
    
    plt.xlabel(x_label)
    plt.ylabel('Loan Amount')
    plt.legend()
    plt.grid(True)
    
    # Save the plot to a file
    filename = f"{title.replace(' ', '_').lower()}_forecast.png"
    plt.savefig(filename)
    plt.close()
    
    return filename

def make_future_forecast(model, last_sequence, n_periods, y_scaler, freq):
    """
    Generate future forecasts
    """
    # Initialize array to store forecasts
    forecasts = []
    
    # Make a copy of the last sequence for forecasting
    current_sequence = last_sequence.copy()
    
    # Generate forecasts one step at a time
    for _ in range(n_periods):
        # Reshape for prediction
        current_sequence_reshaped = current_sequence.reshape(1, current_sequence.shape[0], current_sequence.shape[1])
        
        # Predict next value
        next_pred_scaled = model.predict(current_sequence_reshaped)
        
        # Store the forecast
        forecasts.append(next_pred_scaled[0, 0])
        
        # Update sequence for next prediction (drop oldest, add newest)
        # Assuming the target variable is the first feature
        current_sequence = np.roll(current_sequence, -1, axis=0)
        current_sequence[-1, 0] = next_pred_scaled
    
    # Convert forecasts from scaled to original values
    forecasts_array = np.array(forecasts).reshape(-1, 1)
    forecasts_values = y_scaler.inverse_transform(forecasts_array).flatten()
    
    return forecasts_values

def create_future_dates(last_date, n_periods, freq):
    """
    Create future dates for forecast
    """
    if freq == 'W':
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=7), periods=n_periods, freq='W-FRI')
    elif freq == 'M':
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=n_periods, freq='M')
    elif freq == 'Q':
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=n_periods, freq='Q')
    
    return future_dates

def forecast_timeseries(file_path, frequency, sequence_length, forecast_periods, lstm_units=50):
    """
    Main function to forecast time series at specified frequency
    """
    # Set frequency label
    freq_label = {
        'W': 'Weekly',
        'M': 'Monthly',
        'Q': 'Quarterly'
    }
    
    print(f"\n{'='*50}")
    print(f"Starting {freq_label[frequency]} Forecasting")
    print(f"{'='*50}")
    
    # Load and preprocess data
    df = load_data(file_path)
    
    # Resample to specified frequency
    resampled_df = resample_data(df, frequency)
    
    # Create features
    df_features = create_features(resampled_df)
    
    # Prepare data for LSTM
    data = df_features.copy()
    
    # Select relevant features for model
    selected_features = ['Sales', 'NumberOfAccount', 
                        'sales_lag_1', 'sales_lag_2', 'sales_lag_3',
                        'sales_rolling_mean_3', 'sales_rolling_std_3', 
                        'month', 'quarter', 'year']
    
    # Keep only selected features
    data = data[selected_features]
    
    # Create sequences
    X, y = prepare_sequences(data, 'Sales', sequence_length)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    
    # Scale data
    X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, y_scaler = scale_data(
        X_train, X_test, y_train, y_test
    )
    
    # Build model
    model = build_lstm_model(sequence_length, X_train_scaled.shape[2], lstm_units)
    
    # Train model
    trained_model, history = train_model(model, X_train_scaled, y_train_scaled, 
                                        X_test_scaled, y_test_scaled)
    
    # Evaluate model
    y_pred, y_actual, rmse, mae, mape = evaluate_model(
        trained_model, X_test_scaled, y_test_scaled, y_scaler
    )
    
    # Plot results
    plot_results(y_actual, y_pred, f"{freq_label[frequency]} Forecast Evaluation")
    
    # Generate future forecasts
    last_sequence = X_test_scaled[-1]
    forecasts = make_future_forecast(trained_model, last_sequence, forecast_periods, y_scaler, frequency)
    
    # Create future dates
    future_dates = create_future_dates(resampled_df.index[-1], forecast_periods, frequency)
    
    # Create forecast dataframe
    forecast_df = pd.DataFrame({
        'Forecast': forecasts
    }, index=future_dates)
    
    # Plot historical data with forecast
    plot_forecast(resampled_df['Sales'], forecast_df['Forecast'], f"{freq_label[frequency]} Sales Forecast")
    
    # Save model
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"{frequency.lower()}_forecast_model.h5")
    trained_model.save(model_path)
    
    print(f"\n{freq_label[frequency]} Forecast Summary:")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"MAPE: {mape:.2f}%")
    print(f"Model saved to: {model_path}")
    
    return trained_model, forecast_df, (rmse, mae, mape)

def main():
    """
    Main function to run all forecasting models
    """
    # Set random seed for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Configure visible GPU devices (if available)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    
    # File path
    file_path = 'Sales_Data.csv'
    
    # Create output directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    
    # Weekly forecasting
    weekly_model, weekly_forecast, weekly_metrics = forecast_timeseries(
        file_path, 
        frequency='W', 
        sequence_length=8,  # 8 weeks lookback
        forecast_periods=12,  # Forecast next 12 weeks
        lstm_units=64
    )
    
    # Monthly forecasting
    monthly_model, monthly_forecast, monthly_metrics = forecast_timeseries(
        file_path, 
        frequency='M', 
        sequence_length=12,  # 12 months lookback
        forecast_periods=12,  # Forecast next 12 months
        lstm_units=50
    )
    
    # Quarterly forecasting
    quarterly_model, quarterly_forecast, quarterly_metrics = forecast_timeseries(
        file_path, 
        frequency='Q', 
        sequence_length=8,  # 8 quarters lookback
        forecast_periods=8,  # Forecast next 8 quarters (2 years)
        lstm_units=32
    )
    
    # Print overall summary
    print("\n" + "="*50)
    print("FORECASTING SUMMARY")
    print("="*50)
    print("\nWeekly Forecast:")
    print(f"RMSE: {weekly_metrics[0]:.2f}, MAE: {weekly_metrics[1]:.2f}, MAPE: {weekly_metrics[2]:.2f}%")
    
    print("\nMonthly Forecast:")
    print(f"RMSE: {monthly_metrics[0]:.2f}, MAE: {monthly_metrics[1]:.2f}, MAPE: {monthly_metrics[2]:.2f}%")
    
    print("\nQuarterly Forecast:")
    print(f"RMSE: {quarterly_metrics[0]:.2f}, MAE: {quarterly_metrics[1]:.2f}, MAPE: {quarterly_metrics[2]:.2f}%")
    
    # Save forecast CSVs
    weekly_forecast.to_csv("weekly_forecast.csv")
    monthly_forecast.to_csv("monthly_forecast.csv")
    quarterly_forecast.to_csv("quarterly_forecast.csv")
    
    print("\nForecasts saved to CSV files.")
    print("="*50)

if __name__ == "__main__":
    main() 