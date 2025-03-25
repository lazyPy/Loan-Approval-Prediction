import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os
import joblib
from math import sqrt

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
    df['Sales'] = df['Sales'].astype(str).str.replace(',', '').str.replace('"', '').astype(float)
    df.set_index('Date', inplace=True)
    return df

def resample_data(df, freq):
    if freq == 'W':
        daily_df = df.resample('D').asfreq()
        daily_df = daily_df.ffill()  # Use ffill instead of fillna(method='ffill')
        resampled_df = daily_df.resample('W').sum()
        resampled_df = resampled_df.replace(0, np.nan).ffill()  # Use ffill instead of fillna(method='ffill')
    else:
        # Use updated frequency aliases
        freq_map = {'M': 'ME', 'Q': 'QE'}
        freq_to_use = freq_map.get(freq, freq)
        resampled_df = df.resample(freq_to_use).sum()
    return resampled_df

def create_features(df):
    df['month'] = df.index.month
    df['quarter'] = df.index.quarter
    df['year'] = df.index.year
    
    if 'W' in str(df.index.freq):
        df['dayofweek'] = df.index.dayofweek
    
    for lag in range(1, 4):
        df[f'Sales_lag_{lag}'] = df['Sales'].shift(lag)
    
    df['Sales_rolling_mean_3'] = df['Sales'].rolling(window=3).mean()
    df['Sales_rolling_std_3'] = df['Sales'].rolling(window=3).std()
    return df.dropna()

def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i+seq_length]
        y = data[i+seq_length, 0]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def build_lstm_model(input_shape, lstm_units=50, dropout_rate=0.2):
    # Use Input layer instead of input_shape for Sequential model
    model = Sequential([
        Input(shape=input_shape),
        LSTM(units=lstm_units, return_sequences=True),
        Dropout(dropout_rate),
        LSTM(units=lstm_units, return_sequences=True),
        Dropout(dropout_rate),
        LSTM(units=lstm_units),
        Dropout(dropout_rate),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    return model

def train_lstm_model(X_train, y_train, X_test, y_test, epochs=100, batch_size=32, lstm_units=50, dropout_rate=0.2):
    model = build_lstm_model((X_train.shape[1], X_train.shape[2]), lstm_units, dropout_rate)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    validation_data = (X_test, y_test) if len(X_test) > 0 else None
    
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=min(batch_size, len(X_train)),
        validation_data=validation_data,
        callbacks=[early_stopping],
        verbose=0
    )
    
    return model, history

def evaluate_model(model, X_test, y_test, scaler):
    y_pred = model.predict(X_test, verbose=0)
    
    # Create arrays for inverse transformation
    y_test_array = np.zeros((len(y_test), scaler.scale_.shape[0]))
    y_test_array[:, 0] = y_test.flatten()
    
    y_pred_array = np.zeros((len(y_pred), scaler.scale_.shape[0]))
    y_pred_array[:, 0] = y_pred.flatten()
    
    # Inverse transform to original scale
    y_test_inv = scaler.inverse_transform(y_test_array)[:, 0]
    y_pred_inv = scaler.inverse_transform(y_pred_array)[:, 0]
    
    # Calculate metrics
    rmse = sqrt(mean_squared_error(y_test_inv, y_pred_inv))
    mae = mean_absolute_error(y_test_inv, y_pred_inv)
    mape = np.mean(np.abs((y_test_inv - y_pred_inv) / np.maximum(1e-10, np.abs(y_test_inv)))) * 100
    
    return {
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'y_test': y_test_inv,
        'y_pred': y_pred_inv
    }

def plot_predictions(y_test, y_pred, title='Actual vs Predicted Values'):
    plt.figure(figsize=(12, 6))
    
    y_test_flat = np.array(y_test).flatten()
    y_pred_flat = np.array(y_pred).flatten()
    
    min_len = min(len(y_test_flat), len(y_pred_flat))
    x_axis = np.arange(min_len)
    
    plt.plot(x_axis, y_test_flat[:min_len], label='Actual')
    plt.plot(x_axis, y_pred_flat[:min_len], label='Predicted')
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Sales')
    plt.legend()
    plt.tight_layout()
    
    filepath = f'static/img/{title.replace(" ", "_").lower()}.png'
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    plt.savefig(filepath)
    plt.close()
    return filepath

def generate_future_forecast(model, last_sequence, n_steps, scaler, freq):
    curr_seq = last_sequence.copy()
    future_preds = []
    
    for _ in range(n_steps):
        seq_reshaped = curr_seq.reshape(1, curr_seq.shape[0], curr_seq.shape[1])
        curr_pred = model.predict(seq_reshaped, verbose=0)
        future_preds.append(curr_pred[0, 0])
        
        curr_seq = np.roll(curr_seq, -1, axis=0)
        curr_seq[-1, 0] = curr_pred[0, 0]
    
    # Convert predictions to original scale
    scaled_features = np.zeros((len(future_preds), scaler.scale_.shape[0]))
    scaled_features[:, 0] = future_preds
    future_preds_scaled = scaler.inverse_transform(scaled_features)[:, 0]
    
    # Create future dates
    last_date = pd.to_datetime('2024-12-01')
    
    # Use updated frequency aliases
    freq_map = {'W': 'W', 'M': 'ME', 'Q': 'QE'}
    freq_to_use = freq_map.get(freq, freq)
    
    if freq == 'W':
        future_dates = pd.date_range(start=last_date + pd.Timedelta(weeks=1), periods=n_steps, freq=freq_to_use)
    elif freq == 'M':
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=31), periods=n_steps, freq=freq_to_use)
    elif freq == 'Q':
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=92), periods=n_steps, freq=freq_to_use)
    else:
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=31), periods=n_steps, freq='ME')
    
    forecast_df = pd.DataFrame({
        'Date': future_dates,
        'Forecast': future_preds_scaled
    }).set_index('Date')
    
    return forecast_df

def plot_forecast(historical_data, forecast_data, title='Historical Data and Forecast'):
    plt.figure(figsize=(15, 7))
    
    historical_dates = historical_data.index
    historical_values = historical_data['Sales'].values
    
    forecast_dates = forecast_data.index
    forecast_values = forecast_data['Forecast'].values
    
    plt.plot(historical_dates, historical_values, label='Historical Data')
    plt.plot(forecast_dates, forecast_values, label='Forecast', color='red')
    
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    filepath = f'static/img/{title.replace(" ", "_").lower()}.png'
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    plt.savefig(filepath)
    plt.close()
    return filepath

def save_model(model, scaler, freq, features_list, seq_length, models_dir='app/ml_models'):
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    
    # Use .keras extension without explicit save_format argument
    model_path = f'{models_dir}/lstm_{freq.lower()}_model.keras'
    model.save(model_path)
    
    joblib.dump(scaler, f'{models_dir}/scaler_{freq.lower()}.pkl')
    
    with open(f'{models_dir}/config_{freq.lower()}.txt', 'w') as f:
        f.write(f"Sequence Length: {seq_length}\n")
        f.write(f"Features: {','.join(features_list)}")

def load_saved_model(freq, models_dir='app/ml_models'):
    try:
        # First try .keras format
        model_path = f'{models_dir}/lstm_{freq.lower()}_model.keras'
        if not os.path.exists(model_path):
            # Fall back to .h5 format for backward compatibility
            model_path = f'{models_dir}/lstm_{freq.lower()}_model.h5'
        
        model = load_model(model_path)
        scaler = joblib.load(f'{models_dir}/scaler_{freq.lower()}.pkl')
        
        with open(f'{models_dir}/config_{freq.lower()}.txt', 'r') as f:
            lines = f.readlines()
            seq_length = int(lines[0].split(': ')[1])
            features = lines[1].split(': ')[1].split(',')
        
        return model, scaler, seq_length, features
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None, None, None

def train_and_save_models(file_path='Sales_Data.csv', models_dir='app/ml_models'):
    os.makedirs('static/img', exist_ok=True)
    df = load_and_preprocess_data(file_path)
    results = {}
    
    frequencies = {
        'W': {'seq_length': 4, 'lstm_units': 50, 'dropout_rate': 0.2, 'forecast_steps': 12},
        'M': {'seq_length': 6, 'lstm_units': 64, 'dropout_rate': 0.3, 'forecast_steps': 12},
        'Q': {'seq_length': 3, 'lstm_units': 48, 'dropout_rate': 0.25, 'forecast_steps': 8}
    }
    
    for freq, config in frequencies.items():
        print(f"Training model for {freq} frequency")
        resampled_df = resample_data(df, freq)
        feature_df = create_features(resampled_df)
        
        features = ['Sales'] + [col for col in feature_df.columns if col != 'Sales']
        X = feature_df[features].values
        
        # Skip if not enough data
        if len(X) <= config['seq_length']:
            print(f"Not enough data for {freq} frequency. Skipping.")
            continue
        
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        
        X_seq, y_seq = create_sequences(X_scaled, config['seq_length'])
        
        # Use simple model if not enough sequences
        if len(X_seq) < 2:
            print(f"Not enough sequences for {freq}. Using simple model.")
            # Create simple forecast using last value + average growth
            last_values = resampled_df['Sales'].values[-3:]
            if len(last_values) >= 2:
                avg_growth = np.mean([last_values[i] - last_values[i-1] for i in range(1, len(last_values))])
                forecast_values = [last_values[-1] + avg_growth * (i+1) for i in range(config['forecast_steps'])]
            else:
                forecast_values = [last_values[-1]] * config['forecast_steps']
            
            # Create future dates with updated frequency aliases
            last_date = resampled_df.index[-1]
            freq_map = {'W': 'W', 'M': 'ME', 'Q': 'QE'}
            freq_to_use = freq_map.get(freq, freq)
            
            if freq == 'W':
                future_dates = pd.date_range(start=last_date + pd.Timedelta(weeks=1), periods=config['forecast_steps'], freq=freq_to_use)
            elif freq == 'M':
                future_dates = pd.date_range(start=last_date + pd.Timedelta(days=31), periods=config['forecast_steps'], freq=freq_to_use)
            elif freq == 'Q':
                future_dates = pd.date_range(start=last_date + pd.Timedelta(days=92), periods=config['forecast_steps'], freq=freq_to_use)
            else:
                future_dates = pd.date_range(start=last_date + pd.Timedelta(days=31), periods=config['forecast_steps'], freq='ME')
            
            forecast = pd.DataFrame({
                'Date': future_dates,
                'Forecast': forecast_values
            }).set_index('Date')
            
            plot_forecast(resampled_df, forecast, f'{freq} Frequency - Forecast')
            
            # Save a dummy model with proper Input layer
            dummy_model = Sequential([
                Input(shape=(1,)),
                Dense(1)
            ])
            dummy_model.compile(optimizer='adam', loss='mse')
            save_model(dummy_model, scaler, freq, features, config['seq_length'], models_dir)
            
            results[freq] = {'forecast': forecast}
            continue
        
        # Split into training and testing sets (80:20)
        train_size = max(1, int(len(X_seq) * 0.8))
        X_train, X_test = X_seq[:train_size], X_seq[train_size:]
        y_train, y_test = y_seq[:train_size], y_seq[train_size:]
        
        # Train the model
        model, _ = train_lstm_model(
            X_train, y_train, X_test, y_test, 
            lstm_units=config['lstm_units'], 
            dropout_rate=config['dropout_rate'],
            epochs=50 if freq == 'Q' else 100
        )
        
        # Evaluate if test data exists
        if len(X_test) > 0:
            eval_results = evaluate_model(model, X_test, y_test, scaler)
            plot_predictions(eval_results['y_test'], eval_results['y_pred'], f'{freq} Frequency - Actual vs Predicted')
        
        # Generate forecast
        forecast = generate_future_forecast(model, X_seq[-1], config['forecast_steps'], scaler, freq)
        plot_forecast(resampled_df, forecast, f'{freq} Frequency - Forecast')
        
        # Save model
        save_model(model, scaler, freq, features, config['seq_length'], models_dir)
        results[freq] = {'forecast': forecast}
    
    print("All models trained and saved successfully!")
    return results

if __name__ == "__main__":
    try:
        results = train_and_save_models()
        print("All models trained and saved successfully!")
    except Exception as e:
        print(f"Error in forecasting: {e}") 