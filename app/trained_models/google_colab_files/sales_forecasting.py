import matplotlib

matplotlib.use('Agg')  # Use non-interactive backend to avoid Tkinter issues

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout, Input
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os
import joblib
from math import sqrt
from datetime import datetime, timedelta
import random

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)


def get_historical_data(verbose=True):
    """Load historical data from Sales_Data.csv"""
    try:
        # Load historical data
        historical_df = pd.read_csv('Sales_Data.csv')
        historical_df['Date'] = pd.to_datetime(historical_df['Date'])
        # Convert Sales column to numeric, removing any currency formatting
        historical_df['Sales'] = historical_df['Sales'].astype(str).str.replace(',', '').astype(float)
        historical_df.set_index('Date', inplace=True)

        if verbose:
            print(
                f"Using historical data: {len(historical_df)} records from {historical_df.index.min()} to {historical_df.index.max()}")
        return historical_df

    except Exception as e:
        print(f"Error loading historical data: {e}")
        return pd.DataFrame()


def resample_data(df, freq):
    """
    Resample data to specified frequency
    frequency: 'W' for weekly, 'M' for monthly, 'Q' for quarterly
    """
    try:
        # First handle duplicate dates by aggregating them
        df = df.groupby(df.index).sum()

        # Define resampling function based on frequency
        if freq == 'W':
            # For weekly data - use end of week (Friday)
            resampled_df = df.resample('W-FRI').sum()
        elif freq == 'M':
            # For monthly data - use month end
            resampled_df = df.resample('MS').sum()  # Month Start instead of ME
        elif freq == 'Q':
            # For quarterly data
            resampled_df = df.resample('QS').sum()  # Quarter Start instead of QE
        else:
            raise ValueError(f"Frequency must be 'W', 'M', or 'Q', got {freq}")

        # Fill any missing values using forward fill
        resampled_df = resampled_df.ffill()

        return resampled_df
    except Exception as e:
        print(f"Error processing {freq} frequency: {str(e)}")
        # Return empty DataFrame with same columns as input
        return pd.DataFrame(columns=df.columns)


def create_features(df):
    """Create time-based features for the model"""
    if len(df) == 0:
        return pd.DataFrame()

    df = df.copy()
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
        x = data[i:i + seq_length]
        y = data[i + seq_length, 0]
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


def prepare_last_sequence(df, seq_length, feature_list):
    """Prepare the last sequence for forecasting"""
    if len(df) < seq_length:
        return None

    # Select only the features used during training
    df = df[feature_list]

    # Get the last sequence
    last_sequence = df.iloc[-seq_length:].values

    return last_sequence


def generate_future_forecast(model, last_sequence, n_steps, scaler, freq, df=None):
    """Generate future forecasts using the loaded model"""
    if last_sequence is None:
        return pd.DataFrame()

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

    # Get the last date from our data or use today if empty
    last_date = datetime.now().date()
    if df is not None and len(df) > 0:
        last_date = df.index[-1]

    # Create future dates with correct frequency strings
    if freq == 'W':
        future_dates = pd.date_range(start=last_date + timedelta(days=1),
                                     periods=n_steps,
                                     freq='W-FRI')  # Weekly on Friday
    elif freq == 'M':
        future_dates = pd.date_range(start=last_date + timedelta(days=1),
                                     periods=n_steps,
                                     freq='MS')  # Month Start
    elif freq == 'Q':
        future_dates = pd.date_range(start=last_date + timedelta(days=1),
                                     periods=n_steps,
                                     freq='QS')  # Quarter Start
    else:
        raise ValueError(f"Invalid frequency: {freq}")

    forecast_df = pd.DataFrame({
        'Forecast': future_preds_scaled
    }, index=future_dates)

    return forecast_df


def save_model(model, scaler, freq, features_list, seq_length, models_dir='forecasting_models'):
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    # Use .keras extension without explicit save_format argument
    model_path = f'{models_dir}/lstm_{freq.lower()}_model.keras'
    model.save(model_path)

    joblib.dump(scaler, f'{models_dir}/scaler_{freq.lower()}.pkl')

    with open(f'{models_dir}/config_{freq.lower()}.txt', 'w') as f:
        f.write(f"Sequence Length: {seq_length}\n")
        f.write(f"Features: {','.join(features_list)}")


def load_saved_model(freq, models_dir='forecasting_models'):
    """Load saved model and associated files"""
    try:
        model_path = f'{models_dir}/lstm_{freq.lower()}_model.keras'
        if not os.path.exists(model_path):
            model_path = f'{models_dir}/lstm_{freq.lower()}_model.h5'

        model = load_model(model_path)
        scaler = joblib.load(f'{models_dir}/scaler_{freq.lower()}.pkl')

        with open(f'{models_dir}/config_{freq.lower()}.txt', 'r') as f:
            lines = f.readlines()
            seq_length = int(lines[0].split(': ')[1])
            features = lines[1].split(': ')[1].strip().split(',')

        return model, scaler, seq_length, features
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None, None, None


def train_and_save_models(models_dir='forecasting_models', verbose=True):
    """Train and save forecasting models"""
    df = get_historical_data(verbose=verbose)
    results = {}

    frequencies = {
        'W': {'seq_length': 4, 'lstm_units': 50, 'dropout_rate': 0.2, 'forecast_steps': 12},
        'M': {'seq_length': 6, 'lstm_units': 64, 'dropout_rate': 0.3, 'forecast_steps': 12},
        'Q': {'seq_length': 3, 'lstm_units': 48, 'dropout_rate': 0.25, 'forecast_steps': 8}
    }

    for freq, config in frequencies.items():
        if verbose:
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
                avg_growth = np.mean([last_values[i] - last_values[i - 1] for i in range(1, len(last_values))])
                forecast_values = [last_values[-1] + avg_growth * (i + 1) for i in range(config['forecast_steps'])]
            else:
                forecast_values = [last_values[-1]] * config['forecast_steps']

            # Create future dates
            last_date = resampled_df.index[-1]
            freq_map = {'W': 'W', 'M': 'ME', 'Q': 'QE'}
            freq_to_use = freq_map.get(freq, freq)

            if freq == 'W':
                future_dates = pd.date_range(start=last_date + timedelta(weeks=1), periods=config['forecast_steps'],
                                             freq=freq_to_use)
            elif freq == 'M':
                future_dates = pd.date_range(start=last_date + timedelta(days=31), periods=config['forecast_steps'],
                                             freq=freq_to_use)
            elif freq == 'Q':
                future_dates = pd.date_range(start=last_date + timedelta(days=92), periods=config['forecast_steps'],
                                             freq=freq_to_use)

            forecast = pd.DataFrame({
                'Date': future_dates,
                'Forecast': forecast_values
            }).set_index('Date')

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

        # Generate forecast
        forecast = generate_future_forecast(model, X_seq[-1], config['forecast_steps'], scaler, freq, df)

        # Save model
        save_model(model, scaler, freq, features, config['seq_length'], models_dir)
        results[freq] = {'forecast': forecast}

    print("All models trained and saved successfully!")
    return results


def analyze_forecast_trend(forecast_values):
    """
    Analyze forecast trend to determine if it's increasing or decreasing,
    and provide relevant insights and recommendations.

    Args:
        forecast_values: List of forecast values

    Returns:
        dict: Contains trend information, insights, and recommendations
    """
    if not forecast_values or len(forecast_values) < 2:
        return {
            'trend': 'neutral',
            'insight': 'Insufficient data for trend analysis.',
            'recommendation': 'Add more historical data to improve accuracy.'
        }

    # Calculate slope using linear regression
    x = np.arange(len(forecast_values))
    y = np.array(forecast_values)

    # Use polyfit to get slope
    slope, _ = np.polyfit(x, y, 1)

    # Calculate percent change from first to last value
    first_value = forecast_values[0]
    last_value = forecast_values[-1]

    if first_value <= 0:  # Avoid division by zero
        percent_change = 0
    else:
        percent_change = ((last_value - first_value) / first_value) * 100

    # Determine trend direction and magnitude
    if slope > 0 and percent_change > 5:
        trend = 'increasing'
        insight = 'Loan disbursement is forecasted to rise, indicating growing demand for vehicle financing.'
        recommendation = 'Improve process efficiency, closely monitor credit risk, and ensure adequate fund availability to support the demand.'
    elif slope < 0 and percent_change < -5:
        trend = 'decreasing'
        insight = 'Loan disbursement is forecasted to decline, indicating lower demand for vehicle financing.'
        recommendation = 'Review market trends, strengthen marketing efforts, and consider offering flexible loan options to attract borrowers or applicants.'
    else:
        trend = 'stable'
        insight = 'Loan disbursement is forecasted to remain relatively stable.'
        recommendation = 'Maintain current lending practices while monitoring market conditions for potential changes.'

    return {
        'trend': trend,
        'insight': insight,
        'recommendation': recommendation,
        'percent_change': round(percent_change, 2)
    }


def get_forecast_data(freq, verbose=False):
    """Get forecast data in format suitable for charts"""
    try:
        # Get historical data
        df = get_historical_data(verbose=verbose)
        if len(df) == 0:
            return {
                'historical': {'dates': [], 'values': []},
                'trend': {'dates': [], 'values': []},
                'seasonal': {'dates': [], 'values': []},
                'residual': {'dates': [], 'values': []},
                'forecast': {'dates': [], 'values': []},
                'analysis': {
                    'trend': 'neutral',
                    'insight': 'No historical data available.',
                    'recommendation': 'Check if Sales_Data.csv file exists and is properly formatted.'
                }
            }

        # Resample data
        resampled_df = resample_data(df, freq)

        # Convert dates to string format for JSON
        dates = resampled_df.index.strftime('%Y-%m-%d').tolist()
        values = resampled_df['Sales'].tolist()

        try:
            # Perform decomposition with appropriate period
            if freq == 'W':
                period = 52  # Weekly data (52 weeks in a year)
            elif freq == 'M':
                period = 12  # Monthly data (12 months in a year)
            elif freq == 'Q':
                period = 4  # Quarterly data (4 quarters in a year)
            else:
                period = 1

            # Only perform decomposition if we have enough data points
            if len(resampled_df) >= period * 2:
                decomposition = seasonal_decompose(resampled_df['Sales'], model='additive', period=period)
                trend = decomposition.trend.bfill().ffill().values.tolist()
                seasonal = decomposition.seasonal.bfill().ffill().values.tolist()
                residual = decomposition.resid.bfill().ffill().values.tolist()
            else:
                # If not enough data, use simple moving averages
                trend = resampled_df['Sales'].rolling(
                    window=min(len(resampled_df), 3)).mean().bfill().ffill().values.tolist()
                seasonal = [0] * len(dates)  # No seasonal component
                residual = (resampled_df['Sales'] - pd.Series(trend, index=resampled_df.index)).values.tolist()
        except Exception as e:
            print(f"Error in decomposition: {e}")
            # Fallback to simple trend
            trend = resampled_df['Sales'].rolling(
                window=min(len(resampled_df), 3)).mean().bfill().ffill().values.tolist()
            seasonal = [0] * len(dates)
            residual = [0] * len(dates)

        # Generate forecast
        try:
            model, scaler, seq_length, features = load_saved_model(freq)
            if model is not None:
                feature_df = create_features(resampled_df)
                last_sequence = prepare_last_sequence(feature_df, seq_length, features)
                if last_sequence is not None:
                    last_sequence_scaled = scaler.transform(last_sequence)
                    forecast_steps = {'W': 12, 'M': 12, 'Q': 8}[freq]
                    forecast_df = generate_future_forecast(model, last_sequence_scaled, forecast_steps, scaler, freq,
                                                           df)

                    forecast_dates = forecast_df.index.strftime('%Y-%m-%d').tolist()
                    forecast_values = forecast_df['Forecast'].tolist()

                    # Analyze forecast trend
                    analysis = analyze_forecast_trend(forecast_values)
                else:
                    forecast_dates = []
                    forecast_values = []
                    analysis = {
                        'trend': 'neutral',
                        'insight': 'Insufficient sequence data.',
                        'recommendation': 'Add more historical loan data for better forecasting.'
                    }
            else:
                forecast_dates = []
                forecast_values = []
                analysis = {
                    'trend': 'neutral',
                    'insight': 'No forecast model available.',
                    'recommendation': 'Regenerate new forecasting model.'
                }
        except Exception as e:
            print(f"Error generating forecast: {e}")
            forecast_dates = []
            forecast_values = []
            analysis = {
                'trend': 'neutral',
                'insight': f'Error generating forecast: {str(e)}',
                'recommendation': 'Try regenerating forecasts.'
            }

        return {
            'historical': {
                'dates': dates,
                'values': values
            },
            'trend': {
                'dates': dates,
                'values': trend
            },
            'seasonal': {
                'dates': dates,
                'values': seasonal
            },
            'residual': {
                'dates': dates,
                'values': residual
            },
            'forecast': {
                'dates': forecast_dates,
                'values': forecast_values
            },
            'analysis': analysis
        }

    except Exception as e:
        print(f"Error getting forecast data: {e}")
        # Return empty data structure instead of None
        return {
            'historical': {'dates': [], 'values': []},
            'trend': {'dates': [], 'values': []},
            'seasonal': {'dates': [], 'values': []},
            'residual': {'dates': [], 'values': []},
            'forecast': {'dates': [], 'values': []},
            'analysis': {
                'trend': 'neutral',
                'insight': f'Error analyzing data: {str(e)}',
                'recommendation': 'Try regenerating forecasts.'
            }
        }


def evaluate_model_performance(freq, test_size=0.2, verbose=False):
    """Evaluate model performance on historical data

    Args:
        freq (str): Frequency - 'W', 'M', or 'Q'
        test_size (float): Portion of data to use for testing
        verbose (bool): Whether to print progress messages

    Returns:
        dict: Performance metrics
    """
    try:
        # Get historical data
        df = get_historical_data(verbose=verbose)
        if len(df) == 0:
            return {
                'rmse': None,
                'mae': None,
                'mape': None,
                'r2': None,
                'actual_vs_predicted': {'dates': [], 'actual': [], 'predicted': []}
            }

        # Resample data to the specified frequency
        resampled_df = resample_data(df, freq)

        # Get features
        feature_df = create_features(resampled_df)
        features = ['Sales'] + [col for col in feature_df.columns if col != 'Sales']
        X = feature_df[features].values

        # Load model configuration
        model, scaler, seq_length, _ = load_saved_model(freq)

        if model is None or len(X) <= seq_length:
            return {
                'rmse': None,
                'mae': None,
                'mape': None,
                'r2': None,
                'actual_vs_predicted': {'dates': [], 'actual': [], 'predicted': []}
            }

        # Scale the features
        X_scaled = scaler.transform(X)

        # Create sequences
        X_seq, y_seq = create_sequences(X_scaled, seq_length)

        # Split into train and test sets
        train_size = int(len(X_seq) * (1 - test_size))
        X_train, X_test = X_seq[:train_size], X_seq[train_size:]
        y_train, y_test = y_seq[:train_size], y_seq[train_size:]

        if len(X_test) == 0:
            # Not enough data for test set
            return {
                'rmse': None,
                'mae': None,
                'mape': None,
                'r2': None,
                'actual_vs_predicted': {'dates': [], 'actual': [], 'predicted': []}
            }

        # Generate predictions for test set
        y_pred_scaled = model.predict(X_test)

        # Inverse transform to get original scale
        y_test_transformed = np.zeros((len(y_test), scaler.scale_.shape[0]))
        y_test_transformed[:, 0] = y_test
        y_test_original = scaler.inverse_transform(y_test_transformed)[:, 0]

        y_pred_transformed = np.zeros((len(y_pred_scaled), scaler.scale_.shape[0]))
        y_pred_transformed[:, 0] = y_pred_scaled.flatten()
        y_pred_original = scaler.inverse_transform(y_pred_transformed)[:, 0]

        # Calculate metrics
        rmse = sqrt(mean_squared_error(y_test_original, y_pred_original))
        mae = mean_absolute_error(y_test_original, y_pred_original)

        # Calculate MAPE (Mean Absolute Percentage Error)
        # Only include non-zero actual values in MAPE calculation
        non_zero_indices = y_test_original != 0
        if np.any(non_zero_indices):
            mape = np.mean(np.abs((y_test_original[non_zero_indices] - y_pred_original[non_zero_indices]) /
                                  y_test_original[non_zero_indices])) * 100
        else:
            mape = np.nan  # No valid data for MAPE calculation

        # Calculate R-squared
        ss_total = np.sum((y_test_original - np.mean(y_test_original)) ** 2)
        ss_residual = np.sum((y_test_original - y_pred_original) ** 2)
        r2 = 1 - (ss_residual / ss_total)

        # Get dates for test set
        test_dates = feature_df.index[train_size + seq_length:].strftime('%Y-%m-%d').tolist()

        return {
            'rmse': round(rmse, 2),
            'mae': round(mae, 2),
            'mape': round(float(mape), 2) if not np.isnan(mape) else None,
            'r2': round(r2, 2),
            'actual_vs_predicted': {
                'dates': test_dates,
                'actual': y_test_original.tolist(),
                'predicted': y_pred_original.tolist()
            }
        }
    except Exception as e:
        print(f"Error evaluating model performance: {e}")
        return {
            'rmse': None,
            'mae': None,
            'mape': None,
            'r2': None,
            'actual_vs_predicted': {'dates': [], 'actual': [], 'predicted': []}
        }


def visualize_model_performance(freq, output_dir='forecasting_images', verbose=False):
    """Generate and save model performance visualization

    Args:
        freq (str): Frequency - 'W', 'M', or 'Q'
        output_dir (str): Directory to save images
        verbose (bool): Whether to print progress messages

    Returns:
        dict: Path to generated image and performance metrics
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get model performance metrics
    performance = evaluate_model_performance(freq, verbose=verbose)

    freq_name = {
        'W': 'Weekly',
        'M': 'Monthly',
        'Q': 'Quarterly'
    }[freq]

    if not performance['actual_vs_predicted']['dates']:
        # No performance data available
        return {'image_path': None, 'metrics': performance}

    # Convert string dates back to datetime for plotting
    test_dates = [pd.to_datetime(d) for d in performance['actual_vs_predicted']['dates']]
    actual = performance['actual_vs_predicted']['actual']
    predicted = performance['actual_vs_predicted']['predicted']

    # Plot actual vs predicted values
    plt.figure(figsize=(12, 8))

    # Plot actual and predicted values
    plt.subplot(2, 1, 1)
    plt.plot(test_dates, actual, 'b-', label='Actual')
    plt.plot(test_dates, predicted, 'r--', label='Predicted')
    plt.title(f'{freq_name} Model Performance - Actual vs Predicted Values')
    plt.ylabel('Amount')
    plt.grid(True)
    plt.legend()

    # Plot residuals (errors)
    residuals = np.array(actual) - np.array(predicted)
    plt.subplot(2, 1, 2)
    plt.plot(test_dates, residuals, 'g-')
    plt.axhline(y=0, color='r', linestyle='-')
    plt.title('Prediction Errors (Residuals)')
    plt.xlabel('Date')
    plt.ylabel('Error')
    plt.grid(True)

    plt.tight_layout()

    # Save figure
    performance_path = f"{output_dir}/{freq.lower()}_performance.png"
    plt.savefig(performance_path)
    plt.close()

    # Add metrics table
    plt.figure(figsize=(8, 4))
    plt.axis('tight')
    plt.axis('off')

    metrics_table = [
        ['Metric', 'Value'],
        ['RMSE', str(performance['rmse'])],
        ['MAE', str(performance['mae'])],
        ['MAPE (%)', str(performance['mape'])],
        ['R²', str(performance['r2'])]
    ]

    table = plt.table(cellText=metrics_table, colWidths=[0.4, 0.3], loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.8)

    plt.title(f'{freq_name} Model Performance Metrics', fontsize=14, pad=20)

    # Save metrics table
    metrics_path = f"{output_dir}/{freq.lower()}_metrics.png"
    plt.savefig(metrics_path)
    plt.close()

    return {'image_path': performance_path, 'metrics_path': metrics_path, 'metrics': performance}


def generate_forecast_visualizations(freq, output_dir='forecasting_images', verbose=False):
    """Generate and save visualization images for forecasts

    Args:
        freq (str): Frequency - 'W', 'M', or 'Q'
        output_dir (str): Directory to save images
        verbose (bool): Whether to print progress messages

    Returns:
        dict: Paths to generated images
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get forecast data
    data = get_forecast_data(freq, verbose=verbose)

    if not data['historical']['dates']:
        print(f"No data available for {freq} frequency")
        return {}

    freq_name = {
        'W': 'Weekly',
        'M': 'Monthly',
        'Q': 'Quarterly'
    }[freq]

    # Convert string dates back to datetime for plotting
    historical_dates = [pd.to_datetime(d) for d in data['historical']['dates']]
    forecast_dates = [pd.to_datetime(d) for d in data['forecast']['dates']]

    image_paths = {}

    # Figure 1: Historical Data
    plt.figure(figsize=(12, 6))
    plt.plot(historical_dates, data['historical']['values'], 'b-', label='Historical Data')
    plt.title(f'{freq_name} Loan Disbursement - Historical Data')
    plt.xlabel('Date')
    plt.ylabel('Amount')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # Save figure
    historical_path = f"{output_dir}/{freq.lower()}_historical.png"
    plt.savefig(historical_path)
    plt.close()
    image_paths['historical'] = historical_path

    # Figure 2: Time Series Decomposition
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # Trend
    ax1.plot(historical_dates, data['trend']['values'], 'r-')
    ax1.set_title(f'{freq_name} Loan Disbursement - Trend Component')
    ax1.grid(True)

    # Seasonal
    ax2.plot(historical_dates, data['seasonal']['values'], 'g-')
    ax2.set_title(f'{freq_name} Loan Disbursement - Seasonal Component')
    ax2.grid(True)

    # Residual
    ax3.plot(historical_dates, data['residual']['values'], 'k-')
    ax3.set_title(f'{freq_name} Loan Disbursement - Residual Component')
    ax3.grid(True)
    ax3.set_xlabel('Date')

    plt.tight_layout()

    # Save figure
    decomposition_path = f"{output_dir}/{freq.lower()}_decomposition.png"
    plt.savefig(decomposition_path)
    plt.close()
    image_paths['decomposition'] = decomposition_path

    # Figure 3: Forecast
    plt.figure(figsize=(12, 6))

    # Plot historical data
    plt.plot(historical_dates, data['historical']['values'], 'b-', label='Historical Data')

    # Plot forecast
    if forecast_dates and data['forecast']['values']:
        plt.plot(forecast_dates, data['forecast']['values'], 'r--', label='Forecast')

        # Add confidence intervals (simple implementation - could be improved)
        forecast_values = np.array(data['forecast']['values'])
        std_dev = np.std(data['historical']['values']) if len(data['historical']['values']) > 1 else 0

        plt.fill_between(
            forecast_dates,
            forecast_values - 1.96 * std_dev,
            forecast_values + 1.96 * std_dev,
            color='r', alpha=0.2, label='95% Confidence Interval'
        )

    plt.title(f'{freq_name} Loan Disbursement - Forecast')
    plt.xlabel('Date')
    plt.ylabel('Amount')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # Save figure
    forecast_path = f"{output_dir}/{freq.lower()}_forecast.png"
    plt.savefig(forecast_path)
    plt.close()
    image_paths['forecast'] = forecast_path

    # Add model performance visualization
    performance_result = visualize_model_performance(freq, output_dir, verbose)
    if performance_result['image_path']:
        image_paths['performance'] = performance_result['image_path']
        image_paths['metrics'] = performance_result['metrics_path']

    return image_paths


def generate_all_forecast_visualizations(output_dir='forecasting_images', verbose=False):
    """Generate and save visualizations for all frequencies"""
    results = {}
    for freq in ['W', 'M', 'Q']:
        results[freq] = generate_forecast_visualizations(freq, output_dir, verbose)
    return results


def print_model_performance_metrics(verbose=False):
    """Print performance metrics for all forecasting models

    Args:
        verbose (bool): Whether to print detailed data loading messages
    """
    print("\n----- MODEL PERFORMANCE METRICS -----")

    for freq in ['W', 'M', 'Q']:
        freq_name = {'W': 'Weekly', 'M': 'Monthly', 'Q': 'Quarterly'}[freq]
        print(f"\n{freq_name} Model Performance:")

        # Get performance metrics with the specified verbosity
        metrics = evaluate_model_performance(freq, verbose=verbose)

        if metrics['rmse'] is not None:
            print(f"RMSE (Root Mean Square Error): {metrics['rmse']}")
            print(f"MAE (Mean Absolute Error): {metrics['mae']}")

            if metrics['mape'] is not None:
                print(f"MAPE (Mean Absolute Percentage Error): {metrics['mape']}%")
            else:
                print("MAPE: Not available (division by zero)")

            print(f"R² (Coefficient of Determination): {metrics['r2']}")

            accuracy = max(0, min(100, 100 * (
                        1 - abs(metrics['mae'] / (np.mean(metrics['actual_vs_predicted']['actual']) or 1)))))
            print(f"Approximate Accuracy: {accuracy:.2f}%")

            # Calculate how many predictions were within 10% of actual values
            actual = np.array(metrics['actual_vs_predicted']['actual'])
            predicted = np.array(metrics['actual_vs_predicted']['predicted'])
            non_zero_actual = actual != 0

            if np.any(non_zero_actual):
                percent_errors = np.abs(
                    (predicted[non_zero_actual] - actual[non_zero_actual]) / actual[non_zero_actual]) * 100
                within_10_percent = np.sum(percent_errors <= 10) / len(percent_errors) * 100
                print(f"Predictions within 10% of actual value: {within_10_percent:.2f}%")
            else:
                print("Predictions within 10%: Not available (no non-zero actual values)")
        else:
            print("No performance metrics available. Ensure model is trained and test data is available.")


if __name__ == "__main__":
    try:
        # Train models
        train_and_save_models()

        # Generate visualizations
        generate_all_forecast_visualizations()

        # Print model performance metrics
        print_model_performance_metrics()

        print("\nAll models trained and visualizations generated successfully!")
    except Exception as e:
        print(f"Error in forecasting: {e}")