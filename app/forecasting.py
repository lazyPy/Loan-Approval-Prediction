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
from django.db.models import Sum
from datetime import datetime, timedelta
from .models import Borrower, LoanDisbursementOfficerRemarks, LoanDetails
import random

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)

def get_recent_completed_loans():
    """Get recent completed loans data for forecasting"""
    completed_loans = Borrower.objects.filter(
        loan_disbursement_officer_remarks__status='COMPLETED'
    ).select_related(
        'loan_details',
        'loan_disbursement_officer_remarks'
    ).order_by('loan_disbursement_officer_remarks__disbursement_date')
    
    # Create DataFrame from completed loans
    data = []
    for loan in completed_loans:
        if hasattr(loan, 'loan_details') and hasattr(loan, 'loan_disbursement_officer_remarks'):
            data.append({
                'Date': loan.loan_disbursement_officer_remarks.disbursement_date,
                'Sales': float(loan.loan_details.loan_amount_applied),
                'NumberOfAccount': 1  # Each loan represents one account
            })
    
    df = pd.DataFrame(data)
    if len(df) > 0:
        # Ensure Date is datetime type and properly set as index
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        df.sort_index(inplace=True)
        
    return df

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

def save_model(model, scaler, freq, features_list, seq_length, models_dir='app/trained_models'):
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    
    # Use .keras extension without explicit save_format argument
    model_path = f'{models_dir}/lstm_{freq.lower()}_model.keras'
    model.save(model_path)
    
    joblib.dump(scaler, f'{models_dir}/scaler_{freq.lower()}.pkl')
    
    with open(f'{models_dir}/config_{freq.lower()}.txt', 'w') as f:
        f.write(f"Sequence Length: {seq_length}\n")
        f.write(f"Features: {','.join(features_list)}")

def load_saved_model(freq, models_dir='app/trained_models'):
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

def get_combined_loan_data():
    """Combine historical data from Sales_Data.csv with recent completed loans"""
    try:
        # Load historical data
        historical_df = pd.read_csv('app/trained_models/Sales_Data.csv')
        historical_df['Date'] = pd.to_datetime(historical_df['Date'])
        # Convert Sales column to numeric, removing any currency formatting
        historical_df['Sales'] = historical_df['Sales'].astype(str).str.replace(',', '').astype(float)
        historical_df.set_index('Date', inplace=True)
        
        # Get recent completed loans
        recent_df = get_recent_completed_loans()
        
        # Combine the datasets
        if len(recent_df) > 0:
            # Find the last date in historical data
            last_historical_date = historical_df.index.max()
            
            # Only use recent loans that are after the last historical date
            recent_df = recent_df[recent_df.index > last_historical_date]
            
            if len(recent_df) > 0:
                # Concatenate the datasets
                combined_df = pd.concat([historical_df, recent_df])
                
                # Sort by date
                combined_df.sort_index(inplace=True)
                
                # Remove any duplicates
                combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
                
                print(f"Combined data: {len(combined_df)} records from {combined_df.index.min()} to {combined_df.index.max()}")
                return combined_df
        
        print(f"Using historical data: {len(historical_df)} records from {historical_df.index.min()} to {historical_df.index.max()}")
        return historical_df
        
    except Exception as e:
        print(f"Error combining data: {e}")
        print("Falling back to recent loans only")
        return get_recent_completed_loans()

def train_and_save_models(models_dir='app/trained_models'):
    """Train and save forecasting models"""
    df = get_combined_loan_data()
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
            
            # Create future dates
            last_date = resampled_df.index[-1]
            freq_map = {'W': 'W', 'M': 'ME', 'Q': 'QE'}
            freq_to_use = freq_map.get(freq, freq)
            
            if freq == 'W':
                future_dates = pd.date_range(start=last_date + timedelta(weeks=1), periods=config['forecast_steps'], freq=freq_to_use)
            elif freq == 'M':
                future_dates = pd.date_range(start=last_date + timedelta(days=31), periods=config['forecast_steps'], freq=freq_to_use)
            elif freq == 'Q':
                future_dates = pd.date_range(start=last_date + timedelta(days=92), periods=config['forecast_steps'], freq=freq_to_use)
            
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
            'recommendation': 'Add more completed loans to improve accuracy.'
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

def get_forecast_data(freq):
    """Get forecast data in format suitable for charts"""
    try:
        # Get combined data
        df = get_combined_loan_data()
        if len(df) == 0:
            return {
                'historical': {'dates': [], 'values': []},
                'trend': {'dates': [], 'values': []},
                'seasonal': {'dates': [], 'values': []},
                'residual': {'dates': [], 'values': []},
                'forecast': {'dates': [], 'values': []},
                'analysis': {
                    'trend': 'neutral',
                    'insight': 'No loan data available.',
                    'recommendation': 'Add completed loans to enable forecasting.'
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
                period = 4   # Quarterly data (4 quarters in a year)
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
                trend = resampled_df['Sales'].rolling(window=min(len(resampled_df), 3)).mean().bfill().ffill().values.tolist()
                seasonal = [0] * len(dates)  # No seasonal component
                residual = (resampled_df['Sales'] - pd.Series(trend, index=resampled_df.index)).values.tolist()
        except Exception as e:
            print(f"Error in decomposition: {e}")
            # Fallback to simple trend
            trend = resampled_df['Sales'].rolling(window=min(len(resampled_df), 3)).mean().bfill().ffill().values.tolist()
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
                    forecast_df = generate_future_forecast(model, last_sequence_scaled, forecast_steps, scaler, freq, df)
                    
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
                        'recommendation': 'Add more loan data for better forecasting.'
                    }
            else:
                forecast_dates = []
                forecast_values = []
                analysis = {
                    'trend': 'neutral',
                    'insight': 'No forecast model available.',
                    'recommendation': 'Use "Update Forecasts" button to generate model.'
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

if __name__ == "__main__":
    try:
        # Train models
        results = train_and_save_models()
        print("All models trained and saved successfully!")
    except Exception as e:
        print(f"Error in forecasting: {e}") 