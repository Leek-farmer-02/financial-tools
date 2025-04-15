import numpy as np
import yfinance as yf
from scipy.stats import pearsonr

def garch_volatility_forecast(ticker, omega, alpha, beta, forecast_horizon=5, sigma2_0=None, period='1y'):
    """
    Forecasts future volatility using a GARCH(1,1) model with historical returns from Yahoo Finance.

    Parameters:
    - ticker: Stock ticker symbol as a string (e.g., 'AAPL').
    - omega: Long-run variance component.
    - alpha: Reaction to recent squared returns.
    - beta: Persistence of past volatility.
    - forecast_horizon: Number of periods ahead to forecast volatility.
    - sigma2_0: Initial volatility estimate (optional).
    - period: Period of historical data to retrieve (default '1y').

    Returns:
    - forecasts: Array of forecasted future volatility.
    """

    # Download historical data
    data = yf.download(ticker, period=period)
    returns = data['Close'].pct_change().dropna().values
    T = len(returns)

    # Initialize volatility
    sigma2 = np.zeros(T)
    sigma2[0] = np.var(returns) if sigma2_0 is None else sigma2_0

    # Calculate historical volatility
    for t in range(1, T):
        sigma2[t] = omega + alpha * returns[t - 1] ** 2 + beta * sigma2[t - 1]

    # Forecast future volatility
    forecasts = np.zeros(forecast_horizon)
    forecasts[0] = omega + alpha * returns[-1] ** 2 + beta * sigma2[-1]

    for h in range(1, forecast_horizon):
        forecasts[h] = omega + (alpha + beta) * forecasts[h - 1]

        # Backtesting: compare historical volatility with model predictions
    min_length = min(len(forecasts), len(sigma2[-forecast_horizon:]))
    predicted_volatility = np.sqrt(forecasts[:min_length])

    print(f"Volatility Forecast for {ticker}:")
    for i, vol in enumerate(predicted_volatility, 1):
        print(f"Period {i}: {vol:.6f}")

    return forecasts


import numpy as np
import yfinance as yf
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from arch import arch_model


def garch_forecast(ticker, omega, alpha, beta, period='1y', sigma2_0=None):
    """
    Computes and prints GARCH(1,1) volatility forecasts for multiple time horizons.

    Parameters:
    - ticker: Stock ticker symbol (e.g., 'AAPL').
    - omega: Long-run variance component.
    - alpha: Reaction to recent squared returns.
    - beta: Persistence of past volatility.
    - period: Period of historical data to retrieve (e.g., '1y').
    - sigma2_0: Initial variance value (if None, the sample variance of returns is used).
    
    Returns:
    - forecast_vols: A dictionary with forecasted volatility at different time horizons.
    """
    # Download historical data via yfinance and compute daily percentage returns.
    data = yf.download(ticker, period=period)
    returns = data['Close'].pct_change().dropna().values
    
    # Estimate historical conditional variance using the GARCH(1,1) recursion.
    T = len(returns)
    sigma2 = np.zeros(T)
    sigma2[0] = np.var(returns) if sigma2_0 is None else sigma2_0
    for t in range(1, T):
        sigma2[t] = omega + alpha * (returns[t - 1] ** 2) + beta * sigma2[t - 1]
    
    # Use the last observed return and variance as the starting point.
    last_return = returns[-1]
    last_sigma2 = sigma2[-1]
    
    # Define forecast horizons in terms of trading days.
    horizons = {
        'Day': 1,
        'Week': 5,
        'Month': 21,
        'Quarter': 63,
        'Annually': 252
    }
    forecast_vols = {}
    
    # For each time horizon, compute the volatility forecast recursively.
    for label, h in horizons.items():
        forecast = np.zeros(h)
        # One-step forecast: incorporate the last observation.
        forecast[0] = omega + alpha * (last_return ** 2) + beta * last_sigma2
        # For h > 1, use the simplified recursion
        for i in range(1, h):
            forecast[i] = omega + (alpha + beta) * forecast[i - 1]
        # The h-th forecast (variance forecast) is converted to volatility by taking the square root.
        forecast_vols[label] = np.sqrt(forecast[-1])
    
    # Print the formatted forecast table.
    print(f"Multi-Horizon Volatility Forecast for {ticker}:")
    for label in horizons.keys():
        print(f"{label:8s}: {forecast_vols[label]:.6f}")
    
    return forecast_vols

def garch_forecast_arch(ticker, period='1y'):
    """
    Computes and prints GARCH(1,1) volatility forecasts for multiple time horizons using the arch_model library.

    The function:
      1. Downloads historical data via yfinance and computes daily percentage returns.
      2. Fits a GARCH(1,1) model (with constant mean) using Maximum Likelihood.
      3. Forecasts the conditional variance (and hence volatility) for several horizons.
         The horizons are defined in terms of approximate trading days.
    
    Parameters:
    - ticker: Stock ticker symbol (e.g., 'AAPL').
    - period: Period of historical data to retrieve (e.g., '1y').

    Returns:
    - forecast_vols: A dictionary containing forecasted volatility (square-root of the variance forecast)
                     for different time horizons.
    """
    # Download historical data and compute daily returns.
    data = yf.download(ticker, period=period)
    returns = data['Close'].pct_change().dropna()
    
    # Fit a GARCH(1,1) model using the arch_model library.
    # We use a constant mean model with a normal error distribution.
    am = arch_model(returns, vol='Garch', p=1, q=1, mean='Constant', dist='normal')
    res = am.fit(disp='off')
    
    # Define forecast horizons using approximate trading day counts.
    horizons = {
        'Day': 1,
        'Week': 5,
        'Month': 21,
        'Quarter': 63,
        'Annually': 252
    }
    
    # The maximum horizon determines the forecast length.
    max_horizon = max(horizons.values())
    # Forecast over the required maximum horizon.
    fcast = res.forecast(horizon=max_horizon, reindex=False)
    # Get the variance forecasts from the last available forecast.
    # The forecast object stores the variance forecast in a DataFrame where the columns correspond 
    # to step-ahead predictions. For example, column index 0 is for 1-day ahead, 1 for 2-day ahead, etc.
    forecast_vars = fcast.variance.iloc[-1].values

    forecast_vols = {}
    # For each defined horizon, extract the corresponding variance forecast,
    # then convert it to volatility by taking the square root.
    for label, h in horizons.items():
        # Note: For a horizon h, we use the forecast at step h (i.e. index h-1)
        vol_forecast = np.sqrt(forecast_vars[h - 1])
        forecast_vols[label] = vol_forecast
    
    # Print the formatted forecast table.
    print(f"Multi-Horizon Volatility Forecast for {ticker} (using arch_model):")
    for label in horizons.keys():
        print(f"{label:8s}: {forecast_vols[label]:.6f}")
    
    return forecast_vols

def sliding_window_backtest(ticker, omega, alpha, beta, window_size, forecast_horizon=1, period='2y'):
    """
    Performs sliding window backtesting for a GARCH(1,1) volatility forecast model.

    Parameters:
    - ticker: Stock ticker symbol (e.g., 'AAPL').
    - omega: Long-run variance component.
    - alpha: Reaction to recent squared returns.
    - beta: Persistence of past volatility.
    - window_size: Number of historical days used for each estimation window.
    - forecast_horizon: Number of days ahead to forecast (usually 1 for day-ahead forecasts).
    - period: Time span for fetching historical data (e.g., '2y').

    Returns:
    - forecast_vol_array: Array with forecasted volatility for each sliding window.
    - realized_vol_array: Array with realized volatility corresponding to each forecast.
    - forecast_dates: Dates corresponding to each forecast (taken as the start of the forecast day).
    """
    
    # Download historical data using yfinance
    data = yf.download(ticker, period=period)
    # Compute daily returns from closing prices and drop missing values
    returns = data['Close'].pct_change().dropna().values
    dates = data['Close'].pct_change().dropna().index  # aligned dates for returns
    T = len(returns)
    
    # Initialize lists to collect forecast results and the realized volatility
    forecast_vol_list = []
    realized_vol_list = []
    forecast_dates = []
    
    # Loop over the data using a sliding window
    for i in range(0, T - window_size - forecast_horizon + 1):
        # Extract the returns for the current window
        window_returns = returns[i : i + window_size]
        
        # Initialize the volatility array for the current window
        sigma2_window = np.zeros(window_size)
        # Use sample variance of the window if not otherwise provided
        sigma2_window[0] = np.var(window_returns)
        for t in range(1, window_size):
            sigma2_window[t] = omega + alpha * window_returns[t - 1]**2 + beta * sigma2_window[t - 1]
        
        # Forecast volatility for the next day (forecast_horizon = 1) using the GARCH recursion
        forecasts = np.zeros(forecast_horizon)
        forecasts[0] = omega + alpha * window_returns[-1]**2 + beta * sigma2_window[-1]
        # For horizons > 1, the recursion is:
        for h in range(1, forecast_horizon):
            forecasts[h] = omega + (alpha + beta) * forecasts[h - 1]
        
        # Compute realized volatility for the forecast period.
        # Here, we average the squared returns over the forecast horizon and take a square root.
        realized_returns = returns[i + window_size : i + window_size + forecast_horizon]
        realized_variance = np.mean(realized_returns**2)
        realized_vol = np.sqrt(realized_variance)
        
        # We take the day-ahead forecast (first forecast value) as our prediction.
        forecast_vol = np.sqrt(forecasts[0])
        
        forecast_vol_list.append(forecast_vol)
        realized_vol_list.append(realized_vol)
        # Save the date corresponding to the forecast (the first day after the current window)
        forecast_dates.append(dates[i + window_size])
    
    # Convert the lists to NumPy arrays for performance metric calculations
    forecast_vol_array = np.array(forecast_vol_list)
    realized_vol_array = np.array(realized_vol_list)
    
    # Compute performance metrics: RMSE, covariance, and Pearson correlation between forecast and realization
    mse = np.mean((forecast_vol_array - realized_vol_array) ** 2)
    rmse = np.sqrt(mse)
    covariance = np.cov(forecast_vol_array, realized_vol_array)[0, 1]
    correlation, _ = pearsonr(forecast_vol_array, realized_vol_array)
    
    print(f"Sliding Window Backtesting for {ticker}")
    print(f"Number of forecasts: {len(forecast_vol_array)}")
    print(f"Forecast Horizon (days): {forecast_horizon}")
    print(f"Window Size (days): {window_size}")
    print(f"RMSE: {rmse:.6f}")
    print(f"Covariance: {covariance:.6f}")
    print(f"Correlation: {correlation:.6f}")
    
    # Plot forecasted volatility versus realized volatility
    plt.figure(figsize=(12,6))
    plt.plot(forecast_dates, forecast_vol_array, label='Forecasted Volatility', marker='o')
    plt.plot(forecast_dates, realized_vol_array, label='Realized Volatility', marker='x')
    plt.xlabel('Date')
    plt.ylabel('Volatility')
    plt.title(f'GARCH(1,1) Forecast vs. Realized Volatility for {ticker}')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return forecast_vol_array, realized_vol_array, forecast_dates


# Example usage:
ticker = 'AAPL'
omega = 0.000001
alpha = 0.05
beta = 0.94
forecast_horizon = 5
window_size = 252

#forecast_vols = garch_forecast(ticker, omega, alpha, beta, period='1y')

#forecast_vols = garch_forecast_arch(ticker, period='1y')

#forecast_vol, realized_vol, forecast_dates = sliding_window_backtest(
#    ticker, omega, alpha, beta, window_size, forecast_horizon, period='2y'
#)

forecasted_volatility = garch_volatility_forecast(
    ticker, omega, alpha, beta, forecast_horizon
)
