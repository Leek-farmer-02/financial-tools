import yfinance as yf
import numpy as np
import pandas as pd

def fetch_prices(ticker, start_date, end_date):
    return yf.download(ticker, auto_adjust=False, start=start_date, end=end_date)['Adj Close']

def compute_log_returns(prices):
    return np.log(prices / prices.shift(1)).dropna()

def compute_annualized_volatility(returns, freq):
    periods_per_year = {'daily': 252, 'weekly': 52, 'monthly': 12, 'quarterly': 4, 'annually': 1}
    return returns.std().item() * np.sqrt(periods_per_year[freq])

def compute_ewma_volatility(returns, lambda_=0.94):
    ewma_var = returns.ewm(span=(2 / (1 - lambda_)) - 1).var()
    return np.sqrt(ewma_var).iloc[-1].item()

from arch import arch_model

def garch_forecast(prices, horizon=5):
    log_returns = compute_log_returns(prices)
    model = arch_model(log_returns, mean='Zero', vol='Garch', p=1, q=1)
    res = model.fit(disp='off')
    forecast = res.forecast(horizon=horizon)
    return np.sqrt(forecast.variance.iloc[-1].values)

def compute_all_volatilities(prices_df):
    granularities = ['daily', 'weekly', 'monthly', 'quarterly', 'annually']
    resample_codes = {'daily': 'D', 'weekly': 'W', 'monthly': 'M', 'quarterly': 'Q', 'annually': 'A'}
    periods_per_year = {'daily': 252, 'weekly': 52, 'monthly': 12, 'quarterly': 4, 'annually': 1}

    results = []

    for g in granularities:
        try:
            resampled = prices_df.resample(resample_codes[g]).ffill()
            returns = compute_log_returns(resampled)
            hist_vol = compute_annualized_volatility(returns, g)
            ewma_vol = compute_ewma_volatility(returns)

            # Fit GARCH(1,1) model and forecast volatility
            model = arch_model(returns, mean='Zero', vol='Garch', p=1, q=1)
            res = model.fit(disp='off')
            forecast = res.forecast(horizon=periods_per_year[g])
            garch_vol = np.sqrt(forecast.variance.iloc[-1].values[-1])
        except Exception as e:
            hist_vol, ewma_vol, garch_vol = float('nan'), float('nan'), float('nan')

        results.append(
            f"{g.capitalize()}: Hist Vol = {hist_vol:.4f}, EWMA = {ewma_vol:.4f}, GARCH = {garch_vol:.4f}"
        )
    return results

from scipy.stats import pearsonr

def sliding_window_garch(ticker, omega, alpha, beta, window_size=252, forecast_horizon=1, period='6y'):
    data = yf.download(ticker, period=period)['Close']
    returns = data.pct_change().dropna().values
    dates = data.pct_change().dropna().index

    forecast_list, realized_list, forecast_dates = [], [], []
    for i in range(0, len(returns) - window_size - forecast_horizon + 1):
        win_r = returns[i:i+window_size]
        sigma2 = np.zeros(window_size)
        sigma2[0] = np.var(win_r)

        for t in range(1, window_size):
            sigma2[t] = omega + alpha * win_r[t-1]**2 + beta * sigma2[t-1]

        f = np.zeros(forecast_horizon)
        f[0] = omega + alpha * win_r[-1]**2 + beta * sigma2[-1]
        for h in range(1, forecast_horizon):
            f[h] = omega + (alpha + beta) * f[h - 1]

        realized = returns[i+window_size:i+window_size+forecast_horizon]
        forecast_list.append(np.sqrt(f[0]))
        realized_list.append(np.sqrt(np.mean(realized**2)))
        forecast_dates.append(dates[i+window_size])

    rmse = np.sqrt(np.mean((np.array(forecast_list) - np.array(realized_list))**2))
    corr = pearsonr(forecast_list, realized_list)[0]
    return forecast_dates, forecast_list, realized_list, rmse, corr

from scipy.stats import norm

def black_scholes(S0, K, T, r, sigma, option_type='call'):
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2)*T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        return S0 * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
    else:
        return K * np.exp(-r*T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)
    
class MonteCarloOption:
    def __init__(self, S0, mu, sigma, T, N, M, r, K, option_type='call'):
        self.S0 = S0
        self.mu = mu
        self.sigma = sigma
        self.T = T
        self.N = N
        self.M = M
        self.r = r
        self.K = K
        self.option_type = option_type
        self.dt = T / N

    def simulate_paths(self):
        Z = np.random.standard_normal((self.M, self.N))
        S = np.zeros((self.M, self.N+1))
        S[:, 0] = self.S0
        for t in range(1, self.N + 1):
            S[:, t] = S[:, t-1] * np.exp((self.r - 0.5 * self.sigma ** 2) * self.dt +
                                         self.sigma * np.sqrt(self.dt) * Z[:, t-1])
        return S

    def price(self):
        S = self.simulate_paths()
        payoff = np.maximum(S[:, -1] - self.K, 0) if self.option_type == 'call' else np.maximum(self.K - S[:, -1], 0)
        return np.exp(-self.r * self.T) * np.mean(payoff)
