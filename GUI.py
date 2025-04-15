import sys
from PyQt5.QtWidgets import (QApplication, QWidget, QLabel, QPushButton, QLineEdit,
                             QVBoxLayout, QHBoxLayout, QComboBox, QGridLayout, QTabWidget)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from financial_models import (black_scholes, MonteCarloOption, fetch_prices,
                              compute_log_returns, compute_annualized_volatility,
                              compute_ewma_volatility, garch_forecast,
                              compute_all_volatilities, sliding_window_garch)
import numpy as np

class OptionPricingApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Option & Volatility Calculator")
        self.setGeometry(100, 100, 500, 600)
        self.init_ui()

    def init_ui(self):
        tabs = QTabWidget()
        tabs.addTab(self.init_option_tab(), "Option Pricing")
        tabs.addTab(self.init_volatility_tab(), "Volatility Analysis")

        main_layout = QVBoxLayout()
        main_layout.addWidget(tabs)
        self.setLayout(main_layout)

    def init_option_tab(self):
        option_widget = QWidget()
        layout = QVBoxLayout()

        grid = QGridLayout()
        self.S0_input = QLineEdit("100")
        self.K_input = QLineEdit("100")
        self.T_input = QLineEdit("1")
        self.r_input = QLineEdit("0.05")
        self.sigma_input = QLineEdit("0.2")
        self.option_type_combo = QComboBox()
        self.option_type_combo.addItems(["call", "put"])

        grid.addWidget(QLabel("Spot Price (S0):"), 0, 0)
        grid.addWidget(self.S0_input, 0, 1)
        grid.addWidget(QLabel("Strike Price (K):"), 1, 0)
        grid.addWidget(self.K_input, 1, 1)
        grid.addWidget(QLabel("Time to Maturity (T):"), 2, 0)
        grid.addWidget(self.T_input, 2, 1)
        grid.addWidget(QLabel("Risk-free Rate (r):"), 3, 0)
        grid.addWidget(self.r_input, 3, 1)
        grid.addWidget(QLabel("Volatility (sigma):"), 4, 0)
        grid.addWidget(self.sigma_input, 4, 1)
        grid.addWidget(QLabel("Option Type:"), 5, 0)
        grid.addWidget(self.option_type_combo, 5, 1)

        layout.addLayout(grid)
        bs_button = QPushButton("Price with Black-Scholes")
        mc_button = QPushButton("Price with Monte Carlo")
        bs_button.clicked.connect(self.calculate_black_scholes)
        mc_button.clicked.connect(self.calculate_monte_carlo)
        self.result_label = QLabel("Result: ")
        layout.addWidget(bs_button)
        layout.addWidget(mc_button)
        layout.addWidget(self.result_label)

        option_widget.setLayout(layout)
        return option_widget

    def init_volatility_tab(self):
        vol_widget = QWidget()
        layout = QVBoxLayout()

        vol_grid = QGridLayout()
        self.ticker_input = QLineEdit("AAPL")
        self.start_input = QLineEdit("2020-01-01")
        self.end_input = QLineEdit("2025-01-01")

        vol_grid.addWidget(QLabel("Ticker:"), 0, 0)
        vol_grid.addWidget(self.ticker_input, 0, 1)
        vol_grid.addWidget(QLabel("Start Date (YYYY-MM-DD):"), 1, 0)
        vol_grid.addWidget(self.start_input, 1, 1)
        vol_grid.addWidget(QLabel("End Date (YYYY-MM-DD):"), 2, 0)
        vol_grid.addWidget(self.end_input, 2, 1)

        layout.addLayout(vol_grid)

        vol_button = QPushButton("Compute Volatility")
        vol_button.clicked.connect(self.compute_volatility)
        self.vol_label = QLabel("Volatility Result: ")
        layout.addWidget(vol_button)
        layout.addWidget(self.vol_label)

        garch_button = QPushButton("Show GARCH Forecast vs Realized")
        garch_button.clicked.connect(self.plot_garch_forecast_vs_realized)
        layout.addWidget(garch_button)

        self.canvas = FigureCanvas(Figure(figsize=(5, 3)))
        layout.addWidget(self.canvas)
        self.ax = self.canvas.figure.subplots()

        vol_widget.setLayout(layout)
        return vol_widget

    def get_inputs(self):
        S0 = float(self.S0_input.text())
        K = float(self.K_input.text())
        T = float(self.T_input.text())
        r = float(self.r_input.text())
        sigma = float(self.sigma_input.text())
        option_type = self.option_type_combo.currentText()
        return S0, K, T, r, sigma, option_type

    def calculate_black_scholes(self):
        S0, K, T, r, sigma, option_type = self.get_inputs()
        price = black_scholes(S0, K, T, r, sigma, option_type)
        self.result_label.setText(f"Black-Scholes Price: {price:.4f}")

    def calculate_monte_carlo(self):
        S0, K, T, r, sigma, option_type = self.get_inputs()
        mc = MonteCarloOption(S0=S0, mu=r, sigma=sigma, T=T,
                              N=1000, M=10000, r=r, K=K, option_type=option_type)
        price = mc.price()
        self.result_label.setText(f"Monte Carlo Price: {price:.4f}")

    def compute_volatility(self):
        ticker = self.ticker_input.text()
        start = self.start_input.text()
        end = self.end_input.text()
        prices = fetch_prices(ticker, start, end)

        results = compute_all_volatilities(prices)
        self.vol_label.setText("\n".join(results))

    def plot_garch_forecast_vs_realized(self):
        ticker = self.ticker_input.text()
        forecast_dates, forecast_vol, realized_vol, rmse, corr = sliding_window_garch(
            ticker, omega=0.000001, alpha=0.05, beta=0.94, window_size=252, forecast_horizon=5, period='6y'
        )

        # Downsample every nth point
        step = 10  # show every 10th point
        forecast_dates = forecast_dates[::step]
        forecast_vol = forecast_vol[::step]
        realized_vol = realized_vol[::step]

        self.ax.clear()
        self.ax.plot(forecast_dates, forecast_vol, label='Forecasted Volatility', marker='o')
        self.ax.plot(forecast_dates, realized_vol, label='Realized Volatility', marker='x')
        self.ax.set_title("GARCH Forecast vs Realized Volatility")
        self.ax.set_xlabel("Date")
        self.ax.set_ylabel("Volatility")
        self.ax.legend()
        self.canvas.draw()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = OptionPricingApp()
    window.show()
    sys.exit(app.exec_())
