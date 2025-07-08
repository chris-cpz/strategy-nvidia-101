#!/usr/bin/env python3
"""
NVIDIA 101 - Momentum Trading Strategy

Strategy Type: momentum
Description: NVIDIA 101, buy low sell high, 1 trade per day etc etc 
Created: 2025-07-08T14:49:25.327Z

WARNING: This is a template implementation. Thoroughly backtest before live trading.
"""

import os
import sys
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('strategy.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class NVIDIA101Strategy:
    """
    NVIDIA 101 Implementation
    
    Strategy Type: momentum
    Risk Level: Monitor drawdowns and position sizes carefully
    """
    
    def __init__(self, config=None):
        self.config = config or self.get_default_config()
        self.positions = {}
        self.performance_metrics = {}
        logger.info(f"Initialized NVIDIA 101 strategy")
        
    def get_default_config(self):
        """Default configuration parameters"""
        return {
            'max_position_size': 0.05,  # 5% max position size
            'stop_loss_pct': 0.05,      # 5% stop loss
            'lookback_period': 20,       # 20-day lookback
            'rebalance_freq': 'daily',   # Rebalancing frequency
            'transaction_costs': 0.001,  # 0.1% transaction costs
        }
    
    def load_data(self, symbols, start_date, end_date):
        """Load market data for analysis"""
        try:
            import yfinance as yf
            data = yf.download(symbols, start=start_date, end=end_date)
            logger.info(f"Loaded data for {len(symbols)} symbols")
            return data
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return None

# =============================================================================
# USER'S STRATEGY IMPLEMENTATION
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

# Define the trading strategy class
class Nvidia101:
    def __init__(self, data, initial_capital=10000, position_size=0.1):
        self.data = data
        self.initial_capital = initial_capital
        self.position_size = position_size
        self.capital = initial_capital
        self.positions = []
        self.signals = []
        self.performance = pd.DataFrame(columns=['Date', 'Capital', 'Position', 'Signal'])

    def generate_signals(self):
        # Generate buy/sell signals based on momentum
        self.data['Returns'] = self.data['Close'].pct_change()
        self.data['Signal'] = np.where(self.data['Returns'] > 0, 1, -1)
        self.signals = self.data['Signal'].tolist()

    def execute_trades(self):
        for i in range(1, len(self.data)):
            if self.signals[i] == 1 and self.capital > 0:
                # Buy signal
                shares_to_buy = int(self.capital * self.position_size / self.data['Close'].iloc[i])
                self.capital -= shares_to_buy * self.data['Close'].iloc[i]
                self.positions.append((self.data.index[i], shares_to_buy))
                self.performance = self.performance.append({'Date': self.data.index[i], 'Capital': self.capital, 'Position': shares_to_buy, 'Signal': 'Buy'}, ignore_index=True)
            elif self.signals[i] == -1 and len(self.positions) > 0:
                # Sell signal
                shares_to_sell = self.positions[-1][1]
                self.capital += shares_to_sell * self.data['Close'].iloc[i]
                self.positions.pop()
                self.performance = self.performance.append({'Date': self.data.index[i], 'Capital': self.capital, 'Position': shares_to_sell, 'Signal': 'Sell'}, ignore_index=True)

    def calculate_performance_metrics(self):
        self.performance['Daily_Return'] = self.performance['Capital'].pct_change()
        sharpe_ratio = np.mean(self.performance['Daily_Return']) / np.std(self.performance['Daily_Return'])
        max_drawdown = (self.performance['Capital'].cummax() - self.performance['Capital']).max()
        return sharpe_ratio, max_drawdown

    def backtest(self):
        self.generate_signals()
        self.execute_trades()
        sharpe_ratio, max_drawdown = self.calculate_performance_metrics()
        logging.info("Sharpe Ratio: %f" % sharpe_ratio)
        logging.info("Max Drawdown: %" % max_drawdown)

# Sample data generation for demonstration
dates = pd.date_range(start='2022-01-01', periods=100)
prices = np.random.normal(loc=200, scale=10, size=(100,)).cumsum() + 100
sample_data = pd.DataFrame(data=" + str('Close': prices) + ", index=dates)

# Main execution block
if __name__ == "__main__":
    strategy = Nvidia101(sample_data)
    strategy.backtest()
    plt.plot(strategy.performance['Date'], strategy.performance['Capital'])
    plt.title('Capital Over Time')
    plt.xlabel('Date')
    plt.ylabel('Capital')
    plt.show()

# =============================================================================
# STRATEGY EXECUTION AND TESTING
# =============================================================================

if __name__ == "__main__":
    # Example usage and testing
    strategy = NVIDIA101Strategy()
    print(f"Strategy '{strategyName}' initialized successfully!")
    
    # Example data loading
    symbols = ['SPY', 'QQQ', 'IWM']
    start_date = '2020-01-01'
    end_date = '2023-12-31'
    
    print(f"Loading data for symbols: {symbols}")
    data = strategy.load_data(symbols, start_date, end_date)
    
    if data is not None:
        print(f"Data loaded successfully. Shape: {data.shape}")
        print("Strategy ready for backtesting!")
    else:
        print("Failed to load data. Check your internet connection.")
