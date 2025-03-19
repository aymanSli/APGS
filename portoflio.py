# portfolio.py
import numpy as np
from typing import Dict, List, Optional, Tuple
import pandas as pd
from datetime import datetime
import time


class Portfolio:
    """
    Class for portfolio management of structured products.
    
    Handles:
    - Portfolio construction from delta calculations
    - Tracking positions over time
    - Rebalancing at specified intervals
    - Dividend/payment processing at observation dates
    - Hedging P&L analysis
    """
    
    def __init__(self, market_data, date_handler, product, initial_capital=1000000.0):
        """
        Initialize the Portfolio class.
        
        Parameters:
        -----------
        market_data : MarketData
            MarketData object for pricing assets
        date_handler : DateHandler
            DateHandler for date operations
        product : Product11 or similar
            Product object that defines the structured product
        initial_capital : float, optional
            Initial capital to allocate (default: 1,000,000.0)
        """
        self.market_data = market_data
        self.date_handler = date_handler
        self.product = product
        self.initial_capital = initial_capital
        
        # Portfolio state
        self.cash = initial_capital  # Initial cash position
        self.positions = {}          # Asset positions (qty)
        self.values = {}             # Position values
        self.trade_history = []      # List of all trades
        self.pnl_history = []        # List of P&L
        self.payment_history = []    # List of dividend/payment events
        
        # Portfolio tracking
        self.initial_date = None
        self.current_date = None
        self.current_date_index = None
        self.last_rebalance_date = None
        
        # Risk metrics
        self.delta_history = {}      # Track deltas over time
        
        # Positions by asset type
        self.domestic_positions = {}
        self.foreign_positions = {}
        self.fx_positions = {}
    
    def initialize_portfolio(self, current_date_index, deltas=None):
        """
        Initialize the portfolio at a specific date.
        Can optionally use provided deltas, otherwise assumes flat (no positions).
        
        Parameters:
        -----------
        current_date_index : int
            Index of the current date
        deltas : dict, optional
            Dictionary of deltas for initial hedging (default: None)
            
        Returns:
        --------
        dict
            Initial portfolio state
        """
        self.current_date_index = current_date_index
        self.current_date = self.date_handler.get_date_from_index(current_date_index)
        self.initial_date = self.current_date
        self.last_rebalance_date = self.current_date
        
        # Reset portfolio
        self.cash = self.initial_capital
        self.positions = {}
        self.values = {}
        self.trade_history = []
        self.pnl_history = []
        self.payment_history = []
        
        # Asset categorization
        self.domestic_positions = {}
        self.foreign_positions = {}
        self.fx_positions = {}
        
        # If deltas are provided, set up initial hedge
        if deltas:
            self.rebalance_portfolio(deltas)
        
        return self.get_portfolio_state()
    
    def rebalance_portfolio(self, deltas, trade_costs=0.0001):
        """
        Rebalance the portfolio based on new deltas.
        
        Parameters:
        -----------
        deltas : dict
            Dictionary of deltas for hedging
        trade_costs : float, optional
            Transaction costs as a fraction of trade value (default: 0.0001 = 1bp)
            
        Returns:
        --------
        dict
            Updated portfolio state
        """
        # Record the state before rebalancing for P&L calculation
        pre_rebalance_value = self.get_total_value()
        trades = []
        
        # Get current asset prices
        prices = self._get_current_prices()
        
        # Process each delta
        for asset_key, delta in deltas.items():
            # Parse the asset key to get asset type and name
            parts = asset_key.split('_', 1)
            if len(parts) != 2:
                continue
                
            asset_type, asset_name = parts
            
            # Determine target position for this asset
            # Delta represents sensitivity to price, so target position is -delta
            # to offset the risk (negative for hedging)
            target_position = -delta
            
            # Determine asset price
            price = prices.get(asset_key, 0)
            if price <= 0:
                continue  # Skip if price is invalid
            
            # Calculate required quantity
            target_quantity = (target_position * self.initial_capital) / price
            
            # Get current quantity
            current_quantity = self.positions.get(asset_key, 0)
            
            # Calculate trade quantity
            trade_quantity = target_quantity - current_quantity
            
            # Skip small trades
            if abs(trade_quantity) < 0.001:
                continue
                
            # Calculate trade value
            trade_value = trade_quantity * price
            
            # Calculate transaction costs
            transaction_cost = abs(trade_value) * trade_costs
            
            # Update cash position
            self.cash -= (trade_value + transaction_cost)
            
            # Update position
            self.positions[asset_key] = target_quantity
            self.values[asset_key] = target_quantity * price
            
            # Categorize positions by asset type
            if asset_type == "DOM":
                self.domestic_positions[asset_name] = target_quantity
            elif asset_type == "SX":
                self.foreign_positions[asset_name] = target_quantity
            elif asset_type == "X":
                self.fx_positions[asset_name] = target_quantity
            
            # Record trade
            trade = {
                'date': self.current_date,
                'asset': asset_key,
                'quantity': trade_quantity,
                'price': price,
                'value': trade_value,
                'cost': transaction_cost,
                'type': 'rebalance'
            }
            trades.append(trade)
            self.trade_history.append(trade)
        
        # Update portfolio delta
        self.delta_history[self.current_date] = deltas
        
        # Record P&L from this rebalance
        post_rebalance_value = self.get_total_value()
        rebalance_pnl = post_rebalance_value - pre_rebalance_value
        
        pnl_record = {
            'date': self.current_date,
            'pre_value': pre_rebalance_value,
            'post_value': post_rebalance_value,
            'pnl': rebalance_pnl,
            'type': 'rebalance'
        }
        self.pnl_history.append(pnl_record)
        
        # Update last rebalance date
        self.last_rebalance_date = self.current_date
        
        return {
            'portfolio': self.get_portfolio_state(),
            'trades': trades,
            'pnl': rebalance_pnl
        }
    
    def process_payment_date(self, past_matrix, observation_key=None):
        """
        Process payment at an observation date.
        If it's a payment date, calculates the payment and deducts it from the portfolio.
        
        Parameters:
        -----------
        past_matrix : numpy.ndarray
            Past matrix up to the current date
        observation_key : str, optional
            Key for the observation date (e.g., 'T1', 'T2')
            If None, will check if current date matches any observation date
            
        Returns:
        --------
        dict
            Payment information or None if not a payment date
        """
        # Determine if this is a payment date
        is_payment_date = False
        t_key = None
        
        if observation_key:
            t_key = observation_key
            t_date = self.date_handler.get_key_date(t_key)
            is_payment_date = (t_date == self.current_date)
        else:
            # Check if current date matches any observation date
            for key in ['T1', 'T2', 'T3', 'T4']:
                try:
                    t_date = self.date_handler.get_key_date(key)
                    if t_date == self.current_date:
                        is_payment_date = True
                        t_key = key
                        break
                except (KeyError, ValueError):
                    continue
        
        if not is_payment_date or not t_key:
            return None
        
        # Get the previous observation date key
        prev_key = 'T0' if t_key == 'T1' else f'T{int(t_key[1])-1}'
        
        # Calculate dividend for this observation date
        t0_index = self.date_handler.get_key_date_index('T0')
        current_row = self.current_date_index - t0_index
        
        # Extract this path from past matrix
        path = past_matrix[:current_row+1, :]
        
        # Calculate dividend/payment
        dividend_info = self.product._calculate_dividend_payment(path, t_key, prev_key)
        
        if not dividend_info or dividend_info.get('amount', 0) <= 0:
            return {'date': self.current_date, 'key': t_key, 'amount': 0, 'processed': True}
        
        # Record payment information
        payment_amount = dividend_info.get('amount', 0)
        payment = {
            'date': self.current_date,
            'key': t_key,
            'amount': payment_amount,
            'best_index': dividend_info.get('best_index'),
            'best_return': dividend_info.get('best_return', 0),
            'processed': True
        }
        
        # Deduct payment from cash
        self.cash -= payment_amount
        
        # Record payment in history
        self.payment_history.append(payment)
        
        # Record P&L from this payment
        pre_payment_value = self.get_total_value() + payment_amount
        post_payment_value = self.get_total_value()
        payment_pnl = post_payment_value - pre_payment_value
        
        pnl_record = {
            'date': self.current_date,
            'pre_value': pre_payment_value,
            'post_value': post_payment_value,
            'pnl': payment_pnl,
            'type': 'payment',
            'payment_key': t_key
        }
        self.pnl_history.append(pnl_record)
        
        return payment
    
    def update_position_values(self):
        """
        Update the values of all positions based on current market prices.
        
        Returns:
        --------
        dict
            Updated position values
        """
        prices = self._get_current_prices()
        
        # Update position values
        for asset_key, quantity in self.positions.items():
            price = prices.get(asset_key, 0)
            self.values[asset_key] = quantity * price
        
        return self.values
    
    def get_total_value(self):
        """
        Calculate the total portfolio value (cash + positions).
        
        Returns:
        --------
        float
            Total portfolio value
        """
        # Make sure position values are up to date
        self.update_position_values()
        
        # Sum all position values
        position_value = sum(self.values.values())
        
        # Add cash
        total_value = self.cash + position_value
        
        return total_value
    
    def get_portfolio_state(self):
        """
        Get the current state of the portfolio.
        
        Returns:
        --------
        dict
            Portfolio state information
        """
        self.update_position_values()
        
        return {
            'date': self.current_date,
            'cash': self.cash,
            'positions': self.positions.copy(),
            'values': self.values.copy(),
            'total_value': self.get_total_value(),
            'domestic_positions': self.domestic_positions.copy(),
            'foreign_positions': self.foreign_positions.copy(),
            'fx_positions': self.fx_positions.copy()
        }
    
    def update_to_date(self, new_date_index):
        """
        Update the portfolio to a new date.
        
        Parameters:
        -----------
        new_date_index : int
            Index of the new date
            
        Returns:
        --------
        dict
            Updated portfolio state
        """
        if new_date_index <= self.current_date_index:
            return self.get_portfolio_state()
        
        self.current_date_index = new_date_index
        self.current_date = self.date_handler.get_date_from_index(new_date_index)
        
        # Update position values with current prices
        self.update_position_values()
        
        return self.get_portfolio_state()
    
    def _get_current_prices(self):
        """
        Get current prices for all assets.
        
        Returns:
        --------
        dict
            Dictionary of current prices for all assets
        """
        prices = {}
        
        # Get domestic asset prices
        for i, asset_name in enumerate(self.market_data.domestic_indices):
            price = self.market_data.get_asset_price(asset_name, self.current_date_index)
            prices[f"DOM_{asset_name}"] = price
        
        # Get foreign asset prices (S*X)
        for i, asset_name in enumerate(self.market_data.foreign_indices):
            price = self.market_data.get_asset_price(asset_name, self.current_date_index)
            fx_rate = self.market_data.get_exchange_rate(
                self.market_data.index_currencies[asset_name],
                self.current_date_index
            )
            prices[f"SX_{asset_name}"] = price * fx_rate
        
        # Get FX rates (Xexp(ri*t))
        for i, asset_name in enumerate(self.market_data.foreign_indices):
            currency = self.market_data.index_currencies[asset_name]
            fx_rate = self.market_data.get_exchange_rate(currency, self.current_date_index)
            
            # Calculate time from T0
            t0_index = self.date_handler.get_key_date_index("T0")
            time_fraction = (self.current_date_index - t0_index) / 252  # Trading days
            
            # Get interest rate
            interest_rate = self.market_data.get_interest_rate(currency, self.current_date_index)
            
            # Calculate adjustment
            adjustment = np.exp(interest_rate * time_fraction)
            
            # Store FX rate with interest rate adjustment
            prices[f"X_{currency}"] = fx_rate * adjustment
        
        return prices
    
    def get_performance_summary(self):
        """
        Get a summary of portfolio performance.
        
        Returns:
        --------
        dict
            Performance metrics
        """
        # Calculate portfolio returns
        initial_value = self.initial_capital
        current_value = self.get_total_value()
        total_return = (current_value / initial_value) - 1
        
        # Calculate annualized return
        days_held = (self.current_date - self.initial_date).days
        if days_held > 0:
            years_held = days_held / 365
            annualized_return = (1 + total_return) ** (1 / years_held) - 1
        else:
            annualized_return = 0
        
        # Payment metrics
        total_payments = sum(payment.get('amount', 0) for payment in self.payment_history)
        
        # Calculate P&L breakdown
        rebalance_pnl = sum(pnl.get('pnl', 0) for pnl in self.pnl_history if pnl.get('type') == 'rebalance')
        payment_pnl = sum(pnl.get('pnl', 0) for pnl in self.pnl_history if pnl.get('type') == 'payment')
        
        # Portfolio composition
        positions_value = sum(self.values.values())
        cash_percentage = (self.cash / current_value) * 100 if current_value > 0 else 0
        positions_percentage = (positions_value / current_value) * 100 if current_value > 0 else 0
        
        return {
            'initial_date': self.initial_date,
            'current_date': self.current_date,
            'days_held': days_held,
            'initial_value': initial_value,
            'current_value': current_value,
            'total_return': total_return * 100,  # As percentage
            'annualized_return': annualized_return * 100,  # As percentage
            'total_payments': total_payments,
            'rebalance_pnl': rebalance_pnl,
            'payment_pnl': payment_pnl,
            'cash': self.cash,
            'positions_value': positions_value,
            'cash_percentage': cash_percentage,
            'positions_percentage': positions_percentage,
            'num_positions': len(self.positions),
            'num_trades': len(self.trade_history)
        }
    
    def get_trade_history_df(self):
        """
        Get trade history as a pandas DataFrame.
        
        Returns:
        --------
        pandas.DataFrame
            Trade history
        """
        if not self.trade_history:
            return pd.DataFrame()
        
        return pd.DataFrame(self.trade_history)
    
    def get_payment_history_df(self):
        """
        Get payment history as a pandas DataFrame.
        
        Returns:
        --------
        pandas.DataFrame
            Payment history
        """
        if not self.payment_history:
            return pd.DataFrame()
        
        return pd.DataFrame(self.payment_history)
    
    def get_pnl_history_df(self):
        """
        Get P&L history as a pandas DataFrame.
        
        Returns:
        --------
        pandas.DataFrame
            P&L history
        """
        if not self.pnl_history:
            return pd.DataFrame()
        
        return pd.DataFrame(self.pnl_history)
    
    def get_tracking_error(self, product_value):
        """
        Calculate tracking error between portfolio and product value.
        
        Parameters:
        -----------
        product_value : float
            Current value of the structured product
            
        Returns:
        --------
        float
            Tracking error (portfolio value - product value)
        """
        portfolio_value = self.get_total_value()
        return portfolio_value - product_value