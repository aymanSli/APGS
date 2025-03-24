import numpy as np
from datetime import datetime
from typing import Dict, List, Optional

class DeltaHedgingPortfolio:
    """
    Portfolio class for delta hedging a structured product.
    Uses deltas directly instead of asset quantities for hedging.
    
    Key features:
    - Works with deltas instead of asset quantities
    - Grows cash at the risk-free rate between rebalance dates
    - Handles dividend payments
    - Calculates portfolio value as cash + sum(delta_i * spot_i)
    """
    
    def __init__(self, initial_price: float, risk_free_rate: float):
        """
        Initialize the portfolio with the structured product's price.
        
        Parameters:
        -----------
        initial_price : float
            Initial price of the structured product to be hedged
        risk_free_rate : float
            Domestic risk-free interest rate (annualized)
        """
        self.cash = initial_price  # Initial cash = price of product
        self.deltas = {}           # Empty deltas dictionary
        self.risk_free_rate = risk_free_rate
        self.trade_history = []    # List to track all trades
        self.payment_history = []  # List to track dividend payments
    
    def get_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        """
        Calculate the current total value of the portfolio.
        
        Parameters:
        -----------
        current_prices : Dict[str, float]
            Current prices for all assets
            
        Returns:
        --------
        float
            Total portfolio value (cash + positions)
        """
        # Calculate value of all positions
        position_value = 0.0
        for asset_key, delta in self.deltas.items():
            price = current_prices.get(asset_key, 0.0)
            position_value += delta * price
        
        # Total value = cash + position value
        return self.cash + position_value
    
    def rebalance(self, labeled_deltas: Dict[str, float], current_prices: Dict[str, float], 
                 current_date: datetime, last_rebalance_date: datetime, 
                 trade_costs: float = 0.0001) -> Dict:
        """
        Rebalance the portfolio based on new deltas.
        Updates cash position and deltas directly.
        
        Parameters:
        -----------
        labeled_deltas : Dict[str, float]
            New deltas for each asset from Monte Carlo
        current_prices : Dict[str, float]
            Current prices for all assets
        current_date : datetime
            Current date of rebalancing
        last_rebalance_date : datetime
            Date of the last rebalancing operation
        trade_costs : float, optional
            Transaction costs as a fraction of trade value, default is 0.0001 (1bp)
            
        Returns:
        --------
        Dict
            Dictionary with rebalancing information
        """
        # 1. Calculate time since last rebalance (in years)
        dt = (current_date - last_rebalance_date).days / 365.0
        
        # 2. Grow cash at risk-free rate
        self.cash *= np.exp(self.risk_free_rate * dt)
        
        # 3. Record initial portfolio value for P&L tracking
        initial_value = self.get_portfolio_value(current_prices)
        
        # 4. Track trades for reporting
        trades = []
        
        # 5. Rebalance each asset based on delta changes
        for asset_key, new_delta in labeled_deltas.items():
            # Get current price
            price = current_prices.get(asset_key, 0.0)
            if price <= 0:
                continue  # Skip assets with invalid prices
            
            # Get previous delta (0 if not present)
            previous_delta = self.deltas.get(asset_key, 0.0)
            
            # Calculate delta change
            delta_change = new_delta - previous_delta
            
            # Skip tiny changes (avoids unnecessary trades)
            if abs(delta_change) < 0.0001:
                continue
            
            # Calculate trade value and transaction costs
            trade_value = delta_change * price
            trans_cost = abs(trade_value) * trade_costs
            
            # Update cash (negative delta_change means we're buying, so cash decreases)
            self.cash -= (trade_value + trans_cost)
            
            # Update delta
            self.deltas[asset_key] = new_delta
            
            # Record trade
            trade = {
                'date': current_date,
                'asset': asset_key,
                'delta_change': delta_change,
                'price': price,
                'value': trade_value,
                'cost': trans_cost
            }
            trades.append(trade)
            self.trade_history.append(trade)
        
        # 7. Calculate final portfolio value
        final_value = self.get_portfolio_value(current_prices)
        
        # 8. Return rebalancing information
        return {
            'cash': self.cash,
            'deltas': self.deltas.copy(),
            'trades': trades,
            'initial_value': initial_value,
            'final_value': final_value,
            'pnl': final_value - initial_value
        }
    
    def process_dividend_payment(self, amount: float, current_date: datetime, current_prices: Dict[str, float]) -> Dict:
        """
        Process a dividend payment from the structured product.
        Reduces cash by the dividend amount.
        
        Parameters:
        -----------
        amount : float
            Dividend amount to pay
        current_date : datetime
            Date of the dividend payment
        current_prices : Dict[str, float]
            Current prices for all assets (for portfolio valuation)
            
        Returns:
        --------
        Dict
            Information about the dividend payment
        """
        # Record portfolio value before payment
        pre_payment_value = self.get_portfolio_value(current_prices)
        
        # Deduct dividend from cash
        self.cash -= amount
        
        # Calculate portfolio value after payment
        post_payment_value = self.get_portfolio_value(current_prices)
        
        # Record payment
        payment = {
            'date': current_date,
            'amount': amount,
            'portfolio_value_before': pre_payment_value,
            'portfolio_value_after': post_payment_value
        }
        self.payment_history.append(payment)
        
        return payment
    
    def get_portfolio_state(self, current_prices: Dict[str, float]) -> Dict:
        """
        Get the current state of the portfolio.
        
        Parameters:
        -----------
        current_prices : Dict[str, float]
            Current prices for all assets
            
        Returns:
        --------
        Dict
            Current portfolio state
        """
        # Calculate position values
        position_values = {}
        for asset_key, delta in self.deltas.items():
            price = current_prices.get(asset_key, 0.0)
            position_values[asset_key] = delta * price
        
        total_position_value = sum(position_values.values())
        total_value = self.cash + total_position_value
        
        return {
            'cash': self.cash,
            'deltas': self.deltas.copy(),
            'position_values': position_values,
            'total_position_value': total_position_value,
            'total_value': total_value
        }
    
    def unwind_portfolio(self, current_prices: Dict[str, float], 
                        current_date: datetime, last_rebalance_date: datetime, 
                        trade_costs: float = 0.0001) -> Dict:
        """
        Unwind all positions (convert to cash) at maturity.
        
        Parameters:
        -----------
        current_prices : Dict[str, float]
            Current prices for all assets
        current_date : datetime
            Current date of unwinding
        last_rebalance_date : datetime
            Date of the last rebalancing operation
        trade_costs : float, optional
            Transaction costs as a fraction of trade value
            
        Returns:
        --------
        Dict
            Unwinding information
        """
        # 1. Grow cash at risk-free rate since last rebalance
        dt = (current_date - last_rebalance_date).days / 365.0
        self.cash *= np.exp(self.risk_free_rate * dt)
        
        # 2. Record initial portfolio value
        initial_value = self.get_portfolio_value(current_prices)
        
        # 3. Track unwinding trades
        trades = []
        
        # 4. Unwind each position
        for asset_key, delta in list(self.deltas.items()):  # Use list to avoid modifying during iteration
            price = current_prices.get(asset_key, 0.0)
            if price <= 0:
                continue
            
            # Calculate trade value and costs
            trade_value = delta * price
            trans_cost = abs(trade_value) * trade_costs
            
            # Update cash (selling position, so cash increases)
            self.cash += (trade_value - trans_cost)
            
            # Record trade
            trade = {
                'date': current_date,
                'asset': asset_key,
                'delta_change': -delta,  # Negative because we're reducing delta to zero
                'price': price,
                'value': trade_value,
                'cost': trans_cost
            }
            trades.append(trade)
            self.trade_history.append(trade)
            
            # Remove delta
            del self.deltas[asset_key]
        
        # 6. Return unwinding information
        return {
            'cash': self.cash,
            'trades': trades,
            'initial_value': initial_value,
            'final_value': self.cash,  # After unwinding, all value is in cash
            'pnl': self.cash - initial_value
        }