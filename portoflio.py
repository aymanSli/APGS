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
    
    def __init__(self, initial_price: float, date_handler,market_data):
        """
        Initialize the portfolio with the structured product's price.
        
        Parameters:
        -----------
        initial_price : float
            Initial price of the structured product to be hedged
        risk_free_rate : float
            Domestic risk-free interest rate (annualized)
        """
        self.market_data=market_data
        self.date_handler=date_handler
        self.cash = initial_price  # Initial cash = price of product
        self.deltas = [0.0 for _ in range (9)]           # Empty deltas dictionary
        self.trade_history = []    # List to track all trades
        self.payment_history = []  # List to track dividend payments
        # Get the list of domestic and foreign indices
        self.domestic_indices = market_data.domestic_indices
        self.foreign_indices = market_data.foreign_indices

        # Column names for reference
        self.column_names = (
            self.domestic_indices + 
            [f"{idx}_FX_adjusted" for idx in self.foreign_indices] +
            [f"{market_data.index_currencies[idx]}_FX_adjusted" for idx in self.foreign_indices]
        )

    
    def get_portfolio_value(self, current_prices: List) -> float:
        """
        Calculate the current total value of the portfolio.
        
        Parameters:
        -----------
        current_prices : List
            Current prices for all assets
            
        Returns:
        --------
        float
            Total portfolio value (cash + positions)
        """
        # Calculate value of all positions
        position_value = 0.0
        for i in range (9):
            position_value += self.deltas[i]* current_prices[i]
        
        # Total value = cash + position value
        return self.cash + position_value
    
    def rebalance(self, new_deltas: List , current_prices: List, 
                 current_date: datetime, last_rebalance_date: datetime,risk_free_rate: float) -> Dict:
        """
        Rebalance the portfolio based on new deltas.
        Updates cash position and deltas directly.
        
        Parameters:
        -----------
        labeled_deltas : Dict[str, float]
            New deltas for each asset from Monte Carlo
        current_prices : List
            Current prices for all assets
        current_date : datetime
            Current date of rebalancing
        last_rebalance_date : datetime
            Date of the last rebalancing operation
            
        Returns:
        --------
        Dict
            Dictionary with rebalancing information
        """
        # 1. Calculate time since last rebalance (in years)
        dt = self.date_handler._count_trading_days(last_rebalance_date,current_date)/ 252
        
        # 2. Grow cash at risk-free rate
        if current_date != self.date_handler.key_dates['T0'] :
            self.cash *= np.exp(risk_free_rate * dt)
        
        # 3. Record initial portfolio value for P&L tracking
        initial_value = self.get_portfolio_value(current_prices)
        
        # 4. Track trades for reporting
        trades = []
        
        # 5. Rebalance each asset based on delta changes
        for i in range (9):
            # Get current price
            price = current_prices[i]
            if price <= 0:
                continue  # Skip assets with invalid prices
            
            # Get previous delta (0 if not present)
            previous_delta = self.deltas[i]
            new_delta=new_deltas[i]
            
            # Calculate delta change
            delta_change = new_delta - previous_delta
            
            # Skip tiny changes (avoids unnecessary trades)
            if abs(delta_change) < 0.0001:
                continue
            
            # Calculate trade value and transaction costs
            trade_value = delta_change * price
            
            # Update cash (negative delta_change means we're buying, so cash decreases)
            self.cash -= trade_value 
            
            
            # Record trade
            trade = {
                'date': current_date,
                'asset':self.column_names[i],
                'delta_change': delta_change,
                'price': price,
                'value': trade_value,
            }
            trades.append(trade)
            self.trade_history.append(trade)
        
        self.deltas=new_deltas
        # 7. Calculate final portfolio value
        final_value = self.get_portfolio_value(current_prices)
        
        # 8. Return rebalancing information
        return {
            'cash': self.cash,
            'deltas': self.deltas,
            'trades': trades,
            'initial_value': initial_value,
            'final_value': final_value
        }
    
    def process_dividend_payment(self, amount: float, current_date: datetime, current_prices: List) -> Dict:
        """
        Process a dividend payment from the structured product.
        Reduces cash by the dividend amount.
        
        Parameters:
        -----------
        amount : float
            Dividend amount to pay
        current_date : datetime
            Date of the dividend payment
        current_prices : List
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
    
    def get_portfolio_state(self, current_prices: List) -> Dict:
        """
        Get the current state of the portfolio.
        
        Parameters:
        -----------
        current_prices : List
            Current prices for all assets
            
        Returns:
        --------
        Dict
            Current portfolio state
        """
        # Calculate value of all positions
        position_values =[]
        for i in range (9):
            position_values .append(self.deltas[i]* current_prices[i])
        
        total_position_value = sum(position_values)
        total_value = self.cash + total_position_value
        
        return {
            'cash': self.cash,
            'deltas': self.deltas.copy(),
            'position_values': position_values,
            'total_position_value': total_position_value,
            'total_value': total_value
        }
    
    def unwind_portfolio(self, current_prices: List, 
                        current_date: datetime, last_rebalance_date: datetime,risk_free_rate: float) -> Dict:
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
            
        Returns:
        --------
        Dict
            Unwinding information
        """
        # 1. Grow cash at risk-free rate since last rebalance
        dt = self.date_handler._count_trading_days(last_rebalance_date,current_date)/ 262
        self.cash *= np.exp(risk_free_rate * dt)
        
        # 2. Record initial portfolio value
        initial_value = self.get_portfolio_value(current_prices)
        
        # 3. Track unwinding trades
        trades = []
        
        # 4. Unwind each position
        for i in range (9):
            # Get current price
            price = current_prices[i]
            if price <= 0:
                continue  # Skip assets with invalid prices
            
            # Calculate trade value and costs
            trade_value = self.deltas[i] * price
            
            # Update cash (selling position, so cash increases)
            self.cash += trade_value 
            
            # Record trade
            trade = {
                'date': current_date,
                'asset': self.column_names[i],
                'delta_change': -self.deltas[i],  # Negative because we're reducing delta to zero
                'price': price,
                'value': trade_value,
            }
            trades.append(trade)
            self.trade_history.append(trade)
            
        
        # 6. Return unwinding information
        return {
            'cash': self.cash,
            'trades': trades,
            'initial_value': initial_value,
            'final_value': self.cash,  # After unwinding, all value is in cash
        }