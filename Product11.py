# product11.py
import numpy as np
from datetime import datetime

class Product11:
    """
    Class representing Product 11 structured product.
    
    Key features:
    - 5-year duration
    - Based on a basket of 5 indices: ASX200, DAX, FTSE100, NASDAQ100, SMI
    - Protection against downside risk (limited to -15%)
    - Upside potential capped at 50%
    - 40% participation rate on the final performance
    - 20% minimum guarantee that activates under certain conditions
    - Dividends based on best-performing index at each observation date
    - Best-performing index is excluded from future dividend calculations
    
    Adapted to work with grid dates from the market_data.py class.
    """
    
    def __init__(self, market_data, date_handler):
        """
        Initialize the Product11 class.
        
        Parameters:
        -----------
        market_data : MarketData
            MarketData object containing asset prices, exchange rates, and interest rates
        date_handler : DateHandler
            DateHandler object containing grid dates and key dates
        """
        self.market_data = market_data
        self.date_handler = date_handler
        
        # Product parameters
        self.initial_value = 1000.0  # Initial investment (€)
        self.participation_rate = 0.4  # 40% participation rate
        self.floor = -0.15  # -15% floor
        self.cap = 0.5  # 50% cap
        self.minimum_guarantee = 0.2  # 20% minimum guarantee
        self.dividend_multiplier = 50  # Factor for dividend calculation
        
        # Indices
        self.indices = market_data.indices
        
        # Track excluded indices for dividends
        self.excluded_indices = []
        
        # Track if minimum guarantee has been activated
        self.guarantee_activated = False
        
        # Store interest rates for all currencies
        self.interest_rates = []
        
        # Domestic currency (EUR)
        self.domestic_currency = 'EUR'
        
        # Store grid dates for quick lookups
        self.grid_dates = date_handler.market_dates
        
        # Map key dates to their position in the grid
        self.key_date_to_row = {
            'T0': 0,
            'T1': 1,
            'T2': 2,
            'T3': 3,
            'T4': 4,
            'Tc': 5
        }
        
    
    def _find_date_position_in_grid(self, date):
        """
        Find the position of a date in the grid dates.
        
        Parameters:
        -----------
        date : datetime
            Date to find
            
        Returns:
        --------
        int or None
            Position of the date in the grid, or None if not found
        """
        try:
            return self.grid_dates.index(date)
        except ValueError:
            # Find closest date
            closest_date = min(self.grid_dates, key=lambda d: abs((d - date).total_seconds()))
            print(f"Date {date} not found in grid, using closest date {closest_date}")
            return self.grid_dates.index(closest_date)
    
    def update_interest_rates(self, current_date):
        """
        Update the current interest rates based on the given date.
        
        Parameters:
        -----------
        current_date : datetime
            Current date for which to get interest rates
        """
        # Get the date index in the market data
        date_index = self.market_data.get_date_index(current_date)
        self.interest_rates = self.market_data.rates_matrix[date_index]
    
    def get_domestic_rate(self):
        """
        Get the domestic interest rate.
        
        Returns:
        --------
        float
            Domestic interest rate
        """
        return self.interest_rates[0]
    
    def get_asset_value(self, path, index_name, date_key):
        """
        Get the true asset value from the path, properly adjusting for exchange rates.
        
        Parameters:
        -----------
        path : numpy.ndarray
            Combined matrix of past data and simulated trajectories
        index_name : str
            Name of the index
        date_key : str
            Key name for the date (e.g., 'T0', 'T1')
            
        Returns:
        --------
        float
            Adjusted asset value S
        """
        # Get the row position for this key date
            
        row_idx = self.key_date_to_row[date_key]
        
        if row_idx >= len(path):
            raise ValueError(f"Path does not contain data for {date_key} (row {row_idx})")
        
        # For domestic assets, the value is directly in the path
        if index_name in self.market_data.domestic_indices:
            col_idx = self.indices.index(index_name)
            return path[row_idx, col_idx]
        
        # For foreign assets, we need to extract S from SX and X*exp(ri*(t-T0))
        # S = SX / (X*exp(ri*(t-T0))) * exp(ri*(t-T0))
        
        # Get the column index for the foreign asset (SX value)
        fx_asset_col = len(self.market_data.domestic_indices) + self.market_data.foreign_indices.index(index_name)
        
        # Get the column index for the exchange rate (X*exp(ri*(t-T0)))
        fx_rate_col = len(self.market_data.domestic_indices) + len(self.market_data.foreign_indices) + self.market_data.foreign_indices.index(index_name)
        
        # Extract values from the path
        sx_value = path[row_idx, fx_asset_col]
        x_adj_value = path[row_idx, fx_rate_col]
        r_i=self.market_data.get_interest_rate(
                self.market_data.index_currencies[index_name], 
                self.market_data.get_date_index(self.date_handler.key_dates[date_key]))
        term=self.date_handler._count_trading_days(self.date_handler.key_dates['T0'],self.date_handler.key_dates[date_key]) / 262
        
                   # This is X*exp(ri*(t-T0))
        
        # Calculate S = SX / (X*exp(ri*(t-T0))) * exp(ri*(t-T0))
        if x_adj_value > 0:  # Avoid division by zero
            # The foreign asset price is SX / X
            return (sx_value * np.exp(r_i*term)) / x_adj_value 
        else:
            return 0.0
    
    def calculate_dividend(self, t_i, path,excluded_indices=None):
        """
        Calculate the dividend for a specific observation date.
        
        Parameters:
        -----------
        t_i : str
            Key name for the observation date (e.g., 'T1', 'T2')
        path : numpy.ndarray
            Combined matrix of past data and simulated trajectories
        excluded_indices : List
            list of excluded ( default value is self.excluded_indices)
            
        Returns:
        --------
        tuple
            (dividend_amount, best_index, annual_return)
        """
        if excluded_indices is None:
            excluded_indices = self.excluded_indices
        
        # Get the previous observation date key (Ti-1 or T0 if this is T1)
        prev_key = f"T{int(t_i[1:]) - 1}" if int(t_i[1:]) > 1 else "T0"
        
        # Calculate annual returns for each non-excluded index
        annual_returns = {}
        for idx in self.indices:
            if idx in excluded_indices:
                continue
            
            # Get true asset values for current and previous dates
            current_value = self.get_asset_value(path, idx, t_i)
            prev_value = self.get_asset_value(path, idx, prev_key)
            
            # Calculate annual return
            if prev_value > 0:  # Avoid division by zero
                annual_return = (current_value / prev_value) - 1
                annual_returns[idx] = annual_return
        
        # If all indices are excluded, return 0 dividend
        if not annual_returns:
            return 0, None, 0
        
        # Find the index with the highest annual return
        best_index = max(annual_returns, key=annual_returns.get)
        best_return = annual_returns[best_index]
        
        # Calculate dividend (max(0, 50 * best return))
        dividend = max(0, self.dividend_multiplier * best_return)
        
        # Add the best index to excluded list for future dividends
        self.excluded_indices.append(best_index)
        
        return dividend, best_index, best_return
    
    def check_minimum_guarantee(self, t_i, path):
        """
        Check if minimum guarantee condition is activated at this observation date.
        
        Parameters:
        -----------
        t_i : str
            Key name for the observation date (e.g., 'T1', 'T2')
        path : numpy.ndarray
            Combined matrix of past data and simulated trajectories
            
        Returns:
        --------
        bool
            True if minimum guarantee is activated, False otherwise
        """
        # Get the previous observation date key (Ti-1 or T0 if this is T1)
        prev_key = f"T{int(t_i[1:]) - 1}" if int(t_i[1:]) > 1 else "T0"
        
        # Calculate basket performance (average of all index performances)
        performances = []
        for idx in self.indices:
            # Get true asset values for current and previous dates
            current_value = self.get_asset_value(path, idx, t_i)
            prev_value = self.get_asset_value(path, idx, prev_key)
            
            # Calculate performance
            if prev_value > 0:  # Avoid division by zero
                perf = (current_value / prev_value) - 1
                performances.append(perf)
        
        # Calculate basket performance (average)
        if performances:
            basket_perf = sum(performances) / len(performances)
            
            # Check if basket performance is high enough to activate minimum guarantee
            if basket_perf >= self.minimum_guarantee:
                self.guarantee_activated = True
                return True
        
        return False
    
    def calculate_final_performance(self, path):
        """
        Calculate the final performance of the product.
        
        Parameters:
        -----------
        path : numpy.ndarray
            Combined matrix of past data and simulated trajectories
            
        Returns:
        --------
        float
            Final performance (after applying floor, cap, and guarantee)
        """
        # Calculate final performances for all indices using true asset values
        performances = []
        for idx in self.indices:
            # Get true asset values for T0 and Tc
            t0_value = self.get_asset_value(path, idx, "T0")
            tc_value = self.get_asset_value(path, idx, "Tc")
            
            # Calculate performance from T0 to Tc
            if t0_value > 0:  # Avoid division by zero
                perf = (tc_value / t0_value) - 1
                performances.append(perf)
        
        # Calculate basket performance (average)
        if not performances:
            return 0
        
        basket_perf = sum(performances) / len(performances)
        
        # Apply floor if negative
        if basket_perf < 0:
            basket_perf = max(basket_perf, self.floor)
        # Apply cap if positive
        else:
            basket_perf = min(basket_perf, self.cap)
        
        # Apply minimum guarantee if activated
        if self.guarantee_activated:
            basket_perf = max(basket_perf, self.minimum_guarantee)
        
        return basket_perf
    
    def account_dividend(self, dividend, from_key, to_key="Tc"):
        """
        Discount a dividend from one date to another using the domestic interest rate.
        
        Parameters:
        -----------
        dividend : float
            Dividend amount
        from_key : str
            Key name for the origin date (e.g., 'T1', 'T2')
        to_key : str, optional
            Key name for the target date (default: 'Tc')
            
        Returns:
        --------
        float
            Discounted dividend
        """
        # Get the domestic interest rate
        r_d = self.interest_rates[0]
        
        # Get time in years between key dates using market_data grid
        from_date = self.date_handler.key_dates[from_key]
        to_date = self.date_handler.key_dates[to_key]
        time_fraction = (to_date - from_date).days / 252
        
        # Calculate the discount factor
        discount_factor = np.exp(r_d * time_fraction)
        
        # Apply the discount factor
        return dividend * discount_factor
    
    def calculate_final_payoff(self, path):
        """
        Calculate the final payoff at maturity.
        
        Parameters:
        -----------
        path : numpy.ndarray
            Combined matrix of past data and simulated trajectories
            
        Returns:
        --------
        float
            Final payoff amount
        """
        # Calculate final performance
        final_perf = self.calculate_final_performance(path)
        
        # Apply participation rate and initial value
        final_payoff = self.initial_value * (1 + self.participation_rate * final_perf)
        
        return final_payoff
    
    def calculate_all_dividends(self, path):
        """
        Calculate all dividends for the product.
        
        Parameters:
        -----------
        path : numpy.ndarray
            Combined matrix of past data and simulated trajectories
            
        Returns:
        --------
        dict
            Dictionary with dividend information for each observation date
        """
        # Reset excluded indices
        self.excluded_indices = []
        
        # Reset guarantee activation
        self.guarantee_activated = False
        
        dividends = {}
        
        # Check for each observation date (T1 to T4)
        for i in range(1, 5):
            t_i = f"T{i}"
            
            # Calculate dividend
            dividend, best_index, best_return = self.calculate_dividend(t_i, path)
            
            # Calculate compounded dividend (accounted to Tc)
            compounded_dividend = self.account_dividend(dividend, t_i, "Tc")
            
            # Check if minimum guarantee condition is met
            guarantee_activated = self.check_minimum_guarantee(t_i, path)
            
            # Store dividend information
            dividends[t_i] = {
                'amount': dividend,
                'compounded_amount': compounded_dividend,
                'best_index': best_index,
                'best_return': best_return,
                'guarantee_activated': guarantee_activated
            }
        
        return dividends
    
    def calculate_total_payoff(self, path):
        """
        Calculate the total payoff of the product (dividends + final payoff).
        
        Parameters:
        -----------
        path : numpy.ndarray
            Combined matrix of past data and simulated trajectories
            
        Returns:
        --------
        dict
            Dictionary with payoff information
        """
        # Calculate all dividends
        dividends_info = self.calculate_all_dividends(path)
        
        # Sum all compounded dividend amounts (accounted to Tc)
        total_dividends = sum(info['compounded_amount'] for info in dividends_info.values())
        
        # Calculate final payoff
        final_payoff = self.calculate_final_payoff(path)
        
        # Total payoff = compounded dividends + final payoff
        total_payoff = total_dividends + final_payoff
        
        return {
            'dividends': dividends_info,
            'total_dividends': total_dividends,
            'final_payoff': final_payoff,
            'total_payoff': total_payoff,
            'guarantee_activated': self.guarantee_activated,
            'final_performance': self.calculate_final_performance(path)
        }
    
    def simulate_product_lifecycle(self, path, current_date):
        """
        Simulate the complete lifecycle of the product.
        
        Parameters:
        -----------
        path : numpy.ndarray
            Combined matrix of past data and simulated trajectories
        current_date : datetime
            Current date
            
        Returns:
        --------
        dict
            Dictionary with complete lifecycle information
        """
        # Update interest rates based on the current date
        self.update_interest_rates(current_date)
        
        # Reset state
        self.excluded_indices = []
        self.guarantee_activated = False
        
        # Track the product state at each observation date
        lifecycle = {
            'T0': {
                'date': self.date_handler.key_dates['T0'],
                'excluded_indices': [],
                'guarantee_activated': False
            }
        }
        
        # Process each observation date
        for i in range(1, 5):
            if i < len(path):
                t_i = f"T{i}"
                
                    
                # Calculate dividend
                dividend, best_index, best_return = self.calculate_dividend(t_i, path)
                
                # Calculate compounded dividend (accounted to Tc)
                compounded_dividend = self.account_dividend(dividend, t_i, "Tc")
                
                # Check if minimum guarantee condition is met
                guarantee_check = self.check_minimum_guarantee(t_i, path)
                
                # Store state at this observation date
                lifecycle[t_i] = {
                    'date': self.date_handler.key_dates[t_i],
                    'dividend': dividend,
                    'compounded_dividend': compounded_dividend,
                    'best_index': best_index,
                    'best_return': best_return,
                    'excluded_indices': self.excluded_indices.copy(),
                    'guarantee_activated': self.guarantee_activated,
                    'guarantee_triggered_now': guarantee_check and not lifecycle[f"T{i-1}"]['guarantee_activated']
                }
        
        if len(path)==6:
            final_perf = self.calculate_final_performance(path)
            final_payoff = self.calculate_final_payoff(path)
            total_compounded_dividends = sum(data['compounded_dividend'] for key, data in lifecycle.items() if 'compounded_dividend' in data)
            
            lifecycle['Tc'] = {
                'date': self.date_handler.key_dates['Tc'],
                'final_performance': final_perf,
                'final_payoff': final_payoff,
                'total_compounded_dividends': total_compounded_dividends,
                'total_payoff': final_payoff + total_compounded_dividends,
                'excluded_indices': self.excluded_indices.copy(),
                'guarantee_activated': self.guarantee_activated
            }
        
        return lifecycle
    
    
    
    def print_product_lifecycle(self,lifecycle):
        """
        Print the product lifecycle in a clean, formatted way.
        
        Parameters:
        -----------
        lifecycle : dict
            Dictionary containing the product lifecycle information
        """
        print("\n==== PRODUCT 11 LIFECYCLE SUMMARY ====\n")
        
        # Print initial date info
        print(f"Initial Date (T0): {lifecycle['T0']['date'].strftime('%Y-%m-%d')}")
        print("-" * 40)
        
        # Print information for each observation date
        for i in range(1, 5):
            t_i = f"T{i}"
            
            if t_i not in lifecycle:
                continue
                
            data = lifecycle[t_i]
            
            print(f"\nObservation Date {t_i}: {data['date'].strftime('%Y-%m-%d')}")
            print(f"  Dividend: {data['dividend']:.2f}€")
            
            if 'best_index' in data and data['best_index']:
                print(f"  Best Index: {data['best_index']} (Return: {data['best_return']*100:.2f}%)")
            
            if 'guarantee_triggered_now' in data and data['guarantee_triggered_now']:
                print(f"  *** Minimum guarantee (20%) activated at this date! ***")
                
            if 'excluded_indices' in data:
                excluded = data['excluded_indices']
                if excluded:
                    print(f"  Excluded indices so far: {', '.join(excluded)}")
        
        # Print final information
        if 'Tc' in lifecycle:
            tc_data = lifecycle['Tc']
            
            print("\n" + "=" * 40)
            print(f"Maturity Date (Tc): {tc_data['date'].strftime('%Y-%m-%d')}")
            print(f"  Final Performance: {tc_data['final_performance']*100:.2f}%")
            print(f"  Final Payoff: {tc_data['final_payoff']:.2f}€")
            
            if 'total_compounded_dividends' in tc_data:
                print(f"  Total Dividends: {tc_data['total_compounded_dividends']:.2f}€")
                print(f"  Total Payoff (Dividends + Final): {tc_data['total_payoff']:.2f}€")
            
            if 'guarantee_activated' in tc_data and tc_data['guarantee_activated']:
                print("  Minimum guarantee was activated during the product's lifetime")
        
        print("\n==== END OF LIFECYCLE SUMMARY ====")