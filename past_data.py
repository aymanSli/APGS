# past_data.py
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

class PastData:
    """
    Class for initializing and managing the past matrix for simulations.
    
    The past matrix structure:
    - Rows are for dates in the rebalancing grid (index 0 corresponds to T0)
    - Columns are in this order:
        1. Domestic assets
        2. Foreign assets * respective exchange rates
        3. Exchange rates * exp(ri * dt)
      where t is the current date and T0 is the product's start date
      
    Works directly with filtered grid dates from market_data.py.
    """
    
    def __init__(self, market_data, date_handler):
        """
        Initialize the PastData class.
        
        Parameters:
        -----------
        market_data : MarketData
            MarketData object containing asset prices, exchange rates, and interest rates
            for the rebalancing grid dates
        date_handler : DateHandler
            DateHandler object containing grid dates and key dates
        """
        self.market_data = market_data
        self.date_handler = date_handler
        self.trading_days_per_year = 262
        
        # Get the list of domestic and foreign indices
        self.domestic_indices = market_data.domestic_indices
        self.foreign_indices = market_data.foreign_indices
        
        
        # Store key dates as datetime objects
        self.key_dates = {key: date for key, date in date_handler.key_dates.items()}
        self.grid_dates = [date for key, date in date_handler.key_dates.items()]
        
        # Total number of columns in the past matrix
        self.num_columns = len(market_data.indices) + len(market_data.foreign_indices)
        
        # Initialize the past matrix as an empty list (will hold rows of data)
        self.past_matrix = []
        
        # Column names for reference
        self.column_names = (
            self.domestic_indices + 
            [f"{idx}_FX_adjusted" for idx in self.foreign_indices] + 
            [f"{market_data.index_currencies[idx]}_FX_adjusted" for idx in self.foreign_indices]
        )
    
    def initialize_past_matrix(self, current_date):
        """
        Initialize the past matrix with data up to the current date.
        Uses actual dates from the rebalancing grid.
        
        Parameters:
        -----------
        current_date : datetime, optional
            Current date (default: None, which uses the first date in the grid)
            
        Returns:
        --------
        numpy.ndarray
            The past matrix with rows for each grid date from T0 up to current_date
        """
        # Get T0 date
        t0_date = self.key_dates['T0']
        
        # # If current_date is not provided, use T0
        # if current_date is None:
        #     current_date = t0_date
        
        # Clear the past matrix
        self.past_matrix = []
        
        # Get all grid dates from T0 up to current_date
        valid_dates = [date for date in self.grid_dates if t0_date <= date < current_date]
        valid_dates.append(current_date)

        
        # Sort dates to ensure chronological order
        valid_dates.sort()
        
        # Fill the past matrix with data for each date
        for date in valid_dates:
            self.add_row(date)
            
        
        return np.array(self.past_matrix)
    
    def add_row(self, date):
        """
        Add a new row to the past matrix for the specified date.
        
        Parameters:
        -----------
        date : datetime
            Date to add data for
            
        Returns:
        --------
        list
            The new row added to the past matrix
        """
        # Get T0 date for interest rate calculations
        ref_date ='T0'
    
        
        # Calculate the time fraction (in years) since T0
        # Using actual calendar days divided by 252 for simplicity
        time_in_years = self.date_handler._count_trading_days(self.date_handler.key_dates[ref_date],date) / self.trading_days_per_year
        
        
        # Get grid index for this date
        date_index = self.market_data.get_date_index(date)
        
        # Initialize the new row
        new_row = []
        
        # Add domestic asset prices
        for idx in self.domestic_indices:
            price = self.market_data.get_asset_price(idx, date_index)
            new_row.append(price)
        
        # Add foreign assets * exchange rates
        for idx in self.foreign_indices:
            price = self.market_data.get_asset_price(idx, date_index)
            fx_rate = self.market_data.get_exchange_rate(
                self.market_data.index_currencies[idx], 
                date_index
            )
            new_row.append(price * fx_rate)
        
        # Add exchange rates * exp(ri * (t - T0))
        for idx in self.foreign_indices:
            currency = self.market_data.index_currencies[idx]
            fx_rate = self.market_data.get_exchange_rate(currency, date_index)
            interest_rate = self.market_data.get_interest_rate(currency, date_index)
            
            # Calculate the adjustment factor: 
            adjustment = np.exp(interest_rate * time_in_years)
            
            new_row.append(fx_rate * adjustment)
        
        # Add the new row to the past matrix
        self.past_matrix.append(new_row)
        
        return new_row
    
    
    
    def get_past_matrix(self):
        """
        Get the current past matrix.
        
        Returns:
        --------
        numpy.ndarray
            The past matrix
        """
        return np.array(self.past_matrix)
    
    
    def get_last_row(self):
        """
        Get the last row of the past matrix.
        
        Returns:
        --------
        list
            The last row of the past matrix
        """
        if not self.past_matrix:
            raise ValueError("Past matrix is empty")
        
        return self.past_matrix[-1]
    
    def get_past_dataframe(self,current_date):
        """
        Get the past matrix as a pandas DataFrame with column names.
        
        Returns:
        --------
        pandas.DataFrame
            The past matrix as a DataFrame
        """
        # Get T0 date
        t0_date = self.key_dates['T0']
        
        # Get all grid dates used in the past matrix
        valid_dates = [date for date in self.grid_dates if t0_date <= date < current_date]
        valid_dates.append(current_date)
        valid_dates.sort()
        
        # Create the DataFrame
        return pd.DataFrame(self.past_matrix, index=valid_dates, columns=self.column_names)
    
    def get_spot_prices(self):
        """
        Get the current spot prices for all assets and exchange rates.
        
        Returns:
        --------
        dict
            Dictionary with spot prices
        """
        if not self.past_matrix:
            raise ValueError("Past matrix is empty")
        
        # last_row = self.past_matrix[-1]
        
        # spot_prices = {}
        
        # # Domestic assets
        # for i, idx in enumerate(self.domestic_indices):
        #     spot_prices[idx] = last_row[i]
        
        # # Foreign assets * exchange rates
        # start_idx = len(self.domestic_indices)
        # for i, idx in enumerate(self.foreign_indices):
        #     spot_prices[idx] = last_row[start_idx + i]
        
        # # Exchange rates
        # start_idx = len(self.domestic_indices) + len(self.foreign_indices)
        # for i, idx in enumerate(self.foreign_indices):
        #     currency = self.market_data.index_currencies[idx]
        #     spot_prices[f"FX_{currency}"] = last_row[start_idx + i]
        
        return self.past_matrix[-1]
    
    
        