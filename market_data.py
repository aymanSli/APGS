# market_data.py
import pandas as pd
import numpy as np
from pathlib import Path

class MarketData:
    """
    Class for loading and organizing market data for the 5 indices.
    
    Data is stored in three matrices:
    1. asset_matrix: Historical asset prices (dates as rows, assets as columns)
    2. currency_matrix: Exchange rates (dates as rows, currencies as columns)
    3. rates_matrix: Interest rates (dates as rows, currencies as columns)
    
    The assets are ordered as: domestic assets first, then foreign assets.
    Exchange rates and interest rates follow the same order as the assets.
    """
    
    def __init__(self, data_file_path, date_handler):
        """
        Initialize the MarketData class.
        
        Parameters:
        -----------
        data_file_path : str
            Path to the Excel file containing the market data
        date_handler : DateHandler, optional
            DateHandler object for key date lookups
        """
        self.data_file_path = Path(data_file_path)
        self.date_handler = date_handler
        
        # Default sheet names
        self.sheet_names = {
            'prices': 'ClosePrice',
            'rates': 'TauxInteret',
            'fx': 'XFORPrice'
        }
        
        # Define indices
        self.indices = ['DAX', 'ASX200', 'FTSE100', 'NASDAQ100', 'SMI']
        
        # Define which indices are domestic (EUR-based) and which are foreign
        self.domestic_indices = ['DAX']  # Assuming DAX is the only EUR-based index
        self.foreign_indices = [idx for idx in self.indices if idx not in self.domestic_indices]
        
        # Define currencies for each index
        self.index_currencies = {
            'ASX200': 'AUD',
            'DAX': 'EUR',
            'FTSE100': 'GBP',
            'NASDAQ100': 'USD',
            'SMI': 'CHF'
        }
        
        # Currency codes for FX rates
        self.currency_codes = {
            'AUD': 'XAUD',
            'GBP': 'XGBP',
            'USD': 'XUSD',
            'CHF': 'XCHF'
        }
        
        # Interest rate codes
        self.rate_codes = {
            'AUD': 'RAUD',
            'EUR': 'REUR',
            'GBP': 'RGBP',
            'USD': 'RUSD',
            'CHF': 'RCHF'
        }
        
        # Data containers
        self.asset_matrix = None
        self.currency_matrix = None
        self.rates_matrix = None
        self.dates = None
        
        # Load the data
        self._load_data()
    
    def _load_data(self):
        """
        Load data from the Excel file and organize it into matrices.
        Handles missing values (#N/A) by replacing them with the previous non-#N/A value.
        """
        # Load prices with handling for #N/A values
        prices_df = pd.read_excel(
            self.data_file_path, 
            sheet_name=self.sheet_names['prices'],
            index_col=0,
            na_values=['#N/A', '#N/A N/A', '#NA', '-NaN', 'NaN', 'nan', '']
        )
        
        # Load FX rates with handling for #N/A values
        fx_df = pd.read_excel(
            self.data_file_path, 
            sheet_name=self.sheet_names['fx'],
            index_col=0,
            na_values=['#N/A', '#N/A N/A', '#NA', '-NaN', 'NaN', 'nan', '']
        )
        
        # Load interest rates with handling for #N/A values
        rates_df = pd.read_excel(
            self.data_file_path, 
            sheet_name=self.sheet_names['rates'],
            index_col=0,
            na_values=['#N/A', '#N/A N/A', '#NA', '-NaN', 'NaN', 'nan', '']
        )
        
        # Make sure all indices are parsed as datetime
        prices_df.index = pd.to_datetime(prices_df.index)
        fx_df.index = pd.to_datetime(fx_df.index)
        rates_df.index = pd.to_datetime(rates_df.index)
        
        # Replace #N/A values with the previous non-#N/A value using forward fill
        prices_df = prices_df.fillna(method='ffill')
        fx_df = fx_df.fillna(method='ffill')
        rates_df = rates_df.fillna(method='ffill')
        
        # For any remaining NaN values at the beginning of series, backward fill
        prices_df = prices_df.fillna(method='bfill')
        fx_df = fx_df.fillna(method='bfill')
        rates_df = rates_df.fillna(method='bfill')
        
        # Store all available dates
        self.dates = prices_df.index.tolist()
        
        # Create date_to_index mapping for quick lookup
        self.date_to_index = {date: i for i, date in enumerate(self.dates)}
        
        # Create asset matrix in the order: domestic, then foreign
        asset_columns = self.domestic_indices + self.foreign_indices
        self.asset_matrix = prices_df[asset_columns].values
        
        # Create currency matrix (only for foreign indices)
        currency_columns = [self.currency_codes[self.index_currencies[idx]] for idx in self.foreign_indices]
        self.currency_matrix = fx_df[currency_columns].values
        
        # Create rates matrix (for all indices, in the same order)
        rate_columns = [self.rate_codes[self.index_currencies[idx]] for idx in self.indices]
        self.rates_matrix = rates_df[rate_columns].values

    
    def get_asset_price(self, index_name, date_index):
        """
        Get the price of an asset at a specific date index.
        
        Parameters:
        -----------
        index_name : str
            Name of the index
        date_index : int
            Index of the date
            
        Returns:
        --------
        float
            Price of the asset
        """
        if index_name in self.indices:
            col_index = self.indices.index(index_name)
            return self.asset_matrix[date_index, col_index]
        else:
            raise ValueError(f"Index {index_name} not found")
    
    def get_exchange_rate(self, currency, date_index):
        """
        Get the exchange rate for a currency at a specific date index.
        
        Parameters:
        -----------
        currency : str
            Currency code (e.g., 'AUD', 'GBP')
        date_index : int
            Index of the date
            
        Returns:
        --------
        float
            Exchange rate
        """
        if currency == 'EUR':
            return 1.0  # EUR is the base currency
        
        currency_code = self.currency_codes.get(currency)
        if currency_code:
            # Find which column in the currency matrix corresponds to this currency
            col_index = None
            for i, idx in enumerate(self.foreign_indices):
                if self.index_currencies[idx] == currency:
                    col_index = i
                    break
                    
            if col_index is not None:
                return self.currency_matrix[date_index, col_index]
        
        raise ValueError(f"Currency {currency} not found")
    
    def get_interest_rate(self, currency, date_index):
        """
        Get the interest rate for a currency at a specific date index.
        
        Parameters:
        -----------
        currency : str
            Currency code (e.g., 'AUD', 'EUR')
        date_index : int
            Index of the date
            
        Returns:
        --------
        float
            Interest rate
        """
        if currency in self.index_currencies.values():
            col_index = list(self.index_currencies.values()).index(currency)
            return self.rates_matrix[date_index, col_index]
        else:
            raise ValueError(f"Currency {currency} not found")
    
    def get_date_index(self, date):
        """
        Get the index of a specific date.
        
        Parameters:
        -----------
        date : datetime.date
            The date to look up
            
        Returns:
        --------
        int
            Index of the date
        """
        # Convert to pandas Timestamp if it's not already
        date = pd.Timestamp(date)
        
        if date in self.date_to_index:
            return self.date_to_index[date]
        
        # If date not in our dataset, find the closest one
        closest_date = min(self.dates, key=lambda d: abs((d - date).total_seconds()))
        print(f"Warning: Date {date} not in dataset, using closest date {closest_date}")
        return self.date_to_index[closest_date]
    
    def is_data_complete(self):
        """
        Check if all data matrices are complete (no NaN values).
        
        Returns:
        --------
        bool
            True if all data is complete, False otherwise
        """
        return (
            not np.isnan(self.asset_matrix).any() and
            not np.isnan(self.currency_matrix).any() and
            not np.isnan(self.rates_matrix).any()
        )
    
    
    def get_date_range(self):
        """
        Get the range of dates in the dataset.
        
        Returns:
        --------
        tuple
            (start_date, end_date)
        """
        return (self.dates[0], self.dates[-1])
    
    def get_asset_price_on_date(self, index_name, date):
        """
        Get the price of an asset on a specific date.
        
        Parameters:
        -----------
        index_name : str
            Name of the index
        date : datetime.date
            The date to look up
            
        Returns:
        --------
        float
            Price of the asset
        """
        date_index = self.get_date_index(date)
        return self.get_asset_price(index_name, date_index)
    
    def get_exchange_rate_on_date(self, currency, date):
        """
        Get the exchange rate for a currency on a specific date.
        
        Parameters:
        -----------
        currency : str
            Currency code (e.g., 'AUD', 'GBP')
        date : datetime.date
            The date to look up
            
        Returns:
        --------
        float
            Exchange rate
        """
        date_index = self.get_date_index(date)
        return self.get_exchange_rate(currency, date_index)
    
    def get_interest_rate_on_date(self, currency, date):
        """
        Get the interest rate for a currency on a specific date.
        
        Parameters:
        -----------
        currency : str
            Currency code (e.g., 'AUD', 'EUR')
        date : datetime.date
            The date to look up
            
        Returns:
        --------
        float
            Interest rate
        """
        date_index = self.get_date_index(date)
        return self.get_interest_rate(currency, date_index)
    
    def get_asset_prices_between(self, index_name, start_date, end_date):
        """
        Get the prices of an asset between two dates.
        
        Parameters:
        -----------
        index_name : str
            Name of the index
        start_date : datetime.date
            Start date
        end_date : datetime.date
            End date
            
        Returns:
        --------
        dict
            Dictionary mapping dates to prices
        """
        start_index = self.get_date_index(start_date)
        end_index = self.get_date_index(end_date)
        
        if index_name in self.indices:
            col_index = self.indices.index(index_name)
            prices = self.asset_matrix[start_index:end_index+1, col_index]
            dates = self.dates[start_index:end_index+1]
            return dict(zip(dates, prices))
        else:
            raise ValueError(f"Index {index_name} not found")
    
    def get_prices_for_key_dates(self, index_name):
        """
        Get the prices of an asset on all key dates.
        Requires date_handler to be set.
        
        Parameters:
        -----------
        index_name : str
            Name of the index
            
        Returns:
        --------
        dict
            Dictionary mapping key date names to prices
        """
        if self.date_handler is None:
            raise ValueError("date_handler is not set")
            
        prices = {}
        for key, date in self.date_handler.key_dates.items():
            prices[key] = self.get_asset_price_on_date(index_name, date)
            
        return prices