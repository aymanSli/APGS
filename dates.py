# dates.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class DateHandler:
    """
    Simplified DateHandler class that focuses on key observation dates and current date.
    
    Main functions:
    - Extract actual trading dates from the Excel file
    - Manage key observation dates (T0, T1, T2, T3, T4, Tc)
    - Track current date for simulation and rebalancing
    - Provide date lookups and time calculations
    """
    
    def __init__(self, data_file_path, date_column_name='Date', sheet_name='ClosePrice'):
        """
        Initialize the DateHandler class.
        
        Parameters:
        -----------
        data_file_path : str
            Path to the Excel file containing the dates
        date_column_name : str, optional
            Name of the column containing dates (default: 'Date')
        sheet_name : str, optional
            Name of the sheet containing dates (default: 'ClosePrice')
        """
        self.data_file_path = data_file_path
        
        # Load all trading dates from Excel file
        self.market_dates = self._load_dates(date_column_name, sheet_name)
        print(f"Loaded {len(self.market_dates)} trading dates from {data_file_path}")
        
        # Store key observation dates
        self.key_dates = {}
        
        # Current date - will be set later
        self.current_date = None
        
        # For quick lookups
        self.trading_days_per_year = 262  # Used for time fraction calculations
    
    def _load_dates(self, date_column_name, sheet_name):
        """
        Load dates from Excel file and convert to datetime.date objects.
        
        Parameters:
        -----------
        date_column_name : str
            Name of the column containing dates
        sheet_name : str
            Name of the sheet containing dates
                
        Returns:
        --------
        list
            List of trading dates in chronological order as datetime.date objects
        """
        df = pd.read_excel(
            self.data_file_path, 
            sheet_name=sheet_name,
            parse_dates=[date_column_name]
        )
        
        # Convert pandas Timestamps to datetime.date objects
        date_list = [pd_date.to_pydatetime() for pd_date in df[date_column_name]]
        
        # Sort the dates
        return sorted(date_list)
    
    def set_key_dates(self, date_dict):
        """
        Set the key observation dates (T0, T1, T2, T3, T4, Tc).
        Finds the closest actual trading dates in the Excel file.
        
        Parameters:
        -----------
        date_dict : dict
            Dictionary mapping key date names to calendar dates
            e.g., {'T0': datetime(2009, 1, 5), 'T1': datetime(2010, 1, 4), ...}
            
        Returns:
        --------
        dict
            Dictionary mapping key date names to actual trading dates
        """
        self.key_dates = {}
        
        for key, target_date in date_dict.items():
            # Find the closest trading date
            closest_date = self._find_closest_date(target_date)
            self.key_dates[key] = closest_date
            
            # Print information for verification
            if closest_date != target_date:
                print(f"Note: {key} date adjusted from {target_date.strftime('%Y-%m-%d')} to closest trading date {closest_date.strftime('%Y-%m-%d')}")
            else:
                print(f"Set {key} to trading date {closest_date.strftime('%Y-%m-%d')}")
        
        # Verify that dates are in ascending order
        self._verify_date_order()
        
        # Print time periods
        self._print_time_periods()
        
        # Update date_to_index mapping for quick lookups
        self._update_date_indices()
        
        return self.key_dates
    
    def _update_date_indices(self):
        """
        Update the mapping of dates to their indices in the market_dates list.
        This enables quick lookups of date positions.
        """
        self.date_to_index = {date: i for i, date in enumerate(self.market_dates)}
    
    def _find_closest_date(self, target_date):
        """
        Find the closest trading date to the target date.
        
        Parameters:
        -----------
        target_date : datetime
            Target date to find
            
        Returns:
        --------
        datetime
            Closest trading date
        """
        if target_date in self.market_dates:
            return target_date
        
        # Find the closest date by minimizing the absolute difference
        closest_date = min(self.market_dates, key=lambda date: abs((date - target_date).total_seconds()))
        return closest_date
    
    def _verify_date_order(self):
        """
        Verify that key dates are in the correct order: T0 < T1 < T2 < T3 < T4 < Tc.
        
        Raises:
        -------
        ValueError
            If dates are not in ascending order
        """
        expected_order = ['T0', 'T1', 'T2', 'T3', 'T4', 'Tc']
        
        # Check that all expected keys are present
        if not all(key in self.key_dates for key in expected_order):
            missing = [key for key in expected_order if key not in self.key_dates]
            raise ValueError(f"Missing key dates: {missing}")
        
        # Check for ascending order
        dates = [self.key_dates[key] for key in expected_order]
        for i in range(1, len(dates)):
            if dates[i] <= dates[i-1]:
                raise ValueError(f"{expected_order[i]} ({dates[i].strftime('%Y-%m-%d')}) is not after {expected_order[i-1]} ({dates[i-1].strftime('%Y-%m-%d')})")
    
    def _print_time_periods(self):
        """
        Print information about time periods between key dates.
        """
        expected_order = ['T0', 'T1', 'T2', 'T3', 'T4', 'Tc']
        
        print("\nTime periods between key dates:")
        print(f"{'Period':<10} | {'Trading Days':<15}")
        print("-" * 60)
        
        for i in range(len(expected_order) - 1):
            start_key = expected_order[i]
            end_key = expected_order[i + 1]
            
            if start_key in self.key_dates and end_key in self.key_dates:
                start_date = self.key_dates[start_key]
                end_date = self.key_dates[end_key]
                
                # Calculate days and years
                trading_days = self._count_trading_days(start_date, end_date)
                
                print(f"{start_key}-{end_key:<6} | {trading_days:<15}")
    
    def _count_trading_days(self, start_date, end_date):
        """
        Count the number of trading days between two dates.
        
        Parameters:
        -----------
        start_date : datetime
            Start date
        end_date : datetime
            End date
            
        Returns:
        --------
        int
            Number of trading days
        """
        return sum(1 for date in self.market_dates if start_date <= date <= end_date)
    
    def set_current_date(self, date):
        """
        Set the current date for simulation and rebalancing.
        Finds the closest actual trading date.
        
        Parameters:
        -----------
        date : datetime
            Current date to set
            
        Returns:
        --------
        datetime
            Actual trading date set as current
        """
        self.current_date = self._find_closest_date(date)
        return self.current_date
    
    def is_key_date(self, date=None):
        """
        Check if a date is a key observation date.
        
        Parameters:
        -----------
        date : datetime, optional
            Date to check (default: current_date)
            
        Returns:
        --------
        bool, str
            (is_key_date, key_name) or (False, None) if not a key date
        """
        if date is None:
            date = self.current_date
            
        for key, key_date in self.key_dates.items():
            if key_date == date:
                return True, key
                
        return False, None
    
    def get_next_key_date(self, date=None):
        """
        Find the next key date after the given date.
        
        Parameters:
        -----------
        date : datetime, optional
            Reference date (default: current_date)
            
        Returns:
        --------
        tuple
            (next_key_name, next_key_date) or (None, None) if no next key date
        """
        if date is None:
            date = self.current_date
            
        # Sort key dates
        sorted_keys = ['T0', 'T1', 'T2', 'T3', 'T4', 'Tc']
        sorted_dates = [(key, self.key_dates[key]) for key in sorted_keys if key in self.key_dates]
        
        # Find the next key date
        for key, key_date in sorted_dates:
            if key_date > date:
                return key_date
                
        return  None
    
    def get_previous_key_date(self, date=None):
        """
        Find the previous key date before the given date.
        
        Parameters:
        -----------
        date : datetime, optional
            Reference date (default: current_date)
            
        Returns:
        --------
        tuple
            (prev_key_name, prev_key_date) or (None, None) if no previous key date
        """
        if date is None:
            date = self.current_date
            
        # Sort key dates
        sorted_keys = ['T1', 'T2', 'T3', 'T4', 'Tc']
        sorted_dates = [(key, self.key_dates[key]) for key in sorted_keys if key in self.key_dates]
        
        # Find the previous key date
        prev_date =self.key_dates['T0']
        for key, key_date in sorted_dates:
            print("hey")
            if key_date >= date:
                break
            prev_date = key_date
        print("done")        
        return prev_date
    
    def get_key_date(self, key_name):
        """
        Get the trading date for a specific key date.
        
        Parameters:
        -----------
        key_name : str
            Key date name ('T0', 'T1', 'T2', 'T3', 'T4', 'Tc')
            
        Returns:
        --------
        datetime
            Trading date corresponding to the key date
        """
        if key_name not in self.key_dates:
            raise ValueError(f"Key date {key_name} not set")
        
        return self.key_dates[key_name]
    
    def get_key_date_index(self, key_name):
        """
        Get the index of a key date in the market_dates list.
        
        Parameters:
        -----------
        key_name : str
            Key date name ('T0', 'T1', 'T2', 'T3', 'T4', 'Tc')
            
        Returns:
        --------
        int
            Index of the key date in the market_dates list
        """
        date = self.get_key_date(key_name)
        return self.date_to_index.get(date)
    
    def get_current_date_index(self):
        """
        Get the index of the current date in the market_dates list.
        
        Returns:
        --------
        int
            Index of the current date in the market_dates list
        """
        if self.current_date is None:
            raise ValueError("Current date is not set")
            
        return self.date_to_index.get(self.current_date)
    
    def get_time_fraction(self, start_date, end_date):
        """
        Calculate the time fraction between two dates in years.
        Uses actual trading days for more accurate results.
        
        Parameters:
        -----------
        start_date : datetime
            Start date
        end_date : datetime
            End date
            
        Returns:
        --------
        float
            Time fraction in years
        """
        # Count actual trading days between dates
        trading_days = self._count_trading_days(start_date, end_date)
        
        # Convert to years using trading days per year
        return trading_days / self.trading_days_per_year
    
    def get_key_dates_between(self, start_date, end_date):
        """
        Get all key dates between start_date and end_date (inclusive).
        
        Parameters:
        -----------
        start_date : datetime
            Start date
        end_date : datetime
            End date
            
        Returns:
        --------
        dict
            Dictionary of key dates between start_date and end_date
        """
        return {k: d for k, d in self.key_dates.items() 
                if start_date <= d <= end_date}
    
    def is_after_start(self, date=None):
        """
        Check if a date is after T0.
        
        Parameters:
        -----------
        date : datetime, optional
            Date to check (default: current_date)
            
        Returns:
        --------
        bool
            True if date is after T0
        """
        if date is None:
            date = self.current_date
            
        return date >= self.key_dates.get('T0')
    
    def is_before_end(self, date=None):
        """
        Check if a date is before Tc.
        
        Parameters:
        -----------
        date : datetime, optional
            Date to check (default: current_date)
            
        Returns:
        --------
        bool
            True if date is before Tc
        """
        if date is None:
            date = self.current_date
            
        return date <= self.key_dates.get('Tc')
    
    def is_in_product_period(self, date=None):
        """
        Check if a date is within the product period (between T0 and Tc).
        
        Parameters:
        -----------
        date : datetime, optional
            Date to check (default: current_date)
            
        Returns:
        --------
        bool
            True if date is within product period
        """
        return self.is_after_start(date) and self.is_before_end(date)
    
    def get_date_from_index(self, index):
        """
        Get a date from its index in the market_dates list.
        
        Parameters:
        -----------
        index : int
            Index in the market_dates list
            
        Returns:
        --------
        datetime
            Date at the specified index
        """
        if index < 0 or index >= len(self.market_dates):
            raise ValueError(f"Index {index} out of range for market_dates")
            
        return self.market_dates[index]
    
    def get_all_key_dates(self):
        """
        Get all key dates in chronological order.
        
        Returns:
        --------
        list
            List of (key_name, date) tuples in chronological order
        """
        # Sort by date
        return sorted(self.key_dates.items(), key=lambda x: x[1])
    
    
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
        return self.market_dates.index(date)
        