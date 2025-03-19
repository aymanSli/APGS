# test_past_data.py
import pandas as pd
import numpy as np
from datetime import datetime
from market_data import MarketData
from past_data import PastData
from dates import DateHandler

def test_past_data(data_file_path):
    """
    Test the PastData class by creating and printing the past matrix along rebalancing grid.
    
    Parameters:
    -----------
    data_file_path : str
        Path to the Excel file with market data
    """
    print(f"Testing PastData with file: {data_file_path}")
    print("-" * 80)
    
    # Initialize date handler and set key dates
    date_handler = DateHandler(data_file_path)
    key_dates = {
        'T0': datetime(2009, 1, 5),
        'T1': datetime(2010, 1, 4),
        'T2': datetime(2011, 1, 4),
        'T3': datetime(2012, 1, 4),
        'T4': datetime(2013, 1, 4),
        'Tc': datetime(2014, 1, 6)
    }
    date_handler.set_key_dates(key_dates)
    
    
    # Print grid statistics
    print("\nGrid Statistics:")
    date_handler._print_time_periods()
    
    # Create MarketData with rebalancing grid dates
    market_data = MarketData(data_file_path, date_handler)
    
    # Create PastData
    past_data = PastData(market_data, date_handler)
    
    # Initialize past matrix up to current date (we'll use T2 as an example)
    t2_date = datetime(2010, 1, 13)
    print(t2_date)
    past_matrix = past_data.initialize_past_matrix(t2_date)
    
    print(f"\nPast Matrix initialized up to {t2_date.strftime('%Y-%m-%d')} (T2)")
    print(f"Shape: {past_matrix.shape} (dates × columns)")
    
    # Convert to DataFrame for better visualization
    past_df = past_data.get_past_dataframe(t2_date)
    print(past_df)
    
    
    

if __name__ == "__main__":
    # Change this path to your Excel file location
    file_path = "DonneesGPS2025.xlsx"
    test_past_data(file_path)