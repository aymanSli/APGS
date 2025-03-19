# test_dates.py
# some_file.py
import sys
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, 'C:/Users/Ayman SLIMAN/OneDrive:Bureau/testing/core')

import pandas as pd
from datetime import datetime
from dates import DateHandler

def print_date_handler_lists(excel_path):
    """
    Create a DateHandler and print its various date lists.
    
    Parameters:
    -----------
    excel_path : str
        Path to the Excel file containing market data
    """
    print(f"Loading DateHandler with data from: {excel_path}")
    
    # Create DateHandler instance
    date_handler = DateHandler(excel_path)
    
    # Set key dates according to the third period from the product specification
    key_dates = {
        'T0': datetime(2009, 1, 5),
        'T1': datetime(2010, 1, 4),
        'T2': datetime(2011, 1, 4),
        'T3': datetime(2012, 1, 4),
        'T4': datetime(2013, 1, 4),
        'Tc': datetime(2014, 1, 6)
    }
    
    # Set key dates and get actual trading dates
    print("\n1. Key Dates:")
    actual_key_dates = date_handler.set_key_dates(key_dates)
    
    # Print the actual key dates
    print("\nActual Key Dates:")
    for key, date in actual_key_dates.items():
        print(f"{key}: {date.strftime('%Y-%m-%d')}")
    
    print("\n")
    print("\n")
    print("\n")
    print(date_handler._find_closest_date(datetime(2009, 1, 12)))
    print("\n")
    print(date_handler.get_previous_key_date(datetime(2009, 1, 12)))
    print(datetime(2009, 1, 5))
    print("\n")
    print(date_handler.get_next_key_date(datetime(2009, 1, 12)))
    print(date_handler.market_dates[:5:])
    print(date_handler.get_all_key_dates())
    print(date_handler.market_dates[0])
    print(date_handler._count_trading_days(datetime(2009, 1, 12), date_handler.market_dates[2370]))
    
    return date_handler

if __name__ == "__main__":
    # Path to the Excel file
    excel_path = "DonneesGPS2025.xlsx"
    
    # Print the DateHandler lists
    date_handler = print_date_handler_lists(excel_path)