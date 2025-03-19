# test_market_data.py
import pandas as pd
import numpy as np
from datetime import datetime
from market_data import MarketData

def test_market_data(file_path):
    """
    Test the MarketData class by loading data and printing matrices.
    
    Parameters:
    -----------
    file_path : str
        Path to the Excel file with market data
    """
    print(f"Testing MarketData with file: {file_path}")
    print("-" * 80)
    
    # Load market data with all dates
    data = MarketData(file_path)
    
    # Print basic information
    print(f"Loaded {len(data.dates)} trading dates from {data.data_file_path}")
    print(f"Date range: {data.dates[0].strftime('%Y-%m-%d')} to {data.dates[-1].strftime('%Y-%m-%d')}")
    print(f"Indices: {data.indices}")
    print(f"Domestic indices: {data.domestic_indices}")
    print(f"Foreign indices: {data.foreign_indices}")
    print("-" * 80)
    
    # Print sample of matrices
    print("\nSample of asset_matrix (first 5 rows, all columns):")
    print(f"Shape: {data.asset_matrix.shape} (dates × indices)")
    asset_df = pd.DataFrame(
        data.asset_matrix[:5], 
        index=[d.strftime('%Y-%m-%d') for d in data.dates[:5]],
        columns=data.domestic_indices + data.foreign_indices
    )
    print(asset_df)
    
    print("\nSample of currency_matrix (first 5 rows, all columns):")
    print(f"Shape: {data.currency_matrix.shape} (dates × foreign currencies)")
    currency_df = pd.DataFrame(
        data.currency_matrix[:5],
        index=[d.strftime('%Y-%m-%d') for d in data.dates[:5]],
        columns=[data.currency_codes[data.index_currencies[idx]] for idx in data.foreign_indices]
    )
    print(currency_df)
    
    print("\nSample of rates_matrix (first 5 rows, all columns):")
    print(f"Shape: {data.rates_matrix.shape} (dates × interest rates)")
    rates_df = pd.DataFrame(
        data.rates_matrix[:5],
        index=[d.strftime('%Y-%m-%d') for d in data.dates[:5]],
        columns=[data.rate_codes[data.index_currencies[idx]] for idx in data.indices]
    )
    print(rates_df)
    
    print("-" * 80)
    
    # Test data retrieval functions
    print("\nTesting data retrieval functions:")
    test_date_index = 10  # Use the 10th date for testing
    test_date = data.dates[test_date_index]
    print(f"Test date: {test_date.strftime('%Y-%m-%d')} (index: {test_date_index})")
    
    # Test asset price retrieval
    for idx in data.indices:
        price = data.get_asset_price(idx, test_date_index)
        print(f"{idx} price: {price:.4f}")
    
    # Test exchange rate retrieval
    for currency in data.index_currencies.values():
        if currency != 'EUR':  # Skip EUR as it's the base currency
            rate = data.get_exchange_rate(currency, test_date_index)
            print(f"{currency} exchange rate: {rate:.4f}")
    
    # Test interest rate retrieval
    for currency in data.index_currencies.values():
        rate = data.get_interest_rate(currency, test_date_index)
        print(f"{currency} interest rate: {rate:.4f}")

if __name__ == "__main__":
    # Change this path to your Excel file location
    file_path = "DonneesGPS2025.xlsx"
    test_market_data(file_path)