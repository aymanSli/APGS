# test_sim_params.py

import numpy as np
from datetime import datetime
from dates import DateHandler
from market_data import MarketData
from sim_params import SimulationParameters

def run_test():
    print("Testing SimulationParameters...")
    
    # Initialize date handler and market data
    data_file_path = "DonneesGPS2025.xlsx"  # Adjust path as needed
    date_handler = DateHandler(data_file_path)
    
    # Set key dates (example dates - adjust as needed)
    key_dates = {
        'T0': datetime(2009, 1, 5),
        'T1': datetime(2010, 1, 4),
        'T2': datetime(2011, 1, 4),
        'T3': datetime(2012, 1, 4),
        'T4': datetime(2013, 1, 4),
        'Tc': datetime(2014, 1, 6)
    }
    date_handler.set_key_dates(key_dates)
    
    # Initialize market data with rebalancing grid
    market_data = MarketData(data_file_path,date_handler)
    
    # Initialize simulation parameters
    sim_params = SimulationParameters(market_data, date_handler)
    
    # Test parameter calculation at key dates
    print("\nCalculating parameters at key dates:")

    current_date = datetime(2010, 1, 13)
    
    # Calculate parameters
    volatilities, correlation_matrix, cholesky_matrix = sim_params.calculate_parameters(current_date)
    
    # Print summary information
    print(f"Number of assets: {len(sim_params.all_indices)}")
    print(f"Number of currencies: {len(sim_params.foreign_indices)}")
    print(f"Volatility vector shape: {volatilities.shape}")
    print(f"Correlation matrix shape: {correlation_matrix.shape}")
    print(f"Cholesky matrix shape: {cholesky_matrix.shape}")
    
    # Print a few volatility values as a sample
    print("\nSample volatilities:")
    for i, idx in enumerate(sim_params.domestic_indices[:2] + sim_params.foreign_indices[:2]):
        print(f"  {idx}: {volatilities[i]:.4f}")
    
    # Print correlation matrix summary
    print("\nCorrelation matrix summary:")
    print(f"  Min value: {np.min(correlation_matrix):.4f}")
    print(f"  Max value: {np.max(correlation_matrix):.4f}")
    print(f"  Mean value: {np.mean(correlation_matrix):.4f}")

    print("\nTest complete.")

if __name__ == "__main__":
    run_test()