# visualize_paths.py
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from simulation import Simulation
from dates import DateHandler
from market_data import MarketData
from sim_params import SimulationParameters
from past_data import PastData
from Product11 import Product11

def visualize_paths():
    # Initialize components
    data_file_path = "DonneesGPS2025.xlsx"
    date_handler = DateHandler(data_file_path)
    
    # Set key dates
    key_dates = {
        'T0': datetime(2009, 1, 5),
        'T1': datetime(2010, 1, 4),
        'T2': datetime(2011, 1, 4),
        'T3': datetime(2012, 1, 4),
        'T4': datetime(2013, 1, 4),
        'Tc': datetime(2014, 1, 6)
    }
    date_handler.set_key_dates(key_dates)
    
    # Initialize market data
    market_data = MarketData(data_file_path, date_handler)
    
    # Set current date for testing
    current_date = datetime(2011, 1, 4)  # Mid-way through the product
    
    # Initialize past data manager and get past matrix
    past_data = PastData(market_data, date_handler)
    past_matrix = past_data.initialize_past_matrix(current_date)
    print(f"Past matrix shape: {past_matrix.shape}")
    
    # Calculate simulation parameters
    sim_params = SimulationParameters(market_data, date_handler)
    volatilities, cholesky_matrix = sim_params.calculate_parameters(current_date)
    print(f"Volatilities shape: {volatilities.shape}")
    print(f"Cholesky matrix shape: {cholesky_matrix.shape}")
    
    # Initialize Simulation
    simulator = Simulation(market_data, date_handler)
    
    # Generate paths (3 paths for visualization)
    paths = simulator.simulate_paths(
        past_matrix=past_matrix, 
        current_date=current_date,
        volatilities=volatilities,
        cholesky_matrix=cholesky_matrix,
        num_simulations=3,
        seed=42
    )
    print(f"Simulated paths shape: {paths.shape}")
    
    # Test path shifting
    print("\nTesting path shifting...")
    shifted_path = simulator.shift_path(
        path=paths[:, :, 0],  # Take first simulation path
        asset_idx=2,  # Shift the third asset
        shift_factor=1.05,  # Shift up by 5%
        current_date=current_date
    )
    
    # Get key dates for plotting
    key_dates_list = [date_handler.get_key_date(k) for k in ['T0', 'T1', 'T2', 'T3', 'T4', 'Tc']]
    
    # Create figure with subplots (one for each asset/currency)
    assets_to_show =6  # Limit to a few for clarity
    fig, axes = plt.subplots(assets_to_show, 1, figsize=(12, 12))
    
    # Labels for each subplot
    labels = (
        market_data.domestic_indices + 
        market_data.foreign_indices + 
        [f"FX_{market_data.index_currencies[idx]}" for idx in market_data.foreign_indices]
    )
    
    # Find indices for current date
    current_date_key_idx = None
    for i, date in enumerate(key_dates_list):
        if date >= current_date:
            if date > current_date:
                current_date_key_idx = i
            else:
                current_date_key_idx = i
            break
    
    if current_date_key_idx is None:
        current_date_key_idx = len(key_dates_list) - 1
    
    # Plot each asset
    for i in range(assets_to_show):
        ax = axes[i]
        
        # Plot original simulation paths
        for sim in range(3):
            ax.plot(key_dates_list, paths[:, i, sim], 'b-', alpha=0.5, linewidth=1.5)
        
        # Plot shifted path for asset 2
        if i == 2:
            ax.plot(key_dates_list, shifted_path[:, i], 'r--', linewidth=2, label='Shifted')
        
        # Vertical line at current date
        ax.axvline(x=current_date, color='black', linestyle='-', linewidth=1, label='Current Date')
        
        # Set title and format
        ax.set_title(f"Asset: {labels[i]}")
        ax.set_xlabel('Date')
        ax.set_ylabel('Value')
        ax.grid(True, alpha=0.3)
        
        # Format dates on x-axis
        plt.setp(ax.get_xticklabels(), rotation=45)
         
        # Add legend (only for the shifted asset)
        if i == 2:
            ax.legend(['Path 1', 'Path 2', 'Path 3', 'Shifted', 'Current Date'])
        else:
            ax.legend(['Path 1', 'Path 2', 'Path 3', 'Current Date'])
    
    # ax = axes[5]
        
    # # Plot original simulation paths
    # for sim in range(3):
    #     ax.plot(key_dates_list, paths[:, i, sim], 'b-', alpha=0.5, linewidth=1.5)
    
    # # Plot shifted path for asset 2
    # if i == 2:
    #     ax.plot(key_dates_list, shifted_path[:, i], 'r--', linewidth=2, label='Shifted')
    
    # # Vertical line at current date
    # ax.axvline(x=current_date, color='black', linestyle='-', linewidth=1, label='Current Date')
    
    # # Set title and format
    # ax.set_title(f"Asset: {labels[i]}")
    # ax.set_xlabel('Date')
    # ax.set_ylabel('Value')
    # ax.grid(True, alpha=0.3)
    
    # # Format dates on x-axis
    # plt.setp(ax.get_xticklabels(), rotation=45)
    
    # # Add legend (only for the shifted asset)
    # if i == 2:
    #     ax.legend(['Path 1', 'Path 2', 'Path 3', 'Shifted', 'Current Date'])
    # else:
    #     ax.legend(['Path 1', 'Path 2', 'Path 3', 'Current Date'])
    
    
    
    plt.tight_layout()
    plt.savefig('simulated_paths.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    visualize_paths()