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
    
    # Set key dates (example dates)
    key_dates = {
        'T0': datetime(2009, 1, 5),
        'T1': datetime(2010, 1, 4),
        'T2': datetime(2011, 1, 4),
        'T3': datetime(2012, 1, 4),
        'T4': datetime(2013, 1, 4),
        'Tc': datetime(2014, 1, 6)
    }
    date_handler.set_key_dates(key_dates)
    
    # Generate simulation and rebalancing grids
    sim_dates = date_handler.generate_simulation_grid()
    rebalancing_dates = date_handler.generate_rebalancing_grid()
    
    # Initialize market data
    market_data = MarketData(data_file_path, rebalancing_dates)
    
    # Calculate simulation parameters
    sim_params = SimulationParameters(market_data, date_handler)
    sim_params.calculate_parameters(rebalancing_dates[3])  # 4th date in rebalancing grid
    sim_params.print_parameters()
    
    # Initialize PastData and get past matrix
    past_data = PastData(market_data, date_handler)
    past_matrix = past_data.initialize_past_matrix(current_date=rebalancing_dates[3])
    print("paaaaaaaaaaaast\n")
    print(past_matrix)
    
    
    # # Initialize Simulation
    # simulator = Simulation(market_data, date_handler, sim_params)
    
    # # Generate paths (3 paths for visualization)
    # paths = simulator.generate_paths(past_matrix, rebalancing_dates[3], num_simulations=3, seed=42)
    
    # # Get current date index
    # current_date = rebalancing_dates[3]
    # sim_dates = date_handler.get_simulation_dates()
    # t0_date = date_handler.get_key_date("T0")
    # t0_index = sim_dates.index(t0_date)
    
    # # Find current date index (or closest previous date)
    # current_date_index = -1
    # for i, date in enumerate(sim_dates):
    #     if date > current_date:
    #         break
    #     current_date_index = i
    
    # current_matrix_idx = current_date_index - t0_index
    
    # # Create figure with 9 subplots (one for each asset/currency)
    # fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    # axes = axes.flatten()
    
    # # Labels for each subplot
    # labels = (
    #     market_data.domestic_indices + 
    #     market_data.foreign_indices + 
    #     [f"FX_{market_data.index_currencies[idx]}" for idx in market_data.foreign_indices]
    # )
    
    # # Get x-axis dates
    # t0_absolute_idx = sim_dates.index(t0_date)
    # plot_dates = sim_dates[t0_absolute_idx:t0_absolute_idx+paths.shape[0]]
    
    # # Plot each asset/currency
    # for i in range(9):
    #     ax = axes[i]
        
    #     for sim in range(3):  # 3 paths
    #         # Historical part (solid line)
    #         ax.plot(plot_dates[:current_matrix_idx+1], 
    #                 paths[:current_matrix_idx+1, i, sim], 
    #                 'b-', linewidth=2)
            
    #         # Simulated part (dashed line)
    #         ax.plot(plot_dates[current_matrix_idx:], 
    #                 paths[current_matrix_idx:, i, sim], 
    #                 'r--', linewidth=1.5)
        
    #     # Vertical line at current date
    #     ax.axvline(x=plot_dates[current_matrix_idx], color='black', linestyle='-', linewidth=1)
        
    #     # Set title and format
    #     ax.set_title(labels[i])
    #     ax.set_xlabel('Date')
    #     ax.set_ylabel('Value')
    #     ax.grid(True, alpha=0.3)
        
    #     # Format dates on x-axis
    #     ax.tick_params(axis='x', rotation=45)
    
    # # Add a legend
    # fig.legend(['Historical', 'Simulated', 'Current Date'], 
    #            loc='upper center', 
    #            bbox_to_anchor=(0.5, 0.98), 
    #            ncol=3)
    
    # plt.tight_layout(rect=[0, 0, 1, 0.96])
    # plt.savefig('simulated_paths.png', dpi=300)
    # plt.show()

if __name__ == "__main__":
    visualize_paths()