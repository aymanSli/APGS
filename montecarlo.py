# montecarlo.py
import numpy as np
from typing import Dict, List, Optional
import time


class MonteCarlo:
    """
    Simplified MonteCarlo class that focuses on delta calculation.
    Provides deltas for all assets in a dictionary format.
    """
    
    def __init__(self,date_handler, product, simulation,sim_params, num_samples=10, fd_steps=0.1):
        """
        Initialize the MonteCarlo class.
        
        Parameters:
        -----------
        product : Product11 or similar
            Product object that can calculate payoffs for a given path
        simulation : Simulation
            Simulation object that can generate paths and shift paths
        num_samples : int, optional
            Number of Monte Carlo samples (default: 10000)
        fd_steps : float, optional
            Step size for finite difference calculations (default: 0.1 = 10%)
        """
        self.date_handler=date_handler
        self.product = product
        self.simulation = simulation
        self.sim_params=sim_params
        self.num_samples = num_samples
        self.fd_steps = fd_steps
        
        # Results
        self.price = None
        self.deltas = None
    
    def calculate_deltas(self, past_matrix, current_date, seed=None):
        """
        Calculate the deltas for all assets using finite differences.
        Returns a dictionary with deltas for all asset types.
        
        Parameters:
        -----------
        past_matrix : numpy.ndarray
            Past matrix up to the current date
        current_date_index : int
            Index of the current date
        seed : int, optional
            Random seed for reproducibility (default: None)
            
        Returns:
        --------
        dict
            Dictionary mapping asset indices to their deltas
        """
        
        start_time = time.time()
        t0_date = self.date_handler.get_key_date("T0")
        t1_date = self.date_handler.get_key_date("T1")
        t2_date = self.date_handler.get_key_date("T2")
        t3_date = self.date_handler.get_key_date("T3")
        t4_date = self.date_handler.get_key_date("T4")
        tc_date = self.date_handler.get_key_date("Tc")
        key_dates = [t0_date, t1_date, t2_date, t3_date, t4_date, tc_date]
        # Get the domestic interest rate for discounting
        r_d = self.product.get_domestic_rate()
        
        # Calculate time to maturity
        time_to_maturity = self.date_handler._count_trading_days(current_date,tc_date)/ 262  # Assuming 262 trading days per year
        volatilities,cholesky_matrix = self.sim_params.calculate_parameters(current_date)
        discount_factor = np.exp(-r_d * time_to_maturity)
        
        # Generate paths
        paths = self.simulation.simulate_paths(
            past_matrix=past_matrix,
            current_date=current_date,
            volatilities=volatilities,
            cholesky_matrix=cholesky_matrix
        )
        
        # if current_date not in key_dates : 
        #     current_row=key_dates.index(current_date)
        # else:
        #     current_row=key_dates.index(self.date_handler.get_previous_key_date(current_date))
        
        # Calculate price from base paths (needed for delta computation)
        base_payoffs = []
        for sim in range(self.num_samples):
            path = paths[:, :, sim]
            payoff = self.product.calculate_total_payoff(path)['total_payoff']
            base_payoffs.append(payoff)
        
        # Calculate mean payoff and apply discount factor
        mean_base_payoff = np.mean(base_payoffs)
        self.price = mean_base_payoff * discount_factor
        # print("norm\n")
        # print(base_payoffs)
        
        # Calculate deltas for each asset
        num_assets = past_matrix.shape[1]
        self.deltas = []
        
        for asset_index in range(num_assets):
            # Get current spot price for this asset
            spot = past_matrix[-1, asset_index]
            
            # Create shifted paths for this asset (up)
            shifted_up = self.simulation.shift_all_paths(
                paths, asset_index, 1 + self.fd_steps, current_date
            )
            
            # Calculate payoffs for shifted up paths
            up_payoffs = []
            for sim in range(self.num_samples):
                path_up = shifted_up[:, :, sim]
                payoff_up = self.product.simulate_product_lifecycle(path_up,current_date)['Tc']['total_payoff']
                # print(payoff_up)
                up_payoffs.append(payoff_up)
            
            # Create shifted paths for this asset (down)
            shifted_down = self.simulation.shift_all_paths(
                paths, asset_index, 1 - self.fd_steps, current_date
            )
            # if asset_index==3:
            #     print("norm\n")
            #     print(paths[:, :, 2])
            #     print("up\n")
            #     print(shifted_up[:, :, 2])
            #     print("down\n")
            #     print(shifted_down[:, :, 2])
            # Calculate payoffs for shifted down paths
            down_payoffs = []
            for sim in range(self.num_samples):
                path_down = shifted_down[:, :, sim]
                payoff_down = self.product.simulate_product_lifecycle(path_down,current_date)['Tc']['total_payoff']
                # print(payoff_down)
                down_payoffs.append(payoff_down)
            
            # print("payoffs\n")
            # Calculate mean payoffs and delta
            mean_up_payoff = np.mean(up_payoffs)
            # print("up\n")
            # print(up_payoffs)
            # print(mean_up_payoff)
            mean_down_payoff = np.mean(down_payoffs)
            # print("down\n")
            # print(down_payoffs)
            # print(mean_down_payoff)
            
            # Apply discount factor and calculate delta using central difference
            delta = (mean_up_payoff - mean_down_payoff) * discount_factor / (2.0 * spot * self.fd_steps)
            
            # Store delta
            self.deltas.append(delta)
        
        # Add computation time
        computation_time = time.time() - start_time
        print(f"Delta calculation completed in {computation_time:.2f} seconds")
        
        # Create a dictionary with descriptive keys for easier reference
        # We assume the first num_domestic assets are domestic, followed by
        # foreign assets (SX) and then currencies (Xexp(ri*t))
        # num_domestic = self.simulation.num_domestic
        # num_foreign = self.simulation.num_foreign
        
        # labeled_deltas = {}
        
        # # Add domestic assets
        # for i in range(num_domestic):
        #     idx = i
        #     asset_name = self.simulation.market_data.domestic_indices[i]
        #     labeled_deltas[f"DOM_{asset_name}"] = self.deltas[idx]
        
        # # Add foreign assets (SX)
        # for i in range(num_foreign):
        #     idx = num_domestic + i
        #     asset_name = self.simulation.market_data.foreign_indices[i]
        #     labeled_deltas[f"SX_{asset_name}"] = self.deltas[idx]
        
        # # Add currencies (Xexp(ri*t))
        # for i in range(num_foreign):
        #     idx = num_domestic + num_foreign + i
        #     asset_name = self.simulation.market_data.foreign_indices[i]
        #     currency = self.simulation.market_data.index_currencies[asset_name]
        #     labeled_deltas[f"X_{currency}"] = self.deltas[idx]
        
        return self.deltas
    