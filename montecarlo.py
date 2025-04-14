# montecarlo.py
import numpy as np
from typing import Dict, List, Optional
import time
import multiprocessing as mp
from functools import partial


class MonteCarlo:
    """
    Simplified MonteCarlo class that focuses on delta calculation.
    Provides deltas for all assets in a dictionary format.
    """
    
    def __init__(self, date_handler, product, simulation, sim_params, num_samples=20, fd_steps=0.1):
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
        self.date_handler = date_handler
        self.product = product
        self.simulation = simulation
        self.sim_params = sim_params
        self.num_samples = num_samples
        self.fd_steps = fd_steps
        
        # Results
        self.price = None # valeur liquidative
        self.deltas = None
    
    def _calculate_base_payoffs(self, sim_range, paths):
        """
        Worker function to calculate base payoffs for a range of simulations.
        
        Parameters:
        -----------
        sim_range : range
            Range of simulation indices to process
        paths : numpy.ndarray
            Simulation paths
            
        Returns:
        --------
        list
            List of payoffs for the specified simulations
        """
        payoffs = []
        for sim in sim_range:
            path = paths[:, :, sim]
            payoff = self.product.calculate_total_payoff(path)['total_payoff']
            payoffs.append(payoff)
        return payoffs
    
    def _calculate_shifted_payoffs(self, sim_range, paths, asset_index, shift_factor, current_date):
        """
        Worker function to calculate payoffs for shifted paths.
        
        Parameters:
        -----------
        sim_range : range
            Range of simulation indices to process
        paths : numpy.ndarray
            Simulation paths
        asset_index : int
            Index of the asset to shift
        shift_factor : float
            Factor by which to shift the asset price
        current_date : datetime
            Current date
            
        Returns:
        --------
        list
            List of payoffs for the shifted paths
        """
        payoffs = []
        
        # Process each simulation in the range
        for sim in sim_range:
            # Get the original path for this simulation
            original_path = paths[:, :, sim]
            
            # Create shifted path for this simulation
            shifted_path = self.simulation.shift_path(
                original_path, asset_index, shift_factor, current_date
            )
            
            # Calculate payoff for the shifted path
            payoff = self.product.simulate_product_lifecycle(shifted_path, current_date)['Tc']['total_payoff']
            payoffs.append(payoff)
        
        return payoffs
    
    def calculate_deltas(self, past_matrix, current_date, seed=None):
        """
        Calculate the deltas for all assets using finite differences with parallel processing.
        Returns a list with deltas for all asset types.
        
        Parameters:
        -----------
        past_matrix : numpy.ndarray
            Past matrix up to the current date
        current_date : datetime
            Current date
        seed : int, optional
            Random seed for reproducibility (default: None)
            
        Returns:
        --------
        list
            List of deltas for all assets
        """
        start_time = time.time()
        
        # Get key dates
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
        time_to_maturity = self.date_handler._count_trading_days(current_date, tc_date) / 262
        volatilities, cholesky_matrix = self.sim_params.calculate_parameters(current_date)
        discount_factor = np.exp(-r_d * time_to_maturity)
        
        beg = time.time() - start_time
        print(f"beginning paths generation")
        # Generate paths
        paths = self.simulation.simulate_paths(
            past_matrix=past_matrix,
            current_date=current_date,
            volatilities=volatilities,
            cholesky_matrix=cholesky_matrix
        )
        end = time.time() - start_time
        print(f"paths generation completed in {end-beg:.2f} seconds")
        
        # Determine number of processes and create simulation ranges
        num_processes = min(mp.cpu_count(), self.num_samples)  # Don't use more processes than samples
        chunk_size = max(1, self.num_samples // num_processes)
        sim_ranges = []
        for i in range(0, self.num_samples, chunk_size):
            end = min(i + chunk_size, self.num_samples)
            sim_ranges.append(range(i, end))
        
        # Initialize multiprocessing pool
        pool = mp.Pool(processes=num_processes)
        
        # Calculate base payoffs in parallel
        base_payoff_func = partial(self._calculate_base_payoffs, paths=paths)
        all_base_payoffs = pool.map(base_payoff_func, sim_ranges)
        
        # Flatten the results and calculate mean
        base_payoffs = [payoff for sublist in all_base_payoffs for payoff in sublist]
        mean_base_payoff = np.mean(base_payoffs)
        self.price = mean_base_payoff * discount_factor
        
        # Calculate deltas for each asset
        num_assets = past_matrix.shape[1]
        self.deltas = []
        
        for asset_index in range(num_assets):
            # Get current spot price for this asset
            spot = past_matrix[-1, asset_index]
            
            # Calculate up payoffs in parallel - prepare ranges for each worker
            up_ranges = []
            for sim_range in sim_ranges:
                up_ranges.append((sim_range, paths, asset_index, 1 + self.fd_steps, current_date))
            
            # Run calculations in parallel for up shift
            up_payoffs_lists = pool.starmap(self._calculate_shifted_payoffs, up_ranges)
            up_payoffs = [payoff for sublist in up_payoffs_lists for payoff in sublist]
            mean_up_payoff = np.mean(up_payoffs)
            
            # Calculate down payoffs in parallel - prepare ranges for each worker
            down_ranges = []
            for sim_range in sim_ranges:
                down_ranges.append((sim_range, paths, asset_index, 1 - self.fd_steps, current_date))
            
            # Run calculations in parallel for down shift
            down_payoffs_lists = pool.starmap(self._calculate_shifted_payoffs, down_ranges)
            down_payoffs = [payoff for sublist in down_payoffs_lists for payoff in sublist]
            mean_down_payoff = np.mean(down_payoffs)
            
            # Apply discount factor and calculate delta using central difference
            delta = (mean_up_payoff - mean_down_payoff) * discount_factor / (2.0 * spot * self.fd_steps)
            
            # Store delta
            self.deltas.append(delta)
        
        # Close the pool
        pool.close()
        pool.join()
        
        # Add computation time
        computation_time = time.time() - start_time
        print(f"Delta calculation completed in {computation_time:.2f} seconds")
        
        return self.deltas