# simulation.py
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional, Tuple

class Simulation:
    """
    Class for generating simulated price paths based on the past matrix.
    Provides functionalities to:
    1. Simulate paths from current date to maturity
    2. Shift paths for sensitivity analysis
    """
    
    def __init__(self, market_data, date_handler):
        """
        Initialize the Simulation class.
        
        Parameters:
        -----------
        market_data : MarketData
            MarketData object containing asset prices, exchange rates, and interest rates
        date_handler : DateHandler
            DateHandler object containing key dates and trading dates
        """
        self.market_data = market_data
        self.date_handler = date_handler
        
        # Default simulation parameters
        self.num_simulations = 1000
        self.trading_days_per_year = 262  # Standard convention for annualization
        
        # Indices and asset counts for indexing
        self.domestic_indices = self.market_data.domestic_indices
        self.foreign_indices = self.market_data.foreign_indices
        self.indices = self.market_data.indices
        
        self.num_domestic = len(self.domestic_indices)
        self.num_foreign = len(self.foreign_indices)
        self.num_assets = self.num_domestic + self.num_foreign
        self.num_cols = self.num_assets + self.num_foreign  # Assets + FX rates
        
        # For reproducibility
        self.seed = None
    
    def simulate_paths(self, 
                      past_matrix: np.ndarray,
                      current_date: datetime, 
                      volatilities: np.ndarray,
                      cholesky_matrix: np.ndarray,
                      num_simulations: int = None,
                      seed: int = None) -> np.ndarray:
        """
        Generate simulated price paths from current date to maturity.
        
        Parameters:
        -----------
        past_matrix : Past matrix containing historical data up to current date
        current_date : Current date from which to start simulating
        volatilities : Volatility vector for all assets and currencies
        cholesky_matrix : Cholesky decomposition of correlation matrix
        num_simulations : Number of simulation paths (default: self.num_simulations)
        seed : Random seed for reproducibility (optional)
        
        Returns:
        --------
        np.ndarray : Simulated paths including past data
        """
        # Set simulation parameters
        if num_simulations is not None:
            self.num_simulations = num_simulations
            
        if seed is not None:
            self.seed = seed
            np.random.seed(seed)
        
        # Get key dates
        t0_date = self.date_handler.get_key_date("T0")
        t1_date = self.date_handler.get_key_date("T1")
        t2_date = self.date_handler.get_key_date("T2")
        t3_date = self.date_handler.get_key_date("T3")
        t4_date = self.date_handler.get_key_date("T4")
        tc_date = self.date_handler.get_key_date("Tc")
        key_dates = [t0_date, t1_date, t2_date, t3_date, t4_date, tc_date]
        
        # Determine which key dates are in the future from current date
        future_key_dates = [d for d in key_dates if d > current_date]
        if not future_key_dates:
            raise ValueError(f"Current date {current_date} is after all key dates")
        
        # Check if current date is a key date
        is_key_date = current_date in key_dates
        
        # Get interest rates for all currencies
        interest_rates = self._get_interest_rates(current_date)
        
        # Create combined array for past and future paths
        num_rows = len(key_dates)
        num_cols = past_matrix.shape[1]
        paths = np.zeros((num_rows, num_cols, self.num_simulations))
        
        # Fill in the past matrix data for all simulations
        for i in range(past_matrix.shape[0]):
            for j in range(num_cols):
                paths[i, j, :] = past_matrix[i, j]
        
        # Determine starting point for simulation
        if is_key_date:
            # Simple case: Start from a key date
            start_idx = key_dates.index(current_date)
        else:
            # Complex case: Start from a non-key date
            # Find the previous and next key dates
            prev_key_date = self.date_handler.get_previous_key_date(current_date)
            next_key_date = self.date_handler.get_next_key_date(current_date)
            
            # Find their indices in the key_dates list
            prev_idx = key_dates.index(prev_key_date)
            next_idx = key_dates.index(next_key_date)
            
            # Simulate from current date to next key date
            self._simulate_to_next_key_date(
                paths, past_matrix[-1], prev_idx, next_idx,
                current_date, next_key_date, 
                volatilities, cholesky_matrix, interest_rates
            )
            
            # Update starting point for subsequent simulations
            start_idx = next_idx
        
        # Main simulation loop between key dates
        for i in range(start_idx, len(key_dates) - 1):
            # Calculate time step between key dates
            dt = self.date_handler._count_trading_days(key_dates[i], key_dates[i+1]) / self.trading_days_per_year
            
            # Simulate for this time step
            self._simulate_step(
                paths, i, i+1, dt,
                volatilities, cholesky_matrix, interest_rates
            )
        
        return paths
    
    def shift_path(self, 
                  path: np.ndarray, 
                  asset_idx: int, 
                  shift_factor: float, 
                  current_date: datetime) -> np.ndarray:
        """
        Shift a specific asset's path for sensitivity analysis.
        
        Parameters:
        -----------
        path : Simulated price path
        asset_idx : Index of the asset to shift
        shift_factor : Multiplication factor to apply (e.g., 1.01 for +1%)
        current_date : Current date
        
        Returns:
        --------
        np.ndarray : Shifted path
        """
        # Get key dates
        t0_date = self.date_handler.get_key_date("T0")
        t1_date = self.date_handler.get_key_date("T1")
        t2_date = self.date_handler.get_key_date("T2")
        t3_date = self.date_handler.get_key_date("T3")
        t4_date = self.date_handler.get_key_date("T4")
        tc_date = self.date_handler.get_key_date("Tc")
        key_dates = [t0_date, t1_date, t2_date, t3_date, t4_date, tc_date]
        
        # Check if current date is a key date
        is_key_date = current_date in key_dates
        
        # Determine the starting row for the shift
        if is_key_date:
            # Simple case: Start from this key date
            start_row = key_dates.index(current_date)
        else:
            # Complex case: Start from the next key date
            next_key_date = self.date_handler.get_next_key_date(current_date)
            if next_key_date is None:
                # If there's no next key date, don't shift anything
                return path.copy()
            start_row = key_dates.index(next_key_date)
        
        # Create a copy of the path to avoid modifying the original
        shifted_path = path.copy()
        
        # Apply the shift to all values from the starting row
        for i in range(start_row, path.shape[0]):
            shifted_path[i, asset_idx] *= shift_factor
        
        return shifted_path
    
    def _get_interest_rates(self, current_date: datetime) -> Dict[str, float]:
        """
        Get interest rates for all currencies at the current date.
        
        Parameters:
        -----------
        current_date : Current date
        
        Returns:
        --------
        Dict[str, float] : Dictionary mapping currency codes to interest rates
        """
        date_index = self.market_data.get_date_index(current_date)
        
        # Get rates for all currencies
        rates = {}
        for currency in self.market_data.index_currencies.values():
            rates[currency] = self.market_data.get_interest_rate(currency, date_index)
        
        return rates
    
    def _simulate_to_next_key_date(self,
                                  paths: np.ndarray,
                                  last_row: np.ndarray,
                                  next_idx: int,
                                  current_date: datetime,
                                  next_key_date: datetime,
                                  volatilities: np.ndarray,
                                  cholesky_matrix: np.ndarray,
                                  interest_rates: Dict[str, float]) -> None:
        """
        Special case: Simulate from current non-key date to next key date.
        
        Parameters:
        -----------
        paths : 3D array to store simulation results
        last_row : Last row of past data (current values)
        prev_idx : Index of previous key date
        next_idx : Index of next key date
        current_date : Current date (non-key date)
        next_key_date : Next key date
        volatilities : Volatilities for all assets and currencies
        cholesky_matrix : Cholesky decomposition of correlation matrix
        interest_rates : Interest rates for all currencies
        """
        # Calculate time step to next key date (in years)
        dt = self.date_handler._count_trading_days(current_date, next_key_date) / self.trading_days_per_year
        sqrt_dt = np.sqrt(dt)
        
        # Get domestic interest rate
        r_d = interest_rates.get('EUR', 0.0)
        
        # Create sigma vectors for each asset
        sigma_vectors = self._create_sigma_vectors(volatilities)
        
        # For each simulation path
        for sim in range(self.num_simulations):
            # Generate standard normal random vector
            Z = np.random.standard_normal(len(volatilities))
            
            # Update domestic assets
            for j in range(self.num_domestic):
                col_idx = j
                sigma = volatilities[col_idx]
                prev_value = last_row[col_idx]
                
                # Calculate stochastic term using sigma * L^T * Z
                stochastic_term = np.dot(sigma_vectors[col_idx], cholesky_matrix.T) @ Z * sqrt_dt
                
                # Update path with the GBM formula
                paths[next_idx, col_idx, sim] = prev_value * np.exp(
                    (r_d - 0.5 * sigma**2) * dt + stochastic_term
                )
            
            # Update foreign assets (SX)
            for j in range(self.num_foreign):
                col_idx = self.num_domestic + j
                sigma = volatilities[col_idx]
                prev_value = last_row[col_idx]
                
                # Get currency for this foreign asset
                currency = self.market_data.index_currencies[self.foreign_indices[j]]
                
                # Calculate stochastic term
                stochastic_term = np.dot(sigma_vectors[col_idx], cholesky_matrix.T) @ Z * sqrt_dt
                
                # Update path with the GBM formula
                paths[next_idx, col_idx, sim] = prev_value * np.exp(
                    (r_d - 0.5 * sigma**2) * dt + stochastic_term
                )
            
            # Update exchange rates (X*exp(ri*dt))
            for j in range(self.num_foreign):
                col_idx = self.num_assets + j
                col_idx = self.num_assets + j
                cholesky_asset = cholesky_matrix[j+1, :]  # Ls
                cholesky_fx = cholesky_matrix[col_idx, :]  # Lx
    
                # Calculate dot product (equivalent to pnl_vect_scalar_prod)
                rho = np.dot(cholesky_asset, cholesky_fx)  # Ls * Lx
                asset_sigma = volatilities[j+1]
                fx_sigma = volatilities[col_idx]
                # Calculate the combined sigma
                combined_sigma = np.sqrt(asset_sigma**2 + fx_sigma**2 + 2 * asset_sigma * fx_sigma * rho)
                prev_value = last_row[col_idx]
                
                
                # Calculate stochastic term
                stochastic_term = np.dot(sigma_vectors[col_idx], cholesky_matrix.T) @ Z * sqrt_dt
                
                # Update path with the GBM formula for FX rates
                paths[next_idx, col_idx, sim] = prev_value * np.exp(
                    (r_d - 0.5 * combined_sigma**2) * dt + stochastic_term
                )
    
    def _simulate_step(self,
                      paths: np.ndarray,
                      from_idx: int,
                      to_idx: int,
                      dt: float,
                      volatilities: np.ndarray,
                      cholesky_matrix: np.ndarray,
                      interest_rates: Dict[str, float]) -> None:
        """
        Simulate one key date to the next key date for all paths.
        
        Parameters:
        -----------
        paths : 3D array to store simulation results
        from_idx : Index of starting key date
        to_idx : Index of ending key date
        dt : Time step in years
        volatilities : Volatilities for all assets and currencies
        cholesky_matrix : Cholesky decomposition of correlation matrix
        interest_rates : Interest rates for all currencies
        """
        sqrt_dt = np.sqrt(dt)
        
        # Get domestic interest rate
        r_d = interest_rates.get('EUR', 0.0)
        
        # Create sigma vectors for efficient computation
        sigma_vectors = self._create_sigma_vectors(volatilities)
        
        # For each simulation path
        for sim in range(self.num_simulations):
            # Generate standard normal random vector
            Z = np.random.standard_normal(len(volatilities))
            
            # Update domestic assets
            for j in range(self.num_domestic):
                col_idx = j
                sigma = volatilities[col_idx]
                prev_value = paths[from_idx, col_idx, sim]
                
                # Calculate stochastic term
                stochastic_term = np.dot(sigma_vectors[col_idx], cholesky_matrix.T) @ Z * sqrt_dt
                
                # Update path with the GBM formula
                paths[to_idx, col_idx, sim] = prev_value * np.exp(
                    (r_d - 0.5 * sigma**2) * dt + stochastic_term
                )
            
            # Update foreign assets (SX)
            for j in range(self.num_foreign):
                col_idx = self.num_domestic + j
                sigma = volatilities[col_idx]
                prev_value = paths[from_idx, col_idx, sim]
                
                # Calculate stochastic term
                stochastic_term = np.dot(sigma_vectors[col_idx], cholesky_matrix.T) @ Z * sqrt_dt
                
                # Update path with the GBM formula
                paths[to_idx, col_idx, sim] = prev_value * np.exp(
                    (r_d - 0.5 * sigma**2) * dt + stochastic_term
                )
            
            # Update exchange rates (X*exp(ri*dt))
            for j in range(self.num_foreign):
                col_idx = self.num_assets + j
                cholesky_asset = cholesky_matrix[j+1, :]  # Ls
                cholesky_fx = cholesky_matrix[col_idx, :]  # Lx
    
                # Calculate dot product (equivalent to pnl_vect_scalar_prod)
                rho = np.dot(cholesky_asset, cholesky_fx)  # Ls * Lx
                asset_sigma = volatilities[j+1]
                fx_sigma = volatilities[col_idx]
                # Calculate the combined sigma
                combined_sigma = np.sqrt(asset_sigma**2 + fx_sigma**2 + 2 * asset_sigma * fx_sigma * rho)
                prev_value = paths[from_idx, col_idx, sim]
                
                # Calculate stochastic term
                stochastic_term = np.dot(sigma_vectors[col_idx], cholesky_matrix.T) @ Z * sqrt_dt
                
                # Update path with the GBM formula for FX rates
                paths[to_idx, col_idx, sim] = prev_value * np.exp(
                    (r_d - 0.5 * combined_sigma**2) * dt + stochastic_term
                )
    
    def _create_sigma_vectors(self, volatilities: np.ndarray) -> List[np.ndarray]:
        """
        Create sigma vectors for each asset (for multiplication with L^T).
        
        Parameters:
        -----------
        volatilities : Volatility vector for all assets and currencies
        
        Returns:
        --------
        List[np.ndarray] : List of sigma vectors
        """
        n = len(volatilities)
        sigma_vectors = []
        
        # For each asset/FX rate, create a vector with its volatility
        for i in range(5):
            sigma_vector = np.zeros(n)
            sigma_vector[i] = volatilities[i]
            sigma_vectors.append(sigma_vector)
        for i in range(5,n):
            sigma_vector = np.zeros(n)
            sigma_vector[i-4] = volatilities[i-4]
            sigma_vector[i] = volatilities[i]
            sigma_vectors.append(sigma_vector)
        
        print(sigma_vectors)
        
        return sigma_vectors