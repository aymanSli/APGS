# simulation.py
import numpy as np
from typing import List, Tuple

class Simulation:
    """
    Class for generating simulated price paths based on the past matrix.
    
    Generates a 3D matrix with dimensions:
    - rows: dates (indexed relative to T0, so T0 is row 0)
    - columns: same as past matrix (domestic assets, foreign assets, currencies)
    - 3rd dimension: number of simulation paths
    
    The simulation follows the multi-currency hedging logic.
    """
    
    def __init__(self, market_data, date_handler, sim_params):
        """
        Initialize the Simulation class.
        
        Parameters:
        -----------
        market_data : MarketData
            MarketData object containing asset prices, exchange rates, and interest rates
        date_handler : DateHandler
            DateHandler object containing date indices and key dates
        sim_params : SimulationParameters
            SimulationParameters object containing volatilities and correlation matrix
        """
        self.market_data = market_data
        self.date_handler = date_handler
        self.sim_params = sim_params
        
        # Default parameters
        self.num_simulations = 1000
        self.trading_days_per_year = 252  # For annualization
        
        # Asset counts for indexing
        self.num_domestic = len(market_data.domestic_indices)
        self.num_foreign = len(market_data.foreign_indices)
        self.num_total_assets = self.num_domestic + self.num_foreign
        self.num_total_columns = self.num_total_assets + self.num_foreign  # Adding FX columns
        
        # Seed for reproducibility
        self.seed = None
    
    def generate_paths(self, past_matrix, current_date, num_simulations=None, seed=None):
        """
        Generate simulated price paths based on the past matrix.
        
        Parameters:
        -----------
        past_matrix : numpy.ndarray
            Past matrix up to the current date
        current_date : datetime
            Current date for the simulation
        num_simulations : int, optional
            Number of simulation paths to generate
        seed : int, optional
            Random seed for reproducibility
            
        Returns:
        --------
        numpy.ndarray
            3D array of simulated paths
        """
        if num_simulations is not None:
            self.num_simulations = num_simulations
        
        if seed is not None:
            self.seed = seed
            np.random.seed(seed)
        
        # Get all simulation dates
        sim_dates = self.date_handler.get_simulation_dates()
        
        # Find the index of current date in simulation grid, or closest previous date
        current_date_index = -1
        for i, date in enumerate(sim_dates):
            if date > current_date:
                break
            current_date_index = i
        
        if current_date_index < 0:
            raise ValueError(f"Current date {current_date} is before the first simulation date")
        
        # Get key date indices
        t0_date = self.date_handler.get_key_date("T0")
        tc_date = self.date_handler.get_key_date("Tc")
        
        # Find indices of T0 and Tc in simulation grid
        t0_index = sim_dates.index(t0_date)
        tc_index = sim_dates.index(tc_date)
        
        # Determine dimensions
        num_rows = tc_index - t0_index + 1
        num_cols = past_matrix.shape[1]
        
        # Create 3D array for simulated paths
        paths = np.zeros((num_rows, num_cols, self.num_simulations))
        
        # Determine what part of the past matrix to copy
        past_rows_to_copy = current_date_index - t0_index + 1
        
        # Fill with past matrix data for all simulations
        for i in range(past_rows_to_copy):
            for j in range(num_cols):
                paths[i, j, :] = past_matrix[t0_index + i, j]
        
        # Get simulation parameters
        volatilities, correlation_matrix, cholesky_matrix = self.sim_params.get_parameters()
        
        # If the current date is not a simulation date, adjust the starting point
        if current_date != sim_dates[current_date_index]:
            # We need to adjust for the fractional time step
            # Find the next simulation date after current date
            next_sim_date_index = current_date_index + 1
            if next_sim_date_index >= len(sim_dates):
                next_sim_date_index = current_date_index  # Use the current date if at the end
            
            next_sim_date = sim_dates[next_sim_date_index]
            
            # Calculate the fractional time step in years
            dt_fraction = (next_sim_date - current_date).days / 365.0
            
            # We'll pass this information to the simulation function
            start_is_exact_sim_date = False
            dt_first_step = dt_fraction
        else:
            start_is_exact_sim_date = True
            dt_first_step = None
        
        # Get interest rates for all currencies
        interest_rates = {}
        for currency in self.market_data.index_currencies.values():
            current_idx = self.market_data.get_date_index(current_date)
            interest_rates[currency] = self.market_data.get_interest_rate(currency, current_idx)
        
        # Get domestic interest rate
        r_d = interest_rates.get('EUR', 0.0)
        
        # Simulate future paths from current date to Tc
        for sim in range(self.num_simulations):
            self._simulate_single_path(
                paths, sim, past_rows_to_copy - 1, num_rows, num_cols,
                volatilities, cholesky_matrix, interest_rates, r_d,
                sim_dates, t0_index, current_date_index,
                start_is_exact_sim_date, dt_first_step
            )
        
        return paths
    
    def _create_asset_sigma_vectors(self, volatilities):
        """
        Create asset-specific sigma vectors following the C++ approach.
        
        Parameters:
        -----------
        volatilities : numpy.ndarray
            Vector of volatilities for all assets and currencies
        
        Returns:
        --------
        list of numpy.ndarray
            List of asset-specific sigma vectors
        """
        n_cols = len(volatilities)
        sigma_vectors = []
        
        # Domestic assets
        for i in range(self.num_domestic):
            sigma = np.zeros(n_cols)
            sigma[i] = volatilities[i]
            sigma_vectors.append(sigma)
        
        # Foreign assets (include both asset and corresponding FX volatility)
        for i in range(self.num_foreign):
            asset_idx = self.num_domestic + i
            fx_idx = self.num_total_assets + i
            
            sigma = np.zeros(n_cols)
            sigma[asset_idx] = volatilities[asset_idx]
            sigma[fx_idx] = volatilities[fx_idx]
            sigma_vectors.append(sigma)
        
        # Currencies (FX rates)
        for i in range(self.num_foreign):
            fx_idx = self.num_total_assets + i
            
            sigma = np.zeros(n_cols)
            sigma[fx_idx] = volatilities[fx_idx]
            sigma_vectors.append(sigma)
        
        return sigma_vectors
    
    def _calculate_stochastic_term(self, gaussian_vector, cholesky_matrix, sigma_vector, sqrt_dt):
        """
        Calculate the stochastic term following the C++ approach.
        
        Parameters:
        -----------
        gaussian_vector : numpy.ndarray
            Vector of independent standard normal variables
        cholesky_matrix : numpy.ndarray
            Cholesky decomposition of the correlation matrix
        sigma_vector : numpy.ndarray
            Asset-specific volatility vector
        sqrt_dt : float
            Square root of the time step
        
        Returns:
        --------
        float
            The stochastic term
        """
        # Multiply sigma by the transposed Cholesky matrix
        sigma_L = np.dot(sigma_vector, cholesky_matrix.T)
        
        # Compute the scalar product with Gaussian vector
        stochastic_term = np.dot(sigma_L, gaussian_vector) * sqrt_dt
        
        return stochastic_term
    
    def _simulate_single_path(self, paths, sim_index, start_row, num_rows, num_cols,
                             volatilities, cholesky_matrix, interest_rates, r_d,
                             sim_dates, t0_index, current_date_index,
                             start_is_exact_sim_date, dt_first_step=None):
        """
        Simulate a single path from the current date to Tc.
        
        Parameters:
        -----------
        paths : numpy.ndarray
            3D array of simulated paths to update
        sim_index : int
            Index of the current simulation
        start_row : int
            Starting row index (current date, relative to T0)
        num_rows : int
            Total number of rows (dates)
        num_cols : int
            Number of columns (assets and currencies)
        volatilities : numpy.ndarray
            Volatilities for all assets and currencies
        cholesky_matrix : numpy.ndarray
            Cholesky decomposition of the correlation matrix
        interest_rates : dict
            Interest rates for all currencies
        r_d : float
            Domestic interest rate
        sim_dates : list
            List of simulation dates
        t0_index : int
            Index of T0 in simulation dates
        current_date_index : int
            Index of current date (or closest previous date) in simulation dates
        start_is_exact_sim_date : bool
            Whether the current date is an exact simulation date
        dt_first_step : float, optional
            Time step for the first simulation step if not starting on exact date
        """
        # Create asset-specific sigma vectors (like in the C++ approach)
        sigma_vectors = self._create_asset_sigma_vectors(volatilities)
        
        # Track the current row index in the paths array
        current_row = start_row
        
        # Handle special case for first step if not starting on exact simulation date
        if not start_is_exact_sim_date and dt_first_step is not None:
            # Generate Gaussian random vector
            Z = np.random.standard_normal(num_cols)
            
            # Get sqrt of time step
            sqrt_dt_first = np.sqrt(dt_first_step)
            
            next_row = current_row + 1
            
            # Simulate domestic assets
            for j in range(self.num_domestic):
                prev_spot = paths[current_row, j, sim_index]
                sigma = volatilities[j]
                
                # Calculate stochastic term using our method
                stochastic_term = self._calculate_stochastic_term(Z, cholesky_matrix, sigma_vectors[j], sqrt_dt_first)
                
                # Update the spot price using the Black-Scholes formula
                paths[next_row, j, sim_index] = prev_spot * np.exp(
                    (r_d - 0.5 * sigma**2) * dt_first_step + stochastic_term
                )
            
            # Simulate foreign assets (SX)
            for j in range(self.num_foreign):
                asset_idx = self.num_domestic + j
                fx_idx = self.num_total_assets + j
                
                prev_spot = paths[current_row, asset_idx, sim_index]
                sigma_vector = sigma_vectors[self.num_domestic + j]
                
                # Get the foreign currency for this index
                currency = self.market_data.index_currencies[self.market_data.foreign_indices[j]]
                
                # Calculate stochastic term
                stochastic_term = self._calculate_stochastic_term(Z, cholesky_matrix, sigma_vector, sqrt_dt_first)
                
                # Calculate combined volatility for drift term
                asset_vol = volatilities[asset_idx]
                fx_vol = volatilities[fx_idx]
                asset_row = cholesky_matrix[asset_idx]
                fx_row = cholesky_matrix[fx_idx]
                rho = np.dot(asset_row, fx_row)  # Correlation between asset and its FX
                combined_vol = np.sqrt(asset_vol**2 + fx_vol**2 + 2 * asset_vol * fx_vol * rho)
                
                # Update SX
                paths[next_row, asset_idx, sim_index] = prev_spot * np.exp(
                    (r_d - 0.5 * combined_vol**2) * dt_first_step + stochastic_term
                )
            
            # Simulate exchange rates with interest rate adjustments
            for j in range(self.num_foreign):
                fx_idx = self.num_total_assets + j
                
                prev_spot = paths[current_row, fx_idx, sim_index]
                sigma = volatilities[fx_idx]
                
                # Get the foreign currency for this index
                currency = self.market_data.index_currencies[self.market_data.foreign_indices[j]]
                r_f = interest_rates.get(currency, 0.0)
                
                # For the FX rate, we're simulating X*exp(r_f*t) directly
                # Calculate stochastic term
                stochastic_term = self._calculate_stochastic_term(Z, cholesky_matrix, sigma_vectors[self.num_total_assets + j], sqrt_dt_first)
                
                # Update X*exp(r_f*t)
                paths[next_row, fx_idx, sim_index] = prev_spot * np.exp(
                    (r_d - 0.5 * sigma**2) * dt_first_step + stochastic_term
                )
            
            # Move to the next row
            current_row = next_row
        
        # Continue with regular simulation for remaining time steps
        for t in range(current_row + 1, num_rows):
            # Calculate the time step between simulation dates
            prev_date = sim_dates[t0_index + t - 1]
            curr_date = sim_dates[t0_index + t]
            dt = (curr_date - prev_date).days / 365.0
            sqrt_dt = np.sqrt(dt)
            
            # Generate Gaussian random vector
            Z = np.random.standard_normal(num_cols)
            
            # Simulate domestic assets
            for j in range(self.num_domestic):
                prev_spot = paths[t-1, j, sim_index]
                sigma = volatilities[j]
                
                # Calculate stochastic term
                stochastic_term = self._calculate_stochastic_term(Z, cholesky_matrix, sigma_vectors[j], sqrt_dt)
                
                # Update the spot price
                paths[t, j, sim_index] = prev_spot * np.exp(
                    (r_d - 0.5 * sigma**2) * dt + stochastic_term
                )
            
            # Simulate foreign assets (SX)
            for j in range(self.num_foreign):
                asset_idx = self.num_domestic + j
                fx_idx = self.num_total_assets + j
                
                prev_spot = paths[t-1, asset_idx, sim_index]
                sigma_vector = sigma_vectors[self.num_domestic + j]
                
                # Get the foreign currency for this index
                currency = self.market_data.index_currencies[self.market_data.foreign_indices[j]]
                
                # Calculate stochastic term
                stochastic_term = self._calculate_stochastic_term(Z, cholesky_matrix, sigma_vector, sqrt_dt)
                
                # Calculate combined volatility for drift term
                asset_vol = volatilities[asset_idx]
                fx_vol = volatilities[fx_idx]
                asset_row = cholesky_matrix[asset_idx]
                fx_row = cholesky_matrix[fx_idx]
                rho = np.dot(asset_row, fx_row)  # Correlation between asset and its FX
                combined_vol = np.sqrt(asset_vol**2 + fx_vol**2 + 2 * asset_vol * fx_vol * rho)
                
                # Update SX
                paths[t, asset_idx, sim_index] = prev_spot * np.exp(
                    (r_d - 0.5 * combined_vol**2) * dt + stochastic_term
                )
            
            # Simulate exchange rates with interest rate adjustments
            for j in range(self.num_foreign):
                fx_idx = self.num_total_assets + j
                
                prev_spot = paths[t-1, fx_idx, sim_index]
                sigma = volatilities[fx_idx]
                
                # Get the foreign currency for this index
                currency = self.market_data.index_currencies[self.market_data.foreign_indices[j]]
                r_f = interest_rates.get(currency, 0.0)
                
                # Calculate stochastic term
                stochastic_term = self._calculate_stochastic_term(Z, cholesky_matrix, sigma_vectors[self.num_total_assets + j], sqrt_dt)
                
                # Update X*exp(r_f*t)
                paths[t, fx_idx, sim_index] = prev_spot * np.exp(
                    (r_d - 0.5 * sigma**2) * dt + stochastic_term
                )
    
    def shift_path(self, paths, asset_index, shift_factor, current_date):
        """
        Shift a specific asset's path for delta calculations.
        
        Parameters:
        -----------
        paths : numpy.ndarray
            3D array of simulated paths
        asset_index : int
            Index of the asset to shift
        shift_factor : float
            Factor to shift the path by (e.g., 1.01 for +1%)
        current_date : datetime
            Current date
            
        Returns:
        --------
        numpy.ndarray
            Shifted paths
        """
        # Create a copy of the paths to avoid modifying the original
        shifted_paths = paths.copy()
        
        # Find the row index for the current date
        sim_dates = self.date_handler.get_simulation_dates()
        t0_date = self.date_handler.get_key_date("T0")
        t0_index = sim_dates.index(t0_date)
        
        # Find current date index (or closest previous date)
        current_date_index = -1
        for i, date in enumerate(sim_dates):
            if date > current_date:
                break
            current_date_index = i
        
        if current_date_index < 0:
            raise ValueError(f"Current date {current_date} is before the first simulation date")
        
        # Calculate the row index in the paths array
        start_row = current_date_index - t0_index
        
        # Shift the specified asset from the start row for all simulations
        for sim in range(paths.shape[2]):
            for t in range(start_row, paths.shape[0]):
                shifted_paths[t, asset_index, sim] *= shift_factor
        
        return shifted_paths
    
    def calculate_delta(self, product, paths, current_date, asset_index, h=0.01):
        """
        Calculate delta for a specific asset using finite differences.
        
        Parameters:
        -----------
        product : Product11
            Product11 object to calculate payoffs
        paths : numpy.ndarray
            3D array of simulated paths
        current_date : datetime
            Current date
        asset_index : int
            Index of the asset to calculate delta for
        h : float, optional
            Perturbation size (default: 0.01 = 1%)
            
        Returns:
        --------
        float
            Delta value
        """
        # Find the row index for the current date
        sim_dates = self.date_handler.get_simulation_dates()
        t0_date = self.date_handler.get_key_date("T0")
        t0_index = sim_dates.index(t0_date)
        
        # Find current date index (or closest previous date)
        current_date_index = -1
        for i, date in enumerate(sim_dates):
            if date > current_date:
                break
            current_date_index = i
        
        current_matrix_idx = current_date_index - t0_index
        
        # Calculate payoff for base case
        base_payoffs = []
        for sim in range(paths.shape[2]):
            # Extract this simulation path
            path = paths[:, :, sim]
            
            # Calculate payoff
            payoff_info = product.calculate_total_payoff(path)
            base_payoffs.append(payoff_info['total_payoff'])
        
        # Create up-shifted paths
        up_paths = self.shift_path(paths, asset_index, 1 + h, current_date)
        
        # Calculate payoff for up-shifted case
        up_payoffs = []
        for sim in range(paths.shape[2]):
            # Extract this simulation path
            path = up_paths[:, :, sim]
            
            # Calculate payoff
            payoff_info = product.calculate_total_payoff(path)
            up_payoffs.append(payoff_info['total_payoff'])
        
        # Create down-shifted paths
        down_paths = self.shift_path(paths, asset_index, 1 - h, current_date)
        
        # Calculate payoff for down-shifted case
        down_payoffs = []
        for sim in range(paths.shape[2]):
            # Extract this simulation path
            path = down_paths[:, :, sim]
            
            # Calculate payoff
            payoff_info = product.calculate_total_payoff(path)
            down_payoffs.append(payoff_info['total_payoff'])
        
        # Calculate average payoffs
        avg_base = np.mean(base_payoffs)
        avg_up = np.mean(up_payoffs)
        avg_down = np.mean(down_payoffs)
        
        # Calculate delta using central difference
        current_spot = paths[current_matrix_idx, asset_index, 0]  # Use first simulation for current spot
        delta = (avg_up - avg_down) / (2 * h * current_spot)
        
        return delta
    
    def calculate_all_deltas(self, product, paths, current_date, h=0.01):
        """
        Calculate deltas for all assets.
        
        Parameters:
        -----------
        product : Product11
            Product11 object to calculate payoffs
        paths : numpy.ndarray
            3D array of simulated paths
        current_date : datetime
            Current date
        h : float, optional
            Perturbation size (default: 0.01 = 1%)
            
        Returns:
        --------
        dict
            Dictionary mapping asset names to delta values
        """
        # Number of assets and currencies
        num_domestic = self.num_domestic
        num_foreign = self.num_foreign
        
        # Calculate deltas
        deltas = {}
        
        # Domestic assets
        for i, asset_name in enumerate(self.market_data.domestic_indices):
            deltas[asset_name] = self.calculate_delta(product, paths, current_date, i, h)
        
        # Foreign assets
        for i, asset_name in enumerate(self.market_data.foreign_indices):
            idx = num_domestic + i
            deltas[asset_name] = self.calculate_delta(product, paths, current_date, idx, h)
        
        # Exchange rates
        for i, asset_name in enumerate(self.market_data.foreign_indices):
            idx = num_domestic + num_foreign + i
            currency = self.market_data.index_currencies[asset_name]
            deltas[f"FX_{currency}"] = self.calculate_delta(product, paths, current_date, idx, h)
        
        return deltas