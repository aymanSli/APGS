# sim_params.py
import numpy as np
from typing import List, Dict, Tuple

class SimulationParameters:
    """
    Class for calculating simulation parameters (volatilities and correlation matrix)
    based on historical data for observation dates.
    
    Parameters are calculated using the last 252 trading days (or fewer if not enough data).
    Assets are ordered as: domestic indices, foreign indices, respective currencies.
    """
    
    def __init__(self, market_data, date_handler):
        """
        Initialize the SimulationParameters class.
        
        Parameters:
        -----------
        market_data : MarketData
            MarketData object containing asset prices, exchange rates, and interest rates
        date_handler : DateHandler
            DateHandler object containing date indices and key dates
        """
        self.market_data = market_data
        self.date_handler = date_handler
        
        # Default parameters
        self.trading_days_per_year = 252
        
        # Indices in the same order as the path matrix
        self.domestic_indices = market_data.domestic_indices
        self.foreign_indices = market_data.foreign_indices
        self.all_indices = self.domestic_indices + self.foreign_indices
        
        # Store calculated parameters - will be updated at key dates
        self.volatilities = None
        self.correlation_matrix = None
        self.cholesky_matrix = None
        
        # Track the date for which parameters were last calculated
        self.last_calculation_date = None
    
    def calculate_parameters(self, current_date):
        """
        Calculate volatilities and correlation matrix for the current date
        using the market data matrices.
        
        Parameters:
        -----------
        current_date : datetime
            Current date for which to calculate parameters
            
        Returns:
        --------
        tuple
            (volatilities, correlation_matrix, cholesky_matrix)
        """
        # Get date index in the market data
        current_idx = self.date_handler.get_date_index(current_date)
    
        start_idx = 0
        
        # Extract historical prices and calculate returns
        returns_indices = self._calculate_index_returns(start_idx, current_idx)
        returns_fx = self._calculate_fx_returns(start_idx, current_idx)
        
        # Calculate volatilities
        vol_indices = self._calculate_volatilities_from_returns(returns_indices)
        vol_fx = self._calculate_volatilities_from_returns(returns_fx)
        
        # Combine volatilities in the correct order
        self.volatilities = np.concatenate([vol_indices, vol_fx])
        
        # Calculate correlation matrix
        self.correlation_matrix = self._calculate_correlation_matrix(returns_indices, returns_fx)
        
        # Calculate Cholesky decomposition
        self.cholesky_matrix = self._calculate_cholesky_matrix(self.correlation_matrix)
        
        # Update last calculation date
        self.last_calculation_date = current_date
        
        return (self.volatilities, self.correlation_matrix, self.cholesky_matrix)
    
    def _calculate_index_returns(self, start_idx, end_idx):
        """
        Calculate returns for all indices from the asset matrix.
        
        Parameters:
        -----------
        start_idx : int
            Start index in the market data
        end_idx : int
            End index in the market data
            
        Returns:
        --------
        numpy.ndarray
            Matrix of returns with shape (num_indices, num_periods)
        """
        # Extract asset prices for domestic and foreign indices
        price_matrix = self.market_data.asset_matrix[start_idx:end_idx+1, :]
        
        # Calculate returns: r_t = price_t / price_{t-1} - 1
        returns = np.zeros((price_matrix.shape[1], price_matrix.shape[0] - 1))
        
        for i in range(price_matrix.shape[1]):  # Loop through each index
            for t in range(0, price_matrix.shape[0]-1):  # Loop through time
                # Avoid division by zero
                if price_matrix[t, i] > 0:
                    returns[i, t] = np.log(price_matrix[t+1, i] / price_matrix[t, i])
        
        return returns
    
    def _calculate_fx_returns(self, start_idx, end_idx):
        """
        Calculate returns for foreign exchange rates.
        
        Parameters:
        -----------
        start_idx : int
            Start index in the market data
        end_idx : int
            End index in the market data
            
        Returns:
        --------
        numpy.ndarray
            Matrix of returns with shape (num_currencies, num_periods)
        """
        # Extract FX rates for foreign currencies
        fx_matrix = self.market_data.currency_matrix[start_idx:end_idx+1, :]
        
        # Calculate returns: r_t = fx_t / fx_{t-1} - 1
        returns = np.zeros((fx_matrix.shape[1], fx_matrix.shape[0] - 1))
        
        for i in range(fx_matrix.shape[1]):  # Loop through each currency
            for t in range(0, fx_matrix.shape[0]-1):  # Loop through time
                # Avoid division by zero
                if fx_matrix[t, i] > 0:
                    returns[i, t] = np.log(fx_matrix[t+1, i] / fx_matrix[t, i])
        
        return returns
    
    def _calculate_volatilities_from_returns(self, returns):
        """
        Calculate annualized volatilities from returns.
        
        Parameters:
        -----------
        returns : numpy.ndarray
            Matrix of returns with shape (num_assets, num_periods)
            
        Returns:
        --------
        numpy.ndarray
            Array of annualized volatilities
        """
        # Calculate the annualization factor based on actual number of periods
        # If less than a full year of data is available
        annualization_factor = np.sqrt(self.trading_days_per_year)
        
        # Calculate standard deviation of returns for each asset
        volatilities = np.std(returns, axis=1) * annualization_factor
        
        # Handle any zero volatilities (set to reasonable default)
        volatilities[volatilities == 0] = 0.2
        
        return volatilities
    
    def _calculate_correlation_matrix(self, returns_indices, returns_fx):
        """
        Calculate correlation matrix from returns.
        
        Parameters:
        -----------
        returns_indices : numpy.ndarray
            Matrix of index returns
        returns_fx : numpy.ndarray
            Matrix of FX returns
            
        Returns:
        --------
        numpy.ndarray
            Correlation matrix
        """
        # Combine returns in the correct order
        combined_returns = np.vstack([returns_indices, returns_fx])
        
        # Calculate correlation matrix
        corr_matrix = np.corrcoef(combined_returns)
        
        # Fix any NaN values (replace with zeros)
        corr_matrix = np.nan_to_num(corr_matrix)
        
        # Ensure the correlation matrix is positive definite
        corr_matrix = self._ensure_positive_definite(corr_matrix)
        
        return corr_matrix
    
    def _ensure_positive_definite(self, matrix, epsilon=1e-6):
        """
        Ensure a matrix is positive definite by adding a small value to the diagonal if needed.
        
        Parameters:
        -----------
        matrix : numpy.ndarray
            Input matrix
        epsilon : float
            Small value to add to diagonal
            
        Returns:
        --------
        numpy.ndarray
            Positive definite matrix
        """
        # Check eigenvalues
        eigenvalues = np.linalg.eigvals(matrix)
        
        # If all eigenvalues are positive, matrix is already positive definite
        if np.all(eigenvalues > 0):
            return matrix
        
        # Add a small value to the diagonal to make it positive definite
        n = matrix.shape[0]
        result = matrix.copy()
        for i in range(n):
            result[i, i] += epsilon
        
        return result
    
    def _calculate_cholesky_matrix(self, correlation_matrix):
        """
        Calculate Cholesky decomposition of the correlation matrix.
        
        Parameters:
        -----------
        correlation_matrix : numpy.ndarray
            Correlation matrix
            
        Returns:
        --------
        numpy.ndarray
            Cholesky matrix
        """
        try:
            return np.linalg.cholesky(correlation_matrix)
        except np.linalg.LinAlgError:
            # If Cholesky fails, ensure the matrix is positive definite and try again
            adjusted_matrix = self._ensure_positive_definite(correlation_matrix, epsilon=1e-4)
            return np.linalg.cholesky(adjusted_matrix)
    
    def get_parameters(self, current_date=None):
        """
        Get the calculated parameters. If current_date is provided and parameters
        haven't been calculated yet or were calculated for a different date,
        return None to indicate recalculation is needed.
        
        Parameters:
        -----------
        current_date : datetime, optional
            Current date to check if parameters need recalculation
            
        Returns:
        --------
        tuple or None
            (volatilities, correlation_matrix, cholesky_matrix) or None if recalculation needed
        """
        if self.volatilities is None or self.correlation_matrix is None or self.cholesky_matrix is None:
            return None
        
        if current_date is not None and self.last_calculation_date != current_date:
            # Check if current_date is a key observation date
            is_key_date = False
            for key_name in ['T0', 'T1', 'T2', 'T3', 'T4', 'Tc']:
                if current_date == self.date_handler.get_key_date(key_name):
                    is_key_date = True
                    break
            
            # If it's a key date, indicate recalculation is needed
            if is_key_date:
                return None
        
        return (self.volatilities, self.correlation_matrix, self.cholesky_matrix)
    
    def print_parameters(self):
        """
        Print the calculated parameters in a clear format.
        """
        if self.volatilities is None or self.correlation_matrix is None:
            print("Parameters have not been calculated yet.")
            return
        
        # Print volatilities
        print("\nVolatilities:")
        
        # Domestic indices
        for i, idx in enumerate(self.domestic_indices):
            print(f"  {idx}: {self.volatilities[i]:.4f}")
        
        # Foreign indices
        offset = len(self.domestic_indices)
        for i, idx in enumerate(self.foreign_indices):
            print(f"  {idx}: {self.volatilities[offset + i]:.4f}")
        
        # Currencies
        offset = len(self.domestic_indices) + len(self.foreign_indices)
        for i, idx in enumerate(self.foreign_indices):
            currency = self.market_data.index_currencies[idx]
            print(f"  FX_{currency}: {self.volatilities[offset + i]:.4f}")
        
        # Print correlation matrix summary
        print("\nCorrelation Matrix Summary:")
        print(f"  Size: {self.correlation_matrix.shape[0]}x{self.correlation_matrix.shape[1]}")
        print(f"  Min: {np.min(self.correlation_matrix):.4f}")
        print(f"  Max: {np.max(self.correlation_matrix):.4f}")
        print(f"  Mean: {np.mean(self.correlation_matrix):.4f}")