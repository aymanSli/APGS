#!/usr/bin/env python
# test_structured_product_flow.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import matplotlib.dates as mdates
from pathlib import Path
import sys

# Add project root to the Python path
project_path = Path(__file__).resolve().parent
if str(project_path) not in sys.path:
    sys.path.append(str(project_path))

# Import the modules from our project
from dates import DateHandler
from market_data import MarketData
from past_data import PastData
from sim_params import SimulationParameters
from simulation import Simulation
from Product11 import Product11
from montecarlo import MonteCarlo
from portoflio import Portfolio

# Configuration
DATA_FILE_PATH = "DonneesGPS2025.xlsx"
SEED = 42
NUM_SIMULATIONS = 1000
REBALANCING_FREQUENCY = 30  # Days between rebalancing
INITIAL_CAPITAL = 1000.0  # €1,000,000
OUTPUT_DIR = "results"

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

def setup_test_environment():
    """Initialize all components needed for the test."""
    print("Setting up test environment...")
    
    # Initialize date handler
    date_handler = DateHandler(DATA_FILE_PATH)
    
    # Set key dates for Product 11
    key_dates = {
        'T0': datetime(2009, 1, 5),   # Initial date
        'T1': datetime(2010, 1, 4),   # First observation date
        'T2': datetime(2011, 1, 4),   # Second observation date
        'T3': datetime(2012, 1, 4),   # Third observation date
        'T4': datetime(2013, 1, 4),   # Fourth observation date
        'Tc': datetime(2014, 1, 6)    # Final date
    }
    date_handler.set_key_dates(key_dates)
    
    # Initialize market data
    market_data = MarketData(DATA_FILE_PATH)
    
    # Initialize product
    product = Product11(market_data, date_handler)
    
    # Print information about the initialized components
    print("\nTest environment setup complete!")
    print(f"Data file: {DATA_FILE_PATH}")
    
    print("\nKey Dates:")
    for name, idx in date_handler.key_dates.items():
        print(f"  {name}: {date_handler.dates[idx].strftime('%Y-%m-%d')}")
    
    print("\nIndices:")
    for idx_name in market_data.indices:
        print(f"  {idx_name} ({market_data.index_currencies[idx_name]})")
    
    return date_handler, market_data, product

def run_simulation(date_handler, market_data, product, current_date):
    """Run a Monte Carlo simulation for the given date."""
    print(f"\nRunning simulation for date: {current_date.strftime('%Y-%m-%d')}...")
    
    # Get current date index
    current_date_index = date_handler.get_index_from_date(current_date)
    
    # Initialize past data
    past_data = PastData(market_data, date_handler)
    past_matrix = past_data.initialize_past_matrix('T0', current_date_index)
    
    # Calculate simulation parameters
    sim_params = SimulationParameters(market_data, date_handler)
    sim_params.calculate_parameters(current_date_index)
    
    # Initialize simulation
    simulation = Simulation(market_data, date_handler, sim_params)
    
    # Generate paths
    paths = simulation.generate_paths(
        past_matrix=past_matrix,
        current_date_index=current_date_index,
        num_simulations=NUM_SIMULATIONS,
        seed=SEED
    )
    
    # Initialize Monte Carlo
    monte_carlo = MonteCarlo(product, simulation, NUM_SIMULATIONS)
    
    # Calculate deltas using Monte Carlo
    deltas = monte_carlo.calculate_deltas(past_matrix, current_date_index, SEED)
    
    print(f"Simulation complete. Generated {NUM_SIMULATIONS} paths.")
    
    return past_matrix, paths, deltas

def run_portfolio_hedging(date_handler, market_data, product, past_matrix, deltas, current_date, portfolio=None):
    """Run portfolio hedging based on calculated deltas."""
    print(f"\nRunning portfolio hedging for date: {current_date.strftime('%Y-%m-%d')}...")
    
    # Get current date index
    current_date_index = date_handler.get_index_from_date(current_date)
    
    # Initialize portfolio if needed
    if portfolio is None:
        portfolio = Portfolio(market_data, date_handler, product, INITIAL_CAPITAL)
        portfolio.initialize_portfolio(current_date_index, deltas)
    else:
        # Update portfolio to current date
        portfolio.update_to_date(current_date_index)
        
        # Check if rebalancing is needed
        days_since_last_rebalance = (current_date - portfolio.last_rebalance_date).days
        
        if days_since_last_rebalance >= REBALANCING_FREQUENCY:
            print(f"Rebalancing portfolio (last rebalance: {portfolio.last_rebalance_date.strftime('%Y-%m-%d')})")
            portfolio.rebalance_portfolio(deltas)
    
    # Process any payments due on observation dates
    payment_info = portfolio.process_payment_date(past_matrix)
    if payment_info:
        print(f"Processed payment: €{payment_info['amount']:.2f} on {payment_info['key']}")
    
    # Get portfolio summary
    portfolio_state = portfolio.get_portfolio_state()
    
    print(f"Portfolio total value: €{portfolio_state['total_value']:,.2f}")
    print(f"Cash: €{portfolio_state['cash']:,.2f}")
    print(f"Position value: €{sum(portfolio_state['values'].values()):,.2f}")
    
    return portfolio

def simulate_product_lifecycle():
    """Run a complete simulation of the product lifecycle."""
    print("\n=== STARTING STRUCTURED PRODUCT LIFECYCLE SIMULATION ===\n")
    
    # Setup environment
    date_handler, market_data, product = setup_test_environment()
    
    # Get key dates
    start_date = date_handler.get_key_date('T0')
    end_date = date_handler.get_key_date('Tc')
    observation_dates = [date_handler.get_key_date(f'T{i}') for i in range(1, 5)]
    
    # Create tracking variables
    portfolio = None
    portfolio_values = []
    dates = []
    deltas_history = {}
    
    # Determine simulation dates (monthly steps)
    current_date = start_date
    sim_dates = []
    
    while current_date <= end_date:
        sim_dates.append(current_date)
        current_date = min(end_date, current_date + timedelta(days=30))
    
    # Include observation dates if they're not already included
    for obs_date in observation_dates:
        if obs_date not in sim_dates:
            sim_dates.append(obs_date)
    
    # Sort dates
    sim_dates.sort()
    
    # Run simulation for each date
    for current_date in sim_dates:
        print(f"\n{'='*50}")
        print(f"Processing date: {current_date.strftime('%Y-%m-%d')}")
        
        # Run simulation
        past_matrix, paths, deltas = run_simulation(date_handler, market_data, product, current_date)
        
        # Store deltas for later analysis
        deltas_history[current_date] = deltas
        
        # Portfolio hedging
        portfolio = run_portfolio_hedging(date_handler, market_data, product, past_matrix, deltas, current_date, portfolio)
        
        # Store portfolio value for tracking
        portfolio_state = portfolio.get_portfolio_state()
        portfolio_values.append(portfolio_state['total_value'])
        dates.append(current_date)
        
        # Special handling for observation dates
        is_observation_date = current_date in observation_dates
        if is_observation_date:
            print(f"Observation date T{observation_dates.index(current_date) + 1} reached!")
            
            # Calculate product lifecycle status
            t0_index = date_handler.get_key_date_index('T0')
            current_date_index = date_handler.get_index_from_date(current_date)
            current_row = current_date_index - t0_index
            
            # Extract current path from past matrix
            path = past_matrix[:current_row+1, :]
            
            # Show dividend information
            if len(path) > 0:
                t_i = f"T{observation_dates.index(current_date) + 1}"
                prev_key = f"T{int(t_i[1]) - 1}" if int(t_i[1]) > 1 else "T0"
                
                # Reset excluded indices to ensure correct calculation
                product.excluded_indices = portfolio.product.excluded_indices.copy()
                
                # Calculate dividend
                dividend, best_index, best_return = product.calculate_dividend(t_i, path)
                
                print(f"Dividend payment: €{dividend:.2f}")
                print(f"Best performing index: {best_index} ({best_return*100:.2f}%)")
                print(f"Excluded indices: {product.excluded_indices}")
                
                # Check for guarantee activation
                guarantee_activated = product.check_minimum_guarantee(t_i, path)
                if guarantee_activated and not product.guarantee_activated:
                    print(f"Minimum guarantee of 20% activated!")
    
    # Create final reports
    create_portfolio_evolution_chart(dates, portfolio_values)
    create_delta_evolution_chart(dates, deltas_history)
    create_final_summary(portfolio, product, date_handler, market_data)
    
    print("\n=== STRUCTURED PRODUCT LIFECYCLE SIMULATION COMPLETE ===\n")

def create_portfolio_evolution_chart(dates, values):
    """Create a chart showing portfolio value evolution over time."""
    plt.figure(figsize=(12, 6))
    plt.plot(dates, values, 'b-', linewidth=2)
    
    # Format the chart
    plt.title('Portfolio Value Evolution', fontsize=14)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Portfolio Value (€)', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Format x-axis
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=45)
    
    # Format y-axis with commas for thousands
    plt.gca().yaxis.set_major_formatter(plt.matplotlib.ticker.StrMethodFormatter('{x:,.0f}'))
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/portfolio_evolution.png", dpi=300)
    print(f"Portfolio evolution chart saved to: {OUTPUT_DIR}/portfolio_evolution.png")

def create_delta_evolution_chart(dates, deltas_history):
    """Create charts showing delta evolution for key assets."""
    # Extract delta values for specific assets
    assets_to_plot = []
    
    # Find which assets have data in the first entry
    first_date = list(deltas_history.keys())[0]
    first_deltas = deltas_history[first_date]
    
    # Find domestic indices, foreign indices and currencies with data
    domestic_assets = [key for key in first_deltas.keys() if key.startswith('DOM_')]
    foreign_assets = [key for key in first_deltas.keys() if key.startswith('SX_')]
    currencies = [key for key in first_deltas.keys() if key.startswith('X_')]
    
    # Select assets to plot (take first 2 from each category if available)
    assets_to_plot = (
        domestic_assets[:min(2, len(domestic_assets))] +
        foreign_assets[:min(2, len(foreign_assets))] +
        currencies[:min(2, len(currencies))]
    )
    
    # Create figure with subplots for each asset
    fig, axes = plt.subplots(len(assets_to_plot), 1, figsize=(12, 3 * len(assets_to_plot)), sharex=True)
    
    if len(assets_to_plot) == 1:
        axes = [axes]  # Make sure axes is a list for single asset
    
    # Plot delta evolution for each asset
    for i, asset in enumerate(assets_to_plot):
        asset_deltas = [deltas_history[date].get(asset, 0) for date in dates]
        axes[i].plot(dates, asset_deltas, 'g-', linewidth=2)
        
        # Format subplot
        axes[i].set_title(f'Delta Evolution: {asset}', fontsize=12)
        axes[i].set_ylabel('Delta', fontsize=10)
        axes[i].grid(True, alpha=0.3)
        
        # Add zero line
        axes[i].axhline(y=0, color='r', linestyle='--', alpha=0.7)
    
    # Format x-axis
    axes[-1].set_xlabel('Date', fontsize=10)
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/delta_evolution.png", dpi=300)
    print(f"Delta evolution chart saved to: {OUTPUT_DIR}/delta_evolution.png")

def create_final_summary(portfolio, product, date_handler, market_data):
    """Create a final summary report of the product and portfolio."""
    # Get portfolio performance summary
    performance = portfolio.get_performance_summary()
    
    # Calculate product final payoff
    t0_index = date_handler.get_key_date_index('T0')
    tc_index = date_handler.get_key_date_index('Tc')
    
    # Initialize past data up to Tc
    past_data = PastData(market_data, date_handler)
    past_matrix = past_data.initialize_past_matrix('T0', tc_index)
    
    # Calculate final payoff
    final_payoff_info = product.calculate_total_payoff(past_matrix)
    
    # Create summary report
    report = [
        "=== STRUCTURED PRODUCT FINAL SUMMARY ===",
        "",
        f"Product Name: Performance Monde (Product 11)",
        f"Duration: {(date_handler.get_key_date('Tc') - date_handler.get_key_date('T0')).days} days",
        f"Investment Period: {date_handler.get_key_date('T0').strftime('%Y-%m-%d')} to {date_handler.get_key_date('Tc').strftime('%Y-%m-%d')}",
        "",
        "Product Parameters:",
        f"  Initial Value: €{product.initial_value:.2f}",
        f"  Participation Rate: {product.participation_rate*100:.1f}%",
        f"  Floor: {product.floor*100:.1f}%",
        f"  Cap: {product.cap*100:.1f}%",
        f"  Minimum Guarantee: {product.minimum_guarantee*100:.1f}%",
        f"  Dividend Multiplier: {product.dividend_multiplier:.1f}",
        "",
        "Product Final Performance:",
        f"  Guarantee Activated: {final_payoff_info['guarantee_activated']}",
        f"  Final Performance: {final_payoff_info['final_performance']*100:.2f}%",
        f"  Final Payoff: €{final_payoff_info['final_payoff']:.2f}",
        f"  Total Dividends: €{final_payoff_info['total_dividends']:.2f}",
        f"  Total Payoff: €{final_payoff_info['total_payoff']:.2f}",
        "",
        "Portfolio Performance:",
        f"  Initial Capital: €{performance['initial_value']:,.2f}",
        f"  Final Value: €{performance['current_value']:,.2f}",
        f"  Total Return: {performance['total_return']:.2f}%",
        f"  Annualized Return: {performance['annualized_return']:.2f}%",
        f"  Total Payments: €{performance['total_payments']:,.2f}",
        "",
        "Portfolio Composition at Maturity:",
        f"  Cash: €{portfolio.cash:,.2f} ({performance['cash_percentage']:.1f}%)",
        f"  Positions Value: €{performance['positions_value']:,.2f} ({performance['positions_percentage']:.1f}%)",
        f"  Number of Positions: {performance['num_positions']}",
        f"  Number of Trades: {performance['num_trades']}",
        "",
        "Tracking Error:",
        f"  Portfolio Value - Product Payoff: €{performance['current_value'] - final_payoff_info['total_payoff']:,.2f}"
    ]
    
    # Save report to file
    with open(f"{OUTPUT_DIR}/final_summary.txt", "w") as f:
        f.write("\n".join(report))
    
    print(f"Final summary report saved to: {OUTPUT_DIR}/final_summary.txt")
    
    # Print report to console
    print("\n" + "\n".join(report))

if __name__ == "__main__":
    np.random.seed(SEED)  # Set seed for reproducibility
    simulate_product_lifecycle()