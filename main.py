#!/usr/bin/env python3
# main.py - Interactive Structured Product Simulation
import os
import sys
import numpy as np
from datetime import datetime, timedelta
import time
import pandas as pd

# Import our classes
from dates import DateHandler
from market_data import MarketData
from past_data import PastData
from Product11 import Product11
from sim_params import SimulationParameters
from montecarlo import MonteCarlo
from portoflio import DeltaHedgingPortfolio
from simulation import Simulation  # Using the existing simulation.py

class StructuredProductSimulation:
    """
    Interactive simulation for managing a structured product throughout its lifecycle.
    
    Provides terminal interface to simulate the day-to-day management of a
    delta-hedged structured product portfolio, including key date events,
    dividend payments, and portfolio rebalancing.
    """
    
    def __init__(self, data_file_path):
        """Initialize the simulation with the path to the market data Excel file."""
        self.data_file_path = data_file_path
        
        # Display initialization message
        print("\n======================================================")
        print("      STRUCTURED PRODUCT MANAGEMENT SIMULATION")
        print("======================================================\n")
        print("Initializing components...")
        
        # Initialize all components
        self._init_components()
        
        # Setup simulation variables
        self.running = True
        
        
        # Track important information
        self.excluded_indices = []
        self.guarantee_triggered = False
        self.dividends_paid = []
        
        # Configure display formatting
        self.date_format = "%Y-%m-%d"
        
        print("\nAll components initialized.")
        print("Starting simulation at date:", self.current_date.strftime(self.date_format))
        print("======================================================\n")
        
    def _init_components(self):
        """Initialize all simulation components."""
        
        # Initialize date handler
        print("- Initializing DateHandler...")
        self.date_handler = DateHandler(self.data_file_path)
        
        # Set key dates (T0, T1, T2, T3, T4, Tc)
        t0 = datetime(2009, 1, 5)
        t1 = datetime(2010, 1, 4)
        t2 = datetime(2011, 1, 4)
        t3 = datetime(2012, 1, 4)
        t4 = datetime(2013, 1, 4)
        tc = datetime(2014, 1, 6)
        
        key_dates = {
            'T0': t0,
            'T1': t1,
            'T2': t2,
            'T3': t3,
            'T4': t4,
            'Tc': tc
        }
        self.date_handler.set_key_dates(key_dates)
        
        # Initialize market data
        print("- Initializing MarketData...")
        self.market_data = MarketData(self.data_file_path, self.date_handler)
        
        # Initialize past data matrix
        print("- Initializing PastData...")
        self.past_data = PastData(self.market_data, self.date_handler)
        
        # Initialize product
        print("- Initializing Product11...")
        self.product = Product11(self.market_data, self.date_handler)
        
        # Initialize simulation parameters
        print("- Initializing SimulationParameters...")
        self.sim_params = SimulationParameters(self.market_data, self.date_handler)
        
        # Initialize simulation engine
        print("- Initializing Simulation...")
        self.simulation = Simulation(self.market_data, self.date_handler)
        
        # Initialize Monte Carlo
        print("- Initializing MonteCarlo...")
        self.monte_carlo = MonteCarlo(self.date_handler, self.product, self.simulation, self.sim_params)
        
        self.current_date = self.date_handler.key_dates['T0']
        self.last_rebalancing_date=self.date_handler.key_dates['T0']
        
        # Initialize the past matrix with data up to T0
        self.past_matrix = self.past_data.initialize_past_matrix(self.current_date)
        # print(self.past_matrix)
        
        # Get the domestic interest rate
        date_index = self.market_data.get_date_index(self.current_date)
        self.risk_free_rate = self.market_data.get_interest_rate('EUR', date_index)
        
        # Initialize portfolio with initial product price (1000€)
        print("- Initializing Portfolio...")
        self.portfolio = DeltaHedgingPortfolio(1000.0,self.date_handler,self.market_data)
        
        # Initial rebalance at T0
        initial_current_prices = self.past_data.get_spot_prices()
        self.product.update_interest_rates(self.current_date)
        
        # Calculate initial deltas
        self.deltas = self.monte_carlo.calculate_deltas(
            self.past_matrix, 
            self.current_date
        )
        # print(self.deltas)
        
        # Perform initial rebalance
        self.portfolio.rebalance(
            self.deltas,
            initial_current_prices,
            self.current_date,
            self.last_rebalancing_date,
            self.risk_free_rate
        )
    
    def run(self):
        """Run the main simulation loop."""
        while self.running:
            self.display_status()
            self.get_user_action()
    
    def display_status(self):
        """Display current simulation status."""
        
        # Check if we're at a key date
        is_key, key_name = self.date_handler.is_key_date(self.current_date)
        key_status = f"[KEY DATE: {key_name}]" if is_key else ""
        
        print("\n" + "=" * 60)
        print(f"  CURRENT DATE: {self.current_date.strftime(self.date_format)} {key_status}")
        print("=" * 60)
        
        # Display portfolio status
        spot_prices = self.past_data.get_spot_prices()
        portfolio_state = self.portfolio.get_portfolio_state(spot_prices)
        
        print("\nPORTFOLIO STATUS:")
        print(f"  Cash: €{portfolio_state['cash']:.2f}")
        print(f"  Total Position Value: €{portfolio_state['total_position_value']:.2f}")
        print(f"  Total Portfolio Value: €{portfolio_state['total_value']:.2f}")
        print(f"  Liquidative Value: €{self.monte_carlo.price:.2f}")
        print(f"  PnL: €{portfolio_state['total_value']-self.monte_carlo.price:.2f}")
        print(f"  Interest rate: {self.risk_free_rate*100} %")
        
        
        # Display product status
        print("\nPRODUCT STATUS:")
        
        if self.excluded_indices:
            print(f"  Excluded Indices: {', '.join(self.excluded_indices)}")
        else:
            print("  Excluded Indices: None")
            
        print(f"  Minimum Guarantee Triggered: {'Yes' if self.guarantee_triggered else 'No'}")
        
        if self.dividends_paid:
            print("\n  Dividends Paid:")
            for div in self.dividends_paid:
                print(f"    {div['date'].strftime(self.date_format)}: €{div['amount']:.2f} ({div['index']})")
        
        # Display next key date
        next_key_date = self.date_handler.get_next_key_date(self.current_date)
        if next_key_date:
            days_to_next = self.date_handler._count_trading_days(self.current_date,next_key_date)
            print(f"\nNext key date in {days_to_next} days")
    
    def get_user_action(self):
        """Get and process user action."""
        print("\nACTIONS:")
        print("  1. Move forward X days")
        print("  2. Jump to next key date")
        print("  3. Rebalance portfolio")
        print("  4. View detailed portfolio positions")
        print("  5. View detailed product information")
        print("  0. Exit simulation")
        
        try:
            choice = input("\nSelect an action (0-5): ")
            
            if choice == '0':
                self.running = False
                print("\nExiting simulation. Final results:")
                self.display_final_results()
                
            elif choice == '1':
                days = int(input("Enter number of days to advance: "))
                self.advance_days(days)
                
            elif choice == '2':
                self.jump_to_next_key_date()
                
            elif choice == '3':
                self.rebalance_portfolio()
                
            elif choice == '4':
                self.display_detailed_positions()
                
            elif choice == '5':
                self.display_detailed_product_info()
                
            else:
                print("Invalid choice. Please try again.")
                
        except Exception as e:
            print(f"Error: {e}")
    
    def advance_days(self, days):
        """Advance the simulation by a specified number of days."""
        if days <= 0:
            print("Number of days must be positive.")
            return
        
        curr_index=self.date_handler.get_date_index(self.current_date)
        next_index = curr_index+days
        next_date = self.date_handler.get_date_from_index(next_index)
        
        # Check for key dates in between
        key_dates_between = []
        for key, date in self.date_handler.key_dates.items():
            if self.current_date < date <= next_date and key != 'T0':  # Skip T0
                key_dates_between.append((key, date))
        
        # Sort by date
        key_dates_between.sort(key=lambda x: x[1])
        
        if key_dates_between:
            print(f"\nWarning: There are {len(key_dates_between)} key dates between now and {next_date.strftime(self.date_format)}.")
            print("The simulation will stop at each key date for processing.")
            proceed = input("Proceed? (y/n): ").lower()
            
            if proceed != 'y':
                return
                
            # Process each key date in order
            for key, date in key_dates_between:
                print(f"\nAdvancing to key date {key}: {date.strftime(self.date_format)}...")
                self.current_date = date
                self.update_past_matrix()
                date_index = self.market_data.get_date_index(self.current_date)
                self.risk_free_rate = self.market_data.get_interest_rate('EUR', date_index)
                self.process_key_date(key)
                
            # If next_date is after all key dates, advance to it
            if next_date > key_dates_between[-1][1]:
                print(f"\nAdvancing to {next_date.strftime(self.date_format)}...")
                self.current_date = next_date
                self.update_past_matrix()
                date_index = self.market_data.get_date_index(self.current_date)
                self.risk_free_rate = self.market_data.get_interest_rate('EUR', date_index)
                
        else:
            # No key dates in between, just advance to next_date
            print(f"\nAdvancing to {next_date.strftime(self.date_format)}...")
            self.current_date = next_date
            self.update_past_matrix()
            date_index = self.market_data.get_date_index(self.current_date)
            self.risk_free_rate = self.market_data.get_interest_rate('EUR', date_index)
    
    def jump_to_next_key_date(self):
        """Jump directly to the next key date."""
        next_key_date = self.date_handler.get_next_key_date(self.current_date)
        
        if not next_key_date:
            print("No more key dates. The product has matured.")
            return
            
        # Find the key name for this date
        key_name = None
        for key, date in self.date_handler.key_dates.items():
            if date == next_key_date:
                key_name = key
                break
        
        print(f"\nJumping to key date {key_name}: {next_key_date.strftime(self.date_format)}...")
        self.current_date = next_key_date
        self.update_past_matrix()
        date_index = self.market_data.get_date_index(self.current_date)
        self.risk_free_rate = self.market_data.get_interest_rate('EUR', date_index)
        self.process_key_date(key_name)
    
    def update_past_matrix(self):
        """Update the past matrix to the current date."""
        self.past_matrix = self.past_data.initialize_past_matrix(self.current_date)
    
    def process_key_date(self, key_name):
        """Process events at a key date."""
        print(f"\nProcessing key date {key_name}...")
        
        
        if key_name == 'Tc':
            # Final maturity
            print("\nFinal maturity reached.")
            
            # Calculate final product payoff
            payoff = self.product.calculate_final_payoff(self.past_matrix)
            
            print(f"Final payoff: €{payoff:.2f}")
            
            # Process final payment (unwinding the portfolio)
            spot_prices = self.past_data.get_spot_prices()
            self.product.update_interest_rates(self.current_date)
            
            
            # Unwind portfolio
            unwind_result = self.portfolio.unwind_portfolio(
                spot_prices,
                self.current_date,
                self.last_rebalancing_date,
                self.risk_free_rate
            )
            
            print(f"Portfolio unwound. Final cash: €{unwind_result['cash']:.2f}")
            
            # Calculate final settlement
            final_settlement = unwind_result['cash'] - payoff
            
            print(f"Final settlement (payoff - portfolio value): €{final_settlement:.2f}")
            
            # End the simulation
            self.running = False
            
        else:
            # Calculate dividend for this key date
            exclude_indices = self.excluded_indices.copy()  # Save a copy of excluded indices
            
            # Calculate dividend
            dividend, best_index, best_return = self.product.calculate_dividend(
                key_name, 
                self.past_matrix,
                exclude_indices
            )
            
            # Update excluded indices
            if best_index and best_index not in self.excluded_indices:
                self.excluded_indices.append(best_index)
                print(f"Index {best_index} now excluded from future dividends.")
            
            # Check if minimum guarantee is triggered
            guarantee_check = self.product.check_minimum_guarantee(key_name, self.past_matrix)
            if guarantee_check and not self.guarantee_triggered:
                self.guarantee_triggered = True
                print("Minimum guarantee (20%) has been triggered!")
            
            # Process dividend payment
            if dividend > 0:
                spot_prices = self.past_data.get_spot_prices()
                payment = self.portfolio.process_dividend_payment(
                    dividend,
                    self.current_date,
                    spot_prices
                )
                
                # Record dividend payment
                self.dividends_paid.append({
                    'date': self.current_date,
                    'key': key_name,
                    'amount': dividend,
                    'index': best_index,
                    'return': best_return
                })
                
                print(f"Dividend paid: €{dividend:.2f} from best-performing index {best_index} (return: {best_return*100:.2f}%)")
            else:
                print("No dividend payment for this key date.")
            
            # Rebalance portfolio after key date event
            self.rebalance_portfolio(automatic=True)
    
    def rebalance_portfolio(self, automatic=False):
        """Rebalance the portfolio based on new deltas."""
        print("\nRebalancing portfolio...")
        
        # Calculate new deltas
        self.deltas = self.monte_carlo.calculate_deltas(
            self.past_matrix, 
            self.current_date
        )
        
        # Get spot prices
        spot_prices = self.past_data.get_spot_prices()
        self.product.update_interest_rates(self.current_date)
        
        
        # Perform rebalance
        rebalance_result = self.portfolio.rebalance(
            self.deltas,
            spot_prices,
            self.current_date,
            self.last_rebalancing_date,
            self.risk_free_rate
        )
        self.last_rebalancing_date=self.current_date
        
        print(f"Rebalance complete.")
        print(f"Cash after rebalance: €{rebalance_result['cash']:.2f}")
        print(f"Total value after rebalance: €{rebalance_result['final_value']:.2f}")
        print(f"P&L from rebalance: €{rebalance_result['pnl']:.2f}")
        
        if not automatic:
            # Print trades if manual rebalance
            trades = rebalance_result['trades']
            if trades:
                print("\nTrades executed:")
                for trade in trades:
                    print(f"  {trade['asset']}: Delta change {trade['delta_change']:.4f}, Value €{trade['value']:.2f}, Cost €{trade['cost']:.2f}")
            else:
                print("No trades executed.")
    
    def display_detailed_positions(self):
        """Display detailed portfolio positions."""
        spot_prices = self.past_data.get_spot_prices()
        portfolio_state = self.portfolio.get_portfolio_state(spot_prices)
        
        print("\n" + "=" * 60)
        print("  DETAILED PORTFOLIO POSITIONS")
        print("=" * 60)
        
        print(f"\nCash: €{portfolio_state['cash']:.2f}")
        print(f"  \nInterest rate: {self.risk_free_rate*100} %")
        print("\nDelta Positions:")
        for i in range (9):
            price = spot_prices[i]
            position_value = portfolio_state['position_values'][i]
            print(f"  {self.portfolio.column_names[i]}: Delta {self.portfolio.deltas[i]:.4f}, Price €{price:.2f}, Value €{position_value:.2f}")
        
        print(f"\nTotal Position Value: €{portfolio_state['total_position_value']:.2f}")
        print(f"Total Portfolio Value: €{portfolio_state['total_value']:.2f}")
        
        print("\nDividend History:")
        if self.dividends_paid:
            for div in self.dividends_paid:
                print(f"  {div['date'].strftime(self.date_format)} ({div['key']}): €{div['amount']:.2f} - {div['index']} (return: {div['return']*100:.2f}%)")
        else:
            print("  No dividends paid yet.")
        
        input("\nPress Enter to continue...")
    
    def display_detailed_product_info(self):
        """Display detailed product information."""
        print("\n" + "=" * 60)
        print("  DETAILED PRODUCT INFORMATION")
        print("=" * 60)
        
        # Simulate lifecycle with current data
        lifecycle = self.product.simulate_product_lifecycle(self.past_matrix, self.current_date)
        
        # Print the lifecycle info
        self.product.print_product_lifecycle(lifecycle)
        
        input("\nPress Enter to continue...")
    
    def display_final_results(self):
        """Display final simulation results."""
        print("\n" + "=" * 60)
        print("  SIMULATION FINAL RESULTS")
        print("=" * 60)
        
        print(f"\nSimulation ended at: {self.current_date.strftime(self.date_format)}")
        
        # Display final portfolio status
        spot_prices = self.past_data.get_spot_prices()
        portfolio_state = self.portfolio.get_portfolio_state(spot_prices)
        
        print("\nFinal Portfolio State:")
        print(f"  Cash: €{portfolio_state['cash']:.2f}")
        print(f"  Interest rate: {self.risk_free_rate*100} %")
        print(f"  Total Position Value: €{portfolio_state['total_position_value']:.2f}")
        print(f"  Total Portfolio Value: €{portfolio_state['total_value']:.2f}")
        
        # Display summary of dividend payments
        print("\nDividend Payment Summary:")
        if self.dividends_paid:
            total_dividends = sum(div['amount'] for div in self.dividends_paid)
            print(f"  Total Dividends Paid: €{total_dividends:.2f}")
            print("  Breakdown:")
            for div in self.dividends_paid:
                print(f"    {div['date'].strftime(self.date_format)} ({div['key']}): €{div['amount']:.2f} - {div['index']} (return: {div['return']*100:.2f}%)")
        else:
            print("  No dividends were paid during the simulation.")
        
        # Display excluded indices and guarantee status
        print("\nProduct Final State:")
        print(f"  Excluded Indices: {', '.join(self.excluded_indices) if self.excluded_indices else 'None'}")
        print(f"  Minimum Guarantee Triggered: {'Yes' if self.guarantee_triggered else 'No'}")


if __name__ == "__main__":
    # Check command line arguments for data file path
    data_file = "DonneesGPS2025.xlsx"
    
    if len(sys.argv) > 1:
        data_file = sys.argv[1]
    
    # Check if file exists
    if not os.path.exists(data_file):
        print(f"Error: Data file '{data_file}' not found.")
        print("Usage: python main.py [data_file.xlsx]")
        sys.exit(1)
    
    # Initialize and run simulation
    simulation = StructuredProductSimulation(data_file)
    simulation.run()