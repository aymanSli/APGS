# test_prod11.py
from datetime import datetime
import os

# Import the required components
from dates import DateHandler
from market_data import MarketData
from past_data import PastData
from Product11 import Product11
from simulation import Simulation

def simple_test_product11():
    print("=== SIMPLE PRODUCT 11 TEST ===\n")
    
    # Specify the data file path
    data_file_path = "DonneesGPS2025.xlsx"
    
    # Ensure the file exists
    if not os.path.exists(data_file_path):
        print(f"Error: Data file not found at {data_file_path}")
        return
    
    # 1. Initialize the DateHandler
    date_handler = DateHandler(data_file_path)
    
    # 2. Set the key dates (T0-Tc)
    key_dates = {
        'T0': datetime(2009, 1, 5),
        'T1': datetime(2010, 1, 4),
        'T2': datetime(2011, 1, 4),
        'T3': datetime(2012, 1, 4),
        'T4': datetime(2013, 1, 4),
        'Tc': datetime(2014, 1, 6)
    }
    date_handler.set_key_dates(key_dates)
    
    # 3. Initialize MarketData
    market_data = MarketData(data_file_path, date_handler)
    
    # 4. Initialize PastData to create the path
    past_data = PastData(market_data, date_handler)
    simulation = Simulation(market_data, date_handler)
    
    # 5. Initialize the path with data for all key dates
    # Use the final date (Tc) to get all historical data
    final_date = date_handler.get_key_date('Tc')
    path = past_data.initialize_past_matrix(final_date)
    # path_up = simulation.shift_path(
    #             path, 3, 1 + 0.5, date_handler.get_key_date('T0')
    #         )
    path_down = simulation.shift_path(
                path, 3, 1 - 0.2, date_handler.get_key_date('T0')
            )
    print("\n=== TEST NORM ===\n")
    print(path)
    print("\n=== TEST DOWN ===\n")
    print(path_down)
    
    print(f"Path matrix created with shape: {path.shape}")
    
    # 6. Create Product11 instance
    product = Product11(market_data, date_handler)
    
    # # 7. Calculate payoff using the path
    # print("\nCalculating total payoff...\n")
    # payoff_result = product.calculate_total_payoff(path)
    
    # 8. Print the results
    print("=== PRODUCT 11 RESULTS ===\n")
    
    # # Print dividends for each observation date
    # print("DIVIDENDS:")
    # total_dividends = 0
    # for i in range(1, 5):
    #     t_i = f"T{i}"
    #     dividend_info = payoff_result['dividends'][t_i]
    #     amount = dividend_info['amount']
    #     best_index = dividend_info['best_index']
    #     best_return = dividend_info['best_return']
        
    #     print(f"  {t_i}: {amount:.2f}€ (Best index: {best_index}, Return: {best_return*100:.2f}%)")
    #     total_dividends += amount
    
    # # Print final details
    # print(f"\nTotal dividends: {total_dividends:.2f}€")
    # print(f"Final performance: {payoff_result['final_performance']*100:.2f}%")
    # print(f"Final payoff at maturity: {payoff_result['final_payoff']:.2f}€")
    # print(f"Total payoff (dividends + final): {payoff_result['total_payoff']:.2f}€")
    
    # print(f"\nMinimum guarantee activated: {payoff_result['guarantee_activated']}")
    current_date=datetime(2013, 1, 4)
    print("\n=== TEST NORM ===\n")
    lifecycle=product.simulate_product_lifecycle(path,current_date)
    product.print_product_lifecycle(lifecycle)
    # print("\n=== TEST UP ===\n")
    # lifecycle1=product.simulate_product_lifecycle(path_up,current_date)
    # product.print_product_lifecycle(lifecycle1)
    print("\n=== TEST DOWN ===\n")
    lifecycle2=product.simulate_product_lifecycle(path_down,current_date)
    product.print_product_lifecycle(lifecycle2)
    print("\n=== TEST COMPLETE ===")

if __name__ == "__main__":
    simple_test_product11()