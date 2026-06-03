import itertools
import subprocess
import os

def run_all_experiments():
    """
    Runs main.py with different combinations of N and X.
    """
    N_values = [15, 30, 60]
    X_values = [5, 10, 20]
    
    for N, X in itertools.product(N_values, X_values):
        print(f\"\n--- Running experiment with N={N}, X={X} ---")
        
        # Here we would generate a temporary config file for the specific (N, X) combination
        # and pass it to main.py. For now, it's just a placeholder loop.
        
        # Example of how to call main.py:
        # subprocess.run([\"python\", \"main.py\", \"--config\", \"path_to_temp_config.yaml\"])
        pass

if __name__ == "__main__":
    run_all_experiments()
