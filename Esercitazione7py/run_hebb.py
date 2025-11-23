from simulation import Experiment
from utils import save_results, check_numba_compatibility
import plotting
import os

if __name__ == "__main__":
    check_numba_compatibility()
    
    N = 50
    experiment_runner = Experiment(name="Hebb Experiment")
    
    print(f"Running Hebb Experiment with N={N}...")
    experiment_runner.run_experiment_hebb(N=N)
    
    save_results('experiment_results.json', experiment_runner.results)
    
    plotting.plot_hebb_results(experiment_runner.results)
