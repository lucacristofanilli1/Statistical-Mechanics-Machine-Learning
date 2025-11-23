from simulation import Experiment
from utils import save_results, check_numba_compatibility
import plotting
import os

if __name__ == "__main__":
    check_numba_compatibility()
    
    N = 20
    experiment_runner = Experiment(name="Pseudo-Inverse Experiment")
    
    print(f"Running Pseudo-Inverse Experiment with N={N}...")
    experiment_runner.run_experiment_pseudo_inverse(N=N)
    
    save_results('experiment_results.json', experiment_runner.results)
    
    plotting.plot_pseudo_inverse_results(experiment_runner.results)
