from simulation import Experiment
from utils import save_results, check_numba_compatibility
import plotting
import os

if __name__ == "__main__":
    check_numba_compatibility()
    
    N = 20
    experiment_runner = Experiment(name="Bayes Experiment")
    
    print(f"Running Bayes Experiment with N={N}...")
    experiment_runner.run_experiment_bayes(N=N)
    
    save_results('experiment_results.json', experiment_runner.results)
    
    plotting.plot_bayes_results(experiment_runner.results)
