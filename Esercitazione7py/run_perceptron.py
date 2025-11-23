from simulation import Experiment
from utils import save_results, check_numba_compatibility
import plotting
import os

if __name__ == "__main__":
    check_numba_compatibility()
    
    N = 50
    experiment_runner = Experiment(name="Perceptron Experiment")
    
    print(f"Running Perceptron Experiment with N={N}...")
    experiment_runner.run_experiment_perceptron_noise()
    experiment_runner.run_experiment_perceptron_zero_noise(N=N)
    
    save_results('experiment_results.json', experiment_runner.results)
    
    plotting.plot_advanced_comparison(experiment_runner.results)
