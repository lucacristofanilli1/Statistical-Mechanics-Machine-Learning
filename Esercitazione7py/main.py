import argparse
import os
import json
import numpy as np
from simulation import Experiment
from utils import save_results, load_results, check_numba_compatibility
import plotting

def main():
    parser = argparse.ArgumentParser(description="Statistical Mechanics of Machine Learning - Experiment Runner")
    parser.add_argument('--experiment', type=str, choices=['hebb', 'perceptron', 'adaline', 'pseudo_inverse', 'bayes', 'all'], help='Experiment to run')
    parser.add_argument('--N', type=int, default=20, help='Input dimension N')
    parser.add_argument('--runs', type=int, default=1000, help='Number of runs per point')
    parser.add_argument('--load', action='store_true', help='Load existing results instead of running new simulations')
    parser.add_argument('--plot', action='store_true', help='Plot results after execution/loading')
    parser.add_argument('--save_file', type=str, default='experiment_results.json', help='File to save/load results')
    
    args = parser.parse_args()

    # Check Numba
    check_numba_compatibility()

    experiment_runner = Experiment(name="StatMech ML Experiments")
    
    # Load existing results if requested or if we want to append
    if os.path.exists(args.save_file):
        print(f"Loading existing results from {args.save_file}...")
        loaded_results = load_results(args.save_file)
        # Convert lists to numpy arrays for compatibility
        for key, value in loaded_results.items():
            if isinstance(value, list):
                loaded_results[key] = np.array(value)
        experiment_runner.results.update(loaded_results)

    if args.load and not args.experiment:
        print("Loaded results. Use --plot to visualize or --experiment to run specific experiments.")
    
    if args.experiment:
        if args.experiment == 'hebb' or args.experiment == 'all':
            if not args.load:
                experiment_runner.run_experiment_hebb(N=args.N, runs_num=args.runs)
            if args.plot:
                plotting.plot_hebb_results(experiment_runner.results)

        if args.experiment == 'perceptron' or args.experiment == 'all':
            if not args.load:
                experiment_runner.run_experiment_perceptron_noise(runs_num=args.runs)
                experiment_runner.run_experiment_perceptron_zero_noise(N=args.N, runs_num=args.runs)
            if args.plot:
                # Note: Perceptron plotting might need specific function or use advanced comparison
                plotting.plot_advanced_comparison(experiment_runner.results)

        if args.experiment == 'adaline' or args.experiment == 'all':
            if not args.load:
                experiment_runner.run_experiment_adaline(N=args.N, runs_num=args.runs)
            if args.plot:
                plotting.plot_adaline_results(experiment_runner.results)

        if args.experiment == 'pseudo_inverse' or args.experiment == 'all':
            if not args.load:
                experiment_runner.run_experiment_pseudo_inverse(N=args.N, runs_num=args.runs)
            if args.plot:
                plotting.plot_pseudo_inverse_results(experiment_runner.results)
                
        if args.experiment == 'bayes' or args.experiment == 'all':
            if not args.load:
                experiment_runner.run_experiment_bayes(N=args.N, runs_num=args.runs)
            if args.plot:
                plotting.plot_bayes_results(experiment_runner.results)

        # Save results if new experiments were run
        if not args.load:
            save_results(args.save_file, experiment_runner.results)

if __name__ == "__main__":
    main()
