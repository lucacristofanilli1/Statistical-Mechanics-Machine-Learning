import numpy as np
import time
from joblib import Parallel, delayed
from numpy.random import SeedSequence
from models import Perceptron, SyntheticDataGenerator, define_perfect_perceptron, _core_fit_noise_batch

# ==============================================================================
#     FUNZIONI DI SIMULAZIONE SINGLE RUN
# ==============================================================================

def run_single_perceptron_noise(N: int, P: int, x_val: np.ndarray, y_val: np.ndarray, sigma2: float, seed: int, teacher_perceptron: Perceptron) -> float:
    """
    Esegue UNA singola run di addestramento e valutazione.
    Questa funzione è definita a livello di modulo (top-level) per essere
    efficientemente parallelizzabile con `joblib` e `multiprocessing`.
    """
    # Generatori e perceptron con lo stesso seed (per coerenza interna)
    data_generator = SyntheticDataGenerator(seed=seed, N=N, teacher_perceptron=teacher_perceptron)
    student_perceptron = Perceptron(N=N, seed=seed)

    # Genera dati di training unici per questa run
    x_train, y_train = data_generator.generate_data(P=P)

    # Addestra il modello
    student_perceptron.fit_noise(x_train, y_train, sigma2=sigma2)

    # Valuta l'errore sui dati di validazione comuni
    error = student_perceptron.evaluate(x_val, y_val)
    return error

def run_single_perceptron_zero_noise(N: int, P: int, seed: int, test_size: int) -> float:
    """
    Esegue UNA singola run di addestramento e valutazione.
    Questa funzione è definita a livello di modulo (top-level) per essere
    efficientemente parallelizzabile con `joblib` e `multiprocessing`.
    """
    ss = SeedSequence(seed)
    dataset_seed, student_seed = ss.spawn(2)
    
    data_generator = SyntheticDataGenerator(N=N,seed=dataset_seed)
    student_perceptron = Perceptron(N=N, seed=student_seed)
    
    # Genera dati di training unici per questa run
    x_train, y_train = data_generator.generate_data_generic_teacher(P=P)
    x_val, y_val = data_generator.generate_data_generic_teacher(P=test_size)
    
    # Addestra il modello
    student_perceptron.fit_perceptron(x_train, y_train)

    # Valuta l'errore sui dati di validazione comuni
    error = student_perceptron.evaluate(x_val, y_val)
    return error

def run_single_hebb(N: int, P: int, seed: int, test_size: int) -> tuple:
    """
    Esegue UNA singola run di addestramento e valutazione.
    Questa funzione è definita a livello di modulo (top-level) per essere
    efficientemente parallelizzabile con `joblib` e `multiprocessing`.
    
    return train_error, gen_error
    """
    # Non serve utilizzare due seed, tanto il J_0 viene sempre messo a zero per la Hebb Rule
    data_generator = SyntheticDataGenerator(N=N, seed=seed)
    student_perceptron = Perceptron(N=N)
    
    # Genera dati di training unici per questa run
    x_train, y_train = data_generator.generate_data_generic_teacher(P=P)
    x_val, y_val = data_generator.generate_data_generic_teacher(P=test_size)
    
    # Addestra il modello
    student_perceptron.fit_hebb(x_train, y_train)
    
    # Valuta l'errore sui dati di validazione comuni
    train_error = student_perceptron.evaluate(x_train, y_train)
    gen_error = student_perceptron.evaluate(x_val, y_val)
    return train_error, gen_error

def run_single_pseudo_inverse(N: int, P: int, seed: int, test_size: int) -> tuple:
    """
    Esegue UNA singola run di addestramento e valutazione per la regola di Pseudo Inversa.
    """
    # Non serve utilizzare due seed, tanto il J_0 viene sempre messo a zero per la pseudo inverse rule
    data_generator = SyntheticDataGenerator(N=N, seed=seed)
    student_perceptron = Perceptron(N=N)
    
    # Genera dati di training unici per questa run
    x_train, y_train = data_generator.generate_data_generic_teacher(P=P)
    x_val, y_val = data_generator.generate_data_generic_teacher(P=test_size)
    
    # Addestra il modello
    student_perceptron.fit_pseudo_inverse(x_train, y_train)
    
    # Valuta l'errore sui dati di validazione comuni
    train_error = student_perceptron.evaluate(x_train, y_train)
    gen_error = student_perceptron.evaluate(x_val, y_val)
    return train_error, gen_error

def run_single_adaline(N: int, P: int, seed: int, test_size: int) -> tuple:
    """
    Esegue UNA singola run di addestramento e valutazione per la regola di Pseudo Inversa.
    """
    # Non serve utilizzare due seed, tanto il J_0 viene sempre messo a zero per la pseudo inverse rule
    data_generator = SyntheticDataGenerator(N=N, seed=seed)
    student_perceptron = Perceptron(N=N)
    
    # Genera dati di training unici per questa run
    x_train, y_train = data_generator.generate_data_generic_teacher(P=P)
    x_val, y_val = data_generator.generate_data_generic_teacher(P=test_size)
    
    # Addestra il modello
    student_perceptron.fit_adaline(x_train, y_train)
    
    # Valuta l'errore sui dati di validazione comuni
    train_error = student_perceptron.evaluate(x_train, y_train)
    gen_error = student_perceptron.evaluate(x_val, y_val)
    return train_error, gen_error

def run_single_bayes(N: int, P: int, seed: int, test_size: int, M: int, sigma2: float = 50) -> tuple:
    """
    Esegue UNA singola run di addestramento e valutazione per la regola di Bayes.
    Ottimizzato con training batch vettorializzato su M studenti.
    """
    ss = SeedSequence(seed)
    dataset_seed, student_seed = ss.spawn(2)

    # Inizializza il generatore di dati
    data_generator = SyntheticDataGenerator(N=N, seed=dataset_seed)
    
    # Genera dati di training unici per questa run
    x_train, y_train = data_generator.generate_data_generic_teacher(P=P)

    # Inizializzazione Batch degli studenti
    # Generiamo direttamente una matrice (N, M) di pesi iniziali
    rng_students = np.random.default_rng(student_seed)
    J_batch = rng_students.normal(0, 1, (N, M)).astype(np.float64)
    
    # Seed per Numba
    numba_seed = rng_students.integers(0, 2**31 - 1)
    epochs = 5000
    
    # Training Batch Vettorializzato
    J_batch = _core_fit_noise_batch(
        x_train.astype(np.float64),
        y_train.astype(np.float64),
        J_batch,
        epochs,
        float(sigma2),
        numba_seed
    )
    
    # Calcolo Center of Mass (Media dei pesi)
    J_cm = np.sum(J_batch, axis=1)
    
    # Normalizzazione
    norm = np.linalg.norm(J_cm)
    if norm > 1e-9:
        J_cm /= norm
    J_cm *= np.sqrt(N)

    # Valutazione
    # Usiamo un oggetto Perceptron temporaneo per sfruttare il metodo evaluate esistente
    center_of_mass = Perceptron(N=N, J=J_cm)
    
    x_val, y_val = data_generator.generate_data_generic_teacher(P=test_size)
    
    train_error = center_of_mass.evaluate(x_train, y_train)
    gen_error = center_of_mass.evaluate(x_val, y_val)
    
    return train_error, gen_error


class Experiment:
  def __init__(self, name="Esperimento Perceptrone"):
    """
    Inizializza l'esperimento.
    """
    self.name = name
    self.results = {}

  def simulation_perceptron_noise(self, train_sizes: list[int], test_size: int, runs_num: int, sigma2: float, N: int, master_seed: int = 123456):
    print(f"--- Inizio simulazione perceptron noise parallela per N = {N} ---")
    error_means = []
    error_stds = []

    master_rng = np.random.default_rng(master_seed)

    print(f"Generazione di {test_size} campioni di validazione per N={N}...")
    
    perfect_J = define_perfect_perceptron(N=N)
    perfect_teacher = Perceptron(N=N, J=perfect_J)
    
    data_generator_val = SyntheticDataGenerator(N=N, seed=master_rng.integers(0, 2**31 - 1), teacher_perceptron=perfect_teacher)
    x_val, y_val = data_generator_val.generate_data(P=test_size)

    for P in train_sizes:
        start_time = time.time()
        print(f"Avvio di {runs_num} run in parallelo per N={N}, P={P}...")

        # Genera semi indipendenti per ogni run
        run_seeds = master_rng.integers(0, 2**31 - 1, size=runs_num)
        
        # Esegui in parallelo con i semi specifici
        errors = Parallel(
            n_jobs=-1,
            verbose=10,
            batch_size='auto',
            pre_dispatch='2*n_jobs'
        )(
            delayed(run_single_perceptron_noise)(N, P, x_val, y_val, sigma2, seed=s, teacher_perceptron=perfect_teacher)
            for s in run_seeds
        )

        errors = np.array(errors)
        error_means.append(np.mean(errors))
        error_stds.append(np.std(errors))

        end_time = time.time()
        print(f"Completato per N={N}, P={P} in {end_time - start_time:.2f} secondi. Errore medio: {error_means[-1]:.4f} ± {error_stds[-1]:.4f}")

    return np.array(error_means), np.array(error_stds)

  def simulation_perceptron_zero_noise(self, train_sizes: list[int], test_size: int, runs_num: int, N: int, master_seed: int = 123456):
    print(f"--- Inizio simulazione perceptron noise parallela per N = {N} ---")
    error_means = []
    error_stds = []

    master_rng = np.random.default_rng(master_seed)

    print(f"Generazione di {test_size} campioni di validazione per N={N}...")
    
    for P in train_sizes:
        start_time = time.time()
        print(f"Avvio di {runs_num} run in parallelo per N={N}, P={P}...")

        # Genera semi indipendenti per ogni run
        run_seeds = master_rng.integers(0, 2**31 - 1, size=runs_num)
        
        # Esegui in parallelo con i semi specifici
        errors = Parallel(
            n_jobs=-1,
            verbose=10,
            batch_size='auto',
            pre_dispatch='2*n_jobs'
        )(
            delayed(run_single_perceptron_zero_noise)(N, P, seed=s, test_size=test_size)
            for s in run_seeds
        )

        errors = np.array(errors)
        error_means.append(np.mean(errors))
        error_stds.append(np.std(errors))

        end_time = time.time()
        print(f"Completato per N={N}, P={P} in {end_time - start_time:.2f} secondi. Errore medio: {error_means[-1]:.4f} ± {error_stds[-1]:.4f}")

    return np.array(error_means), np.array(error_stds)

  def simulation_hebb(self, train_sizes: list[int], test_size: int, runs_num: int, N: int, master_seed: int = 123456):
    print(f"--- Inizio simulazione perceptron noise parallela per N = {N} ---")
    train_error_means = []
    train_error_stds = []
    
    gen_error_means = []
    gen_error_stds = []

    master_rng = np.random.default_rng(master_seed)

    print(f"Generazione di {test_size} campioni di validazione per N={N}...")
    
    for P in train_sizes:
        start_time = time.time()
        print(f"Avvio di {runs_num} run in parallelo per N={N}, P={P}...")

        # Genera semi indipendenti per ogni run
        run_seeds = master_rng.integers(0, 2**31 - 1, size=runs_num)
        
        # Esegui in parallelo con i semi specifici
        errors = Parallel(
            n_jobs=-1,
            verbose=0,
            batch_size='auto',
            pre_dispatch='2*n_jobs'
        )(
            delayed(run_single_hebb)(N, P, seed=s, test_size=test_size)
            for s in run_seeds
        )
        train_errors , gen_errors = zip(*errors)
        
        train_errors = np.array(train_errors)
        train_error_means.append(np.mean(train_errors))
        train_error_stds.append(np.std(train_errors))
        
        gen_errors = np.array(gen_errors)
        gen_error_means.append(np.mean(gen_errors))
        gen_error_stds.append(np.std(gen_errors))

        end_time = time.time()
        print(f"Completato per N={N}, P={P} in {end_time - start_time:.2f} secondi.")
        print(f"Errore medio di train: {train_error_means[-1]:.4f} ± {train_error_stds[-1]:.4f}")
        print(f"Errore medio di generalizzazione: {gen_error_means[-1]:.4f} ± {gen_error_stds[-1]:.4f}")

    return np.array(train_error_means), np.array(train_error_stds), np.array(gen_error_means), np.array(gen_error_stds)
    
  def simulation_pseudo_inverse(self, train_sizes: list[int], test_size: int, runs_num: int, N: int, master_seed: int = 123456):
    print(f"--- Inizio simulazione perceptron noise parallela per N = {N} ---")
    train_error_means = []
    train_error_stds = []
    
    gen_error_means = []
    gen_error_stds = []

    master_rng = np.random.default_rng(master_seed)

    print(f"Generazione di {test_size} campioni di validazione per N={N}...")
    
    for P in train_sizes:
        start_time = time.time()
        print(f"Avvio di {runs_num} run in parallelo per N={N}, P={P}...")

        # Genera semi indipendenti per ogni run
        run_seeds = master_rng.integers(0, 2**31 - 1, size=runs_num)
        
        # Esegui in parallelo con i semi specifici
        errors = Parallel(
            n_jobs=-1,
            verbose=0,
            batch_size='auto',
            pre_dispatch='2*n_jobs'
        )(
            delayed(run_single_pseudo_inverse)(N, P, seed=s, test_size=test_size)
            for s in run_seeds
        )
        train_errors , gen_errors = zip(*errors)
        
        train_errors = np.array(train_errors)
        train_error_means.append(np.mean(train_errors))
        train_error_stds.append(np.std(train_errors))
        
        gen_errors = np.array(gen_errors)
        gen_error_means.append(np.mean(gen_errors))
        gen_error_stds.append(np.std(gen_errors))

        end_time = time.time()
        print(f"Completato per N={N}, P={P} in {end_time - start_time:.2f} secondi.")
        print(f"Errore medio di train: {train_error_means[-1]:.4f} ± {train_error_stds[-1]:.4f}")
        print(f"Errore medio di generalizzazione: {gen_error_means[-1]:.4f} ± {gen_error_stds[-1]:.4f}")

    return np.array(train_error_means), np.array(train_error_stds), np.array(gen_error_means), np.array(gen_error_stds)
    
  def simulation_adaline(self, train_sizes: list[int], test_size: int, runs_num: int, N: int, master_seed: int = 123456):
    print(f"--- Inizio simulazione adaline noise parallela per N = {N} ---")
    train_error_means = []
    train_error_stds = []
    
    gen_error_means = []
    gen_error_stds = []

    master_rng = np.random.default_rng(master_seed)

    print(f"Generazione di {test_size} campioni di validazione per N={N}...")
    
    for P in train_sizes:
        start_time = time.time()
        print(f"Avvio di {runs_num} run in parallelo per N={N}, P={P}...")

        # Genera semi indipendenti per ogni run
        run_seeds = master_rng.integers(0, 2**31 - 1, size=runs_num)
        
        # Esegui in parallelo con i semi specifici
        errors = Parallel(
            n_jobs=-1,
            verbose=0,
            batch_size='auto',
            pre_dispatch='2*n_jobs'
        )(
            delayed(run_single_adaline)(N, P, seed=s, test_size=test_size)
            for s in run_seeds
        )
        train_errors , gen_errors = zip(*errors)
        
        train_errors = np.array(train_errors)
        train_error_means.append(np.mean(train_errors))
        train_error_stds.append(np.std(train_errors))
        
        gen_errors = np.array(gen_errors)
        gen_error_means.append(np.mean(gen_errors))
        gen_error_stds.append(np.std(gen_errors))

        end_time = time.time()
        print(f"Completato per N={N}, P={P} in {end_time - start_time:.2f} secondi.")
        print(f"Errore medio di train: {train_error_means[-1]:.4f} ± {train_error_stds[-1]:.4f}")
        print(f"Errore medio di generalizzazione: {gen_error_means[-1]:.4f} ± {gen_error_stds[-1]:.4f}")

    return np.array(train_error_means), np.array(train_error_stds), np.array(gen_error_means), np.array(gen_error_stds)

  def simulation_bayes(self, train_sizes: list[int], test_size: int, runs_num: int, N: int, M: int, sigma2: float, master_seed: int = 123456):
    print(f"--- Inizio simulazione bayes noise parallela per N = {N} ---")
    train_error_means = []
    train_error_stds = []
    
    gen_error_means = []
    gen_error_stds = []

    master_rng = np.random.default_rng(master_seed)

    print(f"Generazione di {test_size} campioni di validazione per N={N}...")
    
    for P in train_sizes:
        start_time = time.time()
        print(f"Avvio di {runs_num} run in parallelo per N={N}, P={P}...")

        # Genera semi indipendenti per ogni run
        run_seeds = master_rng.integers(0, 2**31 - 1, size=runs_num)
        
        # Esegui in parallelo con i semi specifici
        errors = Parallel(
            n_jobs=-1,
            verbose=0,
            batch_size='auto',
            pre_dispatch='2*n_jobs'
        )(
            delayed(run_single_bayes)(N, P, seed=s, test_size=test_size, M=M, sigma2=sigma2)
            for s in run_seeds
        )
        train_errors , gen_errors = zip(*errors)
        
        train_errors = np.array(train_errors)
        train_error_means.append(np.mean(train_errors))
        train_error_stds.append(np.std(train_errors))
        
        gen_errors = np.array(gen_errors)
        gen_error_means.append(np.mean(gen_errors))
        gen_error_stds.append(np.std(gen_errors))

        end_time = time.time()
        print(f"Completato per N={N}, P={P} in {end_time - start_time:.2f} secondi.")
        print(f"Errore medio di train: {train_error_means[-1]:.4f} ± {train_error_stds[-1]:.4f}")
        print(f"Errore medio di generalizzazione: {gen_error_means[-1]:.4f} ± {gen_error_stds[-1]:.4f}")

    return np.array(train_error_means), np.array(train_error_stds), np.array(gen_error_means), np.array(gen_error_stds)
    
    
  def run_experiment_perceptron_noise(self, runs_num=1000):
    print(f"--- Esecuzione Esperimento Perceptron Random ---")
    test_size, sigma2 = 1000, 50
    train_sizes_20 = [1, 10, 20, 50, 100, 150, 200]
    means20, stds20 = self.simulation_perceptron_noise(train_sizes_20, test_size, runs_num, sigma2, N=20)
    train_sizes_40 = [1, 2, 20, 40, 100, 200, 300]
    means40, stds40 = self.simulation_perceptron_noise(train_sizes_40, test_size, runs_num, sigma2, N=40)
    results = {
        'runs_num': runs_num,
        'train_sizes_20': train_sizes_20, 'error_means_20': means20, 'error_stds_20': stds20,
        'train_sizes_40': train_sizes_40, 'error_means_40': means40, 'error_stds_40': stds40,
    }
    self.results.update(results)
    print("\n--- Simulazioni Completate ---")
    return results

  def run_experiment_hebb(self, N, runs_num=1000):
    print(f"--- Esecuzione Esperimento Hebb ---")
    test_size = 1000
    alpha_max = 11
    alpha_resolution = 0.2
    P_max_train = int(alpha_max*N)
    P_resolution = int(alpha_resolution*N)
    train_sizes = np.arange(1, P_max_train, P_resolution)
    train_means_hebb, train_stds_hebb, gen_means_hebb, gen_stds_hebb = self.simulation_hebb(train_sizes=train_sizes, test_size=test_size, runs_num=runs_num, N=N)
    
    results = {
        'runs_num_hebb': runs_num,
        'train_sizes_hebb': train_sizes, 'N_dimension_hebb': N, 'train_error_means_hebb': train_means_hebb, 'train_error_stds_hebb': train_stds_hebb,
        'gen_error_means_hebb': gen_means_hebb, 'gen_error_stds_hebb': gen_stds_hebb
    }
    
    self.results.update(results)
    print("\n --- Simulazioni Completate ---")
    return results

  def run_experiment_perceptron_zero_noise(self, N, runs_num=1000):
    print(f"--- Esecuzione Esperimento Hebb ---")
    test_size = 1000
    alpha_max = 11
    alpha_resolution = 0.2
    P_max_train = int(alpha_max*N)
    P_resolution = int(alpha_resolution*N)
    train_sizes = np.arange(1, P_max_train, P_resolution)
    gen_means_perceptron_zero_noise, gen_stds_perceptron_zero_noise = self.simulation_perceptron_zero_noise(train_sizes=train_sizes, test_size=test_size, runs_num=runs_num, N=N)
    
    results = {
        'runs_num_perceptron_zero_noise': runs_num,
        'train_sizes_perceptron_zero_noise': train_sizes, 
        'N_dimension_perceptron_zero_noise': N, 
        'gen_error_perceptron_zero_noise': gen_means_perceptron_zero_noise, 
        'gen_error_stds_perceptron_zero_noise': gen_stds_perceptron_zero_noise        
    }
    
    self.results.update(results)
    print("\n --- Simulazioni Completate ---")
    return results

  def run_experiment_pseudo_inverse(self, N, alpha_max=1.1, alpha_resolution=0.01, runs_num=1000):
    print(f"--- Esecuzione Esperimento Pseudo Inverse ---")
    test_size = 1000
    P_max_train = int(alpha_max*N)
    P_resolution = max(int(alpha_resolution*N), 1)
    print(f'P_max_train pseudo_inverse: {P_max_train}')
    print(f'P_resolution pseudo_inverse: {P_resolution}')
    train_sizes = np.arange(1, P_max_train, P_resolution)
    train_means_pseudo_inverse, train_stds_pseudo_inverse, gen_means_pseudo_inverse, gen_stds_pseudo_inverse = self.simulation_pseudo_inverse(train_sizes=train_sizes, test_size=test_size, runs_num=runs_num, N=N)
    
    results = {
        'runs_num_pseudo_inverse': runs_num,
        'train_sizes_pseudo_inverse': train_sizes, 'N_dimension_pseudo_inverse': N,
        'train_error_means_pseudo_inverse': train_means_pseudo_inverse, 'train_error_stds_pseudo_inverse': train_stds_pseudo_inverse,
        'gen_error_means_pseudo_inverse': gen_means_pseudo_inverse, 'gen_error_stds_pseudo_inverse': gen_stds_pseudo_inverse
    }
    
    self.results.update(results)
    print("\n --- Simulazioni Completate ---")
    return results

  def run_experiment_adaline(self, N, alpha_max=10, alpha_resolution=0.2, runs_num=1000):
    """
    Esegue l'esperimento Adaline.
    Parametri:
      N: dimensione input
      alpha_max: valore massimo di alpha (P/N)
      alpha_resolution: passo di campionamento per alpha
      runs_num: numero di run per punto
    """
    print(f"--- Esecuzione Esperimento Adaline (N={N}, alpha_max={alpha_max}, res={alpha_resolution}) ---")
    
    test_size = 1000
    
    # Calcolo P basato sui parametri passati
    P_max_train = int(alpha_max * N)
    P_resolution = max(int(alpha_resolution * N), 1)
    
    print(f'P_max_train adaline: {P_max_train}')
    print(f'P_resolution adaline: {P_resolution}')
    
    # Costruzione di train_sizes evitando P=N (alpha=1) per convergenza
    # 1. Da P=1 a P=N-1 (tutti i valori, step=1)
    pre_N = np.arange(1, N)
    # 2. Da P>N in poi con step=P_resolution
    post_N = np.arange(N + P_resolution, P_max_train, P_resolution)
    # 3. Concatenazione (salta P=N)
    train_sizes = np.concatenate([pre_N, post_N])
        
    # Esecuzione simulazione
    train_means, train_stds, gen_means, gen_stds = self.simulation_adaline(
        train_sizes=train_sizes, 
        test_size=test_size, 
        runs_num=runs_num, 
        N=N
    )
    
    # Creazione dizionario risultati
    results = {
        'runs_num_adaline': runs_num,
        'train_sizes_adaline': train_sizes, 
        'N_dimension_adaline': N,
        'train_error_means_adaline': train_means, 
        'train_error_stds_adaline': train_stds,
        'gen_error_means_adaline': gen_means, 
        'gen_error_stds_adaline': gen_stds
    }
    
    # Aggiornamento risultati globali (sovrascrive i vecchi dati Adaline se presenti)
    self.results.update(results)
    print("\n --- Simulazioni Completate ---")
    return results

  def run_experiment_bayes(self, N, alpha_max=10, alpha_resolution=0.2, runs_num=1000):
    """
    Esegue l'esperimento Bayes.
    Parametri:
      N: dimensione input
      alpha_max: valore massimo di alpha (P/N)
      alpha_resolution: passo di campionamento per alpha
      runs_num: numero di run per punto
    """
    print(f"--- Esecuzione Esperimento Bayes (N={N}, alpha_max={alpha_max}, res={alpha_resolution}) ---")
    
    test_size = 1000
    M = 100
    sigma2 = 50.0
    
    # Calcolo P basato sui parametri passati
    P_max_train = int(alpha_max * N)
    P_resolution = max(int(alpha_resolution * N), 1)
    
    print(f'P_max_train bayes: {P_max_train}')
    print(f'P_resolution bayes: {P_resolution}')
    
    train_sizes = np.arange(P_resolution, P_max_train + P_resolution, P_resolution)
        
    # Esecuzione simulazione
    train_means, train_stds, gen_means, gen_stds = self.simulation_bayes(
        train_sizes=train_sizes, 
        test_size=test_size, 
        runs_num=runs_num, 
        N=N,
        M=M,
        sigma2=sigma2
    )
    
    # Creazione dizionario risultati
    results = {
        'runs_num_bayes': runs_num,
        'train_sizes_bayes': train_sizes, 
        'N_dimension_bayes': N,
        'train_error_means_bayes': train_means, 
        'train_error_stds_bayes': train_stds,
        'gen_error_means_bayes': gen_means, 
        'gen_error_stds_bayes': gen_stds
    }
    
    # Aggiornamento risultati globali (sovrascrive i vecchi dati Bayes se presenti)
    self.results.update(results)
    print("\n --- Simulazioni Completate ---")
    return results
