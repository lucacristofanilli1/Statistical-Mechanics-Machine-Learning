import numpy as np
from numba import njit

def vectorized_convert_to_binary(x: np.ndarray, N: int) -> np.ndarray:
    """
    Versione completamente vettorializzata della conversione da coppie di interi
    a vettori binari di tipo Ising (-1, 1).
    """
    num_samples = x.shape[0]
    num_bits_per_int = N // 2

    powers_of_2 = 2 ** np.arange(num_bits_per_int - 1, -1, -1, dtype=np.uint32)
    
    col0_bits = (x[:, 0, None] & powers_of_2) // powers_of_2
    col1_bits = (x[:, 1, None] & powers_of_2) // powers_of_2

    bin_var = np.hstack((col0_bits, col1_bits)).astype(np.int8)
    ising_var = 2 * bin_var - 1

    return ising_var

def define_perfect_perceptron(N = 20):
    """
    Inizializzazione dell'array di pesi del Perceptron perfetto.
    """
    positive_part = 2 ** np.arange((N/2) -1, -1, -1)
    return np.concatenate((positive_part, -positive_part))

@njit(fastmath=True, cache=True)
def _core_fit_perceptron(X, y, J, epochs, seed):
    """Core loop per Perceptron classico."""
    np.random.seed(seed) # Impostiamo il seed locale per Numba
    P, N = X.shape
    idxs = np.arange(P)
    
    for epoch in range(epochs):
        updates_in_epoch = 0
        np.random.shuffle(idxs) # Shuffle supportato in Numba
        
        for i in idxs:
            xi = X[i]
            yi = y[i]
            
            # Calcolo predizione inline per evitare overhead
            dot_prod = 0.0
            for k in range(N):
                dot_prod += xi[k] * J[k]
            
            # Logica sign activation: > 0 -> 1, else -1
            yi_pred = 1.0 if dot_prod >= 0.0 else -1.0
            
            if yi_pred != yi:
                updates_in_epoch += 1
                # J += (yi * xi) / sqrt(N)
                factor = yi / np.sqrt(N)
                for k in range(N):
                    J[k] += factor * xi[k]
                    
        if updates_in_epoch == 0 and epoch > 0:
            break
            
    return J

@njit(fastmath=True, cache=True)
def _core_fit_noise(X, y, J, epochs, sigma, seed):
    """Core loop per Perceptron con noise (il collo di bottiglia principale)."""
    np.random.seed(seed)
    P, N = X.shape
    idxs = np.arange(P)
    sqrt_N = np.sqrt(N)
    sqrt_sigma = np.sqrt(sigma)
    
    for epoch in range(epochs):
        updates_in_epoch = 0
        np.random.shuffle(idxs)
        
        for i in idxs:
            xi = X[i]
            yi = y[i]
            
            # Dot product manuale o numpy (numpy.dot è ottimizzato anche in numba)
            if np.dot(xi, J) >= 0:
                yi_pred = 1.0
            else:
                yi_pred = -1.0

            if yi_pred != yi:
                updates_in_epoch += 1
                
                # Ottimizzazione: Generazione scalare del rumore nel loop
                # Evita allocazione di array temporanei
                factor = yi / sqrt_N
                noise_scale = factor * sqrt_sigma
                
                for k in range(N):
                    # Update rule: J += factor * xi + factor * noise * xi
                    # noise * xi ~ N(0, sigma) independent of sign of xi
                    # So we add signal + independent noise
                    J[k] += factor * xi[k] + np.random.normal() * noise_scale

        if updates_in_epoch == 0 and epoch > 0:
            break
            
    return J

@njit(fastmath=True, cache=True)
def _core_fit_noise_batch(X, y, J_batch, epochs, sigma, seed):
    """
    Versione completamente vettorializzata su M studenti.
    J_batch shape: (N, M)
    """
    np.random.seed(seed)
    P, N = X.shape
    M = J_batch.shape[1]
    idxs = np.arange(P)
    sqrt_N = np.sqrt(N)
    sqrt_sigma = np.sqrt(sigma)
    
    for epoch in range(epochs):
        updates_in_epoch = 0
        np.random.shuffle(idxs)
        
        for i in idxs:
            xi = X[i]     # Shape (N,)
            yi = y[i]
            
            # 1. Predizione Vettorializzata (N dot N,M -> 1,M)
            dot_prods = xi @ J_batch  # Shape (M,)
            
            # 2. Identifica chi ha sbagliato (Vettorializzato)
            # yi_pred != yi <=> yi * dot_prods < 0
            errors_mask = (yi * dot_prods) < 0
            
            n_errors = np.sum(errors_mask)
            
            if n_errors > 0:
                updates_in_epoch += 1
                
                factor = yi / sqrt_N
                noise_scale = factor * sqrt_sigma
                
                # 3. Update Vettorializzato (solo per chi ha sbagliato)
                # Generiamo rumore solo per gli studenti che aggiornano (N, n_errors)
                noise = np.random.normal(0.0, 1.0, (N, n_errors)) * noise_scale
                
                # Termine di segnale: factor * xi (broadcastato su n_errors colonne)
                signal_update = (factor * xi).reshape(-1, 1)
                
                # Aggiornamento in-place usando indicizzazione avanzata
                # J += factor * xi + noise  <-- Equivalente a J += factor * xi * (1 + noise_rel)
                # Dove noise = factor * xi * noise_rel ~ N(0, sigma/N)
                # Qui noise ~ N(0, sigma/N) direttamente, che è statisticamente equivalente
                # dato che xi è +/- 1 e indipendente dal rumore.
                J_batch[:, errors_mask] += signal_update + noise

        if updates_in_epoch == 0 and epoch > 0:
            break
            
    return J_batch

@njit(fastmath=True, cache=True)
def _core_fit_adaline(X, y, J, gamma, max_iter=10000, tol=1e-8):
    """Core loop per Adaline."""
    P, N = X.shape
    
    # Calcolo costo iniziale
    # Cost function: 0.5 * sum((X.J - y)^2) / P
    pred = X.dot(J)
    diff = pred - y
    current_energy = 0.5 * np.sum(diff**2) / P
    
    delta_energy = 1.0
    iter_idx = 0
    
    while delta_energy > tol and iter_idx < max_iter:
        # Gradiente: X.T @ (X.J - y) / P
        # Ricalcoliamo pred error
        pred = X.dot(J)
        error_vect = pred - y
        
        # Calcolo gradiente
        gradient = X.T.dot(error_vect) / P
        
        # Aggiornamento pesi
        J -= gamma * gradient
        
        # Nuova energia
        new_pred = X.dot(J)
        new_diff = new_pred - y
        new_energy = 0.5 * np.sum(new_diff**2) / P
        
        delta_energy = np.abs(new_energy - current_energy)
        
        # Controllo divergenza
        if new_energy > current_energy * 1.5:
            gamma *= 0.5
            # Opzionale: reset J al passo precedente se diverge troppo, 
            # ma per ora manteniamo logica originale
        
        current_energy = new_energy
        iter_idx += 1
        
    return J

class Perceptron:
    def __init__(self, N=20, J=None, seed=None):
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.N = N

        self.numba_seed = self.rng.integers(0, 2**31 - 1) if seed is not None else np.random.randint(0, 2**31 -1)

        if J is None:
            self.J = self.rng.normal(0, 1, self.N).astype(np.float64)
        else:
            self.J = J.astype(np.float64) # Numba ama i float64

        self.evolution_J = []

    def _sign_activation(self, x):
        x = np.asarray(x)
        return np.where(x > 0, 1, -1)

    def predict(self, x_binary):
        """Esegue predizioni usando i coefficienti del modello fittato."""
        return self._sign_activation(np.dot(x_binary, self.J)) 

    def evaluate(self, x_binary, y):
        """Calcola l'errore medio di classificazione."""
        y_pred = self.predict(x_binary)
        y = np.atleast_1d(y)
        return np.mean(y_pred != y)

    def fit_perceptron(self, x_binary, y, epochs=5000):
        """Wrapper per il training Numba standard."""
        self.evolution_J.append(self.J.copy())
        
        # Chiamata a Numba
        # Passiamo self.J direttamente (passaggio per riferimento, array mutabile)
        self.J = _core_fit_perceptron(
            x_binary.astype(np.float64), 
            y.astype(np.float64), 
            self.J, 
            epochs, 
            self.numba_seed
        )
            
        self.evolution_J.append(self.J.copy())
        return self.J.copy()  

    def fit_noise(self, x_binary, y, sigma2=50, epochs=5000):
        """Wrapper per il training Numba con noise."""
        self.evolution_J.append(self.J.copy())

        # Chiamata a Numba
        self.J = _core_fit_noise(
              x_binary.astype(np.float64),
              y.astype(np.float64),
              self.J,
              epochs,
              float(sigma2),
              self.numba_seed
          )

        self.evolution_J.append(self.J.copy())
        return self.J.copy()

    def fit_hebb(self, x_binary, y):
        self.J = 0
        self.J = np.sum(x_binary * y.reshape(-1, 1), axis=0)
        return self.J.copy()

    def fit_pseudo_inverse(self, x_binary, y):
        self.J = 0
        C = np.dot(x_binary*y.reshape(-1, 1), (x_binary*y.reshape(-1, 1)).T)/self.N
        C[np.diag_indices_from(C)] += 1e-10
        try:
            x = np.linalg.solve(C, np.ones(y.shape[0]))
        except np.linalg.LinAlgError:
            x = np.sum(np.linalg.pinv(C), axis=1)

        self.J = np.sum(x.reshape(-1, 1)*x_binary*y.reshape(-1, 1), axis=0)/np.sqrt(self.N)
        return self.J.copy()
  
    def fit_adaline(self, x_binary, y, gamma=0.05):
        """Wrapper per il training Adaline Numba."""
        # Inizializzazione come da tuo codice originale
        self.J = np.sum(x_binary, axis=0).astype(float)
        norm_J = np.linalg.norm(self.J)
        if norm_J > 1e-9:
            self.J /= norm_J
        else:
            self.J = self.rng.normal(0, 0.01, self.N)
            
        # Chiamata a Numba
        self.J = _core_fit_adaline(
            x_binary.astype(np.float64),
            y.astype(np.float64),
            self.J,
            float(gamma)
        )
            
        return self.J.copy()

    def _adaline_cost_function(self, x_binary, y):
        return 0.5*np.sum(((x_binary @ self.J).reshape(-1, 1) - y.reshape(-1, 1))**2)/x_binary.shape[0]

class SyntheticDataGenerator:
    def __init__(self, N=20, seed=None, teacher_perceptron=None):
        self.N = N
        if seed is None:
            self.rng = np.random.default_rng()
        else:
            self.rng = np.random.default_rng(seed)
        
        if teacher_perceptron is None:
            self.teacher = Perceptron(N=N, seed=seed)
        else:
            self.teacher = teacher_perceptron

    def generate_data_generic_teacher(self, P):
        """
        Genera un set completo di dati (x, y) di P campioni.
        
        Shape X: (P, N)
        
        Shape y: (P,)
        """
        x_binary = self.rng.choice([-1, 1], size=(P, self.teacher.N))

        y = self.teacher.predict(x_binary)

        return x_binary, y  
    
    def generate_data(self, P):
        """
        Genera un set completo di dati (x, y) di P campioni,
        assicurandosi che nessuna coppia (a, b) abbia a == b.
        """
        max_val = 2**(self.N//2)

        x = self.rng.integers(max_val, size=(P, 2))

        duplicate_mask = (x[:, 0] == x[:, 1])

        while np.any(duplicate_mask):
            num_duplicates = np.sum(duplicate_mask)  
            # Rigenero solamente il secondo valore
            x[duplicate_mask, 1] = self.rng.integers(max_val, size=num_duplicates) 
            # Aggiorno la maschera dei duplicati
            duplicate_mask = (x[:, 0] == x[:, 1])

        x_binary = vectorized_convert_to_binary(x, self.N)

        y = self.teacher.predict(x_binary)

        return x_binary, y  
