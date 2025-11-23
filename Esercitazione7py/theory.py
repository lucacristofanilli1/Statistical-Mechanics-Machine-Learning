import numpy as np
from scipy.integrate import quad, quad_vec
from scipy.special import log_ndtr

# ==============================================================================
#     DEFINIZIONI MATEMATICHE E TEORICHE (ANNEALED)
# ==============================================================================

def parametric_alphas_annealed(eps):
    """Calcola alpha(eps) per la teoria annealed in modo parametric_solver."""
    safe_eps = np.clip(np.asarray(eps), 1e-9, 0.5 - 1e-9)
    return np.pi * (1 - safe_eps) / (np.tan(np.pi * safe_eps) + 1e-9)

def f1_annealed(eps, alpha):
    """Prima mappa iterativa per la teoria annealed."""
    return 1 - alpha * np.tan(np.pi * eps) / np.pi

def f2_annealed(eps, alpha):
    """Seconda mappa iterativa per la teoria annealed."""
    return np.arctan((np.pi * (1 - eps)) / alpha) / np.pi


# ==============================================================================
#     DEFINIZIONI MATEMATICHE E TEORICHE (QUENCHED)
# ==============================================================================

def I_R(R):
    """Calcola l'integrale I(R) per un singolo valore di R."""
    if R < 0 or R >= 1: return np.nan
    def integrand(v, R_val):
        log_H = log_ndtr(np.sqrt(R_val) * v)
        return np.exp(-0.5 * (1 + R_val) * v**2 - log_H) / np.sqrt(2 * np.pi)
    result, _ = quad(integrand, -np.inf, np.inf, args=(R,))
    return result

def I_R_vec(R_array):
    """Versione vettorizzata di I_R che preserva la forma dell'input."""
    original_shape = R_array.shape
    R_flat = R_array.flatten() # Lavoriamo su un array appiattito

    valid_mask = (R_flat >= 0) & (R_flat < 1)
    results_flat = np.full_like(R_flat, np.nan, dtype=float)

    if np.any(valid_mask):
        valid_R = R_flat[valid_mask]
        res_quad, _ = quad_vec(
            lambda v, r: np.exp(-0.5 * (1 + r) * v**2 - log_ndtr(np.sqrt(r) * v)),
            -np.inf, np.inf, args=(valid_R,)
        )
        results_flat[valid_mask] = res_quad / np.sqrt(2 * np.pi)

    # Rimodelliamo il risultato alla forma originale dell'input
    return results_flat.reshape(original_shape)

def f1_quenched(R, alpha):
    """Funzione f1 corretta per gestire alpha sia come scalare che come array."""
    R = np.asarray(R)
    # alpha può essere uno scalare o un array, il broadcasting gestirà entrambi i casi

    I_val = I_R_vec(R)

    # Calcoliamo il risultato direttamente. Se R, alpha o I_val sono NaN,
    # il risultato sarà NaN, il che è corretto.
    return (alpha / np.pi) * np.sqrt(1 - R) * I_val

def f2_quenched(R, alpha):
    """Funzione f2 corretta per gestire alpha sia come scalare che come array."""
    R = np.asarray(R)
    alpha = np.asarray(alpha) # Convertire in array per ndim

    I_val = I_R_vec(R)

    # Il denominatore è il punto critico
    denominator = alpha**2 * I_val**2

    # Condizione per un calcolo valido
    valid_condition = (denominator > 1e-18) & (R >= 0) & (R < 1)

    # Calcola il rapporto solo dove la condizione è valida, altrimenti metti NaN
    ratio = np.divide(
        np.pi**2 * R**2,
        denominator,
        out=np.full_like(R, np.nan),
        where=valid_condition
    )

    return 1 - ratio

def parametric_solver_quenched(R_values):
    """Calcola alpha data una serie di R (il punto fisso)."""
    R_values = np.asarray(R_values)
    mask = (R_values > 0) & (R_values < 1)
    alphas = np.full_like(R_values, np.nan)
    if np.any(mask):
        valid_R = R_values[mask]
        lhs = (np.pi * valid_R) / np.sqrt(1 - valid_R)
        rhs = I_R_vec(valid_R)
        alphas[mask] = lhs / rhs
    return alphas


# ==============================================================================
#     DEFINIZIONI MATEMATICHE E TEORICHE (HEBB)
# ==============================================================================

def hebb_gen_error(alpha: np.ndarray) -> np.ndarray:
    return np.arccos(np.sqrt((2*alpha) / (2*alpha + np.pi))) / np.pi

def hebb_train_error(alpha: np.ndarray) -> np.ndarray:
    """
    Calculates the theoretical training error for the Hebbian learning rule.

    This function is designed to be vectorized over the input `alpha`.
    """
    
    integrand = lambda x, alpha: np.exp(-0.5 * x**2 + log_ndtr(-1/np.sqrt(alpha) - np.sqrt(2 * alpha / np.pi) * x))
    integral, _ = quad_vec(integrand, 0, np.inf, args=(alpha,))
        
    return np.sqrt(2/np.pi)*integral


# ==============================================================================
#     PROCEDURE NUMERICHE GENERALI
# ==============================================================================

def vectorized_iterative_procedure(alphas, x_0, a, f, max_epochs=2000, tol=1e-8):
    """
    Calcola l'evoluzione di una mappa iterativa per un array di `alphas`.
    Restituisce la matrice della storia dell'evoluzione.
    """
    alphas = np.asarray(alphas)
    num_alphas = len(alphas)
    old_x = np.full(num_alphas, float(x_0))
    history = np.zeros((max_epochs, num_alphas))

    for epoch in range(max_epochs):
        new_x = (1 - a) * old_x + a * f(old_x, alphas)
        history[epoch, :] = new_x

        if np.all(np.abs(new_x - old_x) < tol):
            # print(f"Convergenza raggiunta all'epoca {epoch + 1}.")
            history[epoch+1:, :] = new_x
            return history[:epoch + 1]

        old_x = new_x

    print(f"Attenzione: Max epoche ({max_epochs}) raggiunto.")
    return history

def evolution_vs_a_procedure(alpha, x_0, a_values, f, max_epochs=200):
    """Calcola l'evoluzione per un alpha fisso e diversi valori di `a`."""
    a_values = np.asarray(a_values)
    num_a = len(a_values)
    old_x = np.full(num_a, float(x_0))
    history = np.zeros((max_epochs, num_a))

    for epoch in range(max_epochs):
        f_val = f(old_x, alpha)
        new_x = (1 - a_values) * old_x + a_values * f_val
        history[epoch, :] = new_x
        old_x = new_x

    return history

def numerical_derivative(f, x, alpha, h=1e-6):
    """Calcola la derivata numerica di f(x, alpha) rispetto a x."""
    return (f(x + h, alpha) - f(x - h, alpha)) / (2 * h)

def hybrid_iterative_procedure(
    alphas,
    eps_0,
    a,
    f1,
    f2,
    crossover_alpha,
    max_epochs=2000,
    tol=1e-8
):
    """
    Esegue una procedura iterativa ibrida.
    """
    alphas = np.asarray(alphas)

    # 1. Definiamo la funzione ibrida 'f_hybrid'
    def f_hybrid(eps_array, alpha_array):
        # Inizializziamo l'output con NaN
        results = np.full_like(eps_array, np.nan)

        # Maschera per decidere quale funzione usare
        mask_f1 = (alpha_array <= crossover_alpha)
        mask_f2 = (alpha_array > crossover_alpha)

        # Applica f1 dove la maschera è vera
        if np.any(mask_f1):
            results[mask_f1] = f1(eps_array[mask_f1], alpha_array[mask_f1])

        # Applica f2 dove la maschera è vera
        if np.any(mask_f2):
            results[mask_f2] = f2(eps_array[mask_f2], alpha_array[mask_f2])

        return results

    # 2. Esegui la procedura vettorizzata con la funzione ibrida
    print(f"Esecuzione procedura ibrida con crossover a alpha = {crossover_alpha}...")
    evolution_matrix = vectorized_iterative_procedure(
        alphas,
        eps_0,
        a,
        f_hybrid,
        max_epochs,
        tol
    )

    # 3. Estrai e restituisci l'ultimo valore (il punto fisso) per ogni alpha
    fixed_points = evolution_matrix[-1, :]
    return fixed_points
