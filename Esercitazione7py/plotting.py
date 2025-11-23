import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import root_scalar
from theory import (
    parametric_alphas_annealed, f2_annealed, parametric_solver_quenched, 
    f1_quenched, f2_quenched, vectorized_iterative_procedure, 
    hybrid_iterative_procedure, evolution_vs_a_procedure,
    hebb_train_error, hebb_gen_error
)

# ==============================================================================
#     FUNZIONI DI VISUALIZZAZIONE TEORIA
# ==============================================================================

def compare_dynamics(f1, f2, alpha, x_0, a_values, max_epochs, interval=[0.001, 0.999], is_epsilon=False, save_figure=True):
    """Esegue e plotta la dinamica di convergenza per f1 e f2."""
    print("--- Calcolo Dinamiche di Convergenza ---")
    evo1 = evolution_vs_a_procedure(alpha, x_0, a_values, f1, max_epochs)
    evo2 = evolution_vs_a_procedure(alpha, x_0, a_values, f2, max_epochs)

    print("--- Plotting Dinamiche ---")
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
    colors = plt.cm.viridis(np.linspace(0, 1, len(a_values)))

    for ax, evo, f, name in zip(axes, [evo1, evo2], [f1, f2], ["f_1", "f_2"]):
        try:
            pf = root_scalar(lambda x: x - f(x, alpha), x0=x_0, bracket=interval).root
            ax.axhline(pf, color='black', ls='--', label=f'Punto fisso ≈ {pf:.3f}')
        except ValueError:
            print(f"Nessun punto fisso trovato per {name} nell'intervallo.")

        for i, a_val in enumerate(a_values):
            trajectory = np.concatenate(([x_0], evo[:, i]))
            ax.plot(trajectory, label=f'a = {a_val}', color=colors[i], lw=2.5)

        ax.set_title(f'Dinamica per Funzione ${name}$', fontsize=16)
        ax.set_xlabel('Epoca (i)', fontsize=14)
        ax.legend()
        ax.set_ylim(interval)
        ax.grid(True, ls=':', alpha=0.7)

    if is_epsilon:
        axes[0].set_ylabel(r'Valore di $\epsilon_i$', fontsize=14)
    else:
        axes[0].set_ylabel(r'Valore di $R_i$', fontsize=14)
    fig.suptitle(f'Confronto Dinamiche per $\\alpha={alpha}$ e $x_0={x_0}$', fontsize=20)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    if save_figure:
        filename = f"compare_dynamics_x0_{x_0}_alpha_{alpha}.png"
        print(f"\nSalvataggio figura in: {filename}")
        fig.savefig(filename, dpi=300, bbox_inches='tight')

    plt.show()


def plot_results_annealed(
    f_map=f2_annealed,
    a_fixed=0.9,
    x0_fixed=0.5,
    max_epochs=5000,
    save_figure=False
):
    """
    Genera il grafico che confronta la curva parametrica teorica (Annealed)
    con i punti fissi calcolati tramite la dinamica iterativa.
    """
    print("\n--- Esecuzione Plot di Confronto (Annealed) ---")

    # --- 1. Calcolo Curva Teoria Parametrica ---
    print("Calcolo curva teorica parametrica...")
    eps_parametric = np.linspace(1e-9, 0.5 - 1e-9, 500)
    alpha_parametric = parametric_alphas_annealed(eps_parametric)

    # --- 2. Calcolo Punti Fissi da Mappa Iterativa ---
    print(f"Calcolo punti fissi da mappa iterativa (usando {f_map.__name__})...")
    alphas_for_map = np.linspace(0.01, 10, 100)
    evolution_matrix = vectorized_iterative_procedure(
        alphas=alphas_for_map,
        x_0=x0_fixed,
        a=a_fixed,
        f=f_map,
        max_epochs=max_epochs
    )
    eps_from_map = evolution_matrix[-1, :] if evolution_matrix.shape[0] > 0 else np.full_like(alphas_for_map, np.nan)

    # --- 3. Creazione del Grafico ---
    print("--- Calcolo Completato. Inizio Plotting. ---")
    fig, ax = plt.subplots(figsize=(14, 9))

    # Plotta Curva Teoria Parametrica
    ax.plot(
        alpha_parametric,
        eps_parametric,
        color='black',
        linestyle='--',
        linewidth=3,
        label='Teoria Annealed (Parametrica)'
    )

    # Plotta Punti Fissi Trovati
    ax.plot(
        alphas_for_map,
        eps_from_map,
        'o',
        markersize=8,
        color='limegreen',
        markeredgecolor='black',
        label=f'Punti Fissi da Dinamica ({f_map.__name__}, a={a_fixed})'
    )

    # --- 4. Personalizzazione Grafico ---
    ax.set_title(r"Curva di Apprendimento (Annealed) - $\epsilon^*$ vs $\alpha$", fontsize=20, pad=15)
    ax.set_xlabel(r"$\alpha = p/N$", fontsize=16)
    ax.set_ylabel(r"$\epsilon^*$", fontsize=16)
    ax.legend(fontsize=14)
    ax.grid(True, which='both', linestyle='--', alpha=0.6)
    ax.set_xlim(left=0, right=10)
    ax.set_ylim(bottom=-0.02, top=0.55)
    ax.tick_params(axis='both', which='major', labelsize=12)

    plt.tight_layout()

    if save_figure:
        filename = "annealed_theory_vs_dynamics.png"
        print(f"\nSalvataggio figura in: {filename}")
        fig.savefig(filename, dpi=300, bbox_inches='tight')

    plt.show()

def plot_results_quenched(
    a_fixed=0.5,
    r_zero_fixed=0.5,
    crossover_alpha=2.0,
    max_epochs_fixed=5000,
    plot_in_epsilon=True,
    save_figure=False
    ):
    """
    Genera il grafico che confronta la curva parametrica teorica (quenched)
    con i punti fissi calcolati tramite la procedura iterativa ibrida.
    """
    # --- 1. Parametri per la Simulazione ---

    alphas_for_fixed_point = np.linspace(0.1, 8, 100)

    # --- 2. Calcolo dei Punti Fissi tramite Procedura Ibrida ---
    # La procedura calcola sempre l'overlap R
    fixed_point_R = hybrid_iterative_procedure(
        alphas=alphas_for_fixed_point,
        eps_0=r_zero_fixed,
        a=a_fixed,
        f1=f1_quenched,
        f2=f2_quenched,
        crossover_alpha=crossover_alpha,
        max_epochs=max_epochs_fixed
    )

    # --- 3. Generazione della Curva Parametrica Teorica ---
    print("\n--- Calcolo della Curva Parametrica Teorica ---")
    r_values_parametric = np.linspace(0.001, 0.999, 500)
    alpha_values_parametric = parametric_solver_quenched(r_values_parametric)

    # --- 4. Conversione a Epsilon (se richiesta) e Impostazione Etichette ---
    if plot_in_epsilon:
        # Conversione dei dati
        y_values_parametric = (1 / np.pi) * np.arccos(r_values_parametric)
        y_values_fixed_point = (1 / np.pi) * np.arccos(fixed_point_R)

        # Impostazione etichette e limiti per epsilon
        title = r"Curva di Apprendimento (Quenched) - $\epsilon^*$ vs $\alpha$"
        ylabel = r"$\epsilon^*$"
        ylim = (-0.02, 0.52)
    else:
        # I dati rimangono in termini di R
        y_values_parametric = r_values_parametric
        y_values_fixed_point = fixed_point_R

        # Impostazione etichette e limiti per R
        title = r"Curva di Apprendimento (Quenched) - $R^*$ vs $\alpha$"
        ylabel = r"$R^*$"
        ylim = (-0.05, 1.05)

    # --- 5. Creazione del Grafico ---
    print("--- Calcolo Completato. Inizio Plotting. ---")
    fig, ax = plt.subplots(figsize=(14, 9))

    # Plotta la curva parametrica teorica
    ax.plot(
        alpha_values_parametric,
        y_values_parametric, # <-- Usa i valori y corretti (R o epsilon)
        color='purple',
        linestyle='--',
        linewidth=2.5,
        label='Teoria Quenched (Parametrica)'
    )

    # Plotta i punti fissi trovati iterativamente
    ax.plot(
        alphas_for_fixed_point,
        y_values_fixed_point, # <-- Usa i valori y corretti (R o epsilon)
        'o',
        markersize=7,
        color='limegreen',
        markeredgecolor='black',
        label=f'Punti Fissi da Dinamica Ibrida (a={a_fixed})'
    )

    ax.axvline(x=crossover_alpha, color='red', linestyle=':', linewidth=2, label=f'Crossover $\\alpha_c = {crossover_alpha}$')

    # --- 6. Personalizzazione del Grafico ---
    ax.set_title(title, fontsize=20, pad=15)
    ax.set_xlabel(r"$\alpha = P/N$", fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    ax.legend(fontsize=14)
    ax.grid(True, which='both', linestyle='--', alpha=0.6)

    ax.set_xlim(left=0, right=8)
    ax.set_ylim(bottom=ylim[0], top=ylim[1])
    ax.tick_params(axis='both', which='major', labelsize=12)

    plt.tight_layout()
    
    if save_figure:
        filename = "quenched_comparison_plot_final.png"
        print(f"\nSalvataggio figura in: {filename}")
        fig.savefig(filename, dpi=300, bbox_inches='tight')
    
    plt.show()

# ==============================================================================
#     FUNZIONI DI VISUALIZZAZIONE RISULTATI SPERIMENTALI
# ==============================================================================

def plot_pseudo_inverse_results(results, save_figure=True):
    """
    Plotta le curve di errore sperimentali per l'esperimento con regola della pseudo-inversa.
    """
    print("\n--- Esecuzione Plot dei Risultati della Pseudo-Inverse Rule (Solo Sperimentali) ---")

    res = results
    pseudo_inverse_specific_key = 'runs_num_pseudo_inverse'
    if pseudo_inverse_specific_key not in res:
        raise ValueError("Attenzione, i risultati caricati non sono quelli relativi all'esperimento con pseudo-inversa.")

    # --- Estrazione e calcolo dati ---
    train_sizes = np.array(res['train_sizes_pseudo_inverse'])
    runs_num = res['runs_num_pseudo_inverse']
    N_dim = res['N_dimension_pseudo_inverse']

    alpha_exp = train_sizes / N_dim
    train_error_mean = np.array(res['train_error_means_pseudo_inverse'])
    gen_error_mean = np.array(res['gen_error_means_pseudo_inverse'])
    train_sem = np.array(res['train_error_stds_pseudo_inverse']) / np.sqrt(runs_num)
    gen_sem = np.array(res['gen_error_stds_pseudo_inverse']) / np.sqrt(runs_num)

    # --- FIGURA 1: Errori Sperimentali Separati ---
    fig1, axes = plt.subplots(nrows=1, ncols=2, figsize=(22, 10))
    fig1.suptitle(f'Apprendimento Pseudo-Inversa: Errori Sperimentali (N={N_dim})', fontsize=20)

    # Pannello 1: Errore di Training
    ax1 = axes[0]
    ax1.errorbar(alpha_exp, train_error_mean, yerr=train_sem, fmt='o', markersize=5, capsize=4, label='Dati Sperimentali (con SEM)', color='royalblue')
    ax1.set_title('Errore di Training', fontsize=18)
    ax1.set_xlabel(r'$\alpha = P/N$', fontsize=18)
    ax1.set_ylabel(r'$\epsilon_{train}$', fontsize=18)
    ax1.legend(fontsize=16)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.set_ylim(bottom=-0.005)
    ax1.set_xlim(0, 1.2)
    ax1.tick_params(axis='both', which='major', labelsize=16)

    # Pannello 2: Errore di Generalizzazione
    ax2 = axes[1]
    ax2.errorbar(alpha_exp, gen_error_mean, yerr=gen_sem, fmt='o', markersize=5, capsize=4, label='Dati Sperimentali (con SEM)', color='forestgreen')
    ax2.set_title('Errore di Generalizzazione', fontsize=18)
    ax2.set_xlabel(r'$\alpha = P/N$', fontsize=18)
    ax2.set_ylabel(r'$\epsilon_{gen}$', fontsize=18)
    ax2.legend(fontsize=16)
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.set_ylim(bottom=-0.005)
    ax2.set_xlim(0, 1.2)
    ax2.tick_params(axis='both', which='major', labelsize=16)

    fig1.tight_layout(rect=[0, 0.03, 1, 0.95])
    if save_figure:
        filename1 = f"pseudo_inverse_comparison_N{N_dim}.png"
        fig1.savefig(filename1, dpi=150)
        print(f"Grafico 1 salvato come '{filename1}'")
    plt.show()

def plot_adaline_results(results, save_figure=True):
    """
    Plotta le curve di errore sperimentali per l'esperimento con regola di Adaline.
    """
    print("\n--- Esecuzione Plot dei Risultati della Adaline Rule (Solo Sperimentali) ---")

    res = results
    adaline_specific_key = 'runs_num_adaline'
    if adaline_specific_key not in res:
        raise ValueError("Attenzione, i risultati caricati non sono quelli relativi all'esperimento con Adaline Rule.")

    # --- Estrazione e calcolo dati ---
    train_sizes = np.array(res['train_sizes_adaline'])
    runs_num = res['runs_num_adaline']
    N_dim = res['N_dimension_adaline']

    alpha_exp = train_sizes / N_dim
    train_error_mean = np.array(res['train_error_means_adaline'])
    gen_error_mean = np.array(res['gen_error_means_adaline'])
    train_sem = np.array(res['train_error_stds_adaline']) / np.sqrt(runs_num)
    gen_sem = np.array(res['gen_error_stds_adaline']) / np.sqrt(runs_num)

    # --- FIGURA 1: Errori Sperimentali Separati (Full Range) ---
    fig1, axes = plt.subplots(nrows=1, ncols=2, figsize=(22, 10))
    fig1.suptitle(f'Apprendimento Adaline Rule: Errori Sperimentali (N={N_dim}) - Full Range', fontsize=20)

    # Pannello 1: Errore di Training
    ax1 = axes[0]
    ax1.errorbar(alpha_exp, train_error_mean, yerr=train_sem, fmt='o', markersize=5, capsize=4, label='Dati Sperimentali (con SEM)', color='royalblue')
    ax1.set_title('Errore di Training', fontsize=18)
    ax1.set_xlabel(r'$\alpha = P/N$', fontsize=18)
    ax1.set_ylabel(r'$\epsilon_{train}$', fontsize=18)
    ax1.legend(fontsize=16)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.set_ylim(bottom=-0.005)
    ax1.set_xlim(0, 11)
    ax1.tick_params(axis='both', which='major', labelsize=16)

    # Pannello 2: Errore di Generalizzazione
    ax2 = axes[1]
    ax2.errorbar(alpha_exp, gen_error_mean, yerr=gen_sem, fmt='o', markersize=5, capsize=4, label='Dati Sperimentali (con SEM)', color='forestgreen')
    ax2.set_title('Errore di Generalizzazione', fontsize=18)
    ax2.set_xlabel(r'$\alpha = P/N$', fontsize=18)
    ax2.set_ylabel(r'$\epsilon_{gen}$', fontsize=18)
    ax2.legend(fontsize=16)
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.set_ylim(bottom=-0.005)
    ax2.set_xlim(0, 11)
    ax2.tick_params(axis='both', which='major', labelsize=16)

    fig1.tight_layout(rect=[0, 0.03, 1, 0.95])
    if save_figure:
        filename1 = f"adaline_comparison_N{N_dim}_full.png"
        fig1.savefig(filename1, dpi=150)
        print(f"Grafico 1 salvato come '{filename1}'")
    plt.show()

    # --- FIGURA 2: Errori Sperimentali Separati (Zoom 0-2) ---
    fig2, axes2 = plt.subplots(nrows=1, ncols=2, figsize=(22, 10))
    fig2.suptitle(f'Apprendimento Adaline Rule: Errori Sperimentali (N={N_dim}) - Zoom [0, 2]', fontsize=20)

    # Pannello 1: Errore di Training (Zoom)
    ax3 = axes2[0]
    ax3.errorbar(alpha_exp, train_error_mean, yerr=train_sem, fmt='o', markersize=5, capsize=4, label='Dati Sperimentali (con SEM)', color='royalblue')
    ax3.set_title('Errore di Training (Zoom)', fontsize=18)
    ax3.set_xlabel(r'$\alpha = P/N$', fontsize=18)
    ax3.set_ylabel(r'$\epsilon_{train}$', fontsize=18)
    ax3.legend(fontsize=16)
    ax3.grid(True, linestyle='--', alpha=0.6)
    ax3.set_ylim(bottom=-0.05)
    ax3.set_xlim(0, 2)
    ax3.tick_params(axis='both', which='major', labelsize=16)

    # Pannello 2: Errore di Generalizzazione (Zoom)
    ax4 = axes2[1]
    ax4.errorbar(alpha_exp, gen_error_mean, yerr=gen_sem, fmt='o', markersize=5, capsize=4, label='Dati Sperimentali (con SEM)', color='forestgreen')
    ax4.set_title('Errore di Generalizzazione (Zoom)', fontsize=18)
    ax4.set_xlabel(r'$\alpha = P/N$', fontsize=18)
    ax4.set_ylabel(r'$\epsilon_{gen}$', fontsize=18)
    ax4.legend(fontsize=16)
    ax4.grid(True, linestyle='--', alpha=0.6)
    ax4.set_ylim(bottom=-0.05)
    ax4.set_xlim(0, 2)
    ax4.tick_params(axis='both', which='major', labelsize=16)

    fig2.tight_layout(rect=[0, 0.03, 1, 0.95])
    if save_figure:
        filename2 = f"adaline_comparison_N{N_dim}_zoom.png"
        fig2.savefig(filename2, dpi=150)
        print(f"Grafico 2 salvato come '{filename2}'")
    plt.show()

def plot_comparison_adaline_pseudoinverse(results, save_figure=True):
    """
    Confronta i risultati sperimentali di Adaline e Pseudo-Inverse.
    """
    print("\n--- Esecuzione Plot di Confronto Adaline vs Pseudo-Inverse ---")

    res = results
    if 'runs_num_adaline' not in res:
            raise ValueError("Mancano i dati dell'esperimento Adaline per il confronto.")
    if 'runs_num_pseudo_inverse' not in res:
            raise ValueError("Mancano i dati dell'esperimento Pseudo-Inverse per il confronto.")

    # --- Estrazione Dati Adaline ---
    N_adaline = res['N_dimension_adaline']
    alpha_adaline = np.array(res['train_sizes_adaline']) / N_adaline
    train_err_adaline = np.array(res['train_error_means_adaline'])
    gen_err_adaline = np.array(res['gen_error_means_adaline'])
    
    # --- Estrazione Dati Pseudo-Inverse ---
    N_pinv = res['N_dimension_pseudo_inverse']
    alpha_pinv = np.array(res['train_sizes_pseudo_inverse']) / N_pinv
    train_err_pinv = np.array(res['train_error_means_pseudo_inverse'])
    gen_err_pinv = np.array(res['gen_error_means_pseudo_inverse'])

    # --- FIGURA CONFRONTO ---
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(22, 10))
    fig.suptitle(f'Confronto Adaline vs Pseudo-Inverse', fontsize=20)

    # Pannello 1: Confronto Errore di Training
    ax1 = axes[0]
    ax1.plot(alpha_adaline, train_err_adaline, 'o-', label='Adaline Train Error', color='blue', markersize=5)
    ax1.plot(alpha_pinv, train_err_pinv, 's-', label='Pseudo-Inverse Train Error', color='cyan', markersize=5)
    ax1.set_title('Confronto Errore di Training', fontsize=18)
    ax1.set_xlabel(r'$\alpha = P/N$', fontsize=18)
    ax1.set_ylabel(r'$\epsilon_{train}$', fontsize=18)
    ax1.set_xlim(0, 1.1)
    ax1.set_ylim(bottom=-0.05)
    ax1.legend(fontsize=16)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.tick_params(axis='both', which='major', labelsize=16)

    # Pannello 2: Confronto Errore di Generalizzazione
    ax2 = axes[1]
    ax2.plot(alpha_adaline, gen_err_adaline, 'o-', label='Adaline Gen Error', color='green', markersize=5)
    ax2.plot(alpha_pinv, gen_err_pinv, 's-', label='Pseudo-Inverse Gen Error', color='lime', markersize=5)
    ax2.set_title('Confronto Errore di Generalizzazione', fontsize=18)
    ax2.set_xlabel(r'$\alpha = P/N$', fontsize=18)
    ax2.set_ylabel(r'$\epsilon_{gen}$', fontsize=18)
    ax2.set_xlim(0, 1.1)
    ax2.set_ylim(bottom=-0.05)
    ax2.legend(fontsize=16)
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.tick_params(axis='both', which='major', labelsize=16)

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    if save_figure:
        filename = "adaline_vs_pseudoinverse_comparison.png"
        fig.savefig(filename, dpi=150)
        print(f"Grafico di confronto salvato come '{filename}'")
    plt.show()

def plot_hebb_results(results, save_figure=True):
    """
    Plotta le curve di errore sperimentali e teoriche per l'esperimento di Hebb.
    """
    print("\n--- Esecuzione Plot dei Risultati della Hebb Rule ---")
    
    res = results
    hebb_specific_key = 'runs_num_hebb'
    if hebb_specific_key not in res:
        raise ValueError("Attenzione, i risultati caricati non sono quelli relativi all'esperimento .run_experiment_hebb.")

    # --- Estrazione e calcolo dati ---
    train_sizes = np.array(res['train_sizes_hebb'])
    runs_num = res['runs_num_hebb']
    N_dim = res['N_dimension_hebb']
    
    alpha_exp = train_sizes / N_dim
    train_error_mean = np.array(res['train_error_means_hebb'])
    gen_error_mean = np.array(res['gen_error_means_hebb'])
    train_sem = np.array(res['train_error_stds_hebb']) / np.sqrt(runs_num)
    gen_sem = np.array(res['gen_error_stds_hebb']) / np.sqrt(runs_num)

    alpha_analytical = np.linspace(max(0.01, alpha_exp.min()), alpha_exp.max(), 400)
    train_error_analytical = hebb_train_error(alpha_analytical)
    gen_error_analytical = hebb_gen_error(alpha_analytical)

    # --- FIGURA 1: Errori Separati ---
    fig1, axes = plt.subplots(nrows=1, ncols=2, figsize=(22, 10))
    fig1.suptitle(f'Apprendimento Hebbiano: Confronto Teorico-Sperimentale (N={N_dim})', fontsize=20)

    # Pannello 1: Errore di Training
    ax1 = axes[0]
    ax1.errorbar(alpha_exp, train_error_mean, yerr=train_sem, fmt='o', markersize=5, capsize=4, label='Dati Sperimentali (con SEM)', color='royalblue')
    ax1.plot(alpha_analytical, train_error_analytical, '-', label='Curva Teorica', color='darkorange', linewidth=2.5)
    ax1.set_title('Errore di Training', fontsize=18)
    ax1.set_xlabel(r'$\alpha = P/N$', fontsize=18)
    ax1.set_ylabel(r'$\epsilon_{train}$', fontsize=18)
    ax1.legend(fontsize=16)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.set_ylim(bottom=0)
    ax1.set_xlim(left=0)
    ax1.tick_params(axis='both', which='major', labelsize=16)

    # Pannello 2: Errore di Generalizzazione
    ax2 = axes[1]
    ax2.errorbar(alpha_exp, gen_error_mean, yerr=gen_sem, fmt='o', markersize=5, capsize=4, label='Dati Sperimentali (con SEM)', color='forestgreen')
    ax2.plot(alpha_analytical, gen_error_analytical, '-', label='Curva Teorica', color='crimson', linewidth=2.5)
    ax2.set_title('Errore di Generalizzazione', fontsize=18)
    ax2.set_xlabel(r'$\alpha = P/N$', fontsize=18)
    ax2.set_ylabel(r'$\epsilon_{gen}$', fontsize=18)
    ax2.legend(fontsize=16)
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.set_ylim(bottom=0)
    ax2.set_xlim(left=0)
    ax2.tick_params(axis='both', which='major', labelsize=16)

    fig1.tight_layout(rect=[0, 0.03, 1, 0.95])
    if save_figure:
        filename1 = f"hebb_learning_comparison_N{N_dim}.png"
        fig1.savefig(filename1, dpi=150)
        print(f"Grafico 1 salvato come '{filename1}'")
    plt.show()

    # --- FIGURA 2: Analisi Comparativa degli Errori ---
    
    diff_exp = gen_error_mean - train_error_mean
    diff_sem = np.sqrt(gen_sem**2 + train_sem**2)
    diff_analytical = gen_error_analytical - train_error_analytical

    fig2, ax_main = plt.subplots(figsize=(18, 11))
    
    # Plot della differenza (in primo piano)
    ax_main.errorbar(alpha_exp, diff_exp, yerr=diff_sem, fmt='o', markersize=5, capsize=4, label=r'Sperimentale ($\epsilon_{gen} - \epsilon_{train}$)', color='purple')
    ax_main.plot(alpha_analytical, diff_analytical, '-', label=r'Teorica ($\epsilon_{gen} - \epsilon_{train}$)', color='magenta', linewidth=2.5)
    
    # Aggiunta delle curve di errore individuali (in secondo piano)
    # Errore di generalizzazione
    ax_main.errorbar(alpha_exp, gen_error_mean, yerr=gen_sem, fmt='o', markersize=4, capsize=3, label=r'Sperimentale ($\epsilon_{gen}$)', color='forestgreen', alpha=0.7)
    ax_main.plot(alpha_analytical, gen_error_analytical, '--', label=r'Teorica ($\epsilon_{gen}$)', color='crimson', linewidth=2)
    
    # Errore di training
    ax_main.errorbar(alpha_exp, train_error_mean, yerr=train_sem, fmt='o', markersize=4, capsize=3, label=r'Sperimentale ($\epsilon_{train}$)', color='royalblue', alpha=0.7)
    ax_main.plot(alpha_analytical, train_error_analytical, '--', label=r'Teorica ($\epsilon_{train}$)', color='darkorange', linewidth=2)

    ax_main.set_title(f'Analisi Comparativa degli Errori (N={N_dim})', fontsize=20)
    ax_main.set_xlabel(r'$\alpha = P/N$', fontsize=18)
    ax_main.set_ylabel('Errore', fontsize=18)
    ax_main.legend(fontsize=14) # Leggermente più piccolo per contenere tutte le etichette
    ax_main.grid(True, linestyle='--', alpha=0.6)
    ax_main.tick_params(axis='both', which='major', labelsize=16)
    ax_main.set_xlim(left=0)
    ax_main.set_ylim(bottom=0)

    fig2.tight_layout()
    if save_figure:
        filename2 = f"hebb_error_analysis_N{N_dim}.png"
        fig2.savefig(filename2, dpi=150)
        print(f"Grafico 2 salvato come '{filename2}'")
    plt.show()

def plot_advanced_comparison(results, save_figure=True):
    """
    Plotta il confronto completo tra dati sperimentali (Perceptron, Hebb) e le teorie.
    """
    print("\n--- Esecuzione Plot di Confronto Completo ---")

    res = results

    required_experiments = {
        'hebb': 'runs_num_hebb',
        'perceptron_noise': 'train_sizes_20',
        'perceptron_zero_noise': 'runs_num_perceptron_zero_noise'
    }
    missing_experiments = [name for name, key in required_experiments.items() if key not in res]
    if missing_experiments:
        raise RuntimeError(
            f"Dati incompleti. Mancano i risultati per: {', '.join(missing_experiments)}. "
            "Assicurati di aver eseguito tutte le simulazioni."
        )
    print("Tutti i set di dati necessari sono stati trovati.")

    # --- 2. Preparazione Dati Sperimentali ---
    # Perceptron con rumore (sigma > 0)
    alpha_20 = np.array(res['train_sizes_20']) / 20
    sem_gen_20 = res['error_stds_20'] / np.sqrt(res.get('runs_num', 1000))
    alpha_40 = np.array(res['train_sizes_40']) / 40
    sem_gen_40 = res['error_stds_40'] / np.sqrt(res.get('runs_num', 1000))

    # Perceptron senza rumore (sigma = 0)
    N_pzn = res['N_dimension_perceptron_zero_noise']
    alpha_pzn = np.array(res['train_sizes_perceptron_zero_noise']) / N_pzn
    gen_error_mean_pzn = res['gen_error_perceptron_zero_noise']
    sem_gen_pzn = res['gen_error_stds_perceptron_zero_noise'] / np.sqrt(res['runs_num_perceptron_zero_noise'])
    
    # Hebb (training, generalizzazione e loro differenza)
    N_hebb = res["N_dimension_hebb"]
    alpha_hebb_exp = np.array(res['train_sizes_hebb']) / N_hebb
    gen_error_mean_hebb = np.array(res['gen_error_means_hebb'])
    sem_gen_hebb = np.array(res['gen_error_stds_hebb']) / np.sqrt(res['runs_num_hebb'])
    train_error_mean_hebb = np.array(res['train_error_means_hebb'])
    sem_train_hebb = np.array(res['train_error_stds_hebb']) / np.sqrt(res['runs_num_hebb'])
    
    # Calcolo della differenza e propagazione dell'errore (SEM)
    diff_error_mean_hebb = gen_error_mean_hebb - train_error_mean_hebb
    sem_diff_hebb = np.sqrt(sem_gen_hebb**2 + sem_train_hebb**2)


    # --- 3. Calcolo Curve Teoriche ---
    print("Calcolo teorie Annealed, Quenched e Hebb...")
    alphas_theory = np.linspace(0.01, 15, 400)
    
    # Teorie per l'errore di generalizzazione
    eps_param_annealed = np.linspace(1e-9, 0.5 - 1e-9, 500)
    alpha_param_annealed = parametric_alphas_annealed(eps_param_annealed)
    r_param_quenched = np.linspace(0.001, 0.999, 500)
    alpha_param_quenched = parametric_solver_quenched(r_param_quenched)
    eps_param_quenched = (1 / np.pi) * np.arccos(r_param_quenched)
    
    # Teorie per la regola di Hebb
    eps_hebb_gen_analytical = hebb_gen_error(alphas_theory)
    eps_hebb_train_analytical = hebb_train_error(alphas_theory)
    diff_hebb_analytical = eps_hebb_gen_analytical - eps_hebb_train_analytical

    print("--- Calcoli Teorici Completati. Inizio Plotting. ---")

    # --- 4. Creazione del Grafico ---
    fig, ax = plt.subplots(figsize=(20, 13))
    
    # Plot Teorie
    ax.plot(alpha_param_quenched, eps_param_quenched, color='black', linestyle='--', lw=3, label=r'Teoria Quenched ($\epsilon_{gen}$)')
    ax.plot(alpha_param_annealed, eps_param_annealed, color='deepskyblue', linestyle=':', lw=3, label=r'Teoria Annealed ($\epsilon_{gen}$)')
    ax.plot(alphas_theory, eps_hebb_gen_analytical, color='purple', linestyle='-.', lw=3, label=r'Teoria Hebb ($\epsilon_{gen}$)')
    ax.plot(alphas_theory, eps_hebb_train_analytical, color='magenta', linestyle=':', lw=3, label=r'Teoria Hebb ($\epsilon_{train}$)')
    ax.plot(alphas_theory, diff_hebb_analytical, color='orange', linestyle='--', lw=2.5, label=r'Teoria Hebb ($\epsilon_{gen} - \epsilon_{train}$)')

    # Plot Dati Sperimentali
    label_hebb_gen = f'Sperimentale Hebb N={N_hebb} ' + r'($\epsilon_{gen}$)'
    label_hebb_train = f'Sperimentale Hebb N={N_hebb} ' + r'($\epsilon_{train}$)'
    label_hebb_diff = f'Sperimentale Hebb N={N_hebb} ' + r'($\epsilon_{gen} - \epsilon_{train}$)'
    label_pzn_gen = f'Sperimentale Perceptron N={N_pzn} ($\\sigma=0$) ' + r'($\epsilon_{gen}$)'
    
    ax.errorbar(alpha_20, res['error_means_20'], yerr=sem_gen_20, fmt='o', markersize=8, capsize=5, label=r'Sperimentale Perceptron N=20 ($\sigma>0, \epsilon_{gen}$)', color='royalblue', zorder=10)
    ax.errorbar(alpha_40, res['error_means_40'], yerr=sem_gen_40, fmt='s', markersize=8, capsize=5, label=r'Sperimentale Perceptron N=40 ($\sigma>0, \epsilon_{gen}$)', color='red', zorder=10)
    ax.errorbar(alpha_hebb_exp, gen_error_mean_hebb, yerr=sem_gen_hebb, fmt='D', markersize=8, capsize=5, label=label_hebb_gen, color='darkviolet', zorder=10)
    # Modifica: rimosso linestyle=':' per non collegare i punti
    ax.errorbar(alpha_hebb_exp, train_error_mean_hebb, yerr=sem_train_hebb, fmt='D', markerfacecolor='none', markeredgecolor='darkviolet', markersize=8, capsize=5, label=label_hebb_train, linestyle='None', zorder=9)
    # Aggiunta: nuova curva per la differenza degli errori
    ax.errorbar(alpha_hebb_exp, diff_error_mean_hebb, yerr=sem_diff_hebb, fmt='^', markersize=8, capsize=5, label=label_hebb_diff, color='darkorange', zorder=10)
    ax.errorbar(alpha_pzn, gen_error_mean_pzn, yerr=sem_gen_pzn, fmt='*', markersize=12, capsize=5, label=label_pzn_gen, color='limegreen', markeredgecolor='black', zorder=11)

    # --- 5. Personalizzazione ---
    ax.set_title(r"Confronto Completo: Dati Sperimentali vs Teorie", fontsize=22, pad=15)
    ax.set_xlabel(r"$\alpha = P/N$", fontsize=18)
    ax.set_ylabel(r"$\langle \epsilon \rangle$ (Errore Medio)", fontsize=18)
    
    # Ordine personalizzato per una legenda più leggibile
    handles, labels = ax.get_legend_handles_labels()
    # Mappatura per raggruppare le curve per tipo
    order = [
        # Perceptron
        7, 8, 6,
        # Hebb
        9, 10, 11,
        # Teorie
        0, 1, 2, 3, 4, 5
    ]
    # Filtra indici non validi se il numero di elementi è cambiato
    valid_order = [i for i in order if i < len(handles)]
    ax.legend([handles[idx] for idx in valid_order], [labels[idx] for idx in valid_order], fontsize=14, ncol=2)
    
    ax.grid(True, which='both', linestyle='--', alpha=0.6)
    ax.set_xlim(left=0, right=10.5)
    ax.set_ylim(bottom=-0.02, top=0.55)
    ax.tick_params(axis='both', which='major', labelsize=14)
    
    fig.tight_layout()

    if save_figure:
        filename = "master_comparison_plot_full_with_zero_noise.png"
        print(f"\nSalvataggio figura in: {filename}")
        fig.savefig(filename, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_bayes_results(results, save_figure=True):
    """
    Plotta le curve di errore sperimentali per l'esperimento con regola di Bayes.
    """
    print("\n--- Esecuzione Plot dei Risultati della Bayes Rule (Solo Sperimentali) ---")

    res = results
    bayes_specific_key = 'runs_num_bayes'
    if bayes_specific_key not in res:
        raise ValueError("Attenzione, i risultati caricati non sono quelli relativi all'esperimento con Bayes Rule.")

    # --- Estrazione e calcolo dati ---
    train_sizes = np.array(res['train_sizes_bayes'])
    runs_num = res['runs_num_bayes']
    N_dim = res['N_dimension_bayes']

    alpha_exp = train_sizes / N_dim
    train_error_mean = np.array(res['train_error_means_bayes'])
    gen_error_mean = np.array(res['gen_error_means_bayes'])
    train_sem = np.array(res['train_error_stds_bayes']) / np.sqrt(runs_num)
    gen_sem = np.array(res['gen_error_stds_bayes']) / np.sqrt(runs_num)

    # --- FIGURA 1: Errori Sperimentali Separati ---
    fig1, axes = plt.subplots(nrows=1, ncols=2, figsize=(22, 10))
    fig1.suptitle(f'Apprendimento Bayes Rule: Errori Sperimentali (N={N_dim})', fontsize=20)

    # Pannello 1: Errore di Training
    ax1 = axes[0]
    ax1.errorbar(alpha_exp, train_error_mean, yerr=train_sem, fmt='o', markersize=5, capsize=4, label='Dati Sperimentali (con SEM)', color='royalblue')
    ax1.set_title('Errore di Training', fontsize=18)
    ax1.set_xlabel(r'$\alpha = P/N$', fontsize=18)
    ax1.set_ylabel(r'$\epsilon_{train}$', fontsize=18)
    ax1.legend(fontsize=16)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.set_ylim(bottom=-0.005)
    ax1.tick_params(axis='both', which='major', labelsize=16)

    # Pannello 2: Errore di Generalizzazione
    ax2 = axes[1]
    ax2.errorbar(alpha_exp, gen_error_mean, yerr=gen_sem, fmt='o', markersize=5, capsize=4, label='Dati Sperimentali (con SEM)', color='forestgreen')
    ax2.set_title('Errore di Generalizzazione', fontsize=18)
    ax2.set_xlabel(r'$\alpha = P/N$', fontsize=18)
    ax2.set_ylabel(r'$\epsilon_{gen}$', fontsize=18)
    ax2.legend(fontsize=16)
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.set_ylim(bottom=-0.005)
    ax2.tick_params(axis='both', which='major', labelsize=16)

    fig1.tight_layout(rect=[0, 0.03, 1, 0.95])
    if save_figure:
        filename1 = f"bayes_comparison_N{N_dim}.png"
        fig1.savefig(filename1, dpi=150)
        print(f"Grafico 1 salvato come '{filename1}'")
    plt.show()
