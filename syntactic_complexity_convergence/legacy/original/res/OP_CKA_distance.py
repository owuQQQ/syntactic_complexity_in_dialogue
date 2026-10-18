import pandas as pd
import numpy as np
from scipy.linalg import svdvals
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from scipy.stats import linregress
import os

def min_max_normalize_neg1_pos1(M):
    m_min = np.min(M, axis=0)
    m_max = np.max(M, axis=0)
    diff = m_max - m_min
    diff[diff == 0] = 1.0 
    return 2 * ((M - m_min) / diff) - 1

def interpolate_matrix(M, target_len=40):
    T, D = M.shape
    if T == 1:
        return np.repeat(M, target_len, axis=0)
    x = np.linspace(0, 1, T)
    x_new = np.linspace(0, 1, target_len)
    f = interp1d(x, M, axis=0, kind='linear')
    return f(x_new)

def calculate_cka_distance(X, Y):
    XY = X.T @ Y
    XX = X.T @ X
    YY = Y.T @ Y
    frob_norm_sq = np.linalg.norm(XY, ord='fro')**2
    denom = np.linalg.norm(XX, ord='fro') * np.linalg.norm(YY, ord='fro')
    if denom == 0: return 1.0
    return 1 - (frob_norm_sq / denom)

def process_dataset(dataset_name, n_phases=8, min_turns=None):
    min_len = n_phases if min_turns is None else min_turns
    print(f"\n>>> Processing Dataset: {dataset_name} | phases={n_phases}, min_turns={min_len}")

    init_file = f'{dataset_name}_initiator_symcom.txt'
    foll_file = f'{dataset_name}_follower_symcom.txt'
    
    if not os.path.exists(init_file) or not os.path.exists(foll_file):
        print(f"Skipping {dataset_name}: Files not found.")
        return None

    df_init = pd.read_csv(init_file, sep='\t')
    df_foll = pd.read_csv(foll_file, sep='\t')

    features = ['syn_value', 'dependency_length', 'tree_depth', 'complex_dependency_ratio']
    dialogues = sorted(list(set(df_init['Index'].unique()) & set(df_foll['Index'].unique())))

    final_results = []
    for dia_id in dialogues:
        init_dia = df_init[df_init['Index'] == dia_id][features].values
        foll_dia = df_foll[df_foll['Index'] == dia_id][features].values

        if len(init_dia) >= min_len and len(foll_dia) >= min_len:
            init_phases = np.array_split(init_dia, n_phases)
            foll_phases = np.array_split(foll_dia, n_phases)

            for i in range(n_phases):
                X = min_max_normalize_neg1_pos1(interpolate_matrix(init_phases[i]))
                Y = min_max_normalize_neg1_pos1(interpolate_matrix(foll_phases[i]))
                cka = calculate_cka_distance(X, Y)
                final_results.append({'Dialogue_ID': dia_id, 'Phase': i + 1, 'CKA_Distance': cka})

    if not final_results:
        return None

    results_df = pd.DataFrame(final_results)
    phase_summary = results_df.groupby('Phase')[['CKA_Distance']].mean().reset_index()

    slope, intercept, r_val, p_val, std_err = linregress(phase_summary['Phase'], phase_summary['CKA_Distance'])
    r_sq = r_val**2
    
    print(f"Slope: {slope:.6f}, R-square:{r_sq:.6f}, P-value: {p_val:.6f}")

    plt.figure(figsize=(10, 6))
    plt.scatter(phase_summary['Phase'], phase_summary['CKA_Distance'], color='red', alpha=0.5, label='Mean CKA')
    
    line = slope * phase_summary['Phase'] + intercept
    plt.plot(phase_summary['Phase'], line, color='red', linewidth=2)
    
    stats_text = (f'Slope: {slope:.5f}\n'
                  f'$R^2$: {r_sq:.4f}\n'
                  f'$p$-value: {p_val:.6f}')
    
    plt.text(0.95, 0.95, stats_text, transform=plt.gca().transAxes, 
             fontsize=12, verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.xlabel(f'Dialogue Phase (1-{n_phases})')
    plt.ylabel('CKA Distance')
    plt.title(f'Syntactic Alignment Trend: {dataset_name} (min_turns={min_len})')
    plt.xticks(range(1, n_phases + 1))
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    
    plt.savefig(f'cka_plot_{n_phases}phases_min{min_len}_{dataset_name}.png')
    plt.close()

    return {
        'Dataset': dataset_name,
        'Min_Turns': min_len,
        'Phases': n_phases,
        'Slope': slope,
        'R_Squared': r_sq,
        'Std_Error': std_err,
        'P_Value': p_val
    }

if __name__ == '__main__':
    datasets = ['btsj', 'combined_candor', 'Hazumi', 'A2', 'maptask', 'llm_german', 'llm_english']
    min_turns_list = [16, 24, 32, 36]
    phases_list = [8, 16, 32]
    stats_by_phase_min_turns = {
        n_phases: {min_turns: [] for min_turns in min_turns_list}
        for n_phases in phases_list
    }

    for ds in datasets:
        for n_phases in phases_list:
            for min_turns in min_turns_list:
                stats = process_dataset(ds, n_phases=n_phases, min_turns=min_turns)
                if stats:
                    stats_by_phase_min_turns[n_phases][min_turns].append(stats)

    # Save one CSV per phase/min_turns value
    for n_phases, stats_by_min in stats_by_phase_min_turns.items():
        for min_turns, stats_list in stats_by_min.items():
            out_name = f'all_datasets_statistics_{n_phases}phase_minturns{min_turns}.csv'
            pd.DataFrame(stats_list).to_csv(out_name, index=False)
    print("\n[Done] Results saved to CSV and PNG files.")