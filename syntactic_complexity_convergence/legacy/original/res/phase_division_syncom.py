import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.stats import sem, t
from scipy.stats import friedmanchisquare, wilcoxon

# ---------------------------
# Helper: Extract per-dialogue phase medians
# ---------------------------
def extract_phase_medians(file_path, metrics, phase_labels):
    df = pd.read_csv(file_path, sep='\t')
    data = {}
    for idx, group in df.groupby('Index'):
        if len(group) < len(phase_labels):
            continue
        phases = np.array_split(group, len(phase_labels))
        data[idx] = {metric: {} for metric in metrics}
        for i, phase in enumerate(phases):
            for metric in metrics:
                if metric in phase and not phase[metric].dropna().empty:
                    data[idx][metric][phase_labels[i]] = phase[metric].median()
    return data


def plot_diff_line_box(file1, file2):
    metrics = ['syn_value', 'dependency_length', 'tree_depth', 'complex_dependency_ratio']
    phase_labels = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8']

    data1 = extract_phase_medians(file1, metrics, phase_labels)
    data2 = extract_phase_medians(file2, metrics, phase_labels)
    shared_dialogues = list(set(data1.keys()) & set(data2.keys()))
    
    fig, axes = plt.subplots(1, 4, figsize=(22, 5), sharey=False)

    for i, metric in enumerate(metrics):
        p1_means, p2_means = [], []
        p1_cis, p2_cis = [], []

        for label in phase_labels:
            vals1, vals2 = [], []
            for idx in shared_dialogues:
                if label in data1[idx][metric] and label in data2[idx][metric]:
                    vals1.append(data1[idx][metric][label])
                    vals2.append(data2[idx][metric][label])
            
            def get_stats_with_ci(data_list):
                if data_list is None or len(data_list) < 2:
                    return 0, 0
                arr = np.array(data_list)
                avg = np.mean(arr)
                std_err = sem(arr)
                h = std_err * t.ppf((1 + 0.95) / 2, len(arr) - 1)
                return avg, (h if not np.isnan(h) else 0)

            m1, ci1 = get_stats_with_ci(vals1)
            m2, ci2 = get_stats_with_ci(vals2)
            p1_means.append(m1); p1_cis.append(ci1)
            p2_means.append(m2); p2_cis.append(ci2)

        x = np.arange(1, 9)
        
        # Plotting
        axes[i].plot(x, p1_means, marker='o', label='Initiator', color='tab:blue', linewidth=2)
        axes[i].fill_between(x, np.array(p1_means)-p1_cis, np.array(p1_means)+p1_cis, color='tab:blue', alpha=0.2)
        
        axes[i].plot(x, p2_means, marker='s', linestyle='--', label='Follower', color='tab:orange', linewidth=2)
        axes[i].fill_between(x, np.array(p2_means)-p2_cis, np.array(p2_means)+p2_cis, color='tab:orange', alpha=0.2)

        # --- ADAPTIVE Y-LIMIT LOGIC ---
        # Calculate the absolute min and max including CI boundaries
        all_lows = np.concatenate([np.array(p1_means)-p1_cis, np.array(p2_means)-p2_cis])
        all_highs = np.concatenate([np.array(p1_means)+p1_cis, np.array(p2_means)+p2_cis])
        
        data_min = np.min(all_lows)
        data_max = np.max(all_highs)
        
        # Calculate a buffer (10% of the total data spread)
        data_range = data_max - data_min
        if data_range == 0: data_range = 1.0 
        
        buffer = data_range * 0.15 
        
        # Apply adaptive limits
        axes[i].set_ylim(data_min - buffer, data_max + buffer)

        # Labels and formatting (keep only prominent phase labels on x-axis)
        axes[i].set_xticks(x)
        axes[i].set_xticklabels(phase_labels, fontweight='bold', fontsize=14)
        
        for tick in axes[i].get_yticklabels():
            tick.set_fontweight('bold')

    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    plt.savefig(f"{file1}_vs_{os.path.basename(file2)}_adaptive.png", dpi=300)
    plt.close()
# ---------------------------

def phasewise_within_condition_tests(file_path):
    metrics = ['syn_value', 'dependency_length', 'tree_depth', 'complex_dependency_ratio']
    phase_labels = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8']

    data = extract_phase_medians(file_path, metrics, phase_labels)
    results = []

    def significance_label(p):
        if p < 0.001: return "<0.001"
        elif p < 0.01: return "<0.01"
        elif p < 0.05: return "<0.05"
        else: return "ns"

    for metric in metrics:
        matrix = []
        dialogue_indices = sorted(data.keys())
        
        for idx in dialogue_indices:
            row = [data[idx][metric].get(p) for p in phase_labels]
            if None not in row:
                matrix.append(row)
        
        matrix = np.array(matrix)
        
        if len(matrix) < 5:
            print(f"Skipping {metric} for {os.path.basename(file_path)}: "
                  f"Insufficient complete cases (n={len(matrix)}).")
            continue
        stat, p = friedmanchisquare(*matrix.T)
        
        results.append({
            "metric": metric,
            "test": "Friedman (Repeated Measures)",
            "n_dialogues": len(matrix),
            "n_phases": len(phase_labels),
            "statistic": stat,
            "p_value": p,
            "significance": significance_label(p)
        })

    return pd.DataFrame(results)

if __name__ == '__main__':
    datasets = ['A2', 'btsj', 'combined_candor', 'Hazumi', 'maptask','llm_german','llm_english']

    for ds in datasets:
        print(f"\n{'='*30}")
        print(f" STATISTICAL RESULTS: {ds} ")
        print(f"{'='*30}")

        for role in ['initiator', 'follower']:
            fpath = f"{ds}_{role}_symcom.txt"
            
            # Check if file exists to avoid errors
            if not os.path.exists(fpath):
                print(f"Skipping {role}: {fpath} not found.")
                continue
                
            print(f"\n>>> Role: {role.upper()}")
            df_stats = phasewise_within_condition_tests(fpath)
            
            # Print in a clean format for easy copy-pasting
            # You can also use df_stats.to_latex() if you have Jinja2 installed
            print(df_stats.to_string(index=False))

    # Plotting section remains the same
    for ds in datasets:
        f1 = ds + '_initiator_symcom.txt'
        f2 = ds + '_follower_symcom.txt'
        
        if os.path.exists(f1) and os.path.exists(f2):
            print(f"\nGenerating plots for dataset: {ds}")
            plot_diff_line_box(f1, f2)


