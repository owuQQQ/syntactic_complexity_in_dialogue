import pandas as pd
import numpy as np

def extract_phase_medians(file_path, metrics, phase_labels):
    try:
        df = pd.read_csv(file_path, sep='\t')
    except Exception:
        return None
    
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

def classify_pattern(v1_start, v1_end, v2_start, v2_end, v1_seq=None, v2_seq=None):
    if v1_seq is not None and v2_seq is not None:
        v1_seq = np.asarray(v1_seq, dtype=float)
        v2_seq = np.asarray(v2_seq, dtype=float)
        if v1_seq.size < 2 or v2_seq.size < 2:
            return 0
        if np.any(np.isnan(v1_seq)) or np.any(np.isnan(v2_seq)):
            return 0
        diff = np.abs(v1_seq - v2_seq)
        decreases = np.sum(diff[:-1] > diff[1:])
        if not (diff[0] > diff[-1] and decreases >= 2):
            return 0

    start_diff = abs(v1_start - v2_start)
    end_diff = abs(v1_end - v2_end)

    if end_diff < start_diff:
        if v1_end < v1_start and v2_end < v2_start:
            return 1  # 
        else:
            return 2  # 
    return 0

def compute_convergence(initiator_file, follower_file, dataset_name):
    metrics = ['syn_value', 'dependency_length', 'tree_depth', 'complex_dependency_ratio']
    phase_labels = ['Phase 1', 'Phase 2', 'Phase 3', 'Phase 4', 'Phase 5','Phase 6','Phase 7', 'Phase 8']

    ini = extract_phase_medians(initiator_file, metrics, phase_labels)
    fol = extract_phase_medians(follower_file, metrics, phase_labels)
    if not ini or not fol: return pd.DataFrame()

    shared = sorted(list(set(ini.keys()) & set(fol.keys())))
    results = []

    for metric in metrics:
        p1, p2 = 0, 0
        for idx in shared:
            try:
                ini_seq = [ini[idx][metric].get(p, np.nan) for p in phase_labels]
                fol_seq = [fol[idx][metric].get(p, np.nan) for p in phase_labels]
                pattern = classify_pattern(
                    ini[idx][metric]['Phase 1'], ini[idx][metric]['Phase 8'],
                    fol[idx][metric]['Phase 1'], fol[idx][metric]['Phase 8'],
                    ini_seq, fol_seq
                )
                if pattern == 1: p1 += 1
                elif pattern == 2: p2 += 1
            except KeyError: continue
        
        results.append({
            "Dataset": dataset_name.upper(),
            "Metric": metric.replace('_', ' ').title(),
            "N": len(shared),
            "Both Down (%)": (p1 / len(shared)) * 100,
            "Converged (%)": (p2 / len(shared)) * 100,
            "Total Reduc (%)": ((p1 + p2) / len(shared)) * 100
        })

    overall_p1 = 0
    overall_p2 = 0
    
    for idx in shared:
        any_p1 = False
        any_p2 = False
        for metric in metrics:
            try:
                ini_seq = [ini[idx][metric].get(p, np.nan) for p in phase_labels]
                fol_seq = [fol[idx][metric].get(p, np.nan) for p in phase_labels]
                res = classify_pattern(
                    ini[idx][metric]['Phase 1'], ini[idx][metric]['Phase 8'],
                    fol[idx][metric]['Phase 1'], fol[idx][metric]['Phase 8'],
                    ini_seq, fol_seq
                )
                if res == 1: any_p1 = True
                if res == 2: any_p2 = True
            except KeyError: continue
        
        if any_p1:
            overall_p1 += 1
        elif any_p2: 
            overall_p2 += 1

    results.append({
        "Dataset": dataset_name.upper(),
        "Metric": "\\textbf{OVERALL (any)}",
        "N": len(shared),
        "Both Down (%)": (overall_p1 / len(shared)) * 100,
        "Converged (%)": (overall_p2 / len(shared)) * 100,
        "Total Reduc (%)": ((overall_p1 + overall_p2) / len(shared)) * 100
    })

    return pd.DataFrame(results)

if __name__ == "__main__":
    datasets = ['btsj','A2','Hazumi','combined_candor','maptask','llm_german','llm_english']
    all_dfs = []

    for ds in datasets:
        df_ds = compute_convergence(f"{ds}_initiator_symcom.txt", f"{ds}_follower_symcom.txt", ds)
        if not df_ds.empty:
            all_dfs.append(df_ds)

    if all_dfs:
        final_df = pd.concat(all_dfs, ignore_index=True)
        
        latex_table = final_df.to_latex(
            index=False,
            float_format="%.2f",
            escape=False,
            caption="Syntactic Convergence: Metric-specific and Overall dialogue counts",
            label="tab:convergence_summary",
            column_format="llcccc"
        )
        
        with open("convergence_results.tex", "w") as f:
            f.write(latex_table)
