import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from tqdm import trange
import math
import json
import pickle
import os
import sys

def build_groups_from_df(df, metrics, keep_order=True):
    """
    Build a list of group objects from dataframe grouped by 'Index'.
    Each group is a dict: {'id': idx, 'len': n, 'metrics': {m: np.array([...])}}
    Keeps original order by default.
    """
    groups = []
    for cid, g in df.groupby("Index", sort=False):
        if g.shape[0] == 0:
            continue
        arrs = {}
        for m in metrics:
            arrs[m] = np.asarray(g[m].to_numpy(dtype=float), dtype=float)
        groups.append({"id": cid, "len": g.shape[0], "metrics": arrs})
    return groups    

def strict_decreasing_from_groups(groups1, groups2, metrics, n_phases):
    g1 = {g["id"]: g for g in groups1}
    g2 = {g["id"]: g for g in groups2}
    common_ids = [cid for cid in g1.keys() if cid in g2]

    if len(common_ids) == 0:
        return {m: 0.0 for m in metrics}, {} 
        
    per_metric_sequences = {m: [] for m in metrics}
    for cid in common_ids:
        A = g1[cid]
        B = g2[cid]
        if A["len"] < n_phases or B["len"] < n_phases:
            continue
        for m in metrics:
            a_vals = A["metrics"][m]
            b_vals = B["metrics"][m]
            a_parts = np.array_split(a_vals, n_phases)
            b_parts = np.array_split(b_vals, n_phases)
            if any(p.size == 0 for p in a_parts) or any(p.size == 0 for p in b_parts):
                continue

            a_meds = np.array([np.nanmedian(p) for p in a_parts])
            b_meds = np.array([np.nanmedian(p) for p in b_parts])
            
            if np.any(np.isnan(a_meds)) or np.any(np.isnan(b_meds)):
                continue
            diff = np.abs(a_meds - b_meds)
            per_metric_sequences[m].append(diff)

    strict = {}
    for m in metrics:
        seqs = per_metric_sequences[m]
        n = len(seqs)
        if n == 0:
            strict[m] = 0.0
            continue
            
        strict_count = 0
        for seq in seqs:
            decreases = np.sum(seq[:-1] > seq[1:])
            if seq[0] > seq[-1] and decreases >= 2:
                strict_count += 1
        strict[m] = strict_count / n
        
    return strict, per_metric_sequences

def compute_observed(groups1, groups2, metrics, n_phases):
    strict, per_metric_sequences = strict_decreasing_from_groups(groups1, groups2, metrics, n_phases)
    return strict, per_metric_sequences


def one_permutation(groups1, groups2, metrics, n_phases, rng_seed=None):
    rng = np.random.RandomState(rng_seed)
    permuted_g1 = []
    permuted_g2 = []
    for g in groups1:
        n = g["len"]
        if n == 0:
            permuted_g1.append(g)
            continue
        perm = rng.permutation(n)
        new_metrics = {m: g["metrics"][m][perm] for m in metrics}
        permuted_g1.append({"id": g["id"], "len": n, "metrics": new_metrics})

    for g in groups2:
        n = g["len"]
        if n == 0:
            permuted_g2.append(g)
            continue
        perm = rng.permutation(n)
        new_metrics = {m: g["metrics"][m][perm] for m in metrics}
        permuted_g2.append({"id": g["id"], "len": n, "metrics": new_metrics})

    strict, _ = strict_decreasing_from_groups(permuted_g1, permuted_g2, metrics, n_phases)
    return strict

def choose_B_auto(n_dialogues, B_user=None):
    if B_user is not None:
        return int(B_user)
    if n_dialogues < 50:
        return max(300, n_dialogues * 15)
    elif n_dialogues < 500:
        return 750
    elif n_dialogues < 1000:
        return 1500
    else:
        return 1750

def permutation_test_fast(groups1, groups2, metrics, n_phases, B_user=None, n_jobs=-1, random_state=42, batch_size=100):
    ids1 = {g["id"] for g in groups1}
    ids2 = {g["id"] for g in groups2}
    common_ids = ids1.intersection(ids2)
    n_dialogues = len(common_ids) if len(common_ids) > 0 else max(len(ids1), len(ids2))
    B = choose_B_auto(n_dialogues, B_user)
    rng = np.random.RandomState(random_state)
    null_vals = {m: [] for m in metrics}
    n_done = 0
    seeds = [int(rng.randint(0, 2**31 - 1)) for _ in range(B)]

    for start in range(0, B, batch_size):
        end = min(start + batch_size, B)
        batch_seeds = seeds[start:end]
        batch_results = Parallel(n_jobs=n_jobs)(
            delayed(one_permutation)(
                groups1, groups2, metrics, n_phases, seed
            ) for seed in batch_seeds
        )
        for res in batch_results:
            for m in metrics:
                null_vals[m].append(res[m])

        n_done += len(batch_results)

    null_mean = {m: float(np.mean(null_vals[m])) for m in metrics}
    null_std = {m: float(np.std(null_vals[m], ddof=1)) for m in metrics}
    return {"null_vals": null_vals, "null_mean": null_mean, "null_std": null_std, "B_used": B}

def cohen_d_from_null(observed, null_mean, null_std):
    d = {}
    for m in observed:
        s = null_std.get(m, 0.0)
        if s == 0:
            d[m] = float('nan')
        else:
            d[m] = float((observed[m] - null_mean[m]) / s)
    return d

def bootstrap_ci_diff(groups1, groups2, metrics, n_phases, null_mean, min_turns=None, n_boot=2000, random_state=123):
    g1 = {g["id"]: g for g in groups1}
    g2 = {g["id"]: g for g in groups2}
    
    min_len = n_phases if min_turns is None else min_turns
    common_ids = sorted([
        cid for cid in g1.keys() 
        if cid in g2.keys() and g1[cid]["len"] >= min_len and g2[cid]["len"] >= min_len
    ])
    
    if len(common_ids) == 0:
        return {m: (np.nan, np.nan) for m in metrics}, {m: [] for m in metrics}

    rng = np.random.RandomState(random_state)
    n = len(common_ids)
    boot_diffs = {m: [] for m in metrics}

    for _ in range(n_boot):
        sampled_ids = rng.choice(common_ids, size=n, replace=True)
        g1_sample = []
        g2_sample = []
        for cid in sampled_ids:
            g1_sample.append(g1[cid])
            g2_sample.append(g2[cid])
            
        obs_sample_strict, _ = strict_decreasing_from_groups(g1_sample, g2_sample, metrics, n_phases)
        for m in metrics:
            boot_diffs[m].append(obs_sample_strict[m] - null_mean[m])

    ci = {}
    for m in metrics:
        arr = np.array(boot_diffs[m])
        ci[m] = (float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5)))
    return ci, boot_diffs


def analyze_pair_fast(path1, path2, metrics, n_phases=8, min_turns=None, B_user=None, n_boot=1000, n_jobs=-1, random_state=42, print_latex=False):
    df1 = pd.read_csv(path1, sep='\t')
    df2 = pd.read_csv(path2, sep='\t')
    
    raw_groups1 = build_groups_from_df(df1, metrics)
    raw_groups2 = build_groups_from_df(df2, metrics)
    
    g1_lookup = {g["id"]: g for g in raw_groups1}
    g2_lookup = {g["id"]: g for g in raw_groups2}
    
    min_len = n_phases if min_turns is None else min_turns
    valid_ids = [
        cid for cid in g1_lookup.keys() 
        if cid in g2_lookup and g1_lookup[cid]["len"] >= min_len and g2_lookup[cid]["len"] >= min_len
    ]

    groups1 = [g1_lookup[cid] for cid in valid_ids]
    groups2 = [g2_lookup[cid] for cid in valid_ids]
    n_valid = len(valid_ids) 
    if n_valid == 0:
        print(f"Skipping {path1}: No dialogues meet the minimum turns of {min_len} (phases={n_phases}).")
        return None

    observed, _ = strict_decreasing_from_groups(groups1, groups2, metrics, n_phases)
    perm_res = permutation_test_fast(groups1, groups2, metrics, n_phases, B_user=B_user, n_jobs=n_jobs, random_state=random_state)
    
    null_vals = perm_res["null_vals"]
    null_mean = perm_res["null_mean"]
    null_std = perm_res["null_std"]
    B_used = perm_res["B_used"]
    
    p_values = {m: float((np.array(null_vals[m]) >= observed[m]).sum() / len(null_vals[m])) for m in metrics}
    d = cohen_d_from_null(observed, null_mean, null_std)
    ci_diff, _ = bootstrap_ci_diff(groups1, groups2, metrics, n_phases, null_mean, min_turns=min_len, n_boot=n_boot, random_state=random_state)

    print(f"\n--- RESULTS: {path1} vs {path2} ---")
    print(f"Analyzed Dialogues (Valid N): {n_valid} (Filtered for min {min_len} turns, phases={n_phases})")
    print(f"Permutations: {B_used} | Bootstrap samples: {n_boot}")
    
    print(f"{'Metric':20s} {'Observed':>8s} {'Null mean':>10s} {'Null sd':>9s} {'p-value':>9s} {'Cohen d':>9s} {'CI(diff)':>22s}")
    for m in metrics:
        print(f"{m:20s} {observed[m]:8.4f} {null_mean[m]:10.4f} {null_std[m]:9.4f} {p_values[m]:9.4f} {d[m]:9.3f} [{ci_diff[m][0]:6.4f}, {ci_diff[m][1]:6.4f}]")

    if print_latex:
        dataset_name = path1.split('_')[0]
        # uses n_valid instead of raw group length
        latex_row = f"{dataset_name} & {n_valid} & " + " & ".join([f"{observed[m]:.4f}" for m in metrics]) + " \\\\"
        print("\nLaTeX row:")
        print(latex_row)

    return {
        "observed": observed,
        "p_values": p_values,
        "cohen_d": d,
        "ci_diff": ci_diff,
        "valid_n": n_valid,
        "null_mean": null_mean,
        "null_std": null_std,
        "B_used": B_used
    }

if __name__ == "__main__":
    metrics = ['syn_value', 'dependency_length', 'tree_depth', 'complex_dependency_ratio']
    dataset = ['A2','Hazumi', 'maptask', 'btsj', 'combined_candor', 'llm_german','llm_english']
    #dataset =['llm_english']
    #dataset = ['btsj', 'combined_candor', 'llm_german']
    ## keep , 'llm_english' out given that the data is large

    # for item in dataset:
    #     p1 = item + "_initiator_symcom.txt"
    #     p2 = item + "_follower_symcom.txt"
        
    #     print(f"\n>>> Processing dataset phase = 8: {item}...")
    #     if not (os.path.exists(p1) and os.path.exists(p2)):
    #         print(f"Skipping {item}: Files not found.")
    #         continue

    #     try:
    #         # analyze_pair_fast already contains the print logic
    #         analyze_pair_fast(
    #             p1, p2, metrics, 
    #             n_phases=8, 
    #             B_user=None, 
    #             n_boot=1000, 
    #             n_jobs=-1, 
    #             random_state=42, 
    #             print_latex=True
    #         )
            
    #     except Exception as e:
    #         print(f"Error on {item}: {e}")

    # print("\n====================================")
    min_turns_list = [16, 24, 32, 36, 48, 64]
    phases_list = [8, 16, 32]

    output_file = "pertubation_test_results.txt"
    table_file = "pertubation_test_table.txt"

    class Tee:
        def __init__(self, *streams):
            self.streams = streams

        def write(self, data):
            for s in self.streams:
                s.write(data)
                s.flush()

        def flush(self):
            for s in self.streams:
                s.flush()

    with open(output_file, "w", encoding="utf-8") as f, open(table_file, "w", encoding="utf-8") as tf:
        original_stdout = sys.stdout
        sys.stdout = Tee(original_stdout, f)
        try:
            tf.write("Dataset\tPhases\tMinTurns\tValidN\tMetric\tObserved\tNullMean\tNullSD\tp-value\tCohen_d\tCI_low\tCI_high\n")
            for item in dataset:
                p1 = item + "_initiator_symcom.txt"
                p2 = item + "_follower_symcom.txt"

                if not (os.path.exists(p1) and os.path.exists(p2)):
                    print(f"Skipping {item}: Files not found.")
                    continue

                for n_phases in phases_list:
                    for min_turns in min_turns_list:
                        print(f"\n>>> Processing dataset phases = {n_phases}, min turns = {min_turns}: {item}...")
                        try:
                            # analyze_pair_fast already contains the print logic
                            res = analyze_pair_fast(
                                p1, p2, metrics,
                                n_phases=n_phases,
                                min_turns=min_turns,
                                B_user=None,
                                n_boot=1000,
                                n_jobs=-1,
                                random_state=42,
                                print_latex=True
                            )
                            if res is None:
                                continue

                            observed = res["observed"]
                            null_mean = res["null_mean"]
                            null_std = res["null_std"]
                            p_values = res["p_values"]
                            cohen_d = res["cohen_d"]
                            ci_diff = res["ci_diff"]
                            valid_n = res["valid_n"]

                            for m in metrics:
                                tf.write(
                                    f"{item}\t{n_phases}\t{min_turns}\t{valid_n}\t{m}\t"
                                    f"{observed[m]:.6f}\t{null_mean[m]:.6f}\t{null_std[m]:.6f}\t"
                                    f"{p_values[m]:.6f}\t{cohen_d[m]:.6f}\t"
                                    f"{ci_diff[m][0]:.6f}\t{ci_diff[m][1]:.6f}\n"
                                )
                        except Exception as e:
                            print(f"Error on {item} (min_turns={min_turns}, phases={n_phases}): {e}")
        finally:
            sys.stdout = original_stdout



