import pandas as pd
import numpy as np

def convergence_ratio_calculation(file_path1, file_path2, metrics, phase_labels):
    def get_dialogue_medians(df):
        dialogue_medians = {metric: [] for metric in metrics}

        for _, group in df.groupby('Index'):
            if len(group) < len(phase_labels):
                continue
            phases = np.array_split(group, len(phase_labels))
            dialogue_record = {metric: [] for metric in metrics}

            for phase in phases:
                for metric in metrics:
                    median_val = phase[metric].median() if not phase[metric].dropna().empty else np.nan
                    dialogue_record[metric].append(median_val)

            for metric in metrics:
                if all(pd.notna(dialogue_record[metric])):
                    dialogue_medians[metric].append(dialogue_record[metric])
        return dialogue_medians
        
    df1 = pd.read_csv(file_path1, sep='\t')
    df2 = pd.read_csv(file_path2, sep='\t')
    medians1 = get_dialogue_medians(df1)
    medians2 = get_dialogue_medians(df2)

    abs_diffs = {metric: [] for metric in metrics}
    for metric in metrics:
        for seq1, seq2 in zip(medians1[metric], medians2[metric]):
            diff = np.abs(np.array(seq1) - np.array(seq2))
            abs_diffs[metric].append(diff)

    strict_decreasing = {}
    loose_decreasing = {}

    for metric in metrics:
        sequences = abs_diffs[metric]
        n = len(sequences)
        print(f"total_dialogue_number: {n}")
        if n == 0:
            strict_decreasing[metric] = 0
            loose_decreasing[metric] = 0
            continue

        strict_count = 0
        loose_count = 0

        for seq in sequences:
            decreases = sum(seq[i] > seq[i+1] for i in range(len(seq)-1))
            if seq[0] > seq[-1]:
                loose_count += 1
            if seq[0] > seq[-1] and decreases >= 2:
                strict_count += 1
        strict_decreasing[metric] = strict_count / n
        loose_decreasing[metric] = loose_count / n
    phase_means = {metric: np.mean(np.array(abs_diffs[metric]), axis=0).tolist() for metric in metrics}

    return {
        "phase_means": phase_means,
        "per_dialogue_diffs": abs_diffs,
        "loose_decreasing_ratio": loose_decreasing,
        "strict_decreasing_ratio": strict_decreasing
    }
    
    
if __name__ == '__main__':
    metrics = ['syn_value', 'dependency_length', 'tree_depth', 'complex_dependency_ratio']
    phase_labels2 = ['Phase 1', 'Phase 2', 'Phase 3', 'Phase 4', 'Phase 5','Phase 6','Phase 7','Phase 8']
    
    
    dataset = ['A2', 'btsj','combined_candor','Hazumi','maptask','llm_german','llm_japanese','llm_english']
    #dataset = ['llm_german','llm_japanese','llm_english']
    for item in dataset:
        path1 = item + '_initiator_symcom.txt'
        path2 = item + '_follower_symcom.txt'
        print("==================================================")
        print(item)
        print(convergence_ratio_calculation(path1,path2, metrics, phase_labels2)["strict_decreasing_ratio"])