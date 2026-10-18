import pandas as pd
import os

def resolve_input_path(path):
    if os.path.exists(path):
        return path
    if "_syncom" in path:
        alt = path.replace("_syncom", "_symcom")
        if os.path.exists(alt):
            print(f"[Info] Using existing file: {alt}")
            return alt
    return path

def parse_and_save_syncom(input_path, output_dir, min_turns=9):
    input_path = resolve_input_path(input_path)
    if not os.path.exists(input_path):
        print(f"Error: Input file not found at {input_path}")
        return None

    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created directory: {output_dir}")

    df = pd.read_csv(input_path, sep='\t')
    if 'Index' not in df.columns:
        print(f"Error: 'Index' column not found in {input_path}")
        return None

    metrics = ['syn_value', 'dependency_length', 'tree_depth', 'complex_dependency_ratio']
    for m in metrics:
        if m not in df.columns:
            print(f"Error: '{m}' column not found in {input_path}")
            return None

    base_name = os.path.splitext(os.path.basename(input_path))[0]
    output_paths = {
        m: os.path.join(output_dir, f"{base_name}_{m}_time_series.txt")
        for m in metrics
    }

    counts = df.groupby('Index').size()
    valid_ids = set(counts[counts >= min_turns].index)
    if len(valid_ids) == 0:
        print(f"No dialogues with at least {min_turns} turns in {input_path}")
        return None

    df = df[df['Index'].isin(valid_ids)]
    grouped = df.groupby('Index', sort=False)

    for m in metrics:
        with open(output_paths[m], 'w') as out_f:
            for dia_id, g in grouped:
                values = g[m].tolist()
                if values:
                    line_to_write = " ".join(map(str, values))
                    out_f.write(f"{dia_id}\t{line_to_write}\n")

    print(f"Saved metric files for {input_path}:")
    for m in metrics:
        print(f"  - {output_paths[m]}")

    return output_paths

if __name__ == '__main__':
    output_dir = './combined_con_res'

    input1 = '../json_combined_con/combined_candor_syncom.txt'
    input2 = '../json_combined_con/llm_english_joint_syncom.txt'

    parse_and_save_syncom(input1, output_dir, min_turns=9)
    parse_and_save_syncom(input2, output_dir, min_turns=9)
