import pandas as pd
import numpy as np
import re
import os

def parse_and_save_surprisal(input_path, output_path, min_turns=9):

    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created directory: {output_dir}")

    dialogue_data = {}
    current_index = None
    current_metric = None
    A2_index = 0

    if not os.path.exists(input_path):
        print(f"Error: Input file not found at {input_path}")
        return

    with open(input_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line: continue

            # 1. Identify the Dialogue Index
            match = re.search(r'index\s*:\s*(\S+)', line)
            if match:
                if "A2" in input_path:
                    current_index = str(A2_index)
                    A2_index += 1
                else:
                    current_index = match.group(1)
                
                if current_index not in dialogue_data:
                    dialogue_data[current_index] = []
                continue

            if line in ['surprisal', 'shannon', 'renyi']:
                current_metric = line
                continue

            if current_metric == 'surprisal':
                try:
                    values = [float(v) for v in line.split()]
                    if values:
                        dialogue_data[current_index].extend(values)
                except ValueError:
                    continue

    # 4. Filter by minimum turns
    if min_turns is not None:
        dialogue_data = {
            k: v for k, v in dialogue_data.items()
            if len(v) >= min_turns
        }
        if len(dialogue_data) == 0:
            print(f"No dialogues with at least {min_turns} turns in {input_path}")
            return

    # 5. Save the gathered series with ID label
    with open(output_path, 'w') as out_f:
        sorted_keys = sorted(dialogue_data.keys(), key=lambda x: (not x.isdigit(), int(x) if x.isdigit() else x))
        for dia_id in sorted_keys:
            series = dialogue_data[dia_id]
            if series:
                line_to_write = " ".join(map(str, series))
                out_f.write(f"{dia_id}\t{line_to_write}\n")
    
    print(f"File saved successfully to {output_path}")

if __name__ == '__main__':
    path1 = './combined_con_res/candor_surprisal_time_series.txt'
    path2 = './combined_con_res/llm_english_joint_surprisal_time_series.txt'
    
    parse_and_save_surprisal('./combined_con_res/combined_candor_semcom_gpt.txt', path1, min_turns=9)
    parse_and_save_surprisal('./combined_con_res/llm_english_joint_semcom_gpt.txt', path2, min_turns=9)