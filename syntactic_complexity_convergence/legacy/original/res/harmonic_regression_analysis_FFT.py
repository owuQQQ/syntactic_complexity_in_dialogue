#!/usr/bin/env python3
"""
train_fft.py

Complete runnable script:
- Parses your per-utterance files (same parse_file function as before)
- For each dialogue: estimates K via FFT on the raw 'value' sequence (Option A)
- Per-dialogue K-fold CV, train LearnableHarmonicRegressor per fold
- Optionally initialize model frequencies from FFT peaks

Example:
python harmonic_regression_analysis_FFT.py \
  --info_estimator surprisal \
  --data_ini ./btsj_initiator_semcom_gpt.txt \
  --data_fol ./btsj_follower_semcom_gpt.txt \
  --k_folds 5 \
  --epochs 200
"""

import argparse
import math
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from scipy.signal import find_peaks

import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class LearnableHarmonicRegressor(nn.Module):
    def __init__(self, num_freqs, f_min=0.1, f_max=100.0, num_pos=4):
        super().__init__()
        self.num_freqs = max(1, int(num_freqs))
        # initialize log_freqs on a linspace by default (can be overwritten externally)
        freqs_init = torch.linspace(f_min, f_max, self.num_freqs)
        self.log_freqs = nn.Parameter(freqs_init.log())
        self.a = nn.Parameter(torch.zeros(self.num_freqs))
        self.b = nn.Parameter(torch.zeros(self.num_freqs))
        self.bias = nn.Parameter(torch.tensor(0.0))
        self.pos_emb = nn.Embedding(num_pos, 1)
        self.w_sup = nn.Parameter(torch.tensor(0.0))

    def forward(self, t, utt_pos=None, surprisal=None):
        t = t.unsqueeze(1)  # (N, 1)
        freqs = self.log_freqs.exp().unsqueeze(0)  
        cos_part = torch.cos(2 * math.pi * freqs * t)
        sin_part = torch.sin(2 * math.pi * freqs * t) 
        base = self.bias + cos_part @ self.a + sin_part @ self.b 
        if utt_pos is not None:
            base = base + self.pos_emb(utt_pos).squeeze(-1)
        if surprisal is not None:
            base = base + self.w_sup * surprisal
        return base

def train_model(model, t_train, y_train, utt_train, sup_train,
                t_val, y_val, utt_val, sup_val, lr=0.005, epochs=300, l1_penalty=1e-3, print_every=100):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_val = float('inf')
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        y_pred = model(t_train, utt_pos=utt_train, surprisal=sup_train)
        mse = nn.functional.mse_loss(y_pred, y_train)
        l1 = (model.a.abs().sum() + model.b.abs().sum())
        loss = mse + l1_penalty * l1
        loss.backward()
        optimizer.step()

        if epoch % print_every == 0 or epoch == 1 or epoch == epochs:
            model.eval()
            with torch.no_grad():
                val_pred = model(t_val, utt_pos=utt_val, surprisal=sup_val)
                val_loss = nn.functional.mse_loss(val_pred, y_val)
            if val_loss.item() < best_val:
                best_val = val_loss.item()
    return model

def evaluate(model, t, y, utt_pos, surprisal):
    model.eval()
    with torch.no_grad():
        y_pred = model(t, utt_pos=utt_pos, surprisal=surprisal)
        mse = nn.functional.mse_loss(y_pred, y)
    return mse, y_pred

def get_top3_freq_amp(model):
    with torch.no_grad():
        freqs = model.log_freqs.exp().cpu().numpy()
        amps = (model.a ** 2 + model.b ** 2).sqrt().cpu().numpy()
    top3_idx = np.argsort(amps)[-3:][::-1]
    return freqs[top3_idx], amps[top3_idx]


def parse_file(file_path, min_len=10):
    data = []
    current_index = None
    current_metric = None
    utt_index = 1 
    A2_index = 0
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('index:'):
                if "A2" in file_path:
                    current_index = str(A2_index)
                    A2_index += 1
                    utt_index = 1
                elif "candor" not in file_path:
                    try:
                        current_index = str(int(line.split(':')[1]) - 1)
                        utt_index = 1
                    except Exception:
                        current_index = line.split(':')[1]
                        utt_index = 1
                else:
                    current_index = line.split(':')[1]
                    utt_index = 1
            elif line in ['surprisal', 'shannon', 'renyi']:
                current_metric = line
            else:
                try:
                    values = list(map(float, line.split()))
                    if values:
                        mean_value = np.mean(values)
                        if not np.isnan(mean_value):
                            data.append({
                                'index': current_index,
                                'metric': current_metric,
                                'value': mean_value,
                                'utt_posit': utt_index
                            })
                            utt_index += 1
                except ValueError:
                    continue 

    df = pd.DataFrame(data)
    df = df.dropna(subset=['value'])
    dialogue_lengths = df.groupby('index')['utt_posit'].count()
    valid_indices = dialogue_lengths[dialogue_lengths > min_len].index
    df = df[df['index'].isin(valid_indices)]

    return df

def map_position(x, max_pos):
    # keep same mapping into 4 bins
    rel_pos = x / max_pos
    if rel_pos <= 0.25:
        return 0 
    elif rel_pos <= 0.5:
        return 1  
    elif rel_pos <= 0.75:
        return 2  
    else:
        return 3  

# -----------------------------
# FFT-based K estimation
# -----------------------------
def estimate_k_via_fft(values, height_ratio=0.3, min_prominence=0.0):
    
    """
    values: 1D array-like (raw signal - Option A)
    returns: K (int), peak_freqs (array in cycles/sample), peak_amps (array)
    """
    
    values = np.asarray(values, dtype=float)
    n = values.shape[0]
    if n < 2:
        return 1, np.array([0.0]), np.array([0.0])

    fft_vals = np.abs(np.fft.rfft(values))
    freqs = np.fft.rfftfreq(n)  # cycles per sample (0..0.5)

    fft_vals[0] = 0.0

    height_thr = np.max(fft_vals) * height_ratio if np.max(fft_vals) > 0 else 0.0
    peaks, props = find_peaks(fft_vals, height=height_thr, prominence=min_prominence)
    peak_heights = props.get('peak_heights', fft_vals[peaks])
    if len(peak_heights) == 0 and len(peaks) > 0:
        peak_heights = fft_vals[peaks]

    if len(peaks) == 0:
        peak_idx = int(np.argmax(fft_vals))
        return 1, np.array([freqs[peak_idx]]), np.array([fft_vals[peak_idx]])
    else:
        return len(peaks), freqs[peaks], fft_vals[peaks]

def init_freqs_from_fft_param(model, peak_freqs, f_min=0.1, f_max=100.0):
    """
    Initialize model.log_freqs from peak_freqs (cycles/sample).
    We need to map cycles/sample to a frequency scale consistent with the model usage.
    In the model we multiply freqs * t where t is utterance position (1..N).
    The FFT freqs are in cycles per sample (i.e., cycles per position).
    We'll map them to a similar numeric scale by scaling into [f_min, f_max] proportionally.
    """

    pf = np.asarray(peak_freqs, dtype=float)
    if pf.size == 0:
        return
    pf_norm = (pf - pf.min()) / (pf.max() - pf.min() + 1e-12)
    scaled = f_min + pf_norm * (f_max - f_min)
    with torch.no_grad():
        if model.num_freqs == len(scaled):
            model.log_freqs.data = torch.tensor(scaled, dtype=torch.float32).log()
        else:
            # create an array of freqs to fill model.num_freqs
            freqs_arr = np.linspace(f_min, f_max, model.num_freqs)
            # overwrite the first len(scaled) with scaled (clamped to [f_min,f_max])
            for i, v in enumerate(scaled):
                if i < len(freqs_arr):
                    freqs_arr[i] = float(np.clip(v, f_min, f_max))
            model.log_freqs.data = torch.tensor(freqs_arr, dtype=torch.float32).log()

def run_role(df_role, role_name, args):
    df_role_sup = df_role[df_role['metric'] == args.info_estimator]
    if df_role_sup.empty:
        return

    total_mse = 0.0
    total_folds = 0
    dialogue_count = 0

    unique_dialogues = sorted(df_role_sup['index'].unique(), key=lambda x: int(x) if str(x).isdigit() else x)
    for idx in unique_dialogues:
        if str(idx) == "0":
            continue
        df_dia = df_role_sup[df_role_sup['index'] == str(idx)]
        if len(df_dia) < args.min_utt_per_dialogue:
            continue

        dialogue_count += 1
        max_pos = df_dia['utt_posit'].max()
        df_dia = df_dia.copy()
        df_dia.loc[:, 'utt_pos_cat'] = df_dia['utt_posit'].apply(lambda x: map_position(x, max_pos))

        t_np = df_dia['utt_posit'].to_numpy()
        y_np = df_dia['value'].to_numpy()
        utt_np = df_dia['utt_pos_cat'].to_numpy()
        sup_np = df_dia['value'].to_numpy() 

        K_est, peak_freqs, peak_amps = estimate_k_via_fft(
            y_np, height_ratio=args.fft_height_ratio, min_prominence=args.fft_min_prominence
        )
        K_est = int(max(args.k_min, min(args.k_max, K_est)))
        if K_est < 1:
            K_est = 1

        # ===== DYNAMIC N_SPLITS FOR SMALL DIALOGUES =====
        n_splits = min(args.k_folds, len(t_np))
        if n_splits < 2:
            print(f"Skipping dialogue {idx} with only {len(t_np)} utterances")
            continue
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=args.seed)
        # =================================================

        fold_idx = 0
        for train_index, test_index in kf.split(t_np):
            fold_idx += 1
            t_train = torch.tensor(t_np[train_index], dtype=torch.float32).to(device)
            y_train = torch.tensor(y_np[train_index], dtype=torch.float32).to(device)
            utt_train = torch.tensor(utt_np[train_index], dtype=torch.long).to(device)
            sup_train = torch.tensor(sup_np[train_index], dtype=torch.float32).to(device)

            t_test = torch.tensor(t_np[test_index], dtype=torch.float32).to(device)
            y_test = torch.tensor(y_np[test_index], dtype=torch.float32).to(device)
            utt_test = torch.tensor(utt_np[test_index], dtype=torch.long).to(device)
            sup_test = torch.tensor(sup_np[test_index], dtype=torch.float32).to(device)

            model = LearnableHarmonicRegressor(num_freqs=K_est, f_min=args.f_min, f_max=args.f_max).to(device)

            if args.init_from_fft and len(peak_freqs) > 0:
                try:
                    init_freqs_from_fft_param(model, peak_freqs, f_min=args.f_min, f_max=args.f_max)
                except Exception as e:
                    print("Warning: failed to init freqs from fft:", e)

            model = train_model(
                model,
                t_train, y_train, utt_train, sup_train,
                t_test, y_test, utt_test, sup_test,
                lr=args.lr, epochs=args.epochs, l1_penalty=args.l1, print_every=args.print_every
            )

            mse, y_pred = evaluate(model, t_test, y_test, utt_test, sup_test)
            total_mse += mse.item()
            total_folds += 1

            if args.print_topk > 0:
                freqs, amps = get_top3_freq_amp(model)

    if total_folds > 0:
        avg_mse = total_mse / total_folds
        print(f"\nAverage Test MSE for {role_name} over {total_folds} folds: {avg_mse:.6f} "
              f"across {dialogue_count} dialogues.")
    else:
        print(f"No dialogues evaluated for {role_name}.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_freqs", type=int, default=200, help="(unused if FFT K used) default model freqs")
    parser.add_argument("--f_min", type=float, default=0.1)
    parser.add_argument("--f_max", type=float, default=300.0)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--l1", type=float, default=1e-3)
    parser.add_argument("--info_estimator", type=str, required=True, help="surprisal|shannon|renyi")
    parser.add_argument("--data_ini", type=str, required=True)
    parser.add_argument("--data_fol", type=str)
    parser.add_argument("--k_folds", type=int, default=5)
    parser.add_argument("--k_min", type=int, default=1)
    parser.add_argument("--k_max", type=int, default=50)
    parser.add_argument("--fft_height_ratio", type=float, default=0.3,
                        help="threshold (relative to max FFT) to detect peaks")
    parser.add_argument("--fft_min_prominence", type=float, default=0.0,
                        help="min prominence passed to find_peaks")
    parser.add_argument("--init_from_fft", action="store_true",
                        help="initialize model freqs from FFT peak locations")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min_utt_per_dialogue", type=int, default=5)
    parser.add_argument("--print_every", type=int, default=100)
    parser.add_argument("--print_topk", type=int, default=1, help="print top-k learned freqs per fold (0=off)")
    args = parser.parse_args()

    # reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # read data
    df_initiator = parse_file(args.data_ini).dropna(subset=['value'])
    df_follower = parse_file(args.data_fol).dropna(subset=['value'])

    run_role(df_initiator, "initiator", args)
    run_role(df_follower, "follower", args)


if __name__ == "__main__":
    main()