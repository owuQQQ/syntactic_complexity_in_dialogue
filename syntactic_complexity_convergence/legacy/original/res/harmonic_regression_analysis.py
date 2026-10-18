import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class LearnableHarmonicRegressor(nn.Module):
    def __init__(self, num_freqs, f_min=0.1, f_max=100.0, num_pos=4):
        super().__init__()
        self.num_freqs = num_freqs
        self.log_freqs = nn.Parameter(torch.linspace(f_min, f_max, num_freqs).log())
        self.a = nn.Parameter(torch.zeros(num_freqs))
        self.b = nn.Parameter(torch.zeros(num_freqs))
        self.bias = nn.Parameter(torch.tensor(0.0))
        self.pos_emb = nn.Embedding(num_pos, 1)
        self.w_sup = nn.Parameter(torch.tensor(0.0))

    def forward(self, t, utt_pos=None, surprisal=None):
        t = t.unsqueeze(1)  # (N, 1)
        freqs = self.log_freqs.exp().unsqueeze(0) 
        cos_part = torch.cos(2 * torch.pi * freqs * t) 
        sin_part = torch.sin(2 * torch.pi * freqs * t) 
        base = self.bias + cos_part @ self.a + sin_part @ self.b

        if utt_pos is not None:
            base = base + self.pos_emb(utt_pos).squeeze(-1)

        if surprisal is not None:
            base = base + self.w_sup * surprisal
        return base

def train_model(model, t_train, y_train, utt_train, sup_train,
                t_val, y_val, utt_val, sup_val, lr=0.005, epochs=300, l1_penalty=1e-3):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        y_pred = model(t_train, utt_pos=utt_train, surprisal=sup_train)
        mse = nn.functional.mse_loss(y_pred, y_train)
        l1 = (model.a.abs().sum() + model.b.abs().sum())
        loss = mse + l1_penalty * l1
        loss.backward()
        optimizer.step()
        if epoch % 200 == 0:
            val_pred = model(t_val, utt_pos=utt_val, surprisal=sup_val)
            val_loss = nn.functional.mse_loss(val_pred, y_val)

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

def parse_file(file_path):
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
                    current_index = str(int(line.split(':')[1]) - 1)
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
    return df

### position information  is used as a variable in the regression model
def map_position(x, max_pos):
    rel_pos = x / max_pos
    if rel_pos <= 0.25:
        return 0 
    elif rel_pos <= 0.5:
        return 1  
    elif rel_pos <= 0.75:
        return 2  
    else:
        return 3  
        
def run_role(df_role, role_name, args):
    df_role_sup = df_role[df_role['metric'] == args.info_estimator]
    dialogue_id = df_role['index'].unique()
    max_id = max(list(map(int, dialogue_id)))
    total_mse = 0.0
    count = 0
    for idx in range(max_id):
        if idx == 0:
            continue
        idx = str(idx)
        df_dia = df_role_sup[df_role_sup['index'] == idx]

        if len(df_dia) < 5:
            continue
        max_pos = df_dia['utt_posit'].max()
        df_dia = df_role_sup[df_role_sup['index'] == idx].copy()
        df_dia.loc[:, 'utt_pos_cat'] = df_dia['utt_posit'].apply(lambda x: map_position(x, max_pos))
        t_np, y_np = df_dia['utt_posit'], df_dia['value']
        utt_np = df_dia['utt_pos_cat']
        sup_np = df_dia['value'] 

        t_train_np, t_test_np, y_train_np, y_test_np, utt_train_np, utt_test_np, sup_train_np, sup_test_np = \
            train_test_split(t_np, y_np, utt_np, sup_np, test_size=0.1, random_state=0)

        t_train = torch.tensor(t_train_np.to_numpy(), dtype=torch.float32).to(device)
        y_train = torch.tensor(y_train_np.to_numpy(), dtype=torch.float32).to(device)
        utt_train = torch.tensor(utt_train_np.to_numpy(), dtype=torch.long).to(device)
        sup_train = torch.tensor(sup_train_np.to_numpy(), dtype=torch.float32).to(device)

        t_test = torch.tensor(t_test_np.to_numpy(), dtype=torch.float32).to(device)
        y_test = torch.tensor(y_test_np.to_numpy(), dtype=torch.float32).to(device)
        utt_test = torch.tensor(utt_test_np.to_numpy(), dtype=torch.long).to(device)
        sup_test = torch.tensor(sup_test_np.to_numpy(), dtype=torch.float32).to(device)

        model = LearnableHarmonicRegressor(args.num_freqs, args.f_min, args.f_max).to(device)
        train_model(model, t_train, y_train, utt_train, sup_train,
                    t_test, y_test, utt_test, sup_test,
                    lr=args.lr, epochs=args.epochs, l1_penalty=args.l1)

        mse, _ = evaluate(model, t_test, y_test, utt_test, sup_test)
        total_mse += mse.item()
        count += 1

        freqs, amps = get_top3_freq_amp(model)

    if count > 0:
        avg_mse = total_mse / count
        print(f"\nAverage Test MSE for {role_name} over {count} dialogues: {avg_mse:.6f}")
    else:
        print(f"No dialogues evaluated for {role_name}.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_freqs", type=int, default=200)
    parser.add_argument("--f_min", type=float, default=0.1)
    parser.add_argument("--f_max", type=float, default=300)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--l1", type=float, default=1e-3)
    parser.add_argument("--info_estimator", type=str, required=True)
    parser.add_argument("--data_ini", type=str, required=True)
    parser.add_argument("--data_fol", type=str, required=True)
    args = parser.parse_args()

    df_initiator = parse_file(args.data_ini).dropna(subset=['value'])
    df_follower = parse_file(args.data_fol).dropna(subset=['value'])

    run_role(df_initiator, "initiator", args)
    run_role(df_follower, "follower", args)


if __name__ == "__main__":
    main()
    