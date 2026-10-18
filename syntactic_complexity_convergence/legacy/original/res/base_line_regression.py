import argparse
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.metrics import mean_squared_error


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
    # counts = df.groupby("index")["utt_posit"].max()
    # valid_dialogues = counts[counts > 5].index
    # df = df[df["index"].isin(valid_dialogues)]
    return df


def add_location(df):
    """Create categorical utterance_location: begin/middle1/middle2/end."""
    df = df.copy()
    df['utt_posit'] = df['utt_posit'].astype(int)
    max_per_dialogue = df.groupby('index')['utt_posit'].transform('max')
    rel_pos = df['utt_posit'] / max_per_dialogue

    df['utterance_location'] = pd.cut(
        rel_pos,
        bins=[0, 0.25, 0.5, 0.75, 1],
        labels=["begin", "middle1", "middle2", "end"],
        include_lowest=True
    )
    return df


def run_mixedlm(df, role):
    """Fit mixed effects model for one role and compute MSE."""
    df = add_location(df)
    df_sup = df[df['metric'] == 'surprisal']
    if len(df_sup) < 10:
        print(f"Not enough surprisal data for {role}.")
        return
    model = smf.mixedlm(
        "value ~ utt_posit + utterance_location",
        df_sup,
        groups=df_sup["index"]
    )
    result = model.fit()

    # Compute MSE on the same data (in-sample fit)
    y_true = df_sup["value"].to_numpy()
    y_pred = result.fittedvalues.to_numpy()
    mse = mean_squared_error(y_true, y_pred)

    print(f"\n=== Mixed Effects Model for {role} ===")
    print(result.summary())
    print(f"\nMean Squared Error (in-sample) for {role}: {mse:.6f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_ini", type=str, required=True, help="Path to initiator data")
    parser.add_argument("--data_fol", type=str, required=True, help="Path to follower data")
    args = parser.parse_args()

    # parse files
    df_initiator = parse_file(args.data_ini)
    df_follower = parse_file(args.data_fol)

    # fit separately
    run_mixedlm(df_initiator, role="initiator")
    run_mixedlm(df_follower, role="follower")


if __name__ == "__main__":
    main()