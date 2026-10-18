import pandas as pd
import numpy as np
import math
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import butter, filtfilt
from scipy.fft import fft, ifft, fftfreq
from scipy.stats import sem,norm
from scipy.interpolate import interp1d
from math import exp, sqrt, pi
import seaborn as sns
import matplotlib.pyplot as plt


def parse_file(file_path):
    data = []
    current_index = None
    current_metric = None
    utt_index = 1 
    A2_index =0
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('index:'):
                if "A2" in file_path:
                    current_index = str(A2_index)
                    A2_index+=1
                    itt_index=1
                elif "candor" not in file_path:
                    current_index = str(int(line.split(':')[1])-1)
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

if __name__ == '__main__':
    datasets=['A2', 'btsj','combined_candor','Hazumi','vm2']
    