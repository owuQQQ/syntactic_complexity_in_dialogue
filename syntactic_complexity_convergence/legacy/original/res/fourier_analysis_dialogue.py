import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import butter, filtfilt
from scipy.fft import fftshift, fft, fftfreq
from scipy.interpolate import interp1d
from math import exp, sqrt, pi
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.signal import periodogram
import re

CUTOFF_FREQ = 5
ORDER = 3
FS = 25

def butter_lowpass_filter(data, cutoff=CUTOFF_FREQ, fs=FS, order=ORDER):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

def normal_pdf(x, mu, sigma):
    return (1 / (sigma * sqrt(2 * pi))) * exp(-0.5 * ((x - mu) / sigma) ** 2)
    
def normalize_using_pdf(values):
    mu = np.mean(values)
    sigma = np.std(values)
    pdf_values = [normal_pdf(x, mu, sigma) for x in values]
    min_val = min(pdf_values)
    max_val = max(pdf_values)
    scaled_pdf = [(x - min_val) / (max_val - min_val) for x in pdf_values]
    return scaled_pdf

def interpolate_and_filter(y, target_len=15):
    if len(y) == 1:
        y = y * target_len
        #return y
        return butter_lowpass_filter(y, CUTOFF_FREQ, FS, ORDER)
    elif len(y) < target_len:
        y = np.interp(np.linspace(0, len(y)-1, target_len), np.arange(len(y)), y).tolist()
        #return y
        return butter_lowpass_filter(y, CUTOFF_FREQ, FS, ORDER)
    else:
        #return y
        return butter_lowpass_filter(y, CUTOFF_FREQ, FS, ORDER)

def calculate_freq_and_mag(df,scaler,target_len=13):
    y = pd.to_numeric(df['value'], errors='coerce').dropna().tolist()
    if len(y) == 0:
        return np.zeros(target_len), np.linspace(0, FS/2, target_len)
    y = interpolate_and_filter(y, target_len)
    y = scaler.fit_transform(np.array(y).reshape(-1, 1)).flatten()
    
    fft_values = fftshift(fft(y))
    freqs = fftshift(fftfreq(len(y), 1 / FS))
    magnitude = np.abs(fft_values)
    return magnitude[len(magnitude)//2:], freqs[len(freqs)//2:]


def parse_file(file_path):
    data, utt_index, A2_index = [], 1, 0
    current_index, current_metric = None, None

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            match = re.match(r'index\s*:\s*(\d+)', line)
            if match:
                if "A2" in file_path:
                    current_index = str(A2_index)
                    A2_index += 1
                else:
                    current_index = str(int(match.group(1)))
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

    df = pd.DataFrame(data).dropna(subset=['value'])
    return df


def get_mag_freq(df_follower, df_initiator, indice,scaler):
    mag_fol, mag_init, freqs1, freqs2 = [], [], [], []
    for idx in range(indice):
        if idx==0:
            continue
        else:
            idx = str(idx)
            df1 = df_follower[df_follower['index'] == idx]
            df2 = df_initiator[df_initiator['index'] == idx]
            if len(df1)>5 and len(df2)>5:     
                res1, freq1 = calculate_freq_and_mag(df1,scaler)
                res2, freq2 = calculate_freq_and_mag(df2,scaler)
                mag_fol.append(normalize_using_pdf(res1))
                mag_init.append(normalize_using_pdf(res2))
                freqs1.append(freq1)
                freqs2.append(freq2)
    return mag_fol, mag_init, freqs1, freqs2

def plot_initiator_follower_line(fol, init, freqs1, freqs2, corpus, measure,estimator_model):
    all_freqs = np.concatenate(freqs1)
    f_min, f_max = np.min(all_freqs), np.max(all_freqs)
    common_freqs = np.linspace(f_min, f_max, 100)
    interpolated = []
    for f, m in zip(freqs1, fol):
        interp_func = interp1d(f, m, kind='linear', bounds_error=False, fill_value=np.nan)
        interpolated.append(interp_func(common_freqs))
    interpolated = np.array(interpolated) 
    mean_mag = np.nanmean(interpolated, axis=0)
    std_mag = np.nanstd(interpolated, axis=0)

    all_freqs_init = np.concatenate(freqs2)
    f_min_init, f_max_init = np.min(all_freqs_init), np.max(all_freqs_init)
    common_freqs_init = np.linspace(f_min_init, f_max_init, 100)
    interpolated_init = []
    for f, m in zip(freqs2, init):
        interp_func = interp1d(f, m, kind='linear', bounds_error=False, fill_value=np.nan)
        interpolated_init.append(interp_func(common_freqs_init))

    interpolated_init = np.array(interpolated_init)
    mean_mag_init = np.nanmean(interpolated_init, axis=0)
    std_mag_init = np.nanstd(interpolated_init, axis=0)

    lower_fol = np.clip(mean_mag - std_mag, 0, 1)
    upper_fol = np.clip(mean_mag + std_mag, 0, 1)
    lower_init = np.clip(mean_mag_init - std_mag_init, 0, 1)
    upper_init = np.clip(mean_mag_init + std_mag_init, 0, 1)

    plt.figure(figsize=(7, 4))
    plt.plot(common_freqs, mean_mag, label="Follower", color="blue")
    plt.fill_between(common_freqs, lower_fol, upper_fol, color="blue", alpha=0.05)

    plt.plot(common_freqs_init, mean_mag_init, label="Initiator", color="green")
    plt.fill_between(common_freqs_init, lower_init, upper_init, color="green", alpha=0.05)

    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Normalized magnitude")
    plt.ylim(0, 1)  # Explicitly fix y-axis to [0, 1]
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"res_fourier_plot_{corpus}_{measure}_{estimator_model}.png", dpi=300)
    plt.show()
    plt.close()

def run_all():
    #datasets = ['combined_candor']
    #datasets = ['A2', 'btsj', 'combined_candor', 'Hazumi']
    datasets = ['vm2','texprax','A2', 'btsj', 'combined_candor', 'Hazumi']
    estimators = ['gpt','llama','Qwen']
    scaler = MinMaxScaler()
    for item in datasets:
        for estimator in estimators:
            path_ini = f"{item}_initiator_semcom_{estimator}.txt"
            path_fol = f"{item}_follower_semcom_{estimator}.txt"
            df_ini = parse_file(path_ini)
            df_fol = parse_file(path_fol)
    
            df_ini['role'] = 'initiator'
            df_fol['role'] = 'follower'
            
            df_ini['utt_posit'] = (df_ini['utt_posit']-1) // 3
            df_fol['utt_posit'] = (df_fol['utt_posit']-1) // 3
            dialogue_ids = df_ini['index'].unique()
            dialogue_id =max(list(map(int, dialogue_ids)))
    
            for metric in ['surprisal', 'shannon', 'renyi']:
                df_fol_metric = df_fol[df_fol['metric'] == metric]
                df_ini_metric = df_ini[df_ini['metric'] == metric]

                if metric=='surprisal' and estimator=='llama':
                    frequencies, power = periodogram(df_fol_metric['value'])
                    top_n = 3
                    top_freqs = frequencies[np.argsort(power)[-top_n:]]
                    top_freqs = np.sort(top_freqs)
                    periods = 1 / top_freqs
                    print("Top frequencies and corresponding periods:", list(zip(top_freqs, periods)))
                res1, res2, res3, res4 = get_mag_freq(df_fol_metric, df_ini_metric, dialogue_id,scaler)
                plot_initiator_follower_line(res1, res2, res3, res4, corpus=item, measure=metric, estimator_model=estimator)
            

if __name__ == '__main__':
    run_all()
