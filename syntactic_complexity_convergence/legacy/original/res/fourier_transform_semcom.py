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


cutoff_freq = 3
order = 3
fs = 50

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
    
## in order to apply butter_lowpass_filter, the length should be longer than 12, this is proabbly not okay for datasets except candor, so one way is to concatenate 4 datapoints to one.

def butter_lowpass_filter(data, cutoff, fs, order=3):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

## using linear interpolation to stretch the time-series length
## another method is to combined several results
##calculate frequency and magnitue of the transformed time-series

def calculate_freq_and_mag(df):
    n = len(df['value'])
    if n>13:
        y = list(pd.to_numeric(df['value'], errors='coerce').astype(float).to_numpy())
        y = butter_lowpass_filter(y, cutoff_freq, fs, order)
    if n<=13 and n>1:
        y = list(pd.to_numeric(df['value'], errors='coerce').astype(float).to_numpy())
        y = np.interp(np.linspace(0, len(y)-1, 13), np.arange(len(y)), y).tolist()
        y = butter_lowpass_filter(y, cutoff_freq, fs, order)
    else:
        y = list(pd.to_numeric(df['value']*13, errors='coerce').astype(float).to_numpy())
        y = np.interp(np.linspace(0, len(y)-1, 13), np.arange(len(y)), y).tolist()
        y = butter_lowpass_filter(y, cutoff_freq, fs, order)   
    scaler = MinMaxScaler()
    y = scaler.fit_transform(np.array(y).reshape(-1, 1)).flatten()
    fft_values =  np.fft.fftshift(np.fft.fft(y))
    freqs =  np.fft.fftshift(np.fft.fftfreq(len(y), 1/fs))
    magnitude = np.abs(fft_values)
    return magnitude[len(magnitude)//2:], freqs[len(freqs)//2:]

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
    

def get_mag_freq(df1, df2, max_id):
    magnitude_fol, magnitude_init = [], []
    freqs1, freqs2 = [], []

    for i in range(max_id):
        if i ==0:
            continue
        idx = str(i)
        dataframe1 = df1[df1['index'] == idx]
        dataframe2 = df2[df2['index'] == idx]
        if dataframe1.empty or dataframe2.empty:
            continue
        res1, freq1 = calculate_freq_and_mag(dataframe1)
        res2, freq2 = calculate_freq_and_mag(dataframe2)

        res1 = normalize_using_pdf(res1)
        res2 = normalize_using_pdf(res2)

        magnitude_fol.append(res1)
        magnitude_init.append(res2)
        freqs1.append(freq1)
        freqs2.append(freq2)

    return magnitude_fol, magnitude_init, freqs1, freqs2

def plot_initiator_follower_line(fol, init, freqs1, freqs2, corpus, measure, threshold=0.015, min_sep=0.01):
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
    min_common = max(common_freqs[0], common_freqs_init[0])
    max_common = min(common_freqs[-1], common_freqs_init[-1])
    mask = (common_freqs >= min_common) & (common_freqs <= max_common)
    freq_common = common_freqs[mask]
    
    interp_init_on_fol = interp1d(common_freqs_init, mean_mag_init, kind='linear', bounds_error=False, fill_value=np.nan)
    mean_mag_init_interp = interp_init_on_fol(freq_common)
    mean_mag_fol_trimmed = mean_mag[mask]
    diff = np.abs(mean_mag_fol_trimmed - mean_mag_init_interp)
    diverge_idx = np.where(diff > min_sep)[0]
    if len(diverge_idx) == 0:
        converge_freq = None 
    else:
        post_div_idx = diverge_idx[0]
        post_diff = diff[post_div_idx:]
        converge_locs = np.where(post_diff < threshold)[0]

        if len(converge_locs) == 0:
            converge_freq = None
        else:
            converge_freq = freq_common[post_div_idx + converge_locs[0]]

    plt.figure(figsize=(7, 4))
    valid_mask = ~np.isnan(mean_mag) & ~np.isnan(std_mag)
    plt.plot(common_freqs[valid_mask], mean_mag[valid_mask], label="Follower", color="blue")

    plt.fill_between(common_freqs[valid_mask], mean_mag - std_mag, mean_mag + std_mag, color="blue", alpha=0.1)

    plt.plot(common_freqs_init[valid_mask], mean_mag_init[valid_mask], label="Initiator", color="green")
    plt.fill_between(common_freqs_init[valid_mask], mean_mag_init - std_mag_init, mean_mag_init + std_mag_init, color="green", alpha=0.1)


    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Normalised magnitude")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"fourier_plot {corpus} {measure}.png", dpi=300)
    plt.close()


if __name__ == '__main__':
    dataset = ['A2']
    #dataset =['A2', 'btsj','combined_candor','Hazumi']
    #dataset = ['vm2','texprax']
    for item in dataset:
        path1 = item+'_initiator_semcom_gpt.txt'
        path2 = item+'_follower_semcom_gpt.txt'
        df_initiator = parse_file(path1)
        df_follower = parse_file(path2)
        df_initiator= df_initiator.dropna(subset=['value'])
        df_follower= df_follower.dropna(subset=['value'])
        df_initiator['role'] = 'initiator'
        df_follower['role'] = 'follower'
        df_initiator['utt_posit'] = df_initiator['utt_posit']//3
        df_follower['utt_posit'] = df_follower['utt_posit']//3
        dialogue_id = df_initiator['index'].unique()
        max_id =max(list(map(int, dialogue_id)))
        df_follower_sup = df_follower[df_follower['metric']=='surprisal']
        df_initiator_sup = df_initiator[df_initiator['metric']=='surprisal']
        df_follower_re = df_follower[df_follower['metric']=='renyi']
        df_initiator_re = df_initiator[df_initiator['metric']=='renyi']
        df_follower_shannon = df_follower[df_follower['metric']=='shannon']
        df_initiator_shannon = df_initiator[df_initiator['metric']=='shannon']

        frequencies, power = periodogram(df_follower_sup)
        print(frequencies, power)
        
        
        res1,res2,res3,res4 = get_mag_freq(df_follower_sup,df_initiator_sup,max_id)
        plot_initiator_follower_line(res1,res2,res3,res4, corpus=item, measure='surprisal')
    
        res1,res2,res3,res4 = get_mag_freq(df_follower_shannon,df_initiator_shannon,max_id)
        plot_initiator_follower_line(res1,res2,res3,res4, corpus=item, measure='shannon')
    
        res1,res2,res3,res4 = get_mag_freq(df_follower_re, df_initiator_re,max_id)
        plot_initiator_follower_line(res1,res2,res3,res4, corpus=item, measure='renyi')
