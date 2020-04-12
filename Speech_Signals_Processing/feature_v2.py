import numpy as np
from scipy.io import wavfile
from scipy.fftpack import dct
import warnings
import matplotlib.pyplot as plt

#绘制时域图
def plot_time(singal, sample_rate):
    time = np.arange(0, len(singal)) * (1.0 / sample_rate)
    plt.figure(figsize=(20, 5))
    plt.plot(time, singal)
    plt.xlabel('Time(s)')
    plt.ylabel('Amplitude')
    plt.grid()
    plt.show()

#绘制频域图
def plot_frep(signal, smaple_rate, fft_size=512):
    xf = np.fft.rfft(signal, fft_size) / fft_size
    freqs = np.linspace(0, smaple_rate/2, fft_size/2+1)
    xfp = 20 * np.log10(np.clip(np.abs(xf),1e-20, 1e100))
    plt.figure(figsize=(20, 5))
    plt.plot(freqs, xfp)
    plt.xlabel('Freq(hz)')
    plt.ylabel('dB')
    plt.grid()
    plt.show()

#绘制频谱图
def plot_spectrogram(spec, note):
    fig = plt.figure(figsize=(20,5))
    heatmap = plt.pcolor(spec)
    fig.colorbar(mappable=heatmap)
    plt.xlabel('Time(s)')
    plt.ylabel(note)
    plt.tight_layout()
    plt.show()

# 绘制频谱
def plot_specgram(spec, smaple_rate):
    plt.specgram(spec, Fs=smaple_rate, scale_by_freq=True, sides='default')
    plt.ylabel('Frequency')
    plt.xlabel('Time(s)')
    plt.show()

def delta(feat, N):
    """Compute delta features from a feature vector sequence.
    :param feat: A numpy array of size (NUMFRAMES by number of features) containing features. Each row holds 1 feature vector.
    :param N: For each frame, calculate delta features based on preceding and following N frames
    :returns: A numpy array of size (NUMFRAMES by number of features) containing delta features. Each row holds 1 delta feature vector.
    """
    if N < 1:
        raise ValueError('N must be an integer >= 1')
    NUMFRAMES = len(feat)
    denominator = 2 * sum([i**2 for i in range(1, N+1)])
    delta_feat = np.empty_like(feat)
    padded = np.pad(feat, ((N, N), (0, 0)), mode='edge')   # padded version of feat
    for t in range(NUMFRAMES):
        delta_feat[t] = np.dot(np.arange(-N, N+1), padded[t : t+2*N+1]) / denominator   # [t : t+2*N+1] == [(N+t)-N : (N+t)+N+1]
    return delta_feat

def fbank_and_mfcc(signal, sample_rate):
    #预加重(Pre-Emphasis)
    pre_emphasis = 0.97
    em_phasized_signal = np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])


    #分帧
    frame_size, frame_stride = 0.025, 0.01   #帧长25ms, overlap 10 ms
    frame_length, frame_step = int(round(frame_size * sample_rate)), int(round(frame_stride * sample_rate))
    print(frame_length, frame_step)
    signal_length = len(em_phasized_signal)
    num_frames = int(np.ceil(np.abs(signal_length - frame_length) / frame_step)) + 1
    print(num_frames)
    pad_signal_length = (num_frames - 1) * frame_step + frame_length
    print(pad_signal_length, signal_length)
    z = np.zeros((pad_signal_length - signal_length))
    pad_signal = np.append(em_phasized_signal, z)
    indices = np.arange(0, frame_length).reshape(1,-1) + np.arange(0, num_frames * frame_step, frame_step).reshape(-1,1)
    frames=pad_signal[indices]
    print(frames.shape)

    #加窗(Windows)
    hamming = np.hamming(frame_length)
    frames *= hamming

    #快速傅里叶变换(FFT)
    NFFT = 512
    mag_frames = np.absolute(np.fft.rfft(frames, NFFT))

    #帧能量
    pow_frames = ((1.0 / NFFT) * (mag_frames ** 2))
    print('pow_frames.shape:', pow_frames.shape)
    print(pow_frames[1].shape)

    pow_frames[pow_frames <= 1e-30] = 1e-30
    log_pow_frame = 10 * np.log10(pow_frames)

    log_pow_frame = np.sum(log_pow_frame, axis=1)
    print(log_pow_frame.shape)

    #Mel滤波器
    low_freq_mel = 0
    high_freq_mel = 2595 * np.log10(1 + (sample_rate / 2) / 700)

    nfilt = 40
    mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)
    hz_points = 700 * (10 ** (mel_points / 2595) - 1)

    fbank = np.zeros((nfilt, int(NFFT/2 + 1)))
    bin = (hz_points / (sample_rate / 2)) * (NFFT / 2)
    for i in range(1, nfilt + 1):
        left = int(bin[i-1])
        center = int(bin[i])
        right = int(bin[i+1])
        for j in range(left, center):
            fbank[i-1, j+1] = (j + 1 - bin[i-1]) / (bin[i] - bin[i-1])
        for j in range(center, right):
            fbank[i-1, j+1] = (bin[i+1] - (j+1)) / (bin[i+1] - bin[i])
    print(fbank)



    #FBank
    filter_banks = np.dot(pow_frames, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
    filter_banks = 20 * np.log10(filter_banks)
    print(filter_banks.shape)
    plot_spectrogram(filter_banks.T, 'Filter Banks')

    #MFCC
    num_ceps = 13
    mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1:(num_ceps+1)]
    print(mfcc.shape)
    plot_spectrogram(mfcc.T, 'MFCC Coefficients')

    #一阶差分
    mfcc_d = delta(mfcc, N = 2)

    #二阶差分
    mfcc_dd = delta(mfcc_d, N = 2)

    # N维MFCC参数（N/3 MFCC系数+ N/3 一阶差分参数+ N/3 二阶差分参数）+帧能量
    mfcc_colmn = np.column_stack((mfcc, mfcc_d, mfcc_dd, log_pow_frame))
    print(mfcc_colmn.shape)
    plot_spectrogram(mfcc_colmn.T, 'MFCC Dimension-40')


    #sinusoidal liftering
    cep_lift = 23
    (nframes, ncoeff) = mfcc.shape
    n = np.arange(ncoeff)
    lift = 1 + (cep_lift / 2) * np.sin(np.pi * n / cep_lift)
    mfcc *= lift
    plot_spectrogram(mfcc.T, 'MFCC Sinusoidal')

    #去均值 CMN
    filter_banks -= (np.mean(filter_banks, axis=0) + 1e-8)
    plot_spectrogram(filter_banks.T, 'Filter Banks CMN')

    mfcc -= (np.mean(mfcc, axis=0) + 1e-8)
    plot_spectrogram(mfcc.T, 'MFCC Coefficients CMN')

    return filter_banks, mfcc, mfcc_colmn


if __name__ == '__main__':
    sample_rate, signal = wavfile.read('00001.wav')
    print('sample rate:', sample_rate, ', frame length:', len(signal))
    plot_time(signal, sample_rate)
    plot_frep(signal, sample_rate)
    plot_specgram(signal, sample_rate)
    filter_banks, mfcc, mfcc_colmn = em_phasized_signal = fbank_and_mfcc(signal, sample_rate)

