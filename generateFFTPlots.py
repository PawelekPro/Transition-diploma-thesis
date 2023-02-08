import glob
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft, ifft, fftfreq, rfft
import pandas as pd
import cmath

all_exp_paths = glob.glob("D:/Magisterka/pomiary/*.csv")

for path in all_exp_paths:

    base_freq = 20 * int(path[-7:-5]) / 100

    case_name = path[-12:-4]
    df = pd.read_csv(path)
    time = df['Col0'].to_list()
    angle_left = df['Col1'].to_list()
    angle_right = df['Col2'].to_list()
    angle_left = np.array(angle_left)
    angle_right = np.array(angle_right)

    yf = fft(angle_left) / len(angle_left)
    yf = abs(yf)
    xf = fftfreq(len(angle_left), time[0] * 2) / base_freq
    #ids = np.argwhere(yf > 0.1)
    # y = yf[ids]
    # x = xf[ids]

    plt.plot(xf[1:-1], yf[1:-1])
    plt.grid()
    plt.xlabel('normalised frequency')
    plt.ylabel('Amplitude')
    plt.title('FFT analysis of anlge left for case ' + case_name)
    plt.xlim(-5, 5)
    # plt.ylim(0, 20)
    plt.savefig(case_name + '_fft_plot.png')
    plt.clf()