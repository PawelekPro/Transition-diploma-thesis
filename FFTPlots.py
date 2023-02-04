from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft, fftfreq
import pandas as pd


def FFTPlotsCreate(path):
    PERIOD = 50e-3  # [s]
    BASE_FREQ = 1 / PERIOD  # [Hz]
    AMPLITUDE = 10  # [mm]
    # base_freq = 20 * int(GLOB_PATH[-8:-5]) / 100

    ratio = str(path)[-8:-5]
    frequency = BASE_FREQ * (int(ratio) / 100)
    amplitude = (int(str(path)[-12:-10]) / 100) * AMPLITUDE
    print(path)
    case_name = str(path)[-13:-4]
    df = pd.read_csv(path)
    time = df['Col0'].to_list()
    angle_left = df['Col1'].to_list()
    angle_right = df['Col2'].to_list()
    angle_left = np.array(angle_left)
    angle_right = np.array(angle_right)

    yf = fft(angle_left) / len(angle_left)
    yf = abs(yf)
    xf = fftfreq(len(angle_left), time[0] * 2) / BASE_FREQ
    # ids = np.argwhere(yf > 0.1)
    # y = yf[ids]
    # x = xf[ids]

    plt.figure(figsize=(15, 8))
    plt.plot(xf[1:-1], yf[1:-1], color='mediumslateblue')
    plt.grid()
    plt.xlabel('Normalised frequency', size=14)
    plt.ylabel('Amplitude', size=14)
    plt.title('FFT analysis: Left angle for forcing amplitude: ' + str(amplitude) + '[mm] and frequency: ' +
              str("{:.2f}".format(frequency)) + '[Hz]', size=16)
    plt.xlim(-7.5, 7.5)
    # plt.ylim(0, 20)
    plt.savefig('FFTPlots/' + str(case_name + '_fft_plot.png'))
    plt.close()


def find_dom_parameters(path):
    PERIOD = 50e-3  # [s]
    BASE_FREQ = 1 / PERIOD  # [Hz]
    AMPLITUDE = 10  # [mm]
    # base_freq = 20 * int(GLOB_PATH[-8:-5]) / 100

    ratio = str(path)[-8:-5]
    frequency = BASE_FREQ * (int(ratio) / 100)
    amplitude = (int(str(path)[-12:-10]) / 100) * AMPLITUDE
    case_name = str(path)[-13:-4]
    df = pd.read_csv(path)
    time = df['Col0'].to_list()
    angle_left = df['Col1'].to_list()
    angle_right = df['Col2'].to_list()
    angle_left = np.array(angle_left)
    angle_right = np.array(angle_right)

    yf = fft(angle_left) / len(angle_left)
    yf = abs(yf)
    xf = fftfreq(len(angle_left), time[0] * 2) / BASE_FREQ

    dominant = np.argwhere(yf == max(yf[1:-1]))
    return [frequency, xf[dominant[0]], yf[dominant[0]]]


def main():
    GLOB_PATH = "D:/praca_magisterska/master-s-thesis/csv_data"
    frequency = []
    DomAmplitude = []
    DomFrequency = []

    for path in Path(GLOB_PATH).glob("*.csv"):
        FFTPlotsCreate(path)
        data = find_dom_parameters(path)
        frequency.append(data[0])
        DomFrequency.append(data[1])
        DomAmplitude.append(data[2])

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.scatter(DomAmplitude, DomFrequency, frequency)

    plt.figure(figsize=(20, 10))
    plt.subplot(1, 2, 1)
    plt.plot(frequency, DomFrequency, 'o')
    plt.title("f_DOM(fp) dependency")
    plt.xlabel("fp frequency [Hz]", size=14)
    plt.ylabel("f_DOM frequency (normalised)", size=14)
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(frequency, DomAmplitude, 'o')
    plt.title("A_DOM(fp) dependency")
    plt.xlabel("fp frequency [Hz]", size=14)
    plt.ylabel("A_DOM amplitude", size=14)
    plt.suptitle('Left angle', size=16)
    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()
