from pathlib import Path
from scipy.interpolate import InterpolatedUnivariateSpline
from csv import writer
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
plt.style.use('seaborn-v0_8-dark-palette')

def get_sub(x):
    normal = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+-=()"
    sub_s = "ₐ₈CDₑբGₕᵢⱼₖₗₘₙₒₚQᵣₛₜᵤᵥwₓᵧZₐ♭꜀ᑯₑբ₉ₕᵢⱼₖₗₘₙₒₚ૧ᵣₛₜᵤᵥwₓᵧ₂₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎"
    res = x.maketrans(''.join(normal), ''.join(sub_s))
    return x.translate(res)

def plotIMG(path):
    GLOB_PATH = str(path)
    PERIOD = 50e-3  # [s]
    BASE_FREQ = 1 / PERIOD  # [Hz]
    AMPLITUDE = 10  # [mm]

    df = pd.read_csv(str(path))

    time = df['Col0'].to_list()
    leftAngle = df['Col1'].to_list()
    rightAngle = df['Col2'].to_list()

    ratio = GLOB_PATH[-7:-5]
    frequency = BASE_FREQ * (int(ratio) / 100)
    amplitude = (int(GLOB_PATH[-11:-9]) / 100) * AMPLITUDE

    # Average plot
    leftAng = InterpolatedUnivariateSpline(time, leftAngle)
    qq_left = leftAng.integral(min(time), max(time))/max(time)

    RightAng = InterpolatedUnivariateSpline(time, rightAngle)
    qq_right = RightAng.integral(min(time), max(time))/max(time)
    print("std error [%]: [AVG left, AVG right]: ")
    print(100*(np.mean(leftAngle) - qq_left)/np.mean(leftAngle),
          100*(np.mean(rightAngle) - qq_right)/np.mean(rightAngle))

    return [frequency, min(leftAngle), max(leftAngle), min(rightAngle), max(rightAngle), qq_left, qq_right]

if __name__ == "__main__":
    GLOB_PATH = "D:/praca_magisterska/master-s-thesis"

    frequency = []
    left_min = []
    left_max = []
    right_min = []
    right_max = []
    avg_left = []
    avg_right = []

    for path in Path(GLOB_PATH).glob("*.csv"):
        data = plotIMG(path)
        frequency.append(data[0])
        left_min.append(data[1])
        left_max.append(data[2])
        right_min.append(data[3])
        right_max.append(data[4])
        avg_left.append(data[5])
        avg_right.append(data[6])

    # plot 1
    plt.subplot(1, 2, 1)
    plt.errorbar(frequency, avg_left,
                 [[i - j for i, j in zip(avg_left, left_min)], [j - i for i, j in zip(avg_left, left_max)]],
                 fmt='o', linewidth=2, capsize=6)
    plt.xlim([min(frequency)-0.5, max(frequency)+0.5])
    plt.ylim([0, 140])
    plt.title("LEFT ANGLE")
    plt.xlabel("Frequency [Hz]", size=14)
    plt.ylabel('[min(\u03F4{}), avg(\u03F4{}), max(\u03F4{})]'.format(get_sub('L'), get_sub('L'), get_sub('L')), size=14)
    plt.grid(visible=True, which='both', linestyle='--', linewidth='0.25')
    for a, b in zip(frequency, avg_left):
        plt.text(a, b, str("{:.2f}".format(b)), size=11)
    for a, b in zip(frequency, left_min):
        plt.text(a, b, str("{:.2f}".format(b)), size=11)
    for a, b in zip(frequency, left_max):
        plt.text(a, b, str("{:.2f}".format(b)), size=11)

    # plot 2
    plt.subplot(1, 2, 2)
    plt.errorbar(frequency, avg_right,
                 [[i - j for i, j in zip(avg_right, right_min)], [j - i for i, j in zip(avg_right, right_max)]],
                 fmt='o', linewidth=2, capsize=6)
    plt.xlim([min(frequency) - 0.5, max(frequency) + 0.5])
    plt.ylim([0, 140])
    plt.title("RIGHT ANGLE")
    plt.xlabel("Frequency [Hz]", size=14)
    plt.ylabel('[min(\u03F4{}), avg(\u03F4{}), max(\u03F4{})]'.format(get_sub('P'), get_sub('P'), get_sub('P')), size=14)
    plt.grid(visible=True, which='both', linestyle='--', linewidth='0.25')
    for a, b in zip(frequency, avg_right):
        plt.text(a, b, str("{:.2f}".format(b)), size=11)
    for a, b in zip(frequency, right_min):
        plt.text(a, b, str("{:.2f}".format(b)), size=11)
    for a, b in zip(frequency, right_max):
        plt.text(a, b, str("{:.2f}".format(b)), size=11)

    plt.show()