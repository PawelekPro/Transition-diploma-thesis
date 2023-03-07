from math import exp, sin, cos, pi
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

if __name__ == "__main__":
    GLOB_PATH = "D:/praca_magisterska/LinMot_user_manual"
    amplitude_ = 1
    lambda_ = 0.3
    ang_freq_ = 3
    time = np.linspace(0, 2*2*pi, 1000)
    y = [amplitude_ * exp(-lambda_*x)*(cos(ang_freq_*x))for x in time]
    reverse_arr = np.flip(y)

    time = np.linspace(0, 20*2*pi, 10000)
    y_steady_state = [amplitude_ * cos(ang_freq_*x) for x in time]

    time = np.concatenate((np.linspace(0, 4*2*pi, 1000), np.linspace(4*2*pi, 20*2*pi + 4*2*pi, 10000), np.linspace(20*2*pi + 4*2*pi, 20*2*pi + 8*2*pi, 1000)))
    y_full_scope = np.concatenate((reverse_arr, y_steady_state, y))
    plt.figure(figsize=(15, 8))
    plt.plot(time, y_full_scope)
    plt.show()

    col1 = 'curvePointsValues'
    data = pd.DataFrame({col1: y_full_scope})
    data.to_csv(str(GLOB_PATH + '/testCurve.csv'), index=False, header=False)



