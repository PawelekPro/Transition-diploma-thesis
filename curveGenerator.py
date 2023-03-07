from math import exp, sin, cos, pi, radians
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

if __name__ == "__main__":
    GLOB_PATH = "D:/praca_magisterska/LinMot_user_manual"
    amplitude_ = 1
    lambda_ = 0.3
    ang_freq_ = 4
    transient_scope = 3 # Note: transient_scope must be equal to ang_freq?
    steady_scope = 20

    time = np.linspace(0, transient_scope*2*pi, 1000)
    y = [amplitude_ * exp(-lambda_*x)*(cos(transient_scope*x + pi/2)) for x in time]
    reverse_arr = np.flip(y)

    time = np.linspace(0, steady_scope*2*pi, 10000)
    y_steady_state = [amplitude_ * cos(ang_freq_*x + 3*pi/2) for x in time]

    time = np.concatenate((np.linspace(0, transient_scope*2*pi, 1000), np.linspace(transient_scope*2*pi, steady_scope*2*pi +
                                                                                   transient_scope*2*pi, 10000),
                           np.linspace(steady_scope*2*pi + transient_scope*2*pi, steady_scope*2*pi + 2*transient_scope*2*pi, 1000)))
    y_damped = [-i for i in y]
    y_full_scope = np.concatenate((reverse_arr, y_steady_state, y_damped))
    plt.figure(figsize=(15, 8))
    plt.plot(time, y_full_scope)
    plt.show()

    col1 = 'curvePointsValues'
    data = pd.DataFrame({col1: y_full_scope})
    data.to_csv(str(GLOB_PATH + '/testCurve.csv'), index=False, header=False)




