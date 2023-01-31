from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
plt.style.use('seaborn-v0_8-dark-palette')


def plotIMG(path):
    GLOB_PATH = str(path)
    PERIOD = 50e-3  # [s]
    BASE_FREQ = 1 / PERIOD  # [Hz]
    AMPLITUDE = 10  # [mm]

    # df = pd.read_csv(str(GLOB_PATH[-9:] + '.csv'))
    df = pd.read_csv(str(path))

    time = df['Col0'].to_list()
    leftAngle = df['Col1'].to_list()
    rightAngle = df['Col2'].to_list()
    contactLength = df['Col3'].to_list()

    fig, (ax1, ax2) = plt.subplots(2, gridspec_kw={'height_ratios': [3, 1]}, sharex=True, figsize=(18, 8))
    line1, = ax1.plot(time, leftAngle, label='Left angle', color='navy')
    line2, = ax1.plot(time, rightAngle, label='Right angle', color='deepskyblue')
    line3, = ax2.plot(time, [int(contactLength_) for contactLength_ in contactLength],
                      label='length of droplet contact zone', color='slategray')
    ax1.legend(handles=[line1, line2])
    ax1.grid(visible=True, which='both', linestyle='--', linewidth='0.25')
    ax2.legend(handles=[line3])
    ax1.set(xlabel="time [s]", ylabel="Angle [degrees]")
    ax2.set(xlabel="time [s]", ylabel="Droplet contact zone [pixels]")
    ax2.grid(visible=True, which='both', linestyle='--', linewidth='0.25')

    ratio = GLOB_PATH[-7:-5]
    frequency = BASE_FREQ * (int(ratio) / 100)
    amplitude = (int(GLOB_PATH[-11:-9]) / 100) * AMPLITUDE
    ax1.set_title(
        str('Frequency: ' + str("{:.2f}".format(frequency)) + '[Hz]    ' + 'Amplitude: ' + str(amplitude) + '[mm]'))

    # plt.show()
    fig.savefig(str(GLOB_PATH[0:-4] + '.png'), dpi=100)


if __name__ == "__main__":
    GLOB_PATH = "D:/praca_magisterska/master-s-thesis"

    for path in Path(GLOB_PATH).glob("*.csv"):
        plotIMG(path)




