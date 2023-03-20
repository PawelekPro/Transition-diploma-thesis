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
    x_ref = df['Col4'].to_list()
    avgScope = 5
    avgContactLength = []
    i = 0
    # Uśrednianie długości strefy kontaktu według @param avgScope ostatnich punktów dyskretnych
    while i < len(contactLength) - avgScope + 1:
        bufor = contactLength[i:i + avgScope]
        bufor_avarage = round(sum(bufor) / avgScope, 2)
        avgContactLength.append(bufor_avarage)
        i += 1

    dContactLength = [contactLength_ - contactLength[0] for contactLength_ in avgContactLength]

    fig, (ax1, ax2, ax3) = plt.subplots(3, gridspec_kw={'height_ratios': [4, 2, 1]}, sharex=True, figsize=(20, 8))
    line2, = ax1.plot(time[0:1000], leftAngle[0:1000], label='Left angle', color='navy')
    line3, = ax2.plot(time[0:1000], x_ref[0:1000], label='Actual position', color='slategray')
    # line2, = ax1.plot(time[0:1000], rightAngle[0:1000], label='Right angle', color='deepskyblue')

    line1, = ax1.plot(time[0:1000], [leftAngle[0] for x in leftAngle][0:1000], color='navy', linestyle=(0, (5, 5)), linewidth=1, label='Static angle')

    line4, = ax3.plot(time[0:1000], [int(contactLength_) for contactLength_ in dContactLength][0:1000],
                      label='Length of droplet contact zone', color='deepskyblue', linewidth=1)
    ax1.legend(handles=[line1, line2])
    ax1.grid(visible=True, which='both', linestyle='--', linewidth='0.25')
    ax2.legend(handles=[line3])
    ax1.set(xlabel="time [s]", ylabel="Angle [degrees]")
    ax3.set(xlabel="time [s]", ylabel="Contact zone [pixels]")
    ax2.grid(visible=True, which='both', linestyle='--', linewidth='0.25')
    ax3.grid(visible=True, which='both', linestyle='--', linewidth='0.25')
    ax2.set(xlabel="time [s]", ylabel="Position [pixels]")
    ax3.legend(handles=[line4])
    ax3.set_ylim([-5, 5])

    ratio = GLOB_PATH[-7:-5]
    frequency = 9.9000
    amplitude = '1.00'
    ax1.set_title(
        str('No. measurement: 1    ' + 'Frequency: ' + str("{:.2f}".format(frequency)) + '[Hz]    ' + 'Amplitude: ' + str(amplitude) + '[mm]'))

    # plt.show()
    fig.savefig(str(GLOB_PATH[0:-4] + '_left.png'), dpi=100)


if __name__ == "__main__":
    GLOB_PATH = "D:/praca_magisterska/master-s-thesis"

    for path in Path(GLOB_PATH).glob("*.csv"):
        plotIMG(path)





