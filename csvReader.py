import matplotlib.pyplot as plt
import pandas as pd

GLOB_PATH = "D:/praca_magisterska/a10_f100z"
PERIOD = 50e-3
AMPLITUDE = 10

df = pd.read_csv(str(GLOB_PATH[-9:] + '.csv'))
time = df['Col0'].to_list()
leftAngle = df['Col1'].to_list()
rightAngle = df['Col2'].to_list()
contactLength = df['Col3'].to_list()

fig, (ax1, ax2) = plt.subplots(2, gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
line1, = ax1.plot(time, leftAngle, label='Left angle')
line2, = ax1.plot(time, rightAngle, label='Right angle')
line3, = ax2.plot(time, [int(contactLength_) for contactLength_ in contactLength],label='length of droplet contact zone')
ax1.legend(handles=[line1, line2])
ax1.grid(visible=True, which='both', linestyle='--', linewidth='0.25')
ax2.legend(handles=[line3])
ax1.set(xlabel="time [s]", ylabel="Angle [degrees]")
ax2.set(xlabel="time [s]", ylabel="Droplet contact zone [pixels]")
ax2.grid(visible=True, which='both', linestyle='--', linewidth='0.25')

ratio = GLOB_PATH[-4:-1]
period = PERIOD * (int(ratio)/100)
frequency = 1/period
amplitude = (int(GLOB_PATH[-8:-6]) / 100) * AMPLITUDE

ax1.set_title(str('Frequency: ' + str(frequency) + '[Hz]    ' + 'Amplitude: ' + str(amplitude) + '[mm]'))

plt.show()