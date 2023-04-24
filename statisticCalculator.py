import pandas as pd
from pathlib import Path


# Python program to get average of a list
def average(lst):
    return sum(lst) / len(lst)


def readCSV(path):
    df1 = pd.read_csv(str(path))

    numb_of_frames = df1['Col0'].to_list()
    df = pd.read_csv(path, names=['Col0', 'Col1', 'Col2', 'Col3', 'Col4'])
    # Filter rows that contain '1' or '2' in Col0

    real_numb = []
    dx_mean = []
    dy_mean = []
    area_mean = []
    D_list = []

    for i in range(min(numb_of_frames), max(numb_of_frames) + 1):
        df_filtered = df[df['Col0'].isin([str(i)])]
        # Print the results
        mean_dx = [float(x) for x in df_filtered['Col2'].to_list()]
        mean_dy = [float(x) for x in df_filtered['Col3'].to_list()]
        mean_area = [float(x) for x in df_filtered['Col4'].to_list()]
        # D=(D_x*D_x*D_y)^(1/3)
        D = (average(mean_dx)*average(mean_dx)*average(mean_dy)) ** (1/3)
        print(df_filtered)
        real_numb.append(i)
        dx_mean.append(average(mean_dx))
        dy_mean.append(average(mean_dy))
        area_mean.append(average(mean_area))
        D_list.append(D)

    return [real_numb, dx_mean, dy_mean, area_mean, D_list]

if __name__ == "__main__":
    GLOB_PATH = "D:/praca_magisterska"

    REAL_NUMB_GLOB = []
    DX_MEAN_GLOB = []
    DY_MEAN_GLOB = []
    AREA_MEAN_GLOB = []
    D_LIST_GLOB = []

    for path in Path(GLOB_PATH).glob("*.csv"):
        ret = readCSV(path)
        REAL_NUMB_GLOB += ret[0]
        DX_MEAN_GLOB += ret[1]
        DY_MEAN_GLOB += ret[2]
        AREA_MEAN_GLOB += ret[3]
        D_LIST_GLOB += ret[4]

    df = pd.DataFrame(list(zip(*[REAL_NUMB_GLOB, DX_MEAN_GLOB, DY_MEAN_GLOB, AREA_MEAN_GLOB, D_LIST_GLOB]))).add_prefix('Col')
    df.to_csv('Stat.csv', index=False)

