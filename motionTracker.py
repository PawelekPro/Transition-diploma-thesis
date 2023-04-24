import cv2
from pathlib import Path
from PIL import Image
import glob
import math

from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np
import shutil
import os


def printText(image, value, area, fontScale, printNone=False):
    font = cv2.FONT_HERSHEY_SIMPLEX
    color = (255, 255, 255)
    # Line thickness of 1 px
    thickness = 1

    if printNone is False:
        image = cv2.putText(image, "Center [px, px]: (%.2f, %.2f)" % (value[0][0], value[0][1]), (20, 40), font,
                            fontScale, color, thickness, cv2.LINE_AA)

        image = cv2.putText(image, "Axis length [px]: %.2f" % value[1][0], (45, 60), font,
                            fontScale, color, thickness, cv2.LINE_AA)

        image = cv2.putText(image, "Axis length [px]: %.2f" % value[1][1], (45, 80), font,
                            fontScale, color, thickness, cv2.LINE_AA)
        image = cv2.putText(image, "Area [px2]: %.2f" % area, (20, 100), font,
                            fontScale, color, thickness, cv2.LINE_AA)

    else:
        image = cv2.putText(image, "Center [px, px]: (None, None)", (20, 40), font,
                            fontScale, color, thickness, cv2.LINE_AA)

        image = cv2.putText(image, "Axis length [px]: None", (45, 60), font,
                            fontScale, color, thickness, cv2.LINE_AA)

        image = cv2.putText(image, "Axis length [px]: None", (45, 80), font,
                            fontScale, color, thickness, cv2.LINE_AA)
        image = cv2.putText(image, "Area [px2]: None", (20, 100), font,
                            fontScale, color, thickness, cv2.LINE_AA)
    return image


def drawAxes(img, ellipse):
    # Draw major and minor axis
    (x, y), (a, b), angle = ellipse
    theta = angle * (3.14 / 180.0)
    cos_val = np.cos(theta)
    sin_val = np.sin(theta)
    # End points of major axis
    major_start = (int(x - (a / 2) * cos_val), int(y + (a / 2) * sin_val))
    major_end = (int(x + (a / 2) * cos_val), int(y - (a / 2) * sin_val))
    # End points of minor axis
    minor_start = (int(x + (b / 2) * sin_val), int(y + (b / 2) * cos_val))
    minor_end = (int(x - (b / 2) * sin_val), int(y - (b / 2) * cos_val))
    # Draw major and minor axis
    cv2.line(img, major_start, major_end, (0, 0, 255), 2)
    cv2.line(img, minor_start, minor_end, (255, 0, 0), 2)

    # Add legend
    cv2.line(img, (20, 55), (40, 55), (0, 0, 255), 5)
    cv2.line(img, (20, 75), (40, 75), (255, 0, 0), 5)
    return img


def interpolate_spline(path, frame, fps, numb_of_real, previous_time):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    # cv2.imshow('test_im', img)
    # cv2.waitKey(0)
    time = frame / fps
    if time - previous_time > 0.004:
        numb_of_real += 1

    cv2.putText(img, "Time [s]: %.4f" % time, (20, 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.putText(img, "Number of droplet: %d" % numb_of_real, (20, 120), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 1, cv2.LINE_AA)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    """BINARY TRESHOLDING"""
    _, thresh = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
    thresh = 255 - thresh

    thresh_object = thresh

    contours_object, _ = cv2.findContours(thresh_object, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if len(contours_object) != 0:
        # find the biggest contour (c) by the area
        c_object = max(contours_object, key=cv2.contourArea)
        # cv2.drawContours(img, c_object, -1, (0, 255, 255), 2)
        # (x, y), radius = cv2.minEnclosingCircle(c_object)
        # center = (int(x), int(y))
        # radius = int(radius)
        # cv2.circle(img, center, radius, 255, 2)
        CrossSectionArea = cv2.contourArea(c_object)
        ellipse = cv2.fitEllipse(c_object)
        if abs(ellipse[1][0] - ellipse[1][1]) < 15 and ellipse[1][0] > 100 and \
                ellipse[1][0] / 2 < ellipse[0][1] < 1023 - ellipse[1][0] / 2:
            cv2.ellipse(img, ellipse, (255, 255, 255), 2)
            printText(img, ellipse, CrossSectionArea, 0.5)
            drawAxes(img, ellipse)
            x_obj, y_obj, w_obj, h_obj = cv2.boundingRect(c_object)
            cv2.rectangle(img, [x_obj, y_obj], [x_obj + w_obj, y_obj+h_obj], (255,255,255), 1)
            path_to_save = 'D:/praca_magisterska/conv/' + path[-23:]
            cv2.imwrite(str(path_to_save), img)
            return [True, 1, time, w_obj, h_obj, CrossSectionArea, numb_of_real]
        else:
            printText(img, ellipse, CrossSectionArea, 0.5, printNone=True)
            path_to_save = 'D:/praca_magisterska/conv/' + path[-23:]
            cv2.imwrite(str(path_to_save), img)
            return [True, 2, time, numb_of_real]
    else:
        return [False, 3]


def main():
    FRAME_RATE = 500
    frame = []
    paths = []

    _width = []
    _height = []
    _time = []
    _crossSectionArea = []
    _numb_of_real = []

    shutil.rmtree('D:/praca_magisterska/conv')
    os.mkdir('D:/praca_magisterska/conv')

    GLOB_PATH = "D:/praca_magisterska/zrzut_kropli_10"
    for path in Path(GLOB_PATH).glob("*.png"):
        frame.append(int(str(path)[-9] + str(path)[-8] + str(path)[-7] + str(path)[-6] + str(path)[-5]))
        paths.append(str(path))

    numer_realizacji = 160
    previous_time = 0
    print("\nConverting images:")
    for i in tqdm(range(len(frame)), bar_format='{l_bar}{bar:40}{r_bar}{bar:-10b}'):
        if frame[i] % 1 == 0 and frame[i] < 5500:
            ret = interpolate_spline(str(paths[i]), frame[i], FRAME_RATE, numer_realizacji, previous_time)
            if ret[0] is not False and ret[1] == 1:
                previous_time = ret[2]
                _time.append(ret[2])
                _width.append(ret[3])
                _height.append(ret[4])
                _crossSectionArea.append(ret[5])
                _numb_of_real.append(ret[6])
                numer_realizacji = ret[6]
            elif ret[0] is not False and ret[1] == 2:
                previous_time = ret[2]
                numer_realizacji = ret[3]

    time = [frame_ / FRAME_RATE for frame_ in frame]

    # Csv writer
    import pandas as pd
    df = pd.DataFrame(list(zip(*[_numb_of_real, _time, _width, _height, _crossSectionArea]))).add_prefix('Col')
    df.to_csv(str(GLOB_PATH + '.csv'), index=False)

def make_gif(frame_folder):
    frames = [Image.open(image) for image in glob.glob(f"{frame_folder}/*.png")]
    frame_one = frames[0]
    path_to_save_gif = "anim_001.gif"
    frame_one.save(str(path_to_save_gif), format="GIF", append_images=frames,
                   save_all=True, duration=400, loop=0)


if __name__ == "__main__":
    main()
    # make_gif("D:/praca_magisterska/conv")

# TODO: view calibration
