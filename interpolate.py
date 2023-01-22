import cv2
from pathlib import Path
from PIL import Image
import glob
import math
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import shutil
import os
from scipy.fft import fft, ifft, fftfreq

def printText(image, value, org, fontScale):
    font = cv2.FONT_HERSHEY_SIMPLEX

    color = 255
    # Line thickness of 1 px
    thickness = 1
    if value[0] == 0:
        image = cv2.putText(image, "Left angle: None", (org[0], org[1]), font,
                            fontScale, color, thickness, cv2.LINE_AA)
    else:
        image = cv2.putText(image, "Left angle: %.2f" % value[0], (org[0], org[1]), font,
                        fontScale, color, thickness, cv2.LINE_AA)
    if value[1] == 0:
        image = cv2.putText(image, "Right angle: None", (org[0], org[1] + 16), font,
                            fontScale, color, thickness, cv2.LINE_AA)
    else:
        image = cv2.putText(image, "Right angle: %.2f" % value[1], (org[0], org[1] + 16), font,
                            fontScale, color, thickness, cv2.LINE_AA)
    return image



def interpolate_spline(path):

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    """BINARY TRESHOLDING"""
    _, thresh = cv2.threshold(img, 20, 255, cv2.THRESH_BINARY)
    thresh = 255 - thresh

    thresh_reference = np.zeros_like(img)
    thresh_object = np.zeros_like(img)
    output_reference = np.zeros_like(img)
    output_object = np.zeros_like(img)

    image_cutting_size = 300
    thresh_reference[0:image_cutting_size, :] = thresh[0:image_cutting_size, :]
    thresh_object[image_cutting_size:len(thresh), :] = thresh[image_cutting_size:len(thresh), :]


    GROUND_HEIGHT = 860

    # cutting out everything below the plate
    thresh_object[GROUND_HEIGHT + 1:len(output_object), :] = 0

    contours_reference, _ = cv2.findContours(thresh_reference, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours_object, _ = cv2.findContours(thresh_object, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if len(contours_reference) != 0:
        # find the biggest contour (c) by the area
        c_reference = max(contours_reference, key=cv2.contourArea)
        c_object = max(contours_object, key=cv2.contourArea)
        cv2.drawContours(output_object, c_object, -1, 255, 1)
        cv2.drawContours(output_reference, c_reference, -1, 255, 1)
        if len(contours_object) != 0:
            x_obj, y_obj, w_obj, h_obj = cv2.boundingRect(c_object)

    x_ref, y_ref, w_ref, h_ref = cv2.boundingRect(c_reference)

    # Drawing line which represents plate (ground)
    output_object[GROUND_HEIGHT, :] = 125

    pixels = np.argwhere(output_object == 255)
    x = (pixels[:, 1])
    y = (pixels[:, 0])

    # Selecting set of points on which the spline will be spanned
    x, y = zip(*sorted(zip(x, y)))

    x_select = x[0:len(x):1]
    y_select = y[0:len(y):1]
    y_select = np.asarray(y_select)
    x_select = np.asarray(x_select)

    # Drawing set of selected points
    checkImg = np.zeros_like(img)
    for i in range(0, len(x_select)):
        checkImg = cv2.circle(checkImg, (x_select[i], y_select[i]), radius=1, color=255, thickness=-1)

    checkImg[GROUND_HEIGHT, :] = 125

    # cutoff points below y = 830 for creating approximation polynomials
    y_cutoff_ids = np.where(y_select > 820)
    x_split = x_select[np.argmin(y_select)]
    y_select = y_select[y_cutoff_ids]
    x_select = x_select[y_cutoff_ids]

    # split points to left and right groups in order to create separate polynomials
    idx_left = np.where(x_select < x_split)
    y_left = y_select[idx_left]
    x_left = x_select[idx_left]
    idx_right = np.where(x_select > x_split)
    y_right = y_select[idx_right]
    x_right = x_select[idx_right]

    z_left = np.polyfit(y_left, x_left, 2)
    p_left = np.poly1d(z_left)
    z_right = np.polyfit(y_right, x_right, 2)
    p_right = np.poly1d(z_right)

    # calculate derivatives and use them to get the angle
    dp_left = np.polyder(p_left, 1)
    angle_left = np.pi/2 + np.arctan(dp_left(max(y_left) - 1))

    dp_right = np.polyder(p_right, 1)
    angle_right = np.pi - (np.pi/2 + np.arctan(dp_right(max(y_right) - 1)))

    # draw contour
    splineImg = np.zeros_like(img)
    splineImg[GROUND_HEIGHT, :] = 125
    cv2.drawContours(splineImg, contours_object, -1, 125, 1)

    # draw tangent lines
    line_len = 100
    line_thickness = 1

    p1 = np.array([x_left[np.argmax(y_left)], max(y_left)])
    p2 = p1 + np.array([line_len * np.cos(angle_left), -line_len * np.sin(angle_left)])
    cv2.line(splineImg, (p1[0], p1[1]), (round(p2[0]), round(p2[1])), 	255, thickness=line_thickness)

    p1 = np.array([x_right[np.argmax(y_right)], max(y_right)])
    p2 = p1 + np.array([-line_len * np.cos(angle_right), -line_len * np.sin(angle_right)])
    cv2.line(splineImg, (p1[0], p1[1]), (round(p2[0]), round(p2[1])), 	255, thickness=line_thickness)

    # printText(splineImg, [math.degrees(angle_left),math.degrees(angle_right)], [20, GROUND_HEIGHT - 320], 0.5)

    path_to_save = 'D:/Magisterka/pomiary/conv/' + path[-23:]
    cv2.imwrite(str(path_to_save), splineImg[GROUND_HEIGHT - 350:GROUND_HEIGHT + 70, x_ref:x_ref + 600])

    return [math.degrees(angle_left), math.degrees(angle_right),
            x_right[np.argmax(y_right)] - x_left[np.argmax(y_left)]]


def make_gif(frame_folder, case_name):
    frames = [Image.open(image) for image in glob.glob(f"{frame_folder}/*.png")]
    frame_one = frames[0]
    path_to_save_gif = case_name + '_anim.gif'
    frame_one.save(str(path_to_save_gif), format="GIF", append_images=frames,
                   save_all=True, duration=100, loop=0)


def main():
    FRAME_RATE = 5400
    frame = []
    paths = []

    pomiary_path = 'D:/Magisterka/pomiary'

    exp_paths = os.listdir(pomiary_path)
    exp_paths.pop()

    for i in exp_paths:

        case_name = i
        print('analysing case: ' + case_name + '...\n')

        shutil.rmtree(pomiary_path + '/conv')
        os.mkdir(pomiary_path + '/conv')

        GLOB_PATH = pomiary_path + '/' + case_name
        for path in Path(GLOB_PATH).glob("*.png"):
            frame.append(int(str(path)[-9] + str(path)[-8] + str(path)[-7] + str(path)[-6] + str(path)[-5]))
            paths.append(str(path))

        angleLeft = []
        angleRight = []
        contactLength = []

        print("\nConverting images:")
        for i in tqdm(range(len(frame)), bar_format='{l_bar}{bar:40}{r_bar}{bar:-10b}'):
            if frame[i] % 2 == 0 and frame[i] < 2002:
                angleLeft.append(interpolate_spline(str(paths[i]))[0])
                angleRight.append(interpolate_spline(str(paths[i]))[1])
                contactLength.append(interpolate_spline(str(paths[i]))[2])

        time = [frame_/FRAME_RATE for frame_ in frame]

        # Csv writer
        import pandas as pd
        df = pd.DataFrame(list(zip(*[time[1:len(frame):2], angleLeft, angleRight, contactLength]))).add_prefix('Col')
        df.to_csv(str(GLOB_PATH[-9:] + '.csv'), index=False)

        angle_left = np.array(angleLeft)
        angle_right = np.array(angleRight)
        yf = fft(angle_left) * 0.02137
        xf = fftfreq(len(angle_left),time[0])
        ids = np.argwhere(yf > 5)
        x = np.array(xf[ids])
        y = np.array(yf[ids].real)
        plt.plot(x, y, 'o')
        plt.grid()
        plt.xlabel('frequency [Hz]')
        plt.ylabel('Amplitude [mm]')
        plt.title('FFT analysis for case' + case_name)
        plt.savefig(case_name + '_fft_plot.png')
        plt.clf()

        make_gif(pomiary_path + '/conv', case_name)

if __name__ == "__main__":
    main()
   # make_gif("D:/Magisterka/pomiary/conv")


#TODO: view calibration