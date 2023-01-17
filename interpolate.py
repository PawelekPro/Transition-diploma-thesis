import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import glob
import math
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import shutil
import os
from scipy.interpolate import interp1d, splev, splprep, splrep, BSpline


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

    # Detecting X coordinate for Y_ground
    y = np.asarray(y)
    x = np.asarray(x)
    X_GROUND = np.argwhere(y == GROUND_HEIGHT - 1)
    if len(X_GROUND) > 2:
        X_GROUND[0] = X_GROUND[0]
        X_GROUND[1] = X_GROUND[-1]
        X_GROUND = X_GROUND[0:2]

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

    # f =interp1d(y_left, x_left)


    z_left = np.polyfit(y_left, x_left, 2)
    p_left = np.poly1d(z_left)

    z_right = np.polyfit(y_right, x_right, 2)
    p_right = np.poly1d(z_right)

    # plot polynomials
    # x_approx_left = np.linspace(min(y_left), max(y_left), 100)
    # x_approx_right = np.linspace(min(y_right), max(y_right), 100)
    # plt.plot(x_left, y_left, 'o', label='Data points')
    # plt.plot(x_right, y_right, 'o', label='Data points')
    # plt.plot(p_left(x_approx_left), x_approx_left, label='approx')
    # plt.plot(p_right(x_approx_right), x_approx_right, label='approx')
    # plt.ylim(max(y_select), min(y_select) - 5)
    # ax = plt.gca()
    # ax.set_aspect('equal', adjustable='box')
    # plt.draw()

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
    cv2.line(splineImg, (p1[0], p1[1]),(round(p2[0]), round(p2[1])), 	255, thickness=line_thickness)

    p1 = np.array([x_right[np.argmax(y_right)], max(y_right)])
    p2 = p1 + np.array([-line_len * np.cos(angle_right), -line_len * np.sin(angle_right)])
    cv2.line(splineImg, (p1[0], p1[1]),(round(p2[0]), round(p2[1])), 	255, thickness=line_thickness)

    printText(splineImg, [math.degrees(angle_left),math.degrees(angle_right)], [20, GROUND_HEIGHT - 320], 0.5)

    path_to_save = 'D:/praca_magisterska/conv/' + path[-23:]
    cv2.imwrite(str(path_to_save), splineImg[GROUND_HEIGHT - 350:GROUND_HEIGHT + 70, :])


def make_gif(frame_folder):
    frames = [Image.open(image) for image in glob.glob(f"{frame_folder}/*.png")]
    frame_one = frames[0]
    path_to_save_gif = "anim_001.gif"
    frame_one.save(str(path_to_save_gif), format="GIF", append_images=frames,
                   save_all=True, duration=100, loop=0)


def main():
    frame_rate = 2000
    frame = []
    paths = []

    shutil.rmtree('D:/praca_magisterska/conv')
    os.mkdir('D:/praca_magisterska/conv')

    GLOB_PATH = "D:/praca_magisterska/a10_f135_z"
    for path in Path(GLOB_PATH).glob("*.png"):
        frame.append(int(str(path)[-9] + str(path)[-8] + str(path)[-7] + str(path)[-6] + str(path)[-5]))
        paths.append(str(path))

    print("\nConverting images:")
    for i in tqdm(range(len(frame)), bar_format='{l_bar}{bar:40}{r_bar}{bar:-10b}'):
        if frame[i] % 1 == 0 and frame[i]> 200 and frame[i] < 1000:
            interpolate_spline(str(paths[i]))


if __name__ == "__main__":
    main()
    make_gif("D:/praca_magisterska/conv")


