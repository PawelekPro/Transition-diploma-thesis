import cv2
from pathlib import Path
from PIL import Image
import glob
import math
from tqdm import tqdm
import numpy as np
import shutil
import os


def draw_ellipse(
        img, center, axes, angle,
        startAngle, endAngle, color,
        thickness=1, lineType=cv2.LINE_AA, shift=10):
    # uses the shift to accurately get sub-pixel resolution for arc
    center = (
        int(round(center[0] * 2 ** shift)),
        int(round(center[1] * 2 ** shift))
    )
    axes = (
        int(round(axes[0] * 2 ** shift)),
        int(round(axes[1] * 2 ** shift))
    )
    return cv2.ellipse(
        img, center, axes, angle,
        startAngle, endAngle, color,
        thickness, lineType, shift)

def printText(image, value, org, fontScale):
    font = cv2.FONT_HERSHEY_SIMPLEX

    color = 0
    # Line thickness of 1 px
    thickness = 1
    image = cv2.putText(image, "Measured data:", (org[0], org[1]), font,
                        fontScale, color, thickness, cv2.LINE_AA)

    image = cv2.putText(image, "Left angle [deg]: %.2f" % value[0], (org[0], org[1]+20), font,
                            fontScale, color, thickness, cv2.LINE_AA)

    image = cv2.putText(image, "Right angle [deg]: %.2f" % value[1], (org[0], org[1]+42), font,
                            fontScale, color, thickness, cv2.LINE_AA)
    return image


def get_sub(x):
    normal = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+-=()"
    sub_s = "ₐ₈CDₑբGₕᵢⱼₖₗₘₙₒₚQᵣₛₜᵤᵥwₓᵧZₐ♭꜀ᑯₑբ₉ₕᵢⱼₖₗₘₙₒₚ૧ᵣₛₜᵤᵥwₓᵧ₂₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎"
    res = x.maketrans(''.join(normal), ''.join(sub_s))
    return x.translate(res)


def signature(image, origin, font_scale):
    font = cv2.FONT_HERSHEY_SIMPLEX
    color = 0
    # Line thickness of 1 px
    thickness = 1

    image = cv2.putText(image, "Measurement date: 09.01.2023", (origin[0], origin[1]), font,
                        font_scale, color, thickness, cv2.LINE_AA)
    image = cv2.putText(image, "Frame rate: 5400 fps", (origin[0], origin[1] + 20), font,
                        font_scale, color, thickness, cv2.LINE_AA)
    return image


def drawCSYS(image, org, thickness, scale, color, optional_text=""):
    start_point = (org[0], org[1])
    x_end_point = (org[0], org[1] - int(scale * 50))
    y_end_point = (org[0] + int(scale * 50), org[1])
    image = cv2.arrowedLine(image, start_point, x_end_point, color, thickness, tipLength=0.2)
    image = cv2.arrowedLine(image, start_point, y_end_point, color, thickness, tipLength=0.2)

    if len(optional_text) != 0:
        image = cv2.putText(image, "%s" % optional_text,
                            (org[0] - int(scale*25), org[1] + int(scale*15)), cv2.FONT_HERSHEY_SIMPLEX, scale * 0.4, 0, 1, cv2.LINE_AA)
    return image


def interpolate_spline(path, glob_path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    """BINARY TRESHOLDING"""
    _, thresh = cv2.threshold(img, 30, 255, cv2.THRESH_BINARY)
    thresh = 255 - thresh

    thresh_reference = np.zeros_like(img)
    thresh_object = np.zeros_like(img)
    output_reference = np.zeros_like(img)
    output_object = np.zeros_like(img)

    image_cutting_size = 300
    thresh_reference[0:image_cutting_size, :] = thresh[0:image_cutting_size, :]
    thresh_object[image_cutting_size:len(thresh), :] = thresh[image_cutting_size:len(thresh), :]

    GROUND_HEIGHT = 865

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

    # Obrobka zdjec do raportu
    splineImg = np.zeros_like(img)
    # Drawing set of selected points
    checkImg = np.zeros_like(img)
    for i in range(0, len(x_select)):
        splineImg = cv2.circle(splineImg, (x_select[i], y_select[i]), radius=1, color=255, thickness=-1)

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

    # plot polynomials
    # plt.figure(figsize=(16, 7))
    # x_approx_left = np.linspace(min(y_left), max(y_left), 100)
    # x_approx_right = np.linspace(min(y_right), max(y_right), 100)
    # plt.plot(x, y, marker='.', color='dodgerblue', label='Droplet shape (discrete)', markersize='8')
    # plt.plot(p_left(x_approx_left), x_approx_left, label='Left polynomial', color='red', linewidth=2)
    # plt.plot(p_right(x_approx_right), x_approx_right, label='Right polynomial', color='black', linewidth=2)
    # plt.ylim(max(y), 720)
    # plt.xlim(min(x) - 20, max(x) + 20)
    # plt.plot([min(x) - 20, max(x)],[820, 820], linewidth=1, color='black')
    # plt.legend(loc="upper right")
    # plt.xlabel("X coordinate [pixels]", size=14)
    # plt.ylabel("Y coordinate [pixels]", size=14)
    # plt.grid()
    # plt.savefig( str('approx/' 'approx' + str(path)[-9] + str(path)[-8] + str(path)[-7] + str(path)[-6] + str(path)[-5] +'.png'), dpi=100)
    # plt.close()

    # calculate derivatives and use them to get the angle
    dp_left = np.polyder(p_left, 1)
    angle_left = np.pi / 2 + np.arctan(dp_left(max(y_left) - 1))

    dp_right = np.polyder(p_right, 1)
    angle_right = np.pi - (np.pi / 2 + np.arctan(dp_right(max(y_right) - 1)))

    # draw contour

    splineImg[GROUND_HEIGHT, :] = 125
    cv2.drawContours(splineImg, contours_object, -1, 255, 1)

    # draw tangent lines
    line_len = 150
    line_thickness = 2

    img = cv2.imread(path, cv2.IMREAD_COLOR)
    img[GROUND_HEIGHT, :] = 125
    p1 = np.array([x_left[np.argmax(y_left)], max(y_left)])
    p2 = p1 + np.array([line_len * np.cos(angle_left), -line_len * np.sin(angle_left)])
    cv2.line(img, (p1[0], p1[1]), (round(p2[0]), round(p2[1])), (0,0,255), thickness=line_thickness)

    p1 = np.array([x_right[np.argmax(y_right)], max(y_right)])
    p2 = p1 + np.array([-line_len * np.cos(angle_right), -line_len * np.sin(angle_right)])
    cv2.line(img, (p1[0], p1[1]), (round(p2[0]), round(p2[1])), (0,0,255), thickness=line_thickness)

    draw_ellipse(img, (x_obj, GROUND_HEIGHT), (55, 55), 0, -math.degrees(angle_left), 0, (0,0,255))
    draw_ellipse(img, (x_obj+w_obj, GROUND_HEIGHT), (55, 55), -180, math.degrees(angle_right), 0, (0,0,255))
    img = cv2.putText(img, 'L', (x_obj+25, GROUND_HEIGHT-8), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255,255,255), 1, cv2.LINE_AA)
    img = cv2.putText(img, 'P', (x_obj+w_obj-38, GROUND_HEIGHT-8), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255,255,255), 1, cv2.LINE_AA)

    # printText(splineImg, [math.degrees(angle_left),math.degrees(angle_right)], [20, GROUND_HEIGHT - 320], 0.5)
    # test_img = cv2.bitwise_not(splineImg)
    # cv2.imshow('test_im', test_img)
    # cv2.waitKey(0)

    # Obrobka zdjec do raportu
    # splineImg = cv2.bitwise_not(splineImg)
    # signature(splineImg, [x_ref - 40, GROUND_HEIGHT - 330], 0.5)
    # printText(splineImg, [math.degrees(angle_left), math.degrees(angle_right)], [x_ref + 340, GROUND_HEIGHT - 330], 0.5)
    # drawCSYS(splineImg, [x_ref + 235, GROUND_HEIGHT], 2, 1, 40, "(x0,y0)")
    # splineImg = cv2.rectangle(splineImg, [x_ref-45,GROUND_HEIGHT+65], [x_ref + 545,GROUND_HEIGHT-345], 0, 1)
    # draw_ellipse(splineImg, (x_obj, GROUND_HEIGHT), (55, 55), 0, -math.degrees(angle_left), 0, 50)
    # draw_ellipse(splineImg, (x_obj+w_obj, GROUND_HEIGHT), (55, 55), -180, math.degrees(angle_right), 0, 50)
    # splineImg = cv2.putText(splineImg, 'L', (x_obj+25, GROUND_HEIGHT-8), cv2.FONT_HERSHEY_SIMPLEX,
    #                     0.5, 0, 1, cv2.LINE_AA)
    # splineImg = cv2.putText(splineImg, 'P', (x_obj+w_obj-38, GROUND_HEIGHT-8), cv2.FONT_HERSHEY_SIMPLEX,
    #                     0.5, 0, 1, cv2.LINE_AA)
    #
    # GLOB_PATH = str(glob_path)
    # PERIOD = 50e-3  # [s]
    # BASE_FREQ = 1 / PERIOD  # [Hz]
    # AMPLITUDE = 10  # [mm]
    # ratio = GLOB_PATH[-4:-1]
    # frequency = BASE_FREQ * (int(ratio) / 100)
    # amplitude = (int(GLOB_PATH[-8:-6]) / 100) * AMPLITUDE
    # cv2.putText(splineImg, str('Frequency: ' + str("{:.2f}".format(frequency)) + ' [Hz]'), [x_ref - 40, GROUND_HEIGHT - 290], cv2.FONT_HERSHEY_SIMPLEX,
    #             0.5, 0, 1, cv2.LINE_AA)
    # cv2.putText(splineImg, str('Amplitude: ' + str(amplitude) + ' [mm]'),
    #             [x_ref - 40, GROUND_HEIGHT - 270], cv2.FONT_HERSHEY_SIMPLEX,
    #             0.5, 0, 1, cv2.LINE_AA)
    # cv2.putText(splineImg, 'Frame number: %d' % int(str(path)[-9] + str(path)[-8] + str(path)[-7] + str(path)[-6] + str(path)[-5]),
    #             [x_ref - 40, GROUND_HEIGHT - 250], cv2.FONT_HERSHEY_SIMPLEX,
    #             0.5, 0, 1, cv2.LINE_AA)
    #
    # cv2.line(splineImg, (x_obj, GROUND_HEIGHT), (x_obj, GROUND_HEIGHT + 43) , 0, thickness=1)
    # cv2.line(splineImg, (x_obj+w_obj-1, GROUND_HEIGHT), (x_obj+w_obj-1, GROUND_HEIGHT + 43) , 0, thickness=1)
    # cv2.arrowedLine(splineImg, (x_obj, GROUND_HEIGHT + 40), (x_obj+w_obj-1, GROUND_HEIGHT+40), 0, 1,tipLength=0.03)
    # cv2.arrowedLine(splineImg, (x_obj+w_obj-1, GROUND_HEIGHT+40), (x_obj, GROUND_HEIGHT + 40), 0, 1, tipLength=0.03)
    # cv2.putText(splineImg, 'Contact zone length [px]: ' + str(x_right[np.argmax(y_right)] - x_left[np.argmax(y_left)]),
    #             [x_obj + 100, GROUND_HEIGHT+35], cv2.FONT_HERSHEY_SIMPLEX,
    #             0.4, 0, 1, cv2.LINE_AA)

    path_to_save = 'D:/praca_magisterska/conv/' + path[-23:]
    cv2.imwrite(str(path_to_save), img[GROUND_HEIGHT - 350:GROUND_HEIGHT + 70, x_ref - 50:x_ref + 550])

    return [math.degrees(angle_left), math.degrees(angle_right),
            x_right[np.argmax(y_right)] - x_left[np.argmax(y_left)]]


def make_gif(frame_folder):
    frames = [Image.open(image) for image in glob.glob(f"{frame_folder}/*.png")]
    frame_one = frames[0]
    path_to_save_gif = "anim_001.gif"
    frame_one.save(str(path_to_save_gif), format="GIF", append_images=frames,
                   save_all=True, duration=100, loop=0)


def main():
    FRAME_RATE = 5400
    frame = []
    paths = []

    shutil.rmtree('D:/praca_magisterska/conv')
    os.mkdir('D:/praca_magisterska/conv')

    GLOB_PATH = "D:/praca_magisterska/a10_f139z"
    for path in Path(GLOB_PATH).glob("*.png"):
        frame.append(int(str(path)[-9] + str(path)[-8] + str(path)[-7] + str(path)[-6] + str(path)[-5]))
        paths.append(str(path))

    angleLeft = []
    angleRight = []
    contactLength = []

    print("\nConverting images:")
    for i in tqdm(range(len(frame)), bar_format='{l_bar}{bar:40}{r_bar}{bar:-10b}'):
        if frame[i] % 2 == 0 and frame[i] < 2002:
            angleLeft.append(interpolate_spline(str(paths[i]), GLOB_PATH)[0])
            angleRight.append(interpolate_spline(str(paths[i]), GLOB_PATH)[1])
            contactLength.append(interpolate_spline(str(paths[i]), GLOB_PATH)[2])

    time = [frame_ / FRAME_RATE for frame_ in frame]

    # Csv writer
    import pandas as pd
    df = pd.DataFrame(list(zip(*[time[1:len(frame):2], angleLeft, angleRight, contactLength]))).add_prefix('Col')
    df.to_csv(str(GLOB_PATH[-9:] + '.csv'), index=False)


if __name__ == "__main__":
    main()
    make_gif("D:/praca_magisterska/conv")

# TODO: view calibration
