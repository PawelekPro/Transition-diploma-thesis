import cv2
from pathlib import Path
from PIL import Image
import glob
import math
from tqdm import tqdm
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt


# TODO: aproksymacja konturu za pomocą splajna
#

# TODO: Contour detection - current solution doesn't work for angles > 90 deg
def tangentLine(image, coord_x, coord_y, tangent_line_step, line_length, side):
    directional_coefficient = 0
    if side == 'left':
        output = image[:, coord_x + tangent_line_step]
        output_test = image[coord_y, :]
        for i in range(coord_y):
            if output[i] == 255:
                directional_coefficient = (coord_y - i) / tangent_line_step
                coordX_const_line_length = coord_x + int(line_length / math.sqrt(directional_coefficient + 1))
                coordY_const_line_length = int(
                    -directional_coefficient * coordX_const_line_length + coord_y + directional_coefficient * coord_x)
                real_len = math.sqrt(
                    (coord_x - coordX_const_line_length) ** 2 + (coord_y - coordY_const_line_length) ** 2)
                if real_len < 10 * line_length:
                    cv2.line(image, (coord_x, coord_y), (coordX_const_line_length, coordY_const_line_length), 255, 1)
                    break

    elif side == 'right':
        output = image[:, coord_x - tangent_line_step]
        for i in range(coord_y):
            if output[i] == 255:
                directional_coefficient = (coord_y - i) / tangent_line_step
                coordX_const_line_length = coord_x - int(line_length / math.sqrt(directional_coefficient + 1))
                coordY_const_line_length = int(
                    directional_coefficient * coordX_const_line_length + coord_y - directional_coefficient * coord_x)
                real_len = math.sqrt(
                    (coord_x - coordX_const_line_length) ** 2 + (coord_y - coordY_const_line_length) ** 2)
                if real_len < 10 * line_length:
                    cv2.line(image, (coord_x, coord_y), (coordX_const_line_length, coordY_const_line_length), 255, 1)
                    break
    return image, math.atan(directional_coefficient)


def printAngle(image, value, org, fontScale):
    font = cv2.FONT_HERSHEY_SIMPLEX

    color = 255
    # Line thickness of 1 px
    thickness = 1
    if value[0] == 0:
        image = cv2.putText(image, "Left angle: None", (org[0], org[1]), font,
                            fontScale, color, thickness, cv2.LINE_AA)
    else:
        image = cv2.putText(image, "Left angle [deg]: %.2f" % value[0], (org[0], org[1]), font,
                        fontScale, color, thickness, cv2.LINE_AA)
    if value[1] == 0:
        image = cv2.putText(image, "Right angle: None", (org[0], org[1] + 16), font,
                            fontScale, color, thickness, cv2.LINE_AA)
    else:
        image = cv2.putText(image, "Right angle [deg]: %.2f" % value[1], (org[0], org[1] + 16), font,
                            fontScale, color, thickness, cv2.LINE_AA)
    return image


def signature(image, origin, font_scale):
    font = cv2.FONT_HERSHEY_SIMPLEX
    color = 125
    # Line thickness of 1 px
    thickness = 1

    image = cv2.putText(image, "Measurement date: 12.12.2022", (origin[0], origin[1]), font,
                        font_scale, color, thickness, cv2.LINE_AA)
    image = cv2.putText(image, "Frame rate: 2000 fps", (origin[0], origin[1] + 15), font,
                        font_scale, color, thickness, cv2.LINE_AA)
    return image


def printText(image, value, origin, font_scale):
    font = cv2.FONT_HERSHEY_SIMPLEX

    color = 255
    # Line thickness of 1 px
    thickness = 1

    image = cv2.putText(image, "Coordinates of the object's center of mass: (%.2f, %.2f)" % (value[0], value[1]), (origin[0], origin[1]), font,
                        font_scale, color, thickness, cv2.LINE_AA)

    return image


def drawCSYS(image, org, thickness, scale, color, optional_text = ""):
    start_point = (org[0], org[1])
    x_end_point = (org[0], org[1] - int(scale * 50))
    y_end_point = (org[0] + int(scale * 50), org[1])
    image = cv2.arrowedLine(image, start_point, x_end_point, color, thickness, tipLength = 0.2)
    image = cv2.arrowedLine(image, start_point, y_end_point, color, thickness, tipLength=0.2)

    if len(optional_text) != 0:
        image = cv2.putText(image, "%s" % optional_text,
                            (org[0] - int(scale*25), org[1] + int(scale*15)), cv2.FONT_HERSHEY_SIMPLEX, scale * 0.4, 160, 1, cv2.LINE_AA)
    return image

def interpolate_with_spline(image):
    pixels = np.argwhere(image == 255)
    pixels = np.unique(pixels, axis=1)
    y = (pixels[:,1])
    x = (pixels[:,0])
    x, index = np.unique(x, return_index=True)
    y_filtr=[]
    for i in index:
        y_filtr.append(y[i])

    y_filtr = np.array(y_filtr)
    plt.figure()
    plt.plot(x, y_filtr, 'o')
    plt.show()
    f = interp1d(x, y_filtr, kind='cubic')
    xnew = np.linspace(min(x), max(x), 50)

    plt.figure()
    plt.plot(xnew, f(xnew))


def draw_contour(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    """BINARY TRESHOLDING"""
    _, thresh = cv2.threshold(img, 20, 255, cv2.THRESH_BINARY)

    """ADAPTIVE TRESHOLDING"""
    # thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 19, 24)
    thresh = 255 - thresh

    thresh_reference = np.zeros_like(img)
    thresh_object = np.zeros_like(img)
    output_reference = np.zeros_like(img)
    output_object = np.zeros_like(img)

    image_cutting_size = 300
    thresh_reference[0:image_cutting_size, :] = thresh[0:image_cutting_size, :]
    thresh_object[image_cutting_size:len(thresh), :] = thresh[image_cutting_size:len(thresh), :]

    # TODO: detecting ground position
    GROUND_HEIGHT = 860

    # cutting out everything below the plate
    thresh_object[GROUND_HEIGHT + 1:len(output_object), :] = 0

    contours_reference, _ = cv2.findContours(thresh_reference, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours_object, _ = cv2.findContours(thresh_object, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if len(contours_reference) != 0:
        cv2.drawContours(output_object, contours_object, -1, 255, 1)
        cv2.drawContours(output_reference, contours_reference, -1, 255, 1)

        # find the biggest countour (c) by the area
        c_reference = max(contours_reference, key=cv2.contourArea)
        if len(contours_object) != 0:
            c_object = max(contours_object, key=cv2.contourArea)
            x_obj, y_obj, w_obj, h_obj = cv2.boundingRect(c_object)
            output_object[1: y_obj - 10, :] = 0

        x_ref, y_ref, w_ref, h_ref = cv2.boundingRect(c_reference)
        # cv2.rectangle(output_reference, (x_ref, y_ref), (x_ref + w_ref, y_ref + h_ref), 180, 1)
        cv2.rectangle(output_object, (x_obj, y_obj), (x_obj + w_obj, y_obj + h_obj), 125, 1)

    # [output_object, alpha_1] = tangentLine(output_object, x_obj, GROUND_HEIGHT, 2, 80, 'left')
    # [output_object, alpha_2] = tangentLine(output_object, x_obj + w_obj , GROUND_HEIGHT, 4, 80, 'right')
    # printAngle(output_object, [math.degrees(alpha_1), math.degrees(alpha_2)], [x_ref - 70, GROUND_HEIGHT - 300], 0.4)

    # calculate moments of binary image
    M = cv2.moments(thresh_object)

    # calculate x,y coordinate of center
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    output_object[GROUND_HEIGHT, :] = 125
    signature(output_object, [x_ref - 85, GROUND_HEIGHT - 335], 0.4)
    drawCSYS(output_object, [x_ref + 100, GROUND_HEIGHT], 2, 1, 160, "(x0,y0)")
    drawCSYS(output_object, [x_obj, y_obj], 1, 0.6, 140, "(x1,y1)")
    [oX, oY] = [cX - x_ref - 150, GROUND_HEIGHT - cY]
    # printText(output_object, [oX, oY], [x_ref - 100, GROUND_HEIGHT - 305], 0.4)

    path_to_save = 'D:\praca_magisterska\conv\\' + path[-23:]
    cv2.imwrite(str(path_to_save), output_object[GROUND_HEIGHT - 350:GROUND_HEIGHT + 70, x_ref - 70:x_ref + 350])
    # interpolate_with_spline(output_object)


def make_gif(frame_folder):
    frames = [Image.open(image) for image in glob.glob(f"{frame_folder}/*.PNG")]
    frame_one = frames[0]
    path_to_save_gif = "anim_001.gif"
    frame_one.save(str(path_to_save_gif), format="GIF", append_images=frames,
                   save_all=True, duration=100, loop=0)


def main():
    frame_rate = 2000
    frame = []
    paths = []

    GLOB_PATH = "D/praca_magisterska/python_project/images_1/a100_f200_C001H001S0001"
    for path in Path(GLOB_PATH).glob("*"):
        frame.append(int(str(path)[-9] + str(path)[-8] + str(path)[-7] + str(path)[-6] + str(path)[-5]))
        paths.append(str(path))

    print("\nConverting images:")
    for i in tqdm(range(len(frame)), bar_format='{l_bar}{bar:40}{r_bar}{bar:-10b}'):
        if frame[i] % 5 == 0:
            draw_contour(str(paths[i]))


if __name__ == "__main__":
    main()
    # make_gif("D:\praca_magisterska\conv")
