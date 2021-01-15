import cv2
import numpy as np


def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    return cv2.Canny(blur, 100, 200)  # first number is lower threshold, second number is upper threshold


def region_of_interest(image):
    polygons = np.array([[(0, 69), (0, 50), (66, 17), (116, 20), (160, 48), (160, 69)]])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image


def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)

            cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 4)

    return line_image


def find_average_lines(image, lines):
    left_lines = []
    right_lines = []
    if lines is not None:

        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            parameters = np.polyfit((x1, x2), (y1, y2), 1)
            slope = parameters[0]
            intercept = parameters[1]
            if slope < 0:
                left_lines.append([slope, intercept])
            else:
                right_lines.append([slope, intercept])
    left_lines_average = np.average(left_lines, axis=0)
    right_lines_average = np.average(right_lines, axis=0)
    left_line = make_coordinates(image, left_lines_average)
    right_line = make_coordinates(image, right_lines_average)
    return np.array([left_line, right_line])


def make_coordinates(image, line_parameters):
    if (type(line_parameters) is not np.float64):

        slope, intercept = line_parameters
        y1 = 60
        y2 = 20
        if 0.01 > slope >= 0:
            slope = 0.01
        if 0 > slope > -0.01:
            slope = -0.01
        try:
            x1 = int((y1 - intercept) / slope)
            x2 = int((y2 - intercept) / slope)
        except:
            x1 = 0
            x2 = 0
    else:
        y1 = 0
        y2 = 0
        x1 = 0
        x2 = 0
    return np.array([x1, y1, x2, y2])


def calculate_slope(line):
    rise = line[3]-line[1]
    run = line[2]-line[0]
    if run != 0:
        slope = rise/run
    else:
        slope = None
    return slope


# 160*120
def steering(cX):
    newValue = ((cX-80)/80)
    return newValue

def signs_region_of_interest():
    polygons = np.array([[(120,0), (160,0), (160,80), (120,80)]])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

def stop_sign(img):
    cropped_img = signs_region_of_interest(img)
    lower_red = np.array([0, 50, 50])
    upper_red = np.array([10, 255, 255])
    hsv = cv2.cvtColor(cropped_img, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, lower_red, upper_red)
    lower_red1 = np.array([170, 50, 50])
    upper_red1 = np.array([180, 255, 255])
    hsv1 = cv2.cvtColor(cropped_img, cv2.COLOR_RGB2HSV)
    mask1 = cv2.inRange(hsv1, lower_red1, upper_red1)
    final_mask = cv2.bitwise_and(cropped_img, cropped_img, mask=mask1 + mask)
    gray_mask = cv2.cvtColor(final_mask, cv2.COLOR_BGR2GRAY)
    blur_mask = cv2.GaussianBlur(gray_mask, (5, 5), 0)
    ret, thresh = cv2.threshold(blur_mask, 70, 255, 0)
    thresh2, contours, hierarchy = cv2.findContours(thresh, 1, cv2.CHAIN_APPROX_SIMPLE)
    contours.sort(key=cv2.contourArea, reverse=True)

    if len(contours) > 1:
        if cv.contourArea(contours[1]) > 1:
            return True
        else:
            return False
    else:
        print('not enough contours')
        return False


