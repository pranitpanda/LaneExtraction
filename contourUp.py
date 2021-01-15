#!/usr/bin/env python3
"""
Scripts to drive a donkey 2 car

Usage:
    manage.py (drive)


Options:
    -h --help          Show this screen.    
"""
import os
import time
# from StopFinder import *
# import findStopSign
from docopt import docopt
import numpy as np
import cv2
import imutils
import numpy as np
import argparse
import cv2
import time

import donkeycar as dk
from donkeycar.parts.datastore import TubHandler
from donkeycar.parts.camera import PiCamera

import StopDetect
class MyCVController:
    '''
    CV based controller
    '''



    def run(self, cam_img):

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
            rise = line[3] - line[1]
            run = line[2] - line[0]
            if run != 0:
                slope = rise / run
            else:
                slope = None
            return slope

        # 160*120
        def steering(cX):
            newValue = ((cX - 80) / 80)
            return newValue

        def signs_region_of_interest(image):
            polygons = np.array([[(120, 0), (160, 0), (160, 80), (120, 80)]])
            mask = np.zeros_like(image)
            cv2.fillPoly(mask, polygons, 255)
            masked_image = cv2.bitwise_and(image, mask)
            return masked_image

        def stop_sign(img):
            # cropped_img_stop = signs_region_of_interest(img)
            cropped_img_stop = img
            # lower_red = np.array([0, 50, 50])
            # upper_red = np.array([10, 255, 255])

            lower_red1 = np.array([0, 100, 43])
            upper_red1 = np.array([1, 79, 100])

            # hsv = cv2.cvtColor(cropped_img_stop, cv2.COLOR_RGB2HSV)
            # mask = cv2.inRange(hsv, lower_red, upper_red)
            # lower_red1 = np.array([170, 50, 50])
            # upper_red1 = np.array([180, 255, 255])
            hsv1 = cv2.cvtColor(cropped_img_stop, cv2.COLOR_RGB2HSV)
            mask1 = cv2.inRange(hsv1, lower_red1, upper_red1)
            final_mask = cv2.bitwise_and(cropped_img_stop, cropped_img_stop, mask=mask1)
            gray_mask = cv2.cvtColor(final_mask, cv2.COLOR_BGR2GRAY)
            blur_mask = cv2.GaussianBlur(gray_mask, (5, 5), 0)
            ret, thresh = cv2.threshold(blur_mask, 70, 255, 0)
            thresh2, contours, hierarchy = cv2.findContours(thresh, 1, cv2.CHAIN_APPROX_SIMPLE)
            contours.sort(key=cv2.contourArea, reverse=True)

            if len(contours) > 1:
                if cv2.contourArea(contours[1]) > 1.00: #0------------
                    print("Sign found-------------------")
                    return True

                else:
                    return False
            else:
                # print('not enough contours')
                return False

        #do image processing here. output variables steering and throttle to control vehicle.
        if cam_img is None:
            return 0, 0.2, False
        cropped_img = region_of_interest(cam_img)
        lower_orange = np.array([0, 50, 50])
        upper_orange = np.array([10, 255, 255])
        hsv = cv2.cvtColor(cropped_img, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, lower_orange, upper_orange)

        lower_orange1 = np.array([170, 50, 50])
        upper_orange1 = np.array([180, 255, 255])
        hsv1 = cv2.cvtColor(cropped_img, cv2.COLOR_RGB2HSV)
        mask1 = cv2.inRange(hsv1, lower_orange1, upper_orange1)

        final_mask = cv2.bitwise_and(cropped_img, cropped_img, mask=mask1 + mask)

        gray_mask = cv2.cvtColor(final_mask, cv2.COLOR_BGR2GRAY)
        blur_mask = cv2.GaussianBlur(gray_mask, (5, 5), 0)
        ret, thresh = cv2.threshold(blur_mask, 70, 255, 0)
        
        thresh2, contours, hierarchy = cv2.findContours(thresh, 1, cv2.CHAIN_APPROX_SIMPLE)
        contours.sort(key=cv2.contourArea, reverse=True)

        if len(contours) > 1:
            M = cv2.moments(contours[1])
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            else:
                # print('my boi is 0')
                cX, cY = 80, 0
        else:
            # print('not enough contours')
            cX, cY = 80, 0

    # do image processing here. output variables steering and throttle to control vehicle.
    #     print (cX)
        steering = steering(cX)  # from zero to one
        throttle = 0.2 # from -1 to 1
        # if stop_sign(cam_img):
        #         #     sleep(3);
        #         # recording = False # Set to true if desired to save camera frames
        s = StopFinder()
        if s.findStopSign(cam_img):
            sleep(3)

        return steering, throttle, recording



def drive(cfg):
    '''
    Construct a working robotic vehicle from many parts.
    Each part runs as a job in the Vehicle loop, calling either
    it's run or run_threaded method depending on the constructor flag `threaded`.
    All parts are updated one after another at the framerate given in
    cfg.DRIVE_LOOP_HZ assuming each part finishes processing in a timely manner.
    Parts may have named outputs and inputs. The framework handles passing named outputs
    to parts requesting the same named input.
    '''
    
    #Initialize car
    V = dk.vehicle.Vehicle()

    #Camera
    cam = PiCamera(image_w=cfg.IMAGE_W, image_h=cfg.IMAGE_H, image_d=cfg.IMAGE_DEPTH)
    V.add(cam, outputs=['cam/image_array'], threaded=True)
        
    #Controller
    V.add(MyCVController(), 
          inputs=['cam/image_array'],
          outputs=['steering', 'throttle', 'recording'])

       
    #Sombrero
    if cfg.HAVE_SOMBRERO:
        from donkeycar.parts.sombrero import Sombrero
        s = Sombrero()

        
    #Drive train setup

    from donkeycar.parts.actuator import PCA9685, PWMSteering, PWMThrottle

    steering_controller = PCA9685(cfg.STEERING_CHANNEL, cfg.PCA9685_I2C_ADDR, busnum=cfg.PCA9685_I2C_BUSNUM)
    steering = PWMSteering(controller=steering_controller,
                                    left_pulse=cfg.STEERING_LEFT_PWM, 
                                    right_pulse=cfg.STEERING_RIGHT_PWM)
    
    throttle_controller = PCA9685(cfg.THROTTLE_CHANNEL, cfg.PCA9685_I2C_ADDR, busnum=cfg.PCA9685_I2C_BUSNUM)
    throttle = PWMThrottle(controller=throttle_controller,
                                    max_pulse=cfg.THROTTLE_FORWARD_PWM,
                                    zero_pulse=cfg.THROTTLE_STOPPED_PWM, 
                                    min_pulse=cfg.THROTTLE_REVERSE_PWM)

    V.add(steering, inputs=['steering'])
    V.add(throttle, inputs=['throttle'])
    
    #add tub to save data

    inputs=['cam/image_array',
            'steering', 'throttle']

    types=['image_array',
           'float', 'float']

    th = TubHandler(path=cfg.DATA_PATH)
    tub = th.new_tub_writer(inputs=inputs, types=types)
    V.add(tub, inputs=inputs, outputs=["tub/num_records"], run_condition='recording')

    #run the vehicle
    V.start(rate_hz=cfg.DRIVE_LOOP_HZ, 
            max_loop_count=cfg.MAX_LOOPS)


if __name__ == '__main__':
    args = docopt(__doc__)
    cfg = dk.load_config()
    
    if args['drive']:
        drive(cfg)




#
# Computes mean square error between two n-d matrices. Lower = more similar.
#
class StopFinder:
    def meanSquareError(img1, img2):
        assert img1.shape == img2.shape, "Images must be the same shape."
        error = np.sum((img1.astype("float") - img2.astype("float")) ** 2)
        error = error/float(img1.shape[0] * img1.shape[1] * img1.shape[2])
        return error

    def compareImages(img1, img2):
        return 1/meanSquareError(img1, img2)


    #
    # Computes pyramids of images (starts with the original and down samples).
    # Adapted from:
    # http://www.pyimagesearch.com/2015/03/16/image-pyramids-with-python-and-opencv/
    #
    def pyramid(image, scale = 1.5, minSize = 30, maxSize = 1000):
        yield image
        while True:
            w = int(image.shape[1] / scale)
            image = imutils.resize(image, width = w)
            if(image.shape[0] < minSize or image.shape[1] < minSize):
                break
            if (image.shape[0] > maxSize or image.shape[1] > maxSize):
                continue
            yield image

    #
    # "Slides" a window over the image. See for this url for cool animation:
    # http://www.pyimagesearch.com/2015/03/23/sliding-windows-for-object-detection-with-python-and-opencv/
    #
    def sliding_window(image, stepSize, windowSize):
        for y in xrange(0, image.shape[0], stepSize):
            for x in xrange(0, image.shape[1], stepSize):
                yield (x, y, image[y:y+windowSize[1], x:x+windowSize[1]])


    def findStopSign(img):
        ap = argparse.ArgumentParser()
        ap.add_argument("-i", "--image", required=True, help="Path to the target image")
        ap.add_argument("-p", "--prototype", required=True, help="Path to the prototype object")
        args = vars(ap.parse_args())

        targetImage = cv2.imread(img)
        #targetImage = cv2.GaussianBlur(targetImage, (15, 15), 0)

        targetImage = imutils.resize(targetImage, width=500)
        prototypeImg = cv2.imread('stopPrototype.png')

        maxSim = -1
        maxBox = (0,0,0,0)

        t0 = time.time()

        for p in pyramid(prototypeImg, minSize = 50, maxSize = targetImage.shape[0]):
            for (x, y, window) in sliding_window(targetImage, stepSize = 2, windowSize = p.shape):
                if window.shape[0] != p.shape[0] or window.shape[1] != p.shape[1]:
                    continue

                tempSim = compareImages(p, window)
                if(tempSim > maxSim):
                    maxSim = tempSim
                    maxBox = (x, y, p.shape[0], p.shape[1])

        t1 = time.time()

        print("Execution time: " + str(t1 - t0))
        print(maxSim)
        print(maxBox)
        buff1 = 10
        (x, y, w, h) = maxBox
        if w > 75:
            return True
        else:
            return False

            # cv2.rectangle(targetImage,(x-buff1/2,y-buff1/2),(x+w+buff1/2,y+h+buff1/2),(0,255,0),2)
            #
            #
            # cv2.imshow('image', targetImage)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

    
