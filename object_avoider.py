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

from docopt import docopt
import numpy as np

import donkeycar as dk
from donkeycar.parts.datastore import TubHandler
from donkeycar.parts.camera import PiCamera
from donkeycar.templates.newbruh import *
import time


class MyCVController:
    '''
    CV based controller
    '''

    pastSteer = 0
    pastSpeed = 0

    def run(self, cam_img):
        turningright = False
        # do image processing here. output variables steering and throttle to control vehicle.
        if cam_img is None:
            return self.pastSteer, self.pastSpeed, False

        image_copy = cam_img

        lower_orange = np.array([0, 50, 50])
        upper_orange = np.array([10, 255, 255])
        hsv = cv2.cvtColor(cam_img, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, lower_orange, upper_orange)

        lower_orange1 = np.array([170, 50, 50])
        upper_orange1 = np.array([180, 255, 255])
        hsv1 = cv2.cvtColor(cam_img, cv2.COLOR_RGB2HSV)
        mask1 = cv2.inRange(hsv1, lower_orange1, upper_orange1)

        final_mask = cv2.bitwise_and(cam_img, cam_img, mask=mask1 + mask)

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
                cX, cY = 80, 0
        else:
            cX, cY = 80, 0

        # do image processing here. output variables steering and throttle to control vehicle.
        steering = steeringboi(cX)  # from zero to one
        throttle = 0.2  # from -1 to 1
        recording = True  # Set to true if desired to save camera frames

        lower_blue = np.array([100, 70, 70])
        upper_blue = np.array([120, 255, 200])
        hsv2 = cv2.cvtColor(image_copy, cv2.COLOR_RGB2HSV)
        mask2 = cv2.inRange(hsv2, lower_blue, upper_blue)
        final_mask2 = cv2.bitwise_and(image_copy, image_copy, mask=mask2)
        gray_mask2 = cv2.cvtColor(final_mask2, cv2.COLOR_BGR2GRAY)
        blur_mask2 = cv2.GaussianBlur(gray_mask2, (5, 5), 0)
        cv2.imwrite('test.jpg',blur_mask2)
        ret, thresh = cv2.threshold(blur_mask2, 70, 255, 0)
        thresh2, contours, hierarchy = cv2.findContours(thresh, 1, cv2.CHAIN_APPROX_SIMPLE)
        contours.sort(key=cv2.contourArea, reverse=True)
        if len(contours) >= 1:
            M = cv2.moments(contours[0])
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                if 20 <= cX <= 80:
                    t0 = time.time()
                    print('object on the left: turning right ')
                    steering = (cX-20)/60
                elif 80 < cX <= 140:
                    t1 = time.time()
                    print('object on the right, turning left')
                    steering = (cX-140)/60
        try:
            elapsedtime = time.time() - t0
            print(elapsedtime)
            if elapsedtime < 1.0:
                steering = 0.5
                turningright = True
            else:
                turningright = False
        except NameError:
            elapsedtime = None

        if not turningright:
            try:
                elapsedtime1 = time.time() - t1
                if elapsedtime1 < 1.0:
                    steering = -0.5
            except NameError:
                elapsedtime1 = None

        self.pastSteer = steering
        self.pastSpeed = throttle


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

    # Initialize car
    V = dk.vehicle.Vehicle()

    # Camera
    cam = PiCamera(image_w=cfg.IMAGE_W, image_h=cfg.IMAGE_H, image_d=cfg.IMAGE_DEPTH)
    V.add(cam, outputs=['cam/image_array'], threaded=True)

    # Controller
    V.add(MyCVController(),
          inputs=['cam/image_array'],
          outputs=['steering', 'throttle', 'recording'])

    # Sombrero
    if cfg.HAVE_SOMBRERO:
        from donkeycar.parts.sombrero import Sombrero
        s = Sombrero()

    # Drive train setup

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

    # add tub to save data

    inputs = ['cam/image_array',
              'steering', 'throttle']

    types = ['image_array',
             'float', 'float']

    th = TubHandler(path=cfg.DATA_PATH)
    tub = th.new_tub_writer(inputs=inputs, types=types)
    V.add(tub, inputs=inputs, outputs=["tub/num_records"], run_condition='recording')

    # run the vehicle
    V.start(rate_hz=cfg.DRIVE_LOOP_HZ,
            max_loop_count=cfg.MAX_LOOPS)


if __name__ == '__main__':
    args = docopt(__doc__)
    cfg = dk.load_config()

    if args['drive']:
        drive(cfg)



