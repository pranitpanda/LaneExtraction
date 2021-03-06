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
from donkeycar.templates.newbruhHelen import *

class MyCVController:
    '''
    CV based controller
    '''

    def run(self, cam_img):

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
                print('my boi is 0')
                cX, cY = 80, 0
        else:
            print('not enough contours')
            cX, cY = 80, 0

    # do image processing here. output variables steering and throttle to control vehicle.
        print (cX)
        steering = steering(cX)  # from zero to one
        throttle = 0.3 # from -1 to 1
        if stop_sign(cam_img):
            throttle = 0
        recording = False # Set to true if desired to save camera frames

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
    
