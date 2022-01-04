# -*- coding: utf-8 -*-
# @Time : 20-6-9 下午3:06
# @Author : zhuying
# @Company : Minivision
# @File : test.py
# @Software : PyCharm

import os
import cv2
import numpy as np
import argparse
import warnings
import time
import imutils
import logging

from torch import imag

from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name
warnings.filterwarnings('ignore')


SAMPLE_IMAGE_PATH = "./images/sample/"


# 因为安卓端APK获取的视频流宽高比为3:4,为了与之一致，所以将宽高比限制为3:4
def check_image(image):
    height, width, channel = image.shape
    if width/height - 3/4 > 1e-8:
        print("Image is not appropriate!!!\nHeight/Width should be 4/3. got height %d width %d" % (height, width))
        return False
    else:
        return True

logging.basicConfig()
logging.getLogger().setLevel(logging.DEBUG)
log = logging.getLogger('Face Detector')

model_test = AntiSpoofPredict(0)
image_cropper = CropImage()
log.info("Opening camera")
camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FPS, 10)

log.info("Testing camera")
ret, frame = camera.read()
height, width, channels = frame.shape
log.info("Video shape is %d x %d , %d channels" % (width, height, channels))

isCamScaleFit = False
if width / height - 3/4 < 1e-6:
    isCamScaleFit = True
    log.info("Video shape fits 3:4")
else:
    isCamScaleFit = False
    log.info("Video shape NOT fits 3:4 cutting")

def captureCamera():
    _, image = camera.read()
    if isCamScaleFit:
        return image
    else:
        maxWidth = int(height * (3 / 4))
        beginWidth = int(width / 2 - maxWidth / 2)
        return image[0:height, beginWidth:(beginWidth + maxWidth)]        





if __name__ == "__main__":
    desc = "test"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument(
        "--device_id",
        type=int,
        default=0,
        help="which gpu id, [0/1/2/3]")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="./resources/anti_spoof_models",
        help="model_lib used to test")
    parser.add_argument(
        "--image_name",
        type=str,
        default="image_F1.jpg",
        help="image used to test")
    args = parser.parse_args()
    test(args.model_dir)
