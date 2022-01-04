import os
import time

import cv2
import logging
import imutils

logging.basicConfig()
logging.getLogger().setLevel(logging.DEBUG)
log = logging.getLogger('Face Detector')

log.info("Loading haarcascade")
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
face_cascade.load('./blobs/haarcascade_frontalface_default.xml')

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


log.info("Face detector started")
while True:
    # 调用摄像头，获取图像
    ret, frame = camera.read()
    frame = imutils.resize(frame, height=640, width=480)
    if ret:
        # 转换为灰度图
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 调用人脸检测器检测
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        # 绘制检测出的所有人脸
        for (x, y, w, h) in faces:
            # 画方框
            img = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            # 获取人脸，并转换为200,200的统一格式
            # 原始数据返回的是正方形
            f = cv2.resize(gray[y:y + h, x:x + w], (200, 200))
            cv2.imshow('face', f)
            fE = cv2.equalizeHist(f)
            cv2.imshow('faceE', fE)

            face_area = img[y:y + h, x:x + w]

            # 展示图片
            cv2.imshow('camera0', frame)

    cv2.waitKey(100)
