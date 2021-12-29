import os
import time

import cv2
import logging

camera = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
face_cascade.load('./blobs/haarcascade_frontalface_default.xml')
count = 0

while count < 100:
    # 调用摄像头，获取图像
    ret, frame = camera.read()
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

            # 保存图片
            cv2.imwrite('./data/faces/0/%s.pgm' % str(count), f)
            count += 1

            # 展示图片
            cv2.imshow('camera', frame)

    cv2.waitKey(1)

