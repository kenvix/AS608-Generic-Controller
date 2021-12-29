import os
import time

import cv2
import logging
import numpy as np

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
face_cascade.load('./blobs/haarcascade_frontalface_default.xml')

images = []
labels = []

for i in range(0, 100):
    images.append(cv2.imread("./data/faces/0/%s.pgm" % str(i), cv2.IMREAD_GRAYSCALE))
    labels.append(1)


# 传入测试图片路径，识别器，原始人脸数据库，人脸数据库标签
def showConfidence(imgPath, recognizer, images, labels):
    # 训练识别器
    recognizer.train(images, np.array(labels))
    # 加载测试图片
    predict_image = cv2.imread(imgPath, cv2.IMREAD_GRAYSCALE)
    # 预测并打印结果
    labels, confidence = recognizer.predict(predict_image)
    print("label=", labels)
    print("conficence=", confidence)


imgPath = "./data/faces/out.pgm"

recognizer = cv2.face.EigenFaceRecognizer_create()
showConfidence(imgPath, recognizer, images, labels)
