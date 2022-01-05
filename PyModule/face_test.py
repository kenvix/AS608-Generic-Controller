import os
import time

import cv2
import logging
import numpy as np
import imutils

from face_anti_spoofing.src.anti_spoof_predict import AntiSpoofPredict
from face_anti_spoofing.src.generate_patches import CropImage
from face_anti_spoofing.src.utility import parse_model_name

log = logging.getLogger('Face Detector')


class FaceDetector:
    def __init__(self):
        self.fontSize = 0.8
        self.image_cropper = None
        self.model_test = None
        self.imtitle = "Camera"
        self.imtitle2 = "Camera-Face"
        self.channels = None
        self.width = None
        self.height = None
        self.isCamScaleFit = None
        self.camera = None
        self.face_cascade = None
        self.model_dir = "./face_anti_spoofing/resources/anti_spoof_models"

    def load(self):
        log.info("Loading haarcascade")
        self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        self.face_cascade.load('./blobs/haarcascade_frontalface_default.xml')

        self.model_test = AntiSpoofPredict(0)
        self.image_cropper = CropImage()

        log.info("Opening camera")
        self.camera = cv2.VideoCapture(0)
        self.camera.set(cv2.CAP_PROP_FPS, 10)

        log.info("Testing camera")
        ret, frame = self.camera.read()
        self.height, self.width, self.channels = frame.shape
        log.info("Video shape is %d x %d , %d channels" % (self.width, self.height, self.channels))
        cv2.imshow(self.imtitle, frame)

        self.isCamScaleFit = False
        if self.width / self.height - 3 / 4 < 1e-6:
            self.isCamScaleFit = True
            log.info("Video shape fits 3:4")
        else:
            self.isCamScaleFit = False
            log.info("Video shape NOT fits 3:4 cropping")

    @staticmethod
    def check_image(image):
        height, width, channel = image.shape
        if width / height - 3 / 4 > 1e-8:
            print("Image is not appropriate!!!\nHeight/Width should be 4/3. got height %d width %d" % (height, width))
            return False
        else:
            return True

    def capture_camera(self):
        return self.camera.read()

    def image_fit_liveness(self, image):
        if self.isCamScaleFit:
            return image
        else:
            maxWidth = int(self.height * (3 / 4))
            beginWidth = int(self.width / 2 - maxWidth / 2)
            return image[0:self.height, beginWidth:(beginWidth + maxWidth)]

    def detect_liveness(self, image):
        image_bbox = self.model_test.get_bbox(image)
        prediction = np.zeros((1, 3))
        test_speed = 0
        # sum the prediction from single model's result
        for model_name in os.listdir(self.model_dir):
            h_input, w_input, model_type, scale = parse_model_name(model_name)
            param = {
                "org_img": image,
                "bbox": image_bbox,
                "scale": scale,
                "out_w": w_input,
                "out_h": h_input,
                "crop": True,
            }
            if scale is None:
                param["crop"] = False
            img = self.image_cropper.crop(**param)
            prediction += self.model_test.predict(img, os.path.join(self.model_dir, model_name))


        # draw result of prediction
        label = np.argmax(prediction)
        value = prediction[0][label] / 2
        if label == 1:
            log.debug("Image '{}' is Real Face. Score: {:.2f}.".format("VID", value))
            color = (0, 255, 0)
        else:
            log.debug("Image '{}' is Fake Face. Score: {:.2f}.".format("VID", value))
            color = (0, 0, 255)
        log.debug("Prediction cost {:.2f} s".format(test_speed))
        cv2.rectangle(
            image,
            (image_bbox[0], image_bbox[1]),
            (image_bbox[0] + image_bbox[2], image_bbox[1] + image_bbox[3]),
            color, 1)
        return label, value, (image_bbox[0], image_bbox[1]), (image_bbox[2], image_bbox[3])

    def start_detect(self):
        log.info("Face detector started")
        while True:
            # 调用摄像头，获取图像
            start = time.time()
            ret, frame = self.capture_camera()
            frame = imutils.resize(frame, height=640, width=480)
            if ret:
                # 转换为灰度图
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # 调用人脸检测器检测
                faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
                # 绘制检测出的所有人脸
                face_num = 0
                for (x, y, w, h) in faces:
                    # 画方框
                    img = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 1)
                    # 获取人脸，并转换为200,200的统一格式
                    # 原始数据返回的是正方形
                    # f = cv2.resize(gray[y:y + h, x:x + w], (200, 200))
                    # cv2.imshow('face', f)
                    # fE = cv2.equalizeHist(f)
                    # cv2.imshow('faceE', fE)

                    face_area = img[y:y + h, x:x + w]
                    face_num = face_num + 1

                # Only one face allowed
                if face_num == 1:
                    fit_img = self.image_fit_liveness(frame)
                    liveness_label, liveness_score, face_pos, face_size = self.detect_liveness(fit_img)

                    if liveness_label == 1:
                        cv2.putText(frame, "Face is %s (%f)" % ("Real", liveness_score),
                                    (0, 25),
                                    cv2.FONT_HERSHEY_SIMPLEX, self.fontSize, (10, 255, 10), 2)
                    else:
                        cv2.putText(frame, "Face is %s (%f)" % ("FAKE", liveness_score),
                                    (0, 25),
                                    cv2.FONT_HERSHEY_SIMPLEX, self.fontSize, (10, 10, 255), 2)

                elif face_num == 0:
                    cv2.putText(frame, "No faces",
                                (0, 25),
                                cv2.FONT_HERSHEY_SIMPLEX, self.fontSize, (0, 0, 0), 2)
                else:
                    cv2.putText(frame, "Too many faces: %d" % face_num,
                                (0, 25),
                                cv2.FONT_HERSHEY_SIMPLEX, self.fontSize, (0, 0, 0), 2)

                # 展示图片
                test_speed = time.time() - start
                cv2.putText(frame, "Cost %fs" % test_speed,
                            (0, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, self.fontSize, (0, 255, 255), 2)
                cv2.imshow(self.imtitle, frame)
            else:
                log.error("Camera capture failed: %d" % ret)

            cv2.waitKey(100)


def main():
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)

    os.chdir(os.path.dirname(__file__))

    d = FaceDetector()
    d.load()
    d.start_detect()


if __name__ == '__main__':
    main()
