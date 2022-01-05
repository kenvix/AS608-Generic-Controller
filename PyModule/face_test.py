import os
import time

import cv2
import logging

import face_recognition
import numpy as np
import imutils
import shutil

from face_anti_spoofing.src.anti_spoof_predict import AntiSpoofPredict
from face_anti_spoofing.src.generate_patches import CropImage
from face_anti_spoofing.src.utility import parse_model_name

log = logging.getLogger('Face Detector')


class FaceDetector:
    def __init__(self):
        self.face_tolerance = 0.35
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
        self.should_stop_detect = True
        self.should_stop_add_face = True
        self.model_dir = "./face_anti_spoofing/resources/anti_spoof_models"
        self.face_dir = "./data/faces"
        self.capture_num_each_face = 4
        self.capture_min_x = 95
        self.capture_min_y = 110
        self.known_face_encodings = []
        self.known_face_names = []

    def load(self):
        if os.path.isdir(self.face_dir) is False:
            os.mkdir(self.face_dir)

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

    def reload_known_faces(self):
        self.known_face_encodings = []
        self.known_face_names = []
        for faceName in os.listdir(self.face_dir):
            for facePhoto in os.listdir(self.face_dir + "/" + faceName):
                self.known_face_names.append(faceName)
                image = face_recognition.load_image_file(self.face_dir + "/" + faceName + "/" + facePhoto)
                self.known_face_encodings.append(face_recognition.face_encodings(image)[0])
                logging.debug("Loaded face %s in %s" % (faceName, facePhoto))

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

    def stop_detect(self):
        self.should_stop_detect = True

    def stop_add_new_face(self):
        self.should_stop_add_face = True

    def start_add_new_face(self, tag, overwrite=True):
        log.info("Capturing started")
        self.should_stop_add_face = False
        face_dir = self.face_dir + "/" + tag + "/"
        if os.path.isdir(face_dir):
            if overwrite:
                shutil.rmtree(face_dir)
            else:
                raise FileExistsError(face_dir)

        os.mkdir(face_dir)

        i = 0
        while i < self.capture_num_each_face:
            if self.should_stop_add_face is True:
                shutil.rmtree(face_dir)
                return

            face_area_resized = None
            ret, frame = self.capture_camera()

            if ret:
                # 转换为灰度图
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # 调用人脸检测器检测
                faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
                # 绘制检测出的所有人脸
                face_num = 0
                w = 0
                h = 0
                for (x, y, w, h) in faces:
                    face_area = frame[y:y + h, x:x + w]
                    face_area_resized = frame[max(int(y - h * 0.35), 0):min(int(y + h * 1.35), self.height),
                                        max(int(x - w * 0.25), 0):min(int(x + w * 1.25), self.width)]
                    face_num = face_num + 1

                # Only one face allowed
                if face_num == 1:
                    if w < self.capture_min_x or h < self.capture_min_y:
                        cv2.putText(frame, "Face too small, min %dx%d got %dx%d" % (
                            self.capture_min_x, self.capture_min_y, w, h),
                                    (0, 25),
                                    cv2.FONT_HERSHEY_SIMPLEX, self.fontSize, (10, 10, 255), 2)
                    else:
                        cv2.putText(frame, "Added face %d/%d  %dx%d" % (i, self.capture_num_each_face, w, h),
                                    (0, 25),
                                    cv2.FONT_HERSHEY_SIMPLEX, self.fontSize, (10, 255, 10), 2)

                        logging.info("Writing new face " + face_dir + '%d.jpg' % i)
                        cv2.imwrite(face_dir + '%d.jpg' % i, face_area_resized, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                        i = i + 1
                elif face_num == 0:
                    cv2.putText(frame, "No faces",
                                (0, 25),
                                cv2.FONT_HERSHEY_SIMPLEX, self.fontSize, (0, 0, 0), 2)
                else:
                    cv2.putText(frame, "Too many faces: %d" % face_num,
                                (0, 25),
                                cv2.FONT_HERSHEY_SIMPLEX, self.fontSize, (0, 0, 0), 2)

                cv2.imshow(self.imtitle, frame)

                if face_area_resized is not None:
                    cv2.imshow(self.imtitle2, face_area_resized)

                cv2.waitKey(1000)
            else:
                log.error("Camera capture failed: %d" % ret)

    def start_detect(self):
        log.info("Face detector started")
        self.should_stop_detect = False
        while self.should_stop_detect is False:
            # 调用摄像头，获取图像
            start = time.time()
            face_area_resized = None
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

                    face_area = frame[y:y + h, x:x + w]
                    face_area_resized = frame[max(int(y - h * 0.2), 0):min(int(y + h * 1.2), self.height),
                                        max(int(x - w * 0.2), 0):min(int(x + w * 1.2), self.width)]
                    face_num = face_num + 1

                # Only one face allowed
                if face_num == 1:
                    # Liveness detection
                    fit_img = self.image_fit_liveness(frame)
                    liveness_label, liveness_score, face_pos, face_size = self.detect_liveness(fit_img)

                    if liveness_label == 1:
                        cv2.putText(frame, "Face is %s (%f)" % ("Real", liveness_score),
                                    (0, 25),
                                    cv2.FONT_HERSHEY_SIMPLEX, self.fontSize, (10, 255, 10), 2)

                        # Face recognition
                        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
                        rgb_small_frame = face_area_resized[:, :, ::-1]
                        # Find all the faces and face encodings in the current frame of video
                        face_encodings = face_recognition.face_encodings(rgb_small_frame)
                        if len(face_encodings) >= 1:
                            face_encoding = face_encodings[0]

                            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding, tolerance=self.face_tolerance)

                            if True in matches:
                                # Or instead, use the known face with the smallest distance to the new face
                                face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                                best_match_index = np.argmin(face_distances)
                                name = self.known_face_names[best_match_index]
                                name_distance = np.amin(face_distances)
                                cv2.putText(frame, "N: %s (%f)" % (name, name_distance),
                                            (0, 75),
                                            cv2.FONT_HERSHEY_SIMPLEX, self.fontSize, (10, 255, 10), 2)
                            else:
                                cv2.putText(frame, "Face unmatched",
                                            (0, 75),
                                            cv2.FONT_HERSHEY_SIMPLEX, self.fontSize, (10, 10, 255), 2)

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

                if face_area_resized is not None:
                    cv2.imshow(self.imtitle2, face_area_resized)
            else:
                log.error("Camera capture failed: %d" % ret)

            cv2.waitKey(10)


def main():
    logging.basicConfig()
    logging.getLogger().setLevel(logging.DEBUG)

    os.chdir(os.path.dirname(__file__))

    d = FaceDetector()
    d.reload_known_faces()
    d.load()
    # d.start_add_new_face("Kenvix Zure")
    d.start_detect()


if __name__ == '__main__':
    main()
