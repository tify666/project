import argparse
import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import cv2
import numpy as np
from model import CNN
from utils import index2emotion, cv2_img_add_text
from blazeface import blaze_detect
# Import PyQt5 modules
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QFileDialog, QLabel
from PyQt5.QtGui import QIcon


def load_model():
    """
    加载本地模型
    :return:
    """
    model = CNN()
    model.load_weights('./models/cnn3_best_weights.h5')
    return model


def generate_faces(face_img, img_size=48):
    """
    将探测到的人脸进行增广
    :param face_img: 灰度化的单个人脸图
    :param img_size: 目标图片大小
    :return:
    """

    face_img = face_img / 255.
    face_img = cv2.resize(face_img, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    resized_images = list()
    resized_images.append(face_img)
    resized_images.append(face_img[2:45, :])
    resized_images.append(face_img[1:47, :])
    resized_images.append(cv2.flip(face_img[:, :], 1))

    for i in range(len(resized_images)):
        resized_images[i] = cv2.resize(resized_images[i], (img_size, img_size))
        resized_images[i] = np.expand_dims(resized_images[i], axis=-1)
    resized_images = np.array(resized_images)
    return resized_images


def predict_expression():
    """
    实时预测
    :return:
    """
    # 参数设置
    model = load_model()

    border_color = (0, 0, 0)  # 黑框框
    font_color = (255, 255, 255)  # 白字字
    capture = cv2.VideoCapture(0)  # 指定0号摄像头
    if opt.video_path:
        capture = cv2.VideoCapture(opt.video_path)
    while True:
        print("test")
        _, frame = capture.read()  # 读取一帧视频，返回是否到达视频结尾的布尔值和这一帧的图像
        frame = cv2.cvtColor(cv2.resize(frame, (800, 600)), cv2.COLOR_BGR2RGB)
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 灰度化
        faces = blaze_detect(frame)
        # 如果检测到人脸
        if faces is not None and len(faces) > 0:
            for (x, y, w, h) in faces:
                face = frame_gray[y: y + h, x: x + w]  # 脸部图片
                faces = generate_faces(face)
                results = model.predict(faces)
                result_sum = np.sum(results, axis=0).reshape(-1)
                label_index = np.argmax(result_sum, axis=0)
                emotion = index2emotion(label_index)
                cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), border_color, thickness=2)
                frame = cv2_img_add_text(frame, emotion, x+30, y+30, font_color, 20)
        cv2.imshow("expression recognition(press esc to exit)", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))  # 利用人眼假象

        key = cv2.waitKey(30)  # 等待30ms，返回ASCII码

        # 如果输入esc则退出循环
        if key == 27:
            break
    capture.release()  # 释放摄像头
    cv2.destroyAllWindows()  # 销毁窗口


# Define a class for the front-end page
class App(QWidget):

    def __init__(self):
        super().__init__()
        self.title = 'Real-time Expression Recognition'
        self.left = 100
        self.top = 100
        self.width = 400
        self.height = 200
        self.initUI()

    def initUI(self):
        # Set the window title and size
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        # Create a label to display the instructions
        self.label = QLabel(self)
        self.label.setText('Please choose one of the following options:')
        self.label.move(50, 20)

        # Create a button to choose the camera option
        self.button1 = QPushButton('Camera', self)
        self.button1.move(50, 80)
        # Connect the button to the camera function
        self.button1.clicked.connect(self.camera)

        # Create a button to choose the video option
        self.button2 = QPushButton('Video', self)
        self.button2.move(250, 80)
        # Connect the button to the video function
        self.button2.clicked.connect(self.video)

        # Show the window
        self.show()

    def camera(self):
        # Set the source argument to 0 for camera
        opt.source = 0
        # Close the window and start the recognition
        self.close()
        predict_expression()

    def video(self):
        # Set the source argument to 1 for video
        opt.source = 1
        # Open a file dialog to choose the video file
        opt.video_path, _ = QFileDialog.getOpenFileName(self, 'Open file', '', 'Video files (*.mp4 *.avi)')
        # Close the window and start the recognition
        self.close()
        predict_expression()


parser = argparse.ArgumentParser()
parser.add_argument("--source", type=int, default=0, help="data source, 0 for camera 1 for video")
parser.add_argument("--video_path", type=str, default=None)
opt = parser.parse_args()


# Create an application instance
app = QApplication(sys.argv)
# Create a front-end page instance
ex = App()
# Exit the application when the window is closed
sys.exit(app.exec_())
