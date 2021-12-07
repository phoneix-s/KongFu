import cv2
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import cv2
import matplotlib.pyplot as plt


class Camera(object):
    def __init__(self, frame_count):
        self.capture = cv2.VideoCapture(0)
        self.frame = 0
        self.frame_count = frame_count
        self.h = self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.w = self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    def get_height(self):
        return self.h
    def get_width(self):
        return self.w
    def __iter__(self):
        return self
    def __next__(self):
        key = cv2.waitKey(50)
        ret, frame = self.capture.read()
        # print("ret", ret)
        frame = cv2.flip(frame, 1)
        if not ret:
            raise Exception("未检测到摄像头！")
        if key in [ord('q'), 27, ord('Q')] or self.frame >= self.frame_count:
            self.capture.release()
            cv2.destroyAllWindows()
            raise StopIteration
        self.frame += 1
        return frame


def imshow(img, other=''):
    if other == '':
        if len(img.shape) == 3:
            plt.imshow(img[:, :, ::-1])
        else:
            plt.imshow(img, 'gray')
        plt.axis('off')
        plt.show()
    else:
        cv2.imshow(img, other)
