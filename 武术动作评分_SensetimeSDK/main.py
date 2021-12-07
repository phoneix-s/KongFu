# -*- coding: utf-8 -*-
from class_coding.coding_API import *
from detect_coding.detect_API import *
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QMessageBox, QDialog, QFileDialog
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.Qt import QThread
import pandas as pd
import os
from SenseTime.SDK import *
import cv2
import numpy as np
import glob
import joblib
import copy

g_body_detector = BodyDetector()
g_body_aligner = BodyAligner()


def draw_body_skeleton(show_img, pose):
    cv2.line(show_img, (int(pose[0][0]), int(pose[0][1])), (int(pose[1][0]), int(pose[1][1])), (255, 255, 255), 2)
    cv2.line(show_img, (int(pose[1][0]), int(pose[1][1])), (int(pose[2][0]), int(pose[2][1])), (255, 255, 255), 2)
    cv2.line(show_img, (int(pose[1][0]), int(pose[1][1])), (int(pose[3][0]), int(pose[3][1])), (255, 255, 255), 2)
    cv2.line(show_img, (int(pose[2][0]), int(pose[2][1])), (int(pose[4][0]), int(pose[4][1])), (255, 255, 255), 2)
    # cv2.polylines()
    cv2.line(show_img, (int(pose[4][0]), int(pose[4][1])), (int(pose[6][0]), int(pose[6][1])), (255, 255, 255), 2)
    cv2.line(show_img, (int(pose[3][0]), int(pose[3][1])), (int(pose[5][0]), int(pose[5][1])), (255, 255, 255), 2)
    cv2.line(show_img, (int(pose[5][0]), int(pose[5][1])), (int(pose[7][0]), int(pose[7][1])), (255, 255, 255), 2)
    cv2.line(show_img, (int(pose[1][0]), int(pose[1][1])),
             (int((pose[8][0] + pose[9][0]) / 2), int((pose[8][1] + pose[9][1]) / 2)), (255, 255, 255), 2)
    cv2.line(show_img, (int(pose[8][0]), int(pose[8][1])), (int(pose[9][0]), int(pose[9][1])), (255, 255, 255), 2)
    cv2.line(show_img, (int(pose[8][0]), int(pose[8][1])), (int(pose[10][0]), int(pose[10][1])), (255, 255, 255), 2)
    cv2.line(show_img, (int(pose[9][0]), int(pose[9][1])), (int(pose[11][0]), int(pose[11][1])), (255, 255, 255), 2)
    cv2.line(show_img, (int(pose[10][0]), int(pose[10][1])), (int(pose[12][0]), int(pose[12][1])), (255, 255, 255), 2)
    cv2.line(show_img, (int(pose[11][0]), int(pose[11][1])), (int(pose[13][0]), int(pose[13][1])), (255, 255, 255), 2)


def normalize_keypoint(points):
    points_res = np.array(points)
    min_h = min(points_res[:, 0])
    min_w = min(points_res[:, 1])
    body_h = max(points_res[:, 0]) - min_h
    body_w = max(points_res[:, 1]) - min_w
    points_res = np.ceil((points_res - np.array([min_h, min_w])) / np.array([body_h, body_w]) * np.array([100, 100]))
    return points_res


class Worker(QThread):
    _signal = pyqtSignal(str)

    def __init__(self, window):
        super().__init__()
        self.window = window

    def run(self):
        try:
            post = {}
            post["read_path_dir"], post["save_path"] = [], self.window.saved_csv

            # lucy
            post['read_path_dir'].append("./gongbu")
            post['read_path_dir'].append("./mabu")
            post['read_path_dir'].append("./pubu")
            post['read_path_dir'].append("./chongquan")
            post['read_path_dir'].append("./titui")
            post['read_path_dir'].append("./stand")

            for index, each in enumerate(post['read_path_dir']):
                self._signal.emit("start" + str(index))
                filenames = []
                features = []
                points = []

                filenames = np.sort([os.path.normpath(os.path.join(each, i)) for i in os.listdir(each)])

                skipped_image = 0
                for i in range(len(filenames)):
                    lines = filenames[i]
                    img = self.window.load_img(lines)
                    if img is None:
                        skipped_image += 1
                        continue
                    else:
                        points_res = self.window.extract_points(img)
                        ###############################
                        if isinstance(points_res, list) is True:
                            if len(points_res) > 0:
                                skip_flag = 0
                            else:
                                skip_flag = 1
                        else:
                            skip_flag = 1

                        if skip_flag == 1:
                            skipped_image += 1
                            continue
                        points_res = normalize_keypoint(points_res)
                        features.append(points_res)
                        ###############################
                    self._signal.emit(str(int((i + 1) / len(filenames) * 100)))

                data_mat = np.array(features)
                if len(filenames) - skipped_image != 0:
                    write_csv(post["save_path"][index], data_mat.reshape(len(filenames) - skipped_image, -1))
                else:
                    self._signal.emit("no_image")
                    return
                self._signal.emit("finish")
            self._signal.emit("done")
        except Exception as e:
            self._signal.emit(str(e))


class Ui_MainWindow(QDialog):

    def setupUi(self, MainWindow):
        self.action_num = -1  # lucy
        self.best_score = -1
        self.average = []
        self.evaluate_standard = 0
        self.gongbu()

        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(400, 431)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")

        ####################################

        self.horizontalLayout_training = QtWidgets.QHBoxLayout()
        self.horizontalLayout_training.setObjectName("horizontalLayout_training")
        self.label_training = QtWidgets.QLabel("训练模型：")
        self.label_training.setObjectName("label_training")
        self.horizontalLayout_training.addWidget(self.label_training)
        self.pushButton_training = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_training.setObjectName("pushButton_training")
        self.horizontalLayout_training.addWidget(self.pushButton_training)
        self.progressBar = QtWidgets.QProgressBar(self.centralwidget)
        self.progressBar.setProperty("value", 0)
        self.progressBar.setObjectName("progressBar")
        self.horizontalLayout_training.addWidget(self.progressBar)
        self.label_12 = QtWidgets.QLabel(self.centralwidget)
        self.label_12.setObjectName("label_12")
        self.horizontalLayout_training.addWidget(self.label_12)

        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        # lucy

        self.label_action = QtWidgets.QLabel("第一步->选择动作：")
        self.label_action.setObjectName("label_action")
        self.horizontalLayout_3.addWidget(self.label_action)

        self.pushButton_gongbu = QtWidgets.QPushButton('弓步')
        self.horizontalLayout_3.addWidget(self.pushButton_gongbu)
        self.pushButton_mabu = QtWidgets.QPushButton('马步')
        self.horizontalLayout_3.addWidget(self.pushButton_mabu)
        self.pushButton_pubu = QtWidgets.QPushButton('仆步')
        self.horizontalLayout_3.addWidget(self.pushButton_pubu)

        # self.pushButton_chongquan = QtWidgets.QPushButton('冲拳')
        # self.horizontalLayout_3.addWidget(self.pushButton_chongquan)
        # self.pushButton_titui = QtWidgets.QPushButton('踢腿')
        # self.horizontalLayout_3.addWidget(self.pushButton_titui)

        self.horizontalLayout_test_standard = QtWidgets.QHBoxLayout()
        self.horizontalLayout_test_standard.setObjectName("horizontalLayout_test_standard")
        self.label_standard = QtWidgets.QLabel("第二步->设置评分标准：")
        self.label_standard.setObjectName("label_standard")
        self.horizontalLayout_test_standard.addWidget(self.label_standard)
        self.pushButton_standard_rule = QtWidgets.QPushButton('基于规则')
        self.horizontalLayout_test_standard.addWidget(self.pushButton_standard_rule)
        self.pushButton_standard_ml = QtWidgets.QPushButton('基于机器学习')
        self.horizontalLayout_test_standard.addWidget(self.pushButton_standard_ml)

        self.horizontalLayout_test = QtWidgets.QHBoxLayout()
        self.horizontalLayout_test.setObjectName("horizontalLayout_test")
        self.label_action = QtWidgets.QLabel("第三步->开始评分：")
        self.label_action.setObjectName("label_action")
        self.horizontalLayout_test.addWidget(self.label_action)
        self.pushButton_file = QtWidgets.QPushButton('图片评分')
        self.horizontalLayout_test.addWidget(self.pushButton_file)
        self.pushButton_video = QtWidgets.QPushButton('录像评分')
        self.horizontalLayout_test.addWidget(self.pushButton_video)
        self.pushButton_test_realtime = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_test_realtime.setObjectName("pushButton_4")
        self.horizontalLayout_test.addWidget(self.pushButton_test_realtime)
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setMaximumSize(QtCore.QSize(16777215, 30))
        self.label_3.setObjectName("label_3")
        self.horizontalLayout_test.addWidget(self.label_3)
        self.plainTextEdit_3 = QtWidgets.QPlainTextEdit(self.centralwidget)
        self.plainTextEdit_3.setMaximumSize(QtCore.QSize(30, 30))
        self.plainTextEdit_3.setObjectName("plainTextEdit_3")
        self.horizontalLayout_test.addWidget(self.plainTextEdit_3)
        self.verticalLayout.addLayout(self.horizontalLayout_training)
        self.line_1 = QtWidgets.QFrame(self.centralwidget)
        self.line_1.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_1.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_1.setObjectName("line_1")
        self.verticalLayout.addWidget(self.line_1)
        self.verticalLayout.addLayout(self.horizontalLayout_3)
        self.line_2 = QtWidgets.QFrame(self.centralwidget)
        self.line_2.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_2.setObjectName("line_2")
        self.verticalLayout.addWidget(self.line_2)
        self.verticalLayout.addLayout(self.horizontalLayout_test_standard)
        self.verticalLayout.addLayout(self.horizontalLayout_test)

        self.gridLayout.addLayout(self.verticalLayout, 0, 0, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 400, 25))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "武术基本动作训练"))
        MainWindow.setFixedSize(600, 180)
        MainWindow.move(650, 0)

        self.pushButton_training.setText(_translate("MainWindow", "训练模型"))
        self.pushButton_test_realtime.setText(_translate("MainWindow", "实时评分"))
        self.label_3.setText(_translate("MainWindow", "相机序号"))
        self.plainTextEdit_3.setPlaceholderText(_translate("MainWindow", "0"))
        self.label_12.setText(_translate("MainWindow", "合并完成"))

    def setupFunction(self):
        self.label_12.setText("")

        self.pushButton_training.clicked.connect(self.train)
        self.pushButton_gongbu.clicked.connect(self.gongbu)
        self.pushButton_mabu.clicked.connect(self.mabu)
        self.pushButton_pubu.clicked.connect(self.pubu)
        # self.pushButton_chongquan.clicked.connect(self.chongquan)
        # self.pushButton_titui.clicked.connect(self.titui)
        self.pushButton_standard_rule.clicked.connect(self.set_standard_rule)
        self.pushButton_standard_ml.clicked.connect(self.set_standard_ml)
        self.pushButton_file.clicked.connect(self.test_image)
        self.pushButton_video.clicked.connect(self.test_video)
        self.pushButton_test_realtime.clicked.connect(self.test_realtime)

    def show_pattern_img(self, img_file):
        img = cv2.imread(img_file)
        img = resize_frame(img)
        cv2.imshow("Pattern", img)
        cv2.moveWindow("Pattern", 0, 200)

    def init_best(self):
        self.best_score = -1

    def gongbu(self):
        # self.average=np.array([[61,55,46,62,34,66,48,65,45,60,32,86,0,100],[0,21,26,26,43,43,57,57,56,57,82,70,99,93]])
        self.average = [[61, 55, 46, 62, 34, 66, 48, 65, 45, 60, 32, 86, 0, 100],
                        [0, 21, 26, 26, 43, 43, 57, 57, 56, 57, 82, 70, 99, 93]]
        self.action_num = 1
        self.init_best()
        self.show_pattern_img("./patterns/gongbu.jpg")

    def mabu(self):
        self.average = [[47, 49, 23, 73, 18, 73, 22, 75, 36, 62, 9, 94, 0, 100],
                        [0, 23, 29, 30, 47, 49, 62, 64, 63, 62, 82, 80, 100, 97]]
        self.action_num = 2
        self.init_best()
        self.show_pattern_img("./patterns/mabu.jpg")

    def pubu(self):
        self.average = [[72, 75, 57, 90, 53, 91, 54, 80, 60, 78, 38, 95, 0, 58],
                        [0, 26, 31, 32, 55, 68, 62, 77, 75, 82, 84, 96, 96, 90]]
        self.action_num = 3
        self.init_best()
        self.show_pattern_img("./patterns/pubu.jpg")

    def chongquan(self):
        self.average = np.array([[72, 75, 57, 90, 53, 91, 54, 80, 60, 78, 38, 95, 0, 58],
                                 [0, 26, 31, 32, 55, 68, 62, 77, 75, 82, 84, 96, 96, 90]])
        self.action_num = 0
        self.init_best()
        self.show_pattern_img("./patterns/chongquan.jpg")

    def titui(self):
        self.average = np.array([[72, 75, 57, 90, 53, 91, 54, 80, 60, 78, 38, 95, 0, 58],
                                 [0, 26, 31, 32, 55, 68, 62, 77, 75, 82, 84, 96, 96, 90]])
        self.action_num = 5
        self.init_best()
        self.show_pattern_img("./patterns/titui.jpg")

    def set_standard_ml(self):
        self.evaluate_standard = 1
        self.init_best()

    def set_standard_rule(self):
        self.evaluate_standard = 0
        self.init_best()

    def refresh(self):
        self.label_12.setText("")

    def load_img(self, filename):
        img = cv2.imdecode(np.fromfile(filename, dtype=np.uint8), -1)
        return img

    def select_folder(self, plainTextEdit_n):
        def select_folder_with_text_edit(self):
            folder_path = QFileDialog.getExistingDirectory()
            plainTextEdit_n.setPlainText(folder_path)

        return select_folder_with_text_edit

    def extract_points(self, img):
        rect = g_body_detector.detect(img)
        if len(rect) > 0:
            points = g_body_aligner.align(img, rect[0])
            return points
        else:
            return None

    def get_signal(self, val):
        if val[:5] == "start":
            self.label_12.setText("提取第{0}个文件夹中......".format(int(val[5:]) + 1))
        elif val == "finish":
            self.label_12.setText("提取完毕。")
            self.pushButton_training.setEnabled(True)
            self.pushButton_test_realtime.setEnabled(True)
        elif val == "no_image":
            self.label_12.setText("未找到合适的图片，请重试。")
        elif val == "done":
            X, y = self.combine()
            # X = read_csv("./train_data.csv")
            # y = read_csv("./train_label.csv")
            print(X.shape)
            print(y)
            from sklearn.linear_model import LogisticRegression
            clf = LogisticRegression(solver='lbfgs', max_iter=10000).fit(X, y)
            # from sklearn.neural_network import MLPClassifier
            # X = X.astype(np.float64)
            # clf = MLPClassifier(solver='sgd', activation='relu',alpha=1e-4,hidden_layer_sizes=(10,10), random_state=1,max_iter=500,verbose=10,learning_rate_init=.1).fit(X, y)
            # from sklearn.ensemble import RandomForestRegressor
            # cls = RandomForestRegressor(random_state=0).fit(X, y)            
            self.model_results = clf  # lucy!!!
            self.save('./model' + ".lc")
        else:
            try:
                int_val = int(val)
                self.progressBar.setValue(int_val)
            except Exception:
                self.refresh()
                QMessageBox.critical(self, "错误", str(val))

    def save(self, save_path):
        self.save_path = save_path
        joblib.dump(self.model_results, os.path.normpath(self.save_path))

    def gether_features(self):
        # self.saved_csv = ["./feature1.csv", "./feature2.csv", "./feature3.csv"] # at workers
        self.saved_csv = ["./feature1.csv", "./feature2.csv", "./feature3.csv", "./feature4.csv", "./feature5.csv",
                          "./feature6.csv"]  # Lucy
        try:

            self.pushButton_training.setEnabled(False)
            self.pushButton_test_realtime.setEnabled(False)

            self.worker = Worker(self)
            self.worker._signal.connect(self.get_signal)
            self.worker.start()

        except Exception as e:
            self.refresh()
            QMessageBox.critical(self, "错误", str(e))

    def combine(self):
        try:
            # self.input_dir_names = [self.plainTextEdit.toPlainText(), self.plainTextEdit_2.toPlainText(), self.plainTextEdit_4.toPlainText()]
            # self.input_dir_names = ["D:/AI/线下夏令营/project/gongbu","D:/AI/线下夏令营/project/mabu","D:/AI/线下夏令营/project/pubu","D:/AI/线下夏令营/project/xiebu","D:/AI/线下夏令营/project/xubu","D:/AI/线下夏令营/project/stand"]
            self.input_dir_names = ["./gongbu", "./mabu", "./pubu", "./chongquan", "./titui", "./stand"]
            print("inp", self.input_dir_names)
            features, label_data = [], []
            for each in range(len(self.saved_csv)):
                feature = read_csv(self.saved_csv[each])
                features.append(feature)
                _, dir_name = os.path.split(self.input_dir_names[each])
                label_data += [dir_name] * np.shape(feature)[0]
            concat = np.concatenate(features, axis=0)
            full_set = np.concatenate((concat, np.array(label_data).reshape(-1, 1)), axis=1)
            train_data = full_set[:, :-1]
            train_label = full_set[:, -1]
            self.label_12.setText("训练完成。")
            return train_data, train_label
        except Exception as e:
            self.refresh()
            QMessageBox.critical(self, "错误", str(e))

    def train(self):
        self.gether_features()

    # 基于规则的评分
    def evaluate_by_rule(self, rule, frame):
        rect = detect_body(frame)

        score = 0
        if rect[0] >= 0:
            rect_frame = render_body_rect(frame, rect)
            points = align_body(frame, rect)
            points_res = normalize_keypoint(points)
            # print(points_res)
            # 马步标准坐标[[47,49,23,73,18,73,22,75,36,62,9,94,0,100],[0,23,29,30,47,49,62,64,63,62,82,80,100,97]]
            # 弓步标准坐标[[61,55,46,62,34,66,48,65,45,60,32,86,0,100],[0,21,26,26,43,43,57,57,56,57,82,70,99,93]]
            # 仆步标准坐标[[72,75,57,90,53,91,54,80,60,78,38,95,0,58],[0,26,31,32,55,68,62,77,75,82,84,96,96,90]]
            # 这个没有做分类，只能按设定的类型算分，而且弓步、仆步都是训练集的那个方向。填入下面那行可以更改检测类型
            # 它的问题是一旦画面不正，有倾斜角度，就不靠谱了
            average = np.array(rule)
            font = cv2.FONT_HERSHEY_SIMPLEX

            for i in range(0, 14):
                if abs(points_res[i][0] - average[0][i]) < 7:
                    score = score + 1
                else:
                    if (points_res[i][0] - average[0][i]) < 0:
                        cv2.putText(frame, text="L", org=(int(points[i][0]), int(points[i][1])), fontFace=font,
                                    fontScale=0.6,
                                    color=(0, 255, 0), thickness=1)
                    else:
                        cv2.putText(frame, text="R", org=(int(points[i][0]), int(points[i][1])), fontFace=font,
                                    fontScale=0.6,
                                    color=(0, 255, 0), thickness=1)
                if abs(points_res[i][1] - average[1][i]) < 7:
                    score = score + 1
                else:
                    if (points_res[i][1] - average[1][i]) < 0:
                        cv2.putText(frame, text="U", org=(int(points[i][0]) + 15, int(points[i][1])), fontFace=font,
                                    fontScale=0.6,
                                    color=(0, 255, 0), thickness=1)
                    else:
                        cv2.putText(frame, text="D", org=(int(points[i][0]) + 15, int(points[i][1])), fontFace=font,
                                    fontScale=0.6,
                                    color=(0, 255, 0), thickness=1)
                        # 显示L,R,U,D分别表示坐标相较于标准坐标偏左、右、上、下
                        # 这里+15是为了不让UD和LR重叠，但仍可能和靠得近的其他点重叠，有待解决

            # score= int(score/28*10000)/100 #s是取两位小数的得分
            score = score / 28 * 100

        return score, frame

    # 基于机器学习的评分
    def evaluate_by_ml(self, pose, model, frame):
        font = cv2.FONT_HERSHEY_SIMPLEX
        pose = normalize_keypoint(pose)
        pose = pose.reshape(1, -1)

        result = model.predict(pose)
        # self.action_num =0

        prob = model.predict_proba(pose)[0]
        score = prob[self.action_num] * 100
        # score=100-(100-score)*10
        # if(self.action_num==1 or self.action_num==3):
        # score=100-(100-score)*10
        # cv2.putText(frame, str(int(prob[0]*100)), (50, 50), font, 1.2, (255, 255, 255), 2) #Lucy
        # cv2.putText(frame, str(int(prob[1]*100)), (50, 100), font, 1.2, (255, 255, 255), 2) #Lucy
        # cv2.putText(frame, str(int(prob[2]*100)), (50, 150), font, 1.2, (255, 255, 255), 2) #Lucy
        # cv2.putText(frame, str(int(prob[3]*100)), (50, 200), font, 1.2, (255, 255, 255), 2) #Lucy
        # cv2.putText(frame, str(int(prob[4]*100)), (50, 250), font, 1.2, (255, 255, 255), 2) #Lucy
        # cv2.putText(frame, str(int(prob[5]*100)), (50, 300), font, 1.2, (255, 255, 255), 2) #Lucy

        return score, frame

    def evaluate_image(self, model, img):
        return_value = 0

        show_img = copy.deepcopy(img)
        rects = g_body_detector.detect(img)  # 检测人体位置
        if len(rects) == 0:
            cv2.imshow('Action_Evaluation', show_img)
            cv2.moveWindow('Action_Evaluation', 500, 200)  # lucy
            return_value = 0
        else:
            # show_img = body_detector.render(show_img,rects[0])
            pose = g_body_aligner.align(img, rects[0])  # 对人体进行关键点检测
            ori_pose = copy.deepcopy(pose)
            show_img = g_body_aligner.render(img, pose)  # Lucy
            draw_body_skeleton(show_img, pose)

            font = cv2.FONT_HERSHEY_SIMPLEX

            if self.evaluate_standard == 0:
                score, show_img = self.evaluate_by_rule(self.average, show_img)  # 基于规则的评分
            else:
                score, show_img = self.evaluate_by_ml(pose, model, show_img)  # 基于机器学习的评分
            show_img = render_number(show_img, "Score:" + str(int(score)))
            cv2.imshow('Action_Evaluation', show_img)
            cv2.moveWindow('Action_Evaluation', 500, 200)  # lucy

            if score > self.best_score:
                self.best_score = score
                best_snapshot = copy.deepcopy(show_img)
                # cv2.putText(best_snapshot, "Score: "+ str(int(self.best_score)), (50, 50), font, 1.2, (22,7,201), 2) #Lucy
                cv2.imshow('Best_Snapshot', best_snapshot)
                cv2.moveWindow('Best_Snapshot', 1300, 200)
            return_value = 1
        return return_value

    def test_image(self):
        file_name = QFileDialog.getOpenFileName(self, '选择图片', './examples', 'Images(*.jpg *.jpeg)')[0]
        try:
            model_path = './model.lc'
            model = joblib.load(model_path)
            result = model.predict_proba([np.zeros(28)])[0]
            class_num = result.shape[0]
            if file_name != '':
                self.init_best()
                img = self.load_img(file_name)
                img = resize_frame(img)
                self.evaluate_image(model, img)

        except Exception as e:
            QMessageBox.critical(QMainWindow, "文件错误", str(e))

    def test_video(self):
        try:
            model_path = './model.lc'
            model = joblib.load(model_path)
            result = model.predict_proba([np.zeros(28)])[0]

            file_name = QFileDialog.getOpenFileName(self, '选择录像', './examples', 'Video Files(*.mov *.mp4)')[0]
            if file_name != '':
                self.init_best()
                video = Video(file_name)
                for frame in video:
                    self.evaluate_image(model, frame)
                    k = cv2.waitKey(1)
                    if k == ord("m"):
                        self.evaluate_standard = 1 if self.evaluate_standard == 0 else 0
                        self.init_best()
                    if k == ord("q"):
                        break
            cv2.destroyAllWindows()
        except Exception as e:
            QMessageBox.critical(QMainWindow, "文件错误", str(e))

    def test_realtime(self):
        try:
            self.init_best()
            # ------------输入选择----------------
            camera_num = self.plainTextEdit_3.toPlainText()
            if camera_num == '':
                camera_num = 0
            else:
                camera_num = int(camera_num)
            cap = cv2.VideoCapture(camera_num, cv2.CAP_DSHOW)

            model_path = './model.lc'
            model = joblib.load(model_path)

            result = model.predict_proba([np.zeros(28)])[0]
            class_num = result.shape[0]

            line_color_rgb = [[255, 0, 0], [128, 128, 0], [0, 255, 0], [0, 128, 128], [0, 0, 255], [128, 0, 128]]

            while True and cap.isOpened():
                ret, frame = cap.read()
                show_img = copy.deepcopy(frame)
                if not ret:
                    break

                self.evaluate_image(model, frame)

                k = cv2.waitKey(1)
                if k == ord("m"):
                    self.evaluate_standard = 1 if self.evaluate_standard == 0 else 0
                    self.init_best()
                if k == ord("q"):
                    break

            cap.release()
            cv2.destroyAllWindows()
        except Exception as e:
            QMessageBox.critical(QMainWindow, "错误", str(e))


if __name__ == "__main__":
    import sys
    from PyQt5.QtWidgets import QDialog, QApplication, QMainWindow
    import os
    import glob

    app = QApplication(sys.argv)
    window = QMainWindow()

    ui = Ui_MainWindow()
    ui.setupUi(window)
    ui.setupFunction()

    window.show()
    sys.exit(app.exec_())
