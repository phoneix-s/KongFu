# -*- coding: utf-8 -*-

from numpy.core.fromnumeric import ndim, resize
from numpy.lib.function_base import average
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QMessageBox, QDialog, QFileDialog
from PyQt5.QtCore import Qt, pyqtSignal
from PIL import ImageFont, ImageDraw, Image
from PyQt5.Qt import QThread
import csv
import pandas as pd
import os
import cv2
import numpy as np
import glob
import joblib
import copy

from mmpose.apis import (inference_top_down_pose_model, init_pose_model,
                         process_mmdet_results, vis_pose_result)

try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False


device = 'cuda:0'
det_config = 'configs/yolox_tiny_8x8_300e_coco.py'
det_checkpoint = 'checkpoints/yolox_tiny_8x8_300e_coco_20210806_234250-4ff3b67e.pth'

g_body_detector = init_detector(det_config, det_checkpoint, device=device.lower())
# build the pose model from a config file and a checkpoint file
pose_config = 'configs/hrnet_w32_coco_256x192.py'
pose_checkpoint = 'checkpoints/hrnet_w32_coco_256x192-c78dce93_20200708.pth'
g_body_aligner = init_pose_model(
    pose_config, pose_checkpoint, device=device.lower())

dataset = g_body_aligner.cfg.data['test']['type']
STDHEIGHT = 600


def draw_text(img, text, pos, color=(255,255,255), font_size=40, thickness=1):  #解决openCV不能显示中文的问题
    fontpath = "simsun.ttc"  # 宋体字体文件
    font_1 = ImageFont.truetype(fontpath, font_size)  # 加载字体, 字体大小
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    draw.text(pos, text, font=font_1, fill=color,stroke_width=thickness)#left + 10, bottom - 32
    img = np.array(img_pil)
    return img


def resize_frame(frame):
    if frame.shape[1]> 600:
        resize_rate = 600 / frame.shape[1]
        resized_frame = cv2.resize(frame,(600,int(frame.shape[0]*resize_rate)))
    else:
        resize_rate = STDHEIGHT / frame.shape[0]
        resized_frame = cv2.resize(frame,(int(frame.shape[1]*resize_rate),STDHEIGHT))
    return resized_frame


def normalize_keypoint(points):
    points_res = np.array(points)
    min_h = min(points_res[:,0])
    min_w = min(points_res[:,1])
    body_h = max(points_res[:,0])-min_h
    body_w = max(points_res[:,1])-min_w
    points_res = np.ceil((points_res - np.array([min_h,min_w]))/np.array([body_h,body_w])*np.array([100,100]))
    return points_res


def read_csv(filename):
    f=open(filename, "rb")
    rt = np.loadtxt(f, delimiter=",", skiprows=0)
    return rt


def write_csv(filename, mat):
    with open(filename,'w',newline='') as f:
        writer = csv.writer(f,dialect='excel')
        writer.writerows(mat)  


class Worker(QThread):
    _signal =pyqtSignal(str)
    def __init__(self, window):
        super().__init__()
        self.window = window

    def run(self):
        try:
            post = {}
            post["read_path_dir"], post["save_path"] =[], self.window.saved_csv
                       
            post['read_path_dir'].append("./gongbu") 
            post['read_path_dir'].append("./mabu") 
            post['read_path_dir'].append("./pubu")
            post['read_path_dir'].append("./chongquan") 
            post['read_path_dir'].append("./titui")
            post['read_path_dir'].append("./stand")            
            
            for index, each in enumerate(post['read_path_dir']):
                self._signal.emit("start"+str(index))
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
                        pose_results = self.window.get_pose_results(img)
                        points_res = self.window.get_body_points(pose_results)
                        ###############################
                        
                        if isinstance(points_res,list) is True:
                            if len(points_res)>0:
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
                    self._signal.emit(str(int((i+1)/len(filenames)*100)))

                data_mat = np.array(features)
                if len(filenames)-skipped_image != 0:
                    write_csv(post["save_path"][index], data_mat.reshape(len(filenames)-skipped_image, -1))                   
                else:
                    self._signal.emit("no_image")
                    return
                self._signal.emit("finish")
            self._signal.emit("done")
        except Exception as e:
            self._signal.emit(str(e))


#主窗口UI
class Ui_MainWindow(QDialog):
    def setupUi(self, MainWindow):
        self.action_forms={'gongbu':'弓步','mabu':'马步','pubu':'仆步','chongquan':'冲拳','titui':'踢腿','stand':'站立'}
        self.action_label='' 
        self.best_score=-1
        self.current_rule=[]
        
        self.gongbu()

        MainWindow.setObjectName("MainWindow")        
        #MainWindow.setStyleSheet("#MainWindow{border-image:url(background.webp)}")
        MainWindow.setStyleSheet("#MainWindow{background-color:lightgray}")
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
        
        self.label_action = QtWidgets.QLabel("动作示范：")
        self.label_action.setObjectName("label_action")
        self.horizontalLayout_3.addWidget(self.label_action)

        self.pushButton_gongbu = QtWidgets.QPushButton('弓步')
        self.horizontalLayout_3.addWidget(self.pushButton_gongbu)
        self.pushButton_mabu = QtWidgets.QPushButton('马步')
        self.horizontalLayout_3.addWidget(self.pushButton_mabu)
        self.pushButton_pubu = QtWidgets.QPushButton('仆步')
        self.horizontalLayout_3.addWidget(self.pushButton_pubu)

        self.pushButton_chongquan = QtWidgets.QPushButton('冲拳')
        self.horizontalLayout_3.addWidget(self.pushButton_chongquan)
        self.pushButton_titui = QtWidgets.QPushButton('踢腿')
        self.horizontalLayout_3.addWidget(self.pushButton_titui)

        self.horizontalLayout_test = QtWidgets.QHBoxLayout()
        self.horizontalLayout_test.setObjectName("horizontalLayout_test")       
        self.label_action = QtWidgets.QLabel("动作识别及评分：")
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
        MainWindow.setWindowTitle(_translate("MainWindow", "武术基本动作教学及评分"))
        MainWindow.setFixedSize(600,180)
        MainWindow.move(650,0)
     
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
        self.pushButton_chongquan.clicked.connect(self.chongquan)
        #self.pushButton_chongquan.clicked.connect(self.combine)
        self.pushButton_titui.clicked.connect(self.titui)        
        self.pushButton_file.clicked.connect(self.test_image)
        self.pushButton_video.clicked.connect(self.test_video_file)
        self.pushButton_test_realtime.clicked.connect(self.test_realtime_video)
    

    def show_pattern_img(self,img_file):
        img=cv2.imread(img_file)
        img=resize_frame(img)        
        cv2.rectangle(img,(0,0),(200,80),(64,64,64),-1)
        img=draw_text(img,f'标准示范:{self.action_forms[self.action_label]}',(10,10),color=(255,255,255),font_size=25)
        img=resize_frame(img)
        cv2.imshow("Pattern",img)
        cv2.waitKey(1)
        cv2.moveWindow("Pattern",0,250)

    
    def reset_best_score(self):
        self.best_score=-1

    
    def get_standard_pose_points(self, feature_file):
        feature_data = read_csv(feature_file)
        average =np.mean(feature_data,axis=0)
        x=average[0::2]
        y=average[1::2]
        
        return [x,y]

    
    #弓步
    def gongbu(self):
        self.action_label = 'gongbu'
        self.current_rule=self.get_standard_pose_points('rule1.csv')
        self.reset_best_score()
        self.show_pattern_img("./patterns/gongbu.jpg")


    #马步
    def mabu(self):
        self.action_label = 'mabu'
        self.current_rule=self.get_standard_pose_points('rule2.csv')
        self.reset_best_score()
        self.show_pattern_img("./patterns/mabu.jpg")


    #仆步
    def pubu(self):
        self.action_label = 'pubu'
        self.current_rule=self.get_standard_pose_points('rule3.csv')
        self.reset_best_score()
        self.show_pattern_img("./patterns/pubu.jpg")
       

    #冲拳
    def chongquan(self):
        self.action_label='chongquan'
        self.current_rule=self.get_standard_pose_points('rule4.csv')
        self.reset_best_score()
        self.show_pattern_img("./patterns/chongquan.jpg")


    #踢腿
    def titui(self):
        self.action_label='titui'
        self.current_rule=self.get_standard_pose_points('rule5.csv')
        self.reset_best_score()
        self.show_pattern_img("./patterns/titui.jpg")


    #站立
    def stand(self):    
        self.action_label='stand'
        self.current_rule=self.get_standard_pose_points('rule6.csv')
        self.reset_best_score()
        self.show_pattern_img("./patterns/stand.jpg")

    
    def refresh(self):
        self.label_12.setText("")


    def load_img(self, filename):
        img = cv2.imdecode(np.fromfile(filename, dtype=np.uint8),-1)
        return img


    def select_folder(self, plainTextEdit_n):
        def select_folder_with_text_edit(self):
            folder_path = QFileDialog.getExistingDirectory()
            plainTextEdit_n.setPlainText(folder_path)
        return select_folder_with_text_edit 


    def get_body_points(self, pose_results):
        return_value = None               
        
        if pose_results is not None:            
            pose = pose_results[0]['keypoints'][...,:2]
            return pose.tolist()
        
        return return_value


    def get_pose_results(self, img):
        return_value = None
        
        mmdet_results = inference_detector(g_body_detector, img)
        # keep the person class bounding boxes.
        #person_results = process_mmdet_results(mmdet_results, 1)
        rects = process_mmdet_results(mmdet_results, 1)
               
        if len(rects) > 0:
            pose_results, returned_outputs = inference_top_down_pose_model(
            g_body_aligner,
            img,
            rects,
            #bbox_thr=0.3
            bbox_thr=0.66, #!!!important-lucy  
            format='xyxy',
            dataset=dataset,
            return_heatmap=False,
            outputs=None)        
            
            if len(pose_results) > 0:
                return_value = pose_results            
        
        return return_value


    def get_signal(self, val):
        if val[:5] == "start":
            self.label_12.setText("提取第{0}个文件夹中......".format(int(val[5:])+1))
        elif val == "finish":
            self.label_12.setText("提取完毕。")
            self.pushButton_training.setEnabled(True)
            self.pushButton_test_realtime.setEnabled(True)
        elif val == "no_image":
            self.label_12.setText("未找到合适的图片，请重试。")
        elif val == "done":
            X, y = self.combine()
            
            from sklearn.linear_model import LogisticRegression
            clf = LogisticRegression(solver='lbfgs', max_iter=10000).fit(X,y)
            # from sklearn.neural_network import MLPClassifier
            # X = X.astype(np.float64)
            # clf = MLPClassifier(solver='sgd', activation='relu',alpha=1e-4,hidden_layer_sizes=(10,10), random_state=1,max_iter=500,verbose=10,learning_rate_init=.1).fit(X, y)
            # from sklearn.ensemble import RandomForestRegressor
            # cls = RandomForestRegressor(random_state=0).fit(X, y)            
            self.model_results = clf  
            self.save('./model'+".lc")
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
        self.saved_csv = ["./feature1.csv", "./feature2.csv", "./feature3.csv","./feature4.csv", "./feature5.csv", "./feature6.csv"] 
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
            self.saved_csv = ["./feature1.csv", "./feature2.csv", "./feature3.csv","./feature4.csv", "./feature5.csv", "./feature6.csv"] 
            self.input_dir_names = ["./gongbu","./mabu","./pubu","./chongquan","./titui","./stand"]
            print("inp", self.input_dir_names)
            features, label_data = [], []
            for each in range(len(self.saved_csv)):
                feature = read_csv(self.saved_csv[each])
                
                features.append(feature)
                _, dir_name = os.path.split(self.input_dir_names[each])                
                label_data += [dir_name]*np.shape(feature)[0]
                
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
    

    def draw_arrow(self,image, position, direction):
        triangles=([(0,0), (0,-8), (-20,-4)],       #0,l
                    [(0,0), (8,0), (4,-20)],        #1,u
                    [(0,0), (0,8), (20,4)],         #2,r
                    [(0,0), (8,0), (4,20)],         #3,d
                    [(-3,3), (3,-3), (-14,-14)],    #4,lu
                    [(-3,-3), (3,3), (-14, 14)],    #5,ld
                    [(3,3), (-3,-3), (14, -14)],    #6,ru
                    [(3,-3), (-3,3), (14, 14)] )    #7,rd
        
        triangle_cnt = np.array( triangles[direction] )
        cv2.drawContours(image, [triangle_cnt], 0, (255,255,255), -1,offset = position)


    #基于规则的评分        
    def evaluate_by_rule(self, pose, rule, frame): 
        score=0

        if len(pose) >= 0: 
            points_res = normalize_keypoint(pose) 
            average=np.array(rule)
            for i in range(5,17):                
                x=0
                y=0        
                if abs(points_res[i][0]-average[0][i])<7:
                    score += 1
                else:                                        
                    if(points_res[i][0]-average[0][i])<0:
                        x=-1 #L                        
                    else:
                        x = 1 #R
                if abs(points_res[i][1]-average[1][i])<7:
                    score=score+1
                else:
                    if(points_res[i][1]-average[1][i])<0:
                        y = -1  #U
                    else:
                        y = 1   #D
                        #L,R,U,D分别表示坐标相较于标准坐标偏左、右、上、下                        

                arrow_direction = -1
                if x == -1 and y == 0: arrow_direction = 2
                if x == -1 and y == -1: arrow_direction = 7
                if x == -1 and y == 1: arrow_direction = 6
                if x == 0 and y == -1: arrow_direction = 3
                if x == 0 and y == 1: arrow_direction = 1
                if x == 1 and y == -1: arrow_direction = 5
                if x == 1 and y == 0: arrow_direction = 0
                if x == 1 and y == 1: arrow_direction = 4
                
                if arrow_direction != -1:
                    self.draw_arrow(frame,(int(pose[i][0]),int(pose[i][1])),arrow_direction)

            score= score/24*100 #计算得分

        return score, frame


    #基于机器学习的分类
    def classify_by_ml(self,pose, model):
        pose = normalize_keypoint(pose)
        pose = pose.reshape(1, -1)
        
        result = model.predict(pose)
        if self.action_label != result[0]:
            eval(f'self.{result[0]}()')

        return result


    def evaluate_image(self, model, img, show_best_snapshot = True):
        return_value = 0

        show_img = copy.deepcopy(img)
        
        pose_results = self.get_pose_results(img)      
        if pose_results is None:
            cv2.imshow('Action_Evaluation',show_img)
            cv2.waitKey(1)
            cv2.moveWindow('Action_Evaluation', 700,250)
            return_value = 0
        else:
            # show the results
            show_img = vis_pose_result(
                g_body_aligner,
                img,
                pose_results,
                dataset=dataset,
                kpt_score_thr=0.3,
                radius=4,
                thickness=1,
                show=False)

            pose = self.get_body_points(pose_results)
            
            self.classify_by_ml(pose, model)  #基于机器学习的分类
            score, show_img = self.evaluate_by_rule(pose, self.current_rule, show_img)     #基于规则的评分
            
            show_img=resize_frame(show_img)
            cv2.rectangle(show_img,(0,0),(200,80),(64,64,64),-1)
            show_img=draw_text(show_img,f'动作:{self.action_forms[self.action_label]}',(10,10),color=(255,255,255),font_size=25)
            show_img=draw_text(show_img,f'评分:{int(score)}',(10,50),color=(255,255,255),font_size=25)
            cv2.imshow('Action_Evaluation',show_img)
            cv2.waitKey(1)
            cv2.moveWindow('Action_Evaluation', 700,250)

            if show_best_snapshot == True:
                if score > self.best_score:
                    self.best_score=score
                    best_snapshot=copy.deepcopy(show_img)
                    cv2.rectangle(best_snapshot,(0,0),(200,80),(64,64,64),-1)
                    best_snapshot = draw_text(best_snapshot,f'最高分快照:{self.action_forms[self.action_label]}',(10,10),color=(255,255,255),font_size=25)
                    best_snapshot = draw_text(best_snapshot,f'最高评分:{int(score)}',(10,50),color=(255,255,255),font_size=25)
                    cv2.imshow('Best_Snapshot',best_snapshot)
                    cv2.waitKey(1)
                    cv2.moveWindow('Best_Snapshot', 1300,250)

            return_value = 1    
        return return_value

    
    def test_image(self):
        file_name = QFileDialog.getOpenFileName(self,'选择图片','./examples','Images(*.jpg *.jpeg)')[0]
        try:
            model_path = './model.lc'
            model = joblib.load(model_path)
                        
            if file_name != '':
                self.reset_best_score()
                img = self.load_img(file_name)
                img=resize_frame(img)
                self.evaluate_image(model, img, False)
                
        except Exception as e:
            QMessageBox.critical(QMainWindow, "文件错误", str(e))
    

    def evaluate_video(self, video_source):
        try:            
            model_path = './model.lc'
            model = joblib.load(model_path)
            
            halt_flag = False
            self.reset_best_score()               
            cap = cv2.VideoCapture(video_source)
            while cap.isOpened():
                if halt_flag == True:
                    k = cv2.waitKey(10)
                    if k==ord("h"):
                        halt_flag = not halt_flag
                    continue
                
                flag, img = cap.read()
                if not flag:
                    break                            
                self.evaluate_image(model, img)
                k = cv2.waitKey(10)
                if k==ord("h"):
                    print('h')
                    halt_flag = not halt_flag
                if k == ord("q"):
                    break
            cv2.destroyAllWindows()
        except Exception as e:
            QMessageBox.critical(QMainWindow, "文件错误", str(e))


    def test_video_file(self):
        file_name = QFileDialog.getOpenFileName(self,'选择录像','./examples', 'Video Files(*.mov *.mp4)')[0]
        if file_name != '':            
            self.evaluate_video(file_name)


    def test_realtime_video(self):
            camera_num = self.plainTextEdit_3.toPlainText()
            if camera_num == '':
                camera_num = 0
            else:
                camera_num = int(camera_num)
            self.evaluate_video(camera_num)


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