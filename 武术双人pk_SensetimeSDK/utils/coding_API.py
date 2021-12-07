import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import numpy as np
import time
import os
import cv2
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import joblib
import wave
import sklearn

from SenseTime.SDK import *

def mask(frame,color,x0 = 0,x1 = 1):
    if frame is None:
        return None
    color_f = np.ones(frame.shape,np.uint8)
    mask = cv2.cvtColor(cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY),cv2.COLOR_GRAY2RGB)/255
    
    width = frame.shape[1]
    w1 = int(width*x0)
    w2 = int(width*x1)
    
    if color == 'r':
        color_f[:] = [0,0,100]
    elif color == 'g':
        color_f[:] = [0,100,0]
    elif color == 'b':
        color_f[:] = [100,0,0]
    if w1>0:
        color_f[:,:w1,:] = frame[:,:w1,:]
    if w2<width:
        color_f[:,w2:,:] = frame[:,w2:,:]
    
    frame = (frame*mask + color_f*(1-mask)).astype(np.uint8)
    return frame

class Camera(object):
    def __init__(self, frame_count):
        self.capture = cv2.VideoCapture(0,cv2.CAP_DSHOW)
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
        frame = cv2.flip( frame , 1 )
        if not ret:
            raise Exception("未检测到摄像头！")
        if key in [ord('q'), 27, ord('Q')] or self.frame >= self.frame_count:
            self.capture.release()
            cv2.destroyAllWindows()
            raise StopIteration
        self.frame += 1
        return frame

def save_picture(path):
    cap = cv2.VideoCapture(0)  # 打开本地摄像头
    count = 0  # 计数器，用于统计当前拍照个数
    while True:
        ret, frame = cap.read()  # 读取视频帧
        frame = cv2.flip(frame, 1)  # 对视频帧进行水平翻转
        cv2.imshow("picture", frame)  # 显示视频帧
        if cv2.waitKey(1) == ord("s"):  # 当键入s时保存当前图片，
            cv2.imwrite(path + '/{}.png'.format(count), frame)
            print("已经拍摄 {} 张图片！".format(count + 1))
            count += 1
        elif cv2.waitKey(1) == ord("q"):  # 当键入q时，退出拍照
            cap.release()
            cv2.destroyAllWindows()
            break

def scan_folder(path, n=0):
    filelist = list()
    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:
            if n == 0 or (not 'test' in name):
                filelist.append(os.path.join(root, name))
    return filelist


def scan_cat_and_dog(fold1, fold2):
    list1 = scan_folder(fold1)
    list2 = scan_folder(fold2)
    img_list = list1 + list2
    label = [0] * len(list1) + [1] * len(list2)
    return img_list, label


def train_random_forest(feature, label):
    rf = RandomForestClassifier(n_estimators=10)
    rf = rf.fit(feature, label)
    return rf


def get_acc(model, feature, label):
    score = model.score(feature, label)
    return score


def occlude_img(img, index):
    img_c = img.copy()
    h, w, _ = img.shape
    for ind in index:
        ind = ind % 9
        y_i, x_i = ind // 3, ind % 3
        img_c[int(y_i * h / 3):int(y_i * h / 3 + h / 3), int(x_i * w / 3):int(x_i * w / 3 + w / 3), :] = (0, 0, 0)
    return img_c


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


def load_model(flag):
    return joblib.load('./utils/' + flag + '_model.joblib')


def save_model(flag, model):
    return joblib.dump(model, './utils/' + flag + '_model.joblib')


def sound_and_spectrum(filename):
    wavefile = wave.open(filename, 'r')
    str_data = wavefile.readframes(80000)
    wave_data = np.frombuffer(str_data, dtype=np.int16)
    plt.subplot(211)
    plt.plot(wave_data)
    plt.axis('off')
    plt.subplot(212)
    plt.specgram(wave_data, NFFT=1024, Fs=wavefile.getframerate(), noverlap=900)
    plt.axis('off')
    plt.show()


def accuracy(y_test,y_pred):
    acc = accuracy_score(y_test,y_pred)
    return acc

class classify_API():
    def __init__(self):
        self.detector = FaceDetector()
        # self.detector.init()
        self.ft = ImageNetFeatureExtractor()
        # self.ft.init()

    def extract_npimage(self, img):
        feat = self.ft.extract(img)
        return np.array([feat])

    def extract_imageNet(self, img_list):
        if isinstance(img_list, str):
            img_list = [img_list]
        ft = self.ft
        features = list()
        l = len(img_list)
        for i in range(l):
            impath = img_list[i]
            if l > 100:
                bl = int((i + 1) / l * 100)
                finish = int(bl / 3)
                print('>' * finish + ' ' * (33 - finish) + '{}% [{}/{}]'.format(bl, i + 1, l), end='\r')
            frame = cv2.imread(impath)
            feat = ft.extract(frame)
            features.append(feat)
        return np.array(features)

    def recog_img(self, modeltype):
        detector = self.detector
        model = load_model(modeltype)
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        while True:
            ####摄像头读取图像
            ret, frame = cap.read()
            # frame = cv2.resize()
            if not ret:
                break
            feature = self.ft.extract(frame)
            feature = [feature]  # np.array([feature])
            res = model.predict(feature)
            font = cv2.FONT_HERSHEY_SIMPLEX
            if res[0] == 1:
                cv2.putText(frame, 'dog', (0, 100), font, 2, (0, 255, 0), 2)
            if res[0] == 0:
                cv2.putText(frame, 'cat', (0, 100), font, 2, (0, 255, 0), 2)

            cv2.imshow('img', frame)
            k = cv2.waitKey(1)
            if k == ord("q"):
                break
        cap.release()
        cv2.destroyAllWindows()

    def recog_with_face(self, img, modeltype, modeltype2=None):
        detector = self.detector

        model = load_model(modeltype)
        if modeltype2:
            model2 = load_model(modeltype2)
        cap = cv2.VideoCapture(0)
        img = cv2.imread(img)
        ret, frame = cap.read()
        h_f, w_f, _ = frame.shape
        img = cv2.resize(img, (w_f, h_f))
        while True:
            ####摄像头读取图像
            ret, frame = cap.read()
            # frame = cv2.resize()
            if not ret:
                break

            #####人脸处理
            rects = detector.detect(frame)  # 检测人脸位置
            if len(rects) == 0:
                continue
            img_tmp = img.copy()
            rect = rects[0]
            img_tmp[rect[1]:rect[3], rect[0]:rect[2], :] = frame[rect[1]:rect[3], rect[0]:rect[2], :] * 0.7

            feature = self.ft.extract(img_tmp)
            feature = [feature]  # np.array([feature])
            res = model.predict(feature)[0]
            font = cv2.FONT_HERSHEY_SIMPLEX
            if res == 1:
                cv2.putText(img_tmp, 'dog', (0, 100), font, 2, (0, 255, 0), 2)
            if res == 0:
                cv2.putText(img_tmp, 'cat', (0, 100), font, 2, (0, 255, 0), 2)
            if modeltype2:
                res2 = model2.predict(feature)[0]
                if res2 == 1:
                    cv2.putText(img_tmp, 'dog', (0, 200), font, 2, (240, 155, 0), 2)
                if res2 == 0:
                    cv2.putText(img_tmp, 'cat', (0, 200), font, 2, (240, 155, 0), 2)
            cv2.imshow('img', img_tmp)
            k = cv2.waitKey(1)
            if k == ord("q"):
                break
        cap.release()
        cv2.destroyAllWindows()


def write_csv(filename, mat):
    mat = np.array(mat)
    np.savetxt(filename, mat, delimiter=',')


def read_csv(filename):
    df = pd.read_csv(filename, header=None)
    df = df.dropna(1)
    name_list = df.values.tolist()
    if len(name_list[0]) == 1:
        name_list = [y for x in name_list for y in x]
    name_list = np.array(name_list)
    return name_list


def read_csv_with_title(filename):
    print(filename)
    aa = pd.read_csv(filename)
    feature = aa.iloc[:, :-1].values
    label = aa.iloc[:, -1].values
    return feature, label


h = classify_API()


def extract_imageNet(img_list):
    if isinstance(img_list, np.ndarray):
        return h.extract_npimage(img_list)
    else:
        return h.extract_imageNet(img_list)


def recog_with_face(img, modeltype, modeltype2=None):
    return h.recog_with_face(img, modeltype, modeltype2)


def recog_img(modeltype):
    return h.recog_img(modeltype)


def augment_occlusion(folder, superf, block_area=2 / 3):
    if os.path.exists(superf) == False:
        os.makedirs(superf)
    img_list = os.listdir(folder)
    for imgpath in img_list:
        img = cv2.imread(os.path.join(folder, imgpath))
        for i in range(4):
            oindex = list()
            for j in range(9):
                if np.random.rand() > block_area:
                    oindex.append(j)
            oimg = occlude_img(img, oindex)
            cv2.imwrite(os.path.join(superf, str(i) + '_' + imgpath), oimg)


def search_nearest(feature, query_feat):
    cos_m = -1
    id_n = -1
    for i, f in enumerate(feature):
        num = f.dot(query_feat.T)
        denom = np.linalg.norm(f) * np.linalg.norm(query_feat)
        cos = num / denom
        if cos > cos_m:
            cos_m = cos
            id_n = i
    return id_n



class video():
    def __init__(self):
        self.ok = 1
        self.cid = 1
        self.cap = None
        self.w = 0
        self.h = 0

        
    def start_cap(self,i):
        self.cid = i
        self.cap = cv2.VideoCapture(self.cid)
        self.w = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) 
        self.h = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    def end_cap(self):
        self.cap.release()
        cv2.destroyAllWindows()

    def capture_frame(self,pre_time):
        pre_time = int(pre_time)
        w,h = self.w,self.h
        ret,frame = self.cap.read()
        if ret:
            t = 0
            if pre_time==0:
                return frame

            start = time.time()
            while time.time()-start<pre_time:
                t = t+1
                ret,frame = self.cap.read()
                cv2.rectangle(frame, (round(w*0.05),round(h*0.9)), (round(w*(0.05+(time.time()-start)/pre_time/1.1)),round(h*0.95)), (255,255,0),thickness = -1)  # filled
                cv2.imshow('img',frame)
                cv2.waitKey(1)
            else:
                ret,frame = self.cap.read()
            return frame
        else:
            print('摄像头未开启！')
        return None

    def show_frame(self,frame):
        if frame is not None:
            self.ok = 1
        if self.ok:
            k = cv2.waitKey(1)
            cv2.imshow('img',frame)
            if k == ord('q'):
                self.ok = 0
                self.cap.release()
                cv2.destroyAllWindows()


def plot_on_image(img,l,step = 1):
    h,w,_ = img.shape
    img_c = img.copy()
    for i in range(1,len(l)):
        if l[i-1]*l[i]<0:
            continue
        cv2.line(img_c, (step*(i-1), int(l[i-1])), (step*i,int(l[i])), (0, 255, 0), 3)
    return img_c


class Face_API():
    def __init__(self):
        self.detector =  FaceDetector()
        self.aligner = FaceAligner()

    def get_face_coor(self,img):
        x = -1
        y = -1
        h,w,_ = img.shape
        if img is not None:
            rects = self.detector.detect(img)
            if len(rects)>0:
                rect = rects[0]
                x = (rect[2]+rect[0])/2/w
                y = (rect[3]+rect[1])/2/h
        return x,y
    
    def face_angle(self,img):
        rect = self.detector.detect(img)
        if len(rect) == 0:
        	return None
        pts =  np.array(self.aligner.align(img,rect[0]))
        if len(pts) == 0:
        	return None
        l_face = np.mean(pts[5:7,0])
        r_face = np.mean(pts[26:28,0])
        nose = np.mean(pts[45:47,0]) 

        return 35*np.log((nose-l_face)/(r_face-nose))


class Hand_API():
    def __init__(self):
        self.tracker = HandTracker()

    def get_hand_coor(self,img):
        x = -1
        y = -1
        h,w,_ = img.shape
        if img is not None:
            rects = self.tracker.track(img)
            if len(rects)>0:
                rect = rects[0]
                x = (rect[2]+rect[0])/2/w
                y = (rect[3]+rect[1])/2/h
        return x,y

   

v = video()

def start_cap(i=0):
    v.start_cap(i)

def end_cap():
    v.end_cap()


def capture_frame(i=10):
    return v.capture_frame(i)

def show_frame(f):
    v.show_frame(f)

f = Face_API()
def get_face_coor(img):
	return f.get_face_coor(img)

def seghand(image):
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCR_CB)
    (y, cr, cb) = cv2.split(ycrcb)
    cr1 = cv2.GaussianBlur(cr, (5, 5), 0)
    _, skin = cv2.threshold(cr1,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    dst = cv2.bitwise_and(image, image, mask=skin)
    return dst

def face_angle(img):
	return f.face_angle(img)

h = Hand_API()
def get_hand_coor(img):
	return h.get_hand_coor(img)


def imshow(img):
    if len(img.shape) == 3:
        plt.imshow(img[:,:,::-1])
    else:
        plt.imshow(img,'gray')
    plt.axis('off')
    plt.show()


def draw_rect(frame,x,y):
    cv2.line(frame,(x-100,y-100),(x-100,y+100),(0,255,0),3)
    cv2.line(frame,(x+100,y-100),(x+100,y+100),(0,255,0),3)
    cv2.line(frame,(x-100,y-100),(x+100,y-100),(0,255,0),3)
    cv2.line(frame,(x-100,y+100),(x+100,y+100),(0,255,0),3)


def scan_folder(path, n=0):
    filelist = list()
    for root, dirs, files in os.walk(path, topdown=False):
        for name in files:
            if n == 0 or (not 'test' in name):
                filelist.append(os.path.join(root, name))
    return filelist


def scan_cat_and_dog(fold1, fold2):
    list1 = scan_folder(fold1)
    list2 = scan_folder(fold2)
    img_list = list1 + list2
    label = [0] * len(list1) + [1] * len(list2)
    return img_list, label


def train_random_forest(feature, label):
    rf = RandomForestClassifier(n_estimators=10)
    rf = rf.fit(feature, label)
    return rf


def get_acc(model, feature, label):
    score = model.score(feature, label)
    return score


def occlude_img(img, index):
    img_c = img.copy()
    h, w, _ = img.shape
    for ind in index:
        ind = ind % 9
        y_i, x_i = ind // 3, ind % 3
        img_c[int(y_i * h / 3):int(y_i * h / 3 + h / 3), int(x_i * w / 3):int(x_i * w / 3 + w / 3), :] = (0, 0, 0)
    return img_c


def imshow(img, other=''):
    if other == '':
        if len(img.shape) == 3:
            plt.imshow(img[:, :, ::-1])
        else:
            plt.imshow(img, 'gray')
        plt.axis('off')
        plt.show()
    else:
        cv2.imshow(other, img)


def load_model(flag):
    return joblib.load('./utils/' + flag + '_model.joblib')


def save_model(flag, model):
    return joblib.dump(model, './utils/' + flag + '_model.joblib')


def sound_and_spectrum(filename):
    wavefile = wave.open(filename, 'r')
    str_data = wavefile.readframes(80000)
    wave_data = np.frombuffer(str_data, dtype=np.int16)
    plt.subplot(211)
    plt.plot(wave_data)
    plt.axis('off')
    plt.subplot(212)
    plt.specgram(wave_data, NFFT=1024, Fs=wavefile.getframerate(), noverlap=900)
    plt.axis('off')
    plt.show()


class classify_API():
    def __init__(self):
        self.detector = FaceDetector()
        # self.detector.init()
        self.ft = ImageNetFeatureExtractor()
        # self.ft.init()

    def extract_npimage(self, img):
        feat = self.ft.extract(img)
        return np.array([feat])

    def extract_imageNet(self, img_list):
        if isinstance(img_list, str):
            img_list = [img_list]
        ft = self.ft
        features = list()
        l = len(img_list)
        for i in range(l):
            impath = img_list[i]
            if l > 100:
                bl = int((i + 1) / l * 100)
                finish = int(bl / 3)
                print('>' * finish + ' ' * (33 - finish) + '{}% [{}/{}]'.format(bl, i + 1, l), end='\r')
            frame = cv2.imread(impath)
            feat = ft.extract(frame)
            features.append(feat)
        return np.array(features)

    def recog_img(self, modeltype):
        detector = self.detector
        model = load_model(modeltype)
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        while True:
            ####摄像头读取图像
            ret, frame = cap.read()
            # frame = cv2.resize()
            if not ret:
                break
            feature = self.ft.extract(frame)
            feature = [feature]  # np.array([feature])
            res = model.predict(feature)
            font = cv2.FONT_HERSHEY_SIMPLEX
            if res[0] == 1:
                cv2.putText(frame, 'dog', (0, 100), font, 2, (0, 255, 0), 2)
            if res[0] == 0:
                cv2.putText(frame, 'cat', (0, 100), font, 2, (0, 255, 0), 2)

            cv2.imshow('img', frame)
            k = cv2.waitKey(1)
            if k == ord("q"):
                break
        cap.release()
        cv2.destroyAllWindows()

    def recog_with_face(self, img, modeltype, modeltype2=None):
        detector = self.detector

        model = load_model(modeltype)
        if modeltype2:
            model2 = load_model(modeltype2)
        cap = cv2.VideoCapture(0)
        img = cv2.imread(img)
        ret, frame = cap.read()
        h_f, w_f, _ = frame.shape
        img = cv2.resize(img, (w_f, h_f))
        while True:
            ####摄像头读取图像
            ret, frame = cap.read()
            # frame = cv2.resize()
            if not ret:
                break

            #####人脸处理
            rects = detector.detect(frame)  # 检测人脸位置
            if len(rects) == 0:
                continue
            img_tmp = img.copy()
            rect = rects[0]
            img_tmp[rect[1]:rect[3], rect[0]:rect[2], :] = frame[rect[1]:rect[3], rect[0]:rect[2], :] * 0.7

            feature = self.ft.extract(img_tmp)
            feature = [feature]  # np.array([feature])
            res = model.predict(feature)[0]
            font = cv2.FONT_HERSHEY_SIMPLEX
            if res == 1:
                cv2.putText(img_tmp, 'dog', (0, 100), font, 2, (0, 255, 0), 2)
            if res == 0:
                cv2.putText(img_tmp, 'cat', (0, 100), font, 2, (0, 255, 0), 2)
            if modeltype2:
                res2 = model2.predict(feature)[0]
                if res2 == 1:
                    cv2.putText(img_tmp, 'dog', (0, 200), font, 2, (240, 155, 0), 2)
                if res2 == 0:
                    cv2.putText(img_tmp, 'cat', (0, 200), font, 2, (240, 155, 0), 2)
            cv2.imshow('img', img_tmp)
            k = cv2.waitKey(1)
            if k == ord("q"):
                break
        cap.release()
        cv2.destroyAllWindows()


def write_csv(filename, mat):
    mat = np.array(mat)
    np.savetxt(filename, mat, delimiter=',')


def read_csv(filename):
    df = pd.read_csv(filename, header=None)
    df = df.dropna(1)
    name_list = df.values.tolist()
    if len(name_list[0]) == 1:
        name_list = [y for x in name_list for y in x]
    name_list = np.array(name_list)
    return name_list


def read_csv_with_title(filename):
    print(filename)
    aa = pd.read_csv(filename)
    feature = aa.iloc[:, :-1].values
    label = aa.iloc[:, -1].values
    return feature, label


h = classify_API()

def train_test_split(all_data_set,all_data_label,test_size,random_state):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(all_data_set,all_data_label, test_size=test_size, random_state=random_state)
    return x_train, x_test, y_train, y_test

def train_model(x_train,y_train):
    svm_clf = sklearn.svm.SVC(kernel='poly', degree=3)
    svm_clf.fit(x_train, y_train)
    print("训练完成！")
    return svm_clf

def predict(svm_clf,x_test):
    y_pred = svm_clf.predict(x_test)
    return y_pred


def extract_imageNet(img_list):
    if isinstance(img_list, np.ndarray):
        return h.extract_npimage(img_list)
    else:
        return h.extract_imageNet(img_list)


def recog_with_face(img, modeltype, modeltype2=None):
    return h.recog_with_face(img, modeltype, modeltype2)


def recog_img(modeltype):
    return h.recog_img(modeltype)


def augment_occlusion(folder, superf, block_area=2 / 3):
    if os.path.exists(superf) == False:
        os.makedirs(superf)
    img_list = os.listdir(folder)
    for imgpath in img_list:
        img = cv2.imread(os.path.join(folder, imgpath))
        for i in range(4):
            oindex = list()
            for j in range(9):
                if np.random.rand() > block_area:
                    oindex.append(j)
            oimg = occlude_img(img, oindex)
            cv2.imwrite(os.path.join(superf, str(i) + '_' + imgpath), oimg)


def search_nearest(feature, query_feat):
    cos_m = -1
    id_n = -1
    for i, f in enumerate(feature):
        num = f.dot(query_feat.T)
        denom = np.linalg.norm(f) * np.linalg.norm(query_feat)
        cos = num / denom
        if cos > cos_m:
            cos_m = cos
            id_n = i
    return id_n
def points_trans(points,h,w):
    new_points = []
    length = len(points)
    for i in range(length):
        new_points.append([(points[i][0]/h) * (h//3),(points[i][1]/w) * (w//3)])
    return new_points


def pedal(p1, p2, p3):
    """
    过p3作p1和p2相连直线的垂线, 计算垂足的坐标
    直线1：垂足坐标和p3连线
    直线2: p1和p2连线
    两条直线垂直, 且交点为垂足
    :param p1: (x1, y1)
    :param p2: (x2, y2)
    :param p3: (x3, y3)
    :return: 垂足坐标 (x, y)
    """
    outcome = []

    if p2[0] != p1[0]:
        # 根据点x1和x2计算线性方程的k, b
        k, b = np.linalg.solve([[p1[0], 1], [p2[0], 1]], [p1[1], p2[1]])
        # 原理: 垂直向量数量积为0
        x = np.divide(((p2[0] - p1[0]) * p3[0] + (p2[1] - p1[1]) * p3[1] - b * (p2[1] - p1[1])),
                      (p2[0] - p1[0] + k * (p2[1] - p1[1])))
        y = k * x + b

    else:  # 点p1和p2的连线垂直于x轴时
        x = p1[0]
        y = p3[1]

    outcome.append(int(p3[0]))
    outcome.append(int(2 * y - p3[1]))
    return outcome


def add_feature(img, keypoints, path1 = None, path2 = None):
    # 加载特效素材
    h, w = img.shape[:2]
    keypoints = np.array(keypoints).astype('int32')

    if path1:
        whiskers = cv2.imread(path1)
        # 猫胡子特效
        # 确定特效元素的范围
        length = max(1, keypoints[28][0] - keypoints[4][0])  # 获取长度
        whiskers_resized = cv2.resize(whiskers, (min(length, keypoints[88][0] * 2), min(length, keypoints[88][1] * 2)),
                                      interpolation=cv2.INTER_AREA)

        # 找到猫胡子特效的位置
        rows, cols, channels = whiskers_resized.shape
        roi = img[max(0, keypoints[49][1] - int(rows / 2)):min(h, keypoints[49][1] - int(rows / 2) + rows),
              max(0, keypoints[49][0] - int(cols / 2)):min(w, keypoints[49][0] - int(cols / 2) + cols)]

        whiskers_resized2gray = cv2.cvtColor(whiskers_resized, cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(whiskers_resized2gray, 200, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)
        if roi.shape[:2] == mask.shape:
            img_bg = cv2.bitwise_and(roi, roi, mask=mask)
            whiskers_resized_fg = cv2.bitwise_and(whiskers_resized, whiskers_resized, mask=mask_inv)
            dst = cv2.add(img_bg, whiskers_resized_fg)
            img[keypoints[49][1] - int(rows / 2):keypoints[49][1] - int(rows / 2) + rows,
            keypoints[49][0] - int(cols / 2):keypoints[49][0] - int(cols / 2) + cols] = dst

    if path2:
        # 猫耳朵特效
        ear = cv2.imread(path2)
        whis_info = pedal(keypoints[38], keypoints[37], keypoints[46])
        length = max(1, keypoints[42][0] - keypoints[33][0])

        ear_resized = cv2.resize(ear, (min(length, max(1, whis_info[0])), min(length, max(1, whis_info[1] * 2))),
                                 interpolation=cv2.INTER_AREA)
        rows, cols, channels = ear_resized.shape

        roi = img[max(0, whis_info[1] - int(rows / 2)):min(h, whis_info[1] - int(rows / 2) + rows),
              max(0, whis_info[0] - int(cols / 2)):min(w, whis_info[0] - int(cols / 2) + cols)]
        ear_resized2gray = cv2.cvtColor(ear_resized, cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(ear_resized2gray, 200, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)
        if roi.shape[:2] == mask.shape:
            img1_bg = cv2.bitwise_and(roi, roi, mask=mask)
            ear_resized_fg = cv2.bitwise_and(ear_resized, ear_resized, mask=mask_inv)
            dst = cv2.add(img1_bg, ear_resized_fg)
            img[whis_info[1] - int(rows / 2):whis_info[1] - int(rows / 2) + rows,
            whis_info[0] - int(cols / 2):whis_info[0] - int(cols / 2) + cols] = dst

    return img


def full_lips(frame, landmarks):
    # 上嘴唇
    lips1 = landmarks[85:90] + landmarks[97:100]
    lip_hull1 = cv2.convexHull(np.int32(lips1))  # 将嘴唇的关键点绘制成几何图形
    color = (92, 92, 205)  # BRG
    frame = cv2.fillConvexPoly(frame, lip_hull1.astype(np.int), color)  ## 对嘴唇区域填充颜色

    # 下嘴唇
    lips2 = landmarks[91:96] + landmarks[101:104]
    lip_hull2 = cv2.convexHull(np.int32(lips2))
    color = (92, 92, 205)  # BRG
    frame = cv2.fillConvexPoly(frame, lip_hull2.astype(np.int), color)

    return frame