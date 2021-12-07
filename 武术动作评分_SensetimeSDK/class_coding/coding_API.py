import cv2
import matplotlib.pyplot as plt
import numpy as np
import time
import os

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

class Video(object):#!!!
    def __init__(self, filepath, height = 600):
        self.capture = cv2.VideoCapture(filepath)
        if not self.capture.isOpened():
            raise Exception("未能打开视频！")
        self.h = self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.w = self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.setHeight = height
    def get_height(self):
        return self.h
    def get_width(self):
        return self.w
    def __iter__(self):
        return self
    def __next__(self):
        key = cv2.waitKey(1)
        ret, frame = self.capture.read()

        if key in [ord('q'), 27, ord('Q')] or frame is None:
            self.capture.release()
            cv2.destroyAllWindows()
            raise StopIteration
        return cv2.resize(frame, (int(frame.shape[1]*self.setHeight/frame.shape[0]), self.setHeight))


def take_photo(color = 1):
    pre_time = 5
    cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
    ret,frame = cap.read()
    if ret:
        start = time.time()
        w,h,_ = frame.shape
        while time.time()-start<pre_time:
            time_left = int(pre_time - (time.time()-start))
            ret,frame = cap.read()
            if time_left >= 1 : 
                time_left = str(time_left)
                cv2.putText(frame,time_left,(50,150),cv2.FONT_ITALIC,6,(0,0,255),25)
            # cv2.rectangle(frame, (round(w*0.05),round(h*0.9)), (round(w*(0.05+(time.time()-start)/pre_time/1.1)),round(h*0.95)), (255,255,0),thickness = -1)  # filled
            cv2.imshow('img',frame)
            cv2.waitKey(10)
        cap.release()
        cv2.destroyAllWindows()    
        if color == 0:
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)      
        return frame
    else:
        print('摄像头未开启！')
        cap.release()
        cv2.destroyAllWindows()  
        return None


def plot_on_image(img,l,step = 1):
    h,w,_ = img.shape
    img_c = img.copy()
    if np.max(l)<=1 and np.min(l)>=-1:
        l = np.array(l)*h
    for i in range(1,len(l)):
        if l[i-1]*l[i]<0:
            continue
        cv2.line(img_c, (step*(i-1), int(l[i-1])), (step*i,int(l[i])), (0, 255, 0), 3)
    return img_c

def find_max(rects):
    area = -1
    rect_max = None
    for rect in rects:
        area_t = (rect[2]-rect[0])*(rect[3]-rect[1])
        if area_t > area:
            area = area_t
            rect_max = rect
    return rect_max

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
                rect = find_max(rects)
                x = (rect[2]+rect[0])/2/w
                y = (rect[3]+rect[1])/2/h
        return x,y

    def face_angle(self,img):
        rect = self.detector.detect(img)
        if len(rect) == 0:
            return None
        else:
            rect_m = find_max(rect)
            pts =  np.array(self.aligner.align(img,rect_m))
            if len(pts) == 0:
                return None
            l_face = np.mean(pts[5:7,0])
            r_face = np.mean(pts[26:28,0])
            nose = np.mean(pts[45:47,0]) 
        return 35*np.log((nose-l_face)/(r_face-nose)),rect_m


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

   

f = Face_API()
def get_face_coor(img):
    return f.get_face_coor(img)

def face_angle(img):
    return f.face_angle(img)

h = Hand_API()
def get_hand_coor(img):
    return h.get_hand_coor(img)

def imshow(img, other = ''):
    if other == '':
        if len(img.shape) == 3:
            plt.imshow(img[:,:,::-1])
        else:
            plt.imshow(img,'gray')
        plt.axis('off')
        plt.show()
    else:
        cv2.imshow(img, other)
    

def draw_rect(frame,x,y):
    cv2.line(frame,(x-100,y-100),(x-100,y+100),(0,255,0),3)
    cv2.line(frame,(x+100,y-100),(x+100,y+100),(0,255,0),3)
    cv2.line(frame,(x-100,y-100),(x+100,y-100),(0,255,0),3)
    cv2.line(frame,(x-100,y+100),(x+100,y+100),(0,255,0),3)

def imread(frame):
    return cv2.imread(frame)

def line(frame,point1,point2):
    cv2.line(frame,point1,point2,(0,255,0),3)

def find_local_peaks(res):
    ids = []
    for i in range(len(res)):
        if res[i] > 0.8:
            if res[(i-1)%len(res)] < res[i] and res[(i+1)%len(res)] < res[i]:
                ids.append(i)
    return ids

def generate_sensor_signal(totallength):
    signal1 = np.zeros(totallength)
    signal2 = np.zeros(totallength)

    begin1 = int(totallength*0.1)
    end1 = int(totallength*0.4)
    begin2 = int(totallength*0.6)
    end2 = int(totallength*0.9)

    
    for i in range(int(totallength*0.3)):
        signal1[begin1+i] = 1
        signal2[begin2+i] = 1

    signal1 += np.random.rand(totallength)/8
    signal2 += np.random.rand(totallength)/8
    return signal1,signal2



def square_norm(arr):
    full_signal = np.array(arr)
    full_signal = full_signal/np.sqrt(np.sum(full_signal*full_signal))
    return full_signal