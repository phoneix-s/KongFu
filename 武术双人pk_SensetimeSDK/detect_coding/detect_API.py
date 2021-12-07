import cv2
from SenseTime.SDK import *

pose_avg = [[7.0229276895943595, 7.146384479717816, -44.13756613756613, 52.73897707231041, -57.45855379188712,
            64.64021164021163, -69.43386243386244, 65.81305114638447, -28.03880070546738, 24.232804232804227,
            -34.07583774250442, 23.232804232804234, -36.16225749559083, 24.479717813051145, -279.5511463844798,
            -187.09435626102288, -150.64991181657845, -152.48941798941797, -67.06966490299824, -66.09435626102294,
            13.955026455026456, 12.424162257495592, 11.22663139329806, 13.300705467372135, 151.43650793650795,
            155.2760141093475, 270.18959435626095, 275.1402116402116],
           [4.251846546261848, 8.605307518279524, -22.528276563614636, 41.70146890787468, -65.46874354194887,
            72.23550985257687, -108.38466461944856, 75.57251585889341, -17.488998070459555, 24.80504755757673,
            -31.0822184094426, 28.020997400732583, -43.62130523061175, 33.38151279333031, -188.76748611004277,
            -116.57701913170851, -99.44901701351017, -104.88949320398636, -76.92827067548521, -82.01645435014325,
            -72.2738181203936, -78.67152560457548, 33.35343395114967, 36.94572653828762, 137.42969919427296,
            142.04021458687063, 231.89421164611718, 237.9097982931471],
           [-0.7866071428571433, 1.713392857142857, -34.92410714285713, 36.363392857142856, -39.39910714285715,
            47.00089285714287, -17.624107142857138, 26.53839285714286, -21.97410714285714, 18.85089285714286,
            -23.074107142857137, 23.97589285714286, -26.41160714285714, 9.750892857142855, -161.3803571428572,
            -100.51785714285715, -84.60535714285717, -82.51785714285715, -32.817857142857136, -31.95535714285713,
            -33.905357142857135, -24.780357142857138, 20.182142857142857, 23.93214285714286, 90.15714285714283,
            91.00714285714284, 164.4946428571428, 162.7071428571428]]

CLASSNUM = 3
STDHEIGHT = 600

body_aligner = BodyAligner()
body_detector = BodyDetector()
face_aligner = FaceAligner()
face_detector = FaceDetector()

def minuslist(a, b):
    c = []
    for i in range(len(a)):
        c.append(abs(a[i] - b[i]))
    return c


def resize_frame(frame):
    resize_rate = STDHEIGHT / frame.shape[0]
    resized_frame = cv2.resize(frame,(int(frame.shape[1]*resize_rate),STDHEIGHT))
    return resized_frame


def centralize(points):
    x_mean = 0
    y_mean = 0
    for p in points:
        x_mean += p[0]
        y_mean += p[1]
    x_mean = x_mean / len(points)
    y_mean = y_mean / len(points)
    centered_points = []
    for p in points:
        centered_points.append([p[0] - x_mean, p[1] - y_mean])

    return centered_points


def get_extend_feature(points):
    x_points = []
    y_points = []
    for k in points:
        x_points.append(k[0])
        y_points.append(k[1])
    format_points = x_points + y_points

    exs = []
    for i in range(CLASSNUM):
        c = minuslist(format_points, pose_avg[i])
        exs.append(sum(c) / len(c))

    return exs



def render_number(frame, pose_type):
    textSize = 2
    textThick = 3
    color = (0, 255, 0)
    return cv2.putText(frame, str(pose_type), (30, 80), cv2.FONT_HERSHEY_SIMPLEX, textSize, color, textThick)

def detect_face(frame):
    rects = face_detector.detect(frame)
    if len(rects) >= 1:
        return rects[0]
    else:
        return [-1,-1,-1,-1]

def align_face(frame,rect):
    points = face_aligner.align(frame,rect)
    return points

def render_face_rect(frame,rect):
    rect_frame = face_detector.render(frame, rect)
    return rect_frame

def render_face_points(frame,points):
    points_frame = face_aligner.render(frame, points)
    return points_frame

def detect_body(frame):
    rects =  body_detector.detect(frame)
    if len(rects) >= 1:
        return rects[0]
    else:
        return [-1, -1, -1, -1]

def align_body(frame, rect):
    points = body_aligner.align(frame, rect)
    return points

def render_body_rect(frame,rect):
    rect_frame = body_detector.render(frame, rect)
    return rect_frame

def render_body_points(frame,points):
    points_frame = body_aligner.render(frame, points)
    return points_frame