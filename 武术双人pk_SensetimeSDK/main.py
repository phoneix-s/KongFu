from PIL.Image import new
from numpy import true_divide
from class_coding.coding_API import *
from detect_coding.detect_API import *
import sys
import copy
import math
from animation import PkAnimator, SCREEN_WIDTH, SCREEN_HEIGHT, sound_ko
import time

font = cv2.FONT_HERSHEY_SIMPLEX
STDHEIGHT = 600

score_list = [0, 3, 2, 1, 1]


def normalize_keypoint(points):
    points_res = np.array(points)
    min_h = min(points_res[:, 0])
    min_w = min(points_res[:, 1])
    body_h = max(points_res[:, 0]) - min_h
    body_w = max(points_res[:, 1]) - min_w
    points_res = np.ceil((points_res - np.array([min_h, min_w])) / np.array([body_h, body_w]) * np.array([100, 100]))
    return points_res


def check_body_status(pose):
    status = 0  # stand
    if pose[6][1] < pose[0][1]:
        status = 5  # left hand up
    elif pose[7][1] < pose[0][1]:
        status = 6  # left hand up
    elif pose[12][1] < 95:
        status = 1  # left leg up
    elif pose[13][1] < 95:
        status = 2  # right leg up
    elif pose[6][1] < 15:
        status = 3  # left arm up
    elif pose[7][1] < 15:
        status = 4  # right arm up
    else:
        status = 0

    return status


def resize_frame(frame):
    resize_rate = STDHEIGHT / frame.shape[0]
    resized_frame = cv2.resize(frame, (int(frame.shape[1] * resize_rate), STDHEIGHT))
    return resized_frame


def render_number(frame, pose_type):
    textSize = 2
    textThick = 3
    color = (0, 255, 0)
    return cv2.putText(frame, str(pose_type), (30, 80), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, textSize, color, textThick)


empty_playground = np.ones((SCREEN_HEIGHT, SCREEN_WIDTH, 3), np.uint8)
cv2.imshow("Playground", empty_playground)

pk_animator = PkAnimator('background/back_pk.jpg')
pk_animator.set_player1_animation(6)  # animation_standby
pk_animator.set_player2_animation(6)
pk_animator.run()

cam = Camera(200000)
for frame in cam:
    # t_start=time.time()
    # if pk_animator.game_start_flag == True: 
    #     pk_animator.set_player1_animation(0)
    #     pk_animator.set_player2_animation(0)
    # else:
    #     pk_animator.set_player1_animation(6)
    #     pk_animator.set_player2_animation(6)

    if pk_animator.player1_ready == False:
        pk_animator.set_player1_animation(6)
    else:
        pk_animator.set_player1_animation(0)

    if pk_animator.player2_ready == False:
        pk_animator.set_player2_animation(6)
    else:
        pk_animator.set_player2_animation(0)

    color = (255, 255, 255)
    bg_color = (100, 33, 3)
    frame = resize_frame(frame)
    image = frame
    img_shape = image.shape  # Image Shape：return[rows，columns]
    img_height = img_shape[0]  # height（rows）
    img_width = img_shape[1]  # width（columns）

    a = 0  # x start
    b = int(img_height)  # x end
    c = 0  # y start
    d = int(img_width / 2)  # y end
    crop_img_1 = image[a:b, c:d]  # Crop Image

    a = 0
    b = int(img_height)
    c = int(img_width / 2)
    d = int(img_width)
    crop_img_2 = image[a:b, c:d]

    frame = crop_img_1
    frame = resize_frame(frame)
    rect = detect_body(frame)
    new_body_status = 0
    if rect[0] >= 0:
        pose = align_body(frame, rect)
        if len(pose) > 0:
            frame = render_body_points(frame, pose)
            pose = normalize_keypoint(pose)
            new_body_status = check_body_status(pose)
            if pk_animator.game_start_flag == True:
                if new_body_status != 5 and new_body_status != 6:
                    pk_animator.set_player1_animation(new_body_status)
                    pk_animator.score1 += score_list[new_body_status]
                    if pk_animator.score1 > 100:
                        pk_animator.score1 = 100
                    pk_animator.health1 = 100 - pk_animator.score2
                    pk_animator.health2 = 100 - pk_animator.score1
            else:
                if new_body_status == 5 or new_body_status == 6:
                    pk_animator.player1_ready = True
                    pk_animator.set_player1_animation(0)
                if pk_animator.player1_ready == True and pk_animator.player2_ready == True:
                    pk_animator.game_start_flag = True
                    pk_animator.winner_text = ''
                    pk_animator.show_help(False)

    frame = cv2.resize(frame, (300, 450))
    cv2.imshow("Player1", frame)
    cv2.moveWindow('Player1', 0, SCREEN_HEIGHT - 480)
    # player1_win_rect=cv2.getWindowImageRect('Player1')
    # cv2.moveWindow('Player1',0,SCREEN_HEIGHT-player1_win_rect[3])

    #########################################################

    frame2 = crop_img_2
    frame2 = resize_frame(frame2)
    rect2 = detect_body(frame2)  # rect的四个值为左上右下的坐标
    new_body_status = 0
    if rect2[0] >= 0:
        pose2 = align_body(frame2, rect2)
        if len(pose2) > 0:
            frame2 = render_body_points(frame2, pose2)
            pose2 = normalize_keypoint(pose2)
            new_body_status = check_body_status(pose2)
            if pk_animator.game_start_flag == True:
                if new_body_status != 5 and new_body_status != 6:
                    pk_animator.set_player2_animation(new_body_status)
                    pk_animator.score2 += score_list[new_body_status]
                    if pk_animator.score2 > 100:
                        pk_animator.score2 = 100
                    pk_animator.health1 = 100 - pk_animator.score2
                    pk_animator.health2 = 100 - pk_animator.score1
            else:
                if new_body_status == 5 or new_body_status == 6:
                    pk_animator.player2_ready = True
                    pk_animator.set_player2_animation(0)
                if pk_animator.player1_ready == True and pk_animator.player2_ready == True:
                    pk_animator.game_start_flag = True
                    pk_animator.winner_text = ''
                    pk_animator.show_help(False)

    frame2 = cv2.resize(frame2, (300, 450))
    cv2.imshow("Player2", frame2)
    cv2.moveWindow('Player2', SCREEN_WIDTH - 300, SCREEN_HEIGHT - 480)

    if pk_animator.health1 <= 0:
        pk_animator.winner_text = 'Player2 Wins!!!'
        sound_ko.play()
        pk_animator.initialize()
    if pk_animator.health2 <= 0:
        pk_animator.winner_text = 'Player1 Wins!!!'
        sound_ko.play()
        pk_animator.initialize()

    # print(time.time()-t_start)
    k = cv2.waitKey(1)
    if k == ord("q") or k == 27:  # 27:"Esc"
        pk_animator.exit_flag = True
        break

sys.exit()
