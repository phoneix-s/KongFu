from utils.coding_API import *
import sys

from animation import PkAnimator, SCREEN_WIDTH, SCREEN_HEIGHT, sound_ko
from mmdet.apis import init_detector, inference_detector
from mmpose.apis import init_pose_model, inference_top_down_pose_model, process_mmdet_results, vis_pose_result

device = 'cuda:0'

config_file = 'configs/yolox_tiny_8x8_300e_coco.py'
checkpoint_file = 'checkpoints/yolox_tiny_8x8_300e_coco_20210806_234250-4ff3b67e.pth'
body_detector = init_detector(config_file, checkpoint_file, device=device)

config = 'configs/hrnet_w32_coco_256x192.py'
checkpoint = 'checkpoints/hrnet_w32_coco_256x192-c78dce93_20200708.pth'
body_aligner = init_pose_model(config, checkpoint, device=device)

dataset = body_aligner.cfg.data['test']['type']

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


def render_body_points(frame, pose):
    for i in pose:
        cv2.circle(frame, (int(i[0]), int(i[1])), 2, (0,0,255))
    return frame


def main():
    empty_playground = np.ones((SCREEN_HEIGHT, SCREEN_WIDTH, 3), np.uint8)
    cv2.imshow("Playground", empty_playground)

    pk_animator = PkAnimator('background/back_pk.jpg')
    pk_animator.set_player1_animation(6)  # animation_standby
    pk_animator.set_player2_animation(6)
    # pk_animator.run()

    cam = Camera(200000)
    for frame in cam:

        if not pk_animator.player1_ready:
            pk_animator.set_player1_animation(6)
        else:
            pk_animator.set_player1_animation(0)

        pk_animator.play_animation_1_step()

        if not pk_animator.player2_ready:
            pk_animator.set_player2_animation(6)
        else:
            pk_animator.set_player2_animation(0)

        pk_animator.play_animation_2_step()

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
        mmmdet_result = inference_detector(body_detector, frame)
        rects = process_mmdet_results(mmmdet_result, 1)
        rect = rects[0]['bbox'][:4] if len(rects) >= 1 else []
        rect = np.array(rect, dtype=int)
        new_body_status = 0

        if len(rect) > 0:
            pose_result, output = inference_top_down_pose_model(body_aligner, frame, rects, bbox_thr=0.66, format='xyxy',
                                                                    dataset=dataset)
            if len(pose_result) > 0:
                pose = pose_result[0]['keypoints'][:, :2]
                frame = render_body_points(frame, pose)
                pose = normalize_keypoint(pose)

                new_body_status = check_body_status(pose)
                if pk_animator.game_start_flag:
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

                pk_animator.play_animation_1_step()

        frame = cv2.resize(frame, (300, 450))
        cv2.imshow("Player1", frame)
        cv2.moveWindow('Player1', 0, SCREEN_HEIGHT - 480)

        #########################################################

        frame2 = crop_img_2
        frame2 = resize_frame(frame2)

        mmmdet_result2 = inference_detector(body_detector, frame2)
        rects2 = process_mmdet_results(mmmdet_result2, 1)

        if len(rects2) > 0:
            rect2 = rects2[0]['bbox'][:4] if len(rects2) >= 1 else [-1, -1, -1, -1]
            rect2 = np.array(rect2, dtype=int)

        new_body_status = 0
        if rect2[0] >= 0:
            pose_result2, output = inference_top_down_pose_model(body_aligner, frame2, rects2, bbox_thr=0.66, format='xyxy',
                                                                dataset=dataset)

            if len(pose_result2) > 0:
                pose2 = pose_result2[0]['keypoints'][:, :2]
                frame2 = render_body_points(frame2, pose2)
                pose2 = normalize_keypoint(pose2)

                new_body_status = check_body_status(pose2)
                if pk_animator.game_start_flag:
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

                pk_animator.play_animation_2_step()

        frame2 = cv2.resize(frame2, (300, 450))
        imshow("Player2", frame2)
        cv2.moveWindow('Player2', SCREEN_WIDTH - 300, SCREEN_HEIGHT - 480)

        if pk_animator.health1 <= 0:
            pk_animator.winner_text = 'Player2 Wins!!!'
            sound_ko.play()
            pk_animator.initialize()
        if pk_animator.health2 <= 0:
            pk_animator.winner_text = 'Player1 Wins!!!'
            sound_ko.play()
            pk_animator.initialize()

        k = cv2.waitKey(1)
        if k == ord("q") or k == 27:  # 27:"Esc"
            pk_animator.exit_flag = True
            break

    sys.exit()


if __name__ == "__main__":
    main()
