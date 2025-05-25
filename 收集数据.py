import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp
mp_holistic =mp.solutions.holistic #holistic model
mp_drawing =mp.solutions.drawing_utils # Drawing utilities

def mediapipe_detection(image,model):
    image =cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image.flags.writeable=False
    results =model.process(image)
    image.flags.writeable =True
    image =cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
    return image,results
def draw_landmarks(image,results):
    mp_drawing.draw_landmarks(image,results.face_landmarks,mp_holistic.FACEMESH_TESSELATION)
    mp_drawing.draw_landmarks(image,results.pose_landmarks,mp_holistic.POSE_CONNECTIONS)
    mp_drawing.draw_landmarks(image,results.left_hand_landmarks,mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(image,results.right_hand_landmarks,mp_holistic.HAND_CONNECTIONS)
def draw_styled_landmarks(image, results):
    # 绘制面部连接
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
                              mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                              mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1))

    # 绘制姿势连接
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2),
                              mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4))

    # 绘制左手连接
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=2),
                              mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2))

    # 绘制右手连接|
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                              mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1))
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in
                     results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(
        33 * 4)  # 33个关键点，每个点有x, y, z, visibility

    face = np.array([[res.x, res.y, res.z] for res in
                     results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(
        1404)  # 假设有468个面部关键点，每个点有x, y, z

    lh = np.array([[res.x, res.y, res.z] for res in
                   results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(
        21 * 3)  # 21个手部关键点，每个点有x, y, z

    rh = np.array([[res.x, res.y, res.z] for res in
                   results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(
        21 * 3)  # 21个手部关键点，每个点有x, y, z


def extract_keypoints(results):  # 提取特征
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in
                     results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(
        33 * 4)  # 33个关键点，每个点有x, y, z, visibility

    face = np.array([[res.x, res.y, res.z] for res in
                     results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(
        1404)  # 假设有468个面部关键点，每个点有x, y, z

    lh = np.array([[res.x, res.y, res.z] for res in
                   results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(
        21 * 3)  # 21个手部关键点，每个点有x, y, z

    rh = np.array([[res.x, res.y, res.z] for res in
                   results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(
        21 * 3)  # 21个手部关键点，每个点有x, y, z
    return np.concatenate([pose, face, lh, rh])

#动作检测
#动作检测与其他计算机视觉任务的一个区别是，检测使用的是数据序列，而不是单个帧。
# Path for exported data, numpy arrays
DATA_PATH = os.path.join('手语数据集')  # 用正确的引号
# Actions that we try to detect
actions = np.array(['你好', '谢谢', '对不起','可以','帮助','玩'])  # 用正确的引号和括号
# Thirty videos worth of data
no_sequences = 65
# Videos are going to be 30 frames in length   要收集数据就改变no_sequences，或者添加动作
sequence_length = 40
for action in actions:
    for sequence in range(no_sequences):
        try:
            # 创建子文件夹
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
        except:
            pass  # 如果文件夹已经存在，则跳过创建

from PIL import ImageFont, ImageDraw, Image

# 读取字体

font = ImageFont.truetype("C:/Windows/Fonts/simhei.ttf", 30)  # 请使用支持中文的字体
cap = cv2.VideoCapture(0)
# Set mediapipe model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    # NEW LOOP
    # Loop through actions
    for action in actions:

        for sequence in range(no_sequences):
            sequence_path = os.path.join(DATA_PATH, action, str(sequence))

            if os.path.exists(sequence_path):
                print(f"跳过现有序列: {action}/{sequence}")
                continue  # 跳到下一个序列
            os.makedirs(sequence_path, exist_ok=True)  # 创建文件夹，如果已存在则不报错
            # 检查序列是否已存在
            sequence_path = os.path.join(DATA_PATH, action, str(sequence))

            # 循环遍历视频长度（序列长度）
            for frame_num in range(sequence_length):

                # Read feed
                ret, frame = cap.read()

                # Make detections
                image, results = mediapipe_detection(frame, holistic)
                #                 print(results)

                # Draw landmarks
                draw_styled_landmarks(image, results)

                # NEW Apply wait logic
                if frame_num == 0:
                    # 转换为PIL图像来使用中文
                    img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                    draw = ImageDraw.Draw(img_pil)
                    draw.text((120, 200), '开始采集', font=font, fill=(0, 255, 0))
                    draw.text((15, 12), f'收集 {action} 视频编号 {sequence}', font=font, fill=(0, 0, 255))

                    # 转回为OpenCV格式
                    image = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

                    cv2.imshow('OpenCV Feed', image)
                    cv2.waitKey(2000)
                else:
                    # 转换为PIL图像来使用中文
                    img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                    draw = ImageDraw.Draw(img_pil)
                    draw.text((15, 12), f'收集 {action} 视频编号 {sequence}', font=font, fill=(0, 0, 255))

                    # 转回为OpenCV格式
                    image = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

                    cv2.imshow('OpenCV Feed', image)

                # NEW Export keypoints
                keypoints = extract_keypoints(results)
                npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                np.save(npy_path, keypoints)

                # Break gracefully
                # Pause functionality
                key = cv2.waitKey(10) & 0xFF
                if key == ord('q'):  # 'q' to quit
                    break
                elif key == ord('p'):  # 'p' to pause
                    cv2.putText(image, 'PAUSED', (120, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
                    cv2.imshow('OpenCV Feed', image)
                    while True:
                        if cv2.waitKey(1) & 0xFF == ord('p'):  # Press 'p' again to resume
                            break

    cap.release()
    cv2.destroyAllWindows()

