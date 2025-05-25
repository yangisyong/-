from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
model = load_model('手语数据训练参数.h5')

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


# 动态生成颜色
def generate_colors(num_actions):
    """
    根据动作数量动态生成颜色。
    每种颜色在 RGB 空间中均匀分布。
    """
    import random
    random.seed(42)
    colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(num_actions)]
    return colors


# 使用 PIL 渲染中文的 prob_viz 函数
def prob_viz(res, actions, input_frame, colors=None, font_path="C:/Windows/Fonts/simhei.ttf", font_size=20):
    """
    可视化概率和动作，支持中文。
    参数:
    - res: 每个动作的预测概率 (如 [0.8, 0.1, 0.1])。
    - actions: 动作列表 (如 ["跑步", "跳跃", "站立"])。
    - input_frame: OpenCV 图像帧。
    - colors: 动作对应的颜色列表 (可选)。
    - font_path: 字体路径，用于支持中文渲染。
    - font_size: 字体大小。
    """
    output_frame = input_frame.copy()

    # 确保 colors 和 actions 数量一致
    num_actions = len(actions)
    if colors is None or len(colors) < num_actions:
        colors = generate_colors(num_actions)

    # 转换为 PIL 图像
    img_pil = Image.fromarray(cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    font = ImageFont.truetype(font_path, font_size)

    # 绘制概率条和动作名称
    for num, prob in enumerate(res):
        # 绘制矩形表示概率
        start_x, start_y = 0, 60 + num * 40
        end_x = int(prob * 300)
        end_y = 90 + num * 40
        cv2.rectangle(output_frame, (start_x, start_y), (end_x, end_y), colors[num], -1)

        # 使用 PIL 绘制中文
        draw.text((10, end_y + 5), f"{actions[num]} ({prob:.2f})", font=font, fill=(255, 255, 255))

    # 转回 OpenCV 格式
    output_frame = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    return output_frame

#动作检测
#动作检测与其他计算机视觉任务的一个区别是，检测使用的是数据序列，而不是单个帧。
# Path for exported data, numpy arrays
DATA_PATH = os.path.join('手语数据集')  # 用正确的引号
# Actions that we try to detect
actions = np.array(['你好', '谢谢', '对不起','可以','帮助','玩'])  # 用正确的引号和括号
# Thirty videos worth of data
no_sequences = 65
# Videos are going to be 30 frames in length
sequence_length = 40
# 1. New detection variables
sequence = []
sentence = []
threshold = 0.90
from PIL import ImageFont, ImageDraw, Image

# 读取字体

font_path = "C:/Windows/Fonts/simhei.ttf"  # 替换为实际字体路径
font = ImageFont.truetype(font_path, 32)  # 设置字体和大小
cap = cv2.VideoCapture(0)
# Set mediapipe model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():

        # Read feed
        ret, frame = cap.read()

        # Make detections
        image, results = mediapipe_detection(frame, holistic)
        print(results)

        # Draw landmarks
        draw_styled_landmarks(image, results)

        # 2. Prediction logic
        keypoints = extract_keypoints(results)
        #         sequence.insert(0,keypoints)
        #         sequence = sequence[:30]
        sequence.append(keypoints)
        sequence = sequence[-40:]

        if len(sequence) == 40:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            print(actions[np.argmax(res)])

            # 3. Viz logic
            if res[np.argmax(res)] > threshold:
                if len(sentence) > 0:
                    if actions[np.argmax(res)] != sentence[-1]:
                        sentence.append(actions[np.argmax(res)])
                else:
                    sentence.append(actions[np.argmax(res)])

            if len(sentence) > 5:
                sentence = sentence[-5:]
            # 显示概率可视化
            image = prob_viz(res, actions, image)
            # 使用 PIL 绘制中文文本
            img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # 转换为 PIL 格式
            draw = ImageDraw.Draw(img_pil)

            # 绘制中文和英文文本
            if sentence:
                draw.text((120, 200), '实时测试', font=font, fill=(0, 255, 0))
                draw.text((15, 12), f'动作: {sentence[-1]}', font=font, fill=(0, 0, 255))
                draw.text((3, 30), ' '.join(sentence), font=font, fill=(255, 255, 255))
                # 转回为 OpenCV 格式
            image = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

        # Show to screen
        cv2.imshow('OpenCV Feed', image)

        # Break gracefully
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
