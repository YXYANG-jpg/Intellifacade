import cv2
import mediapipe as mp
import time
import numpy as np

# --- 抑制 MediaPipe 警告日志 (可选) ---
try:
    import logging
    logging.getLogger('mediapipe').setLevel(logging.ERROR)
except:
    pass

# MediaPipe 初始化
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# --- 核心控制参数 ---
CONTROL_MODE_WAIT_TIME = 3.0    # 激活/锁定手势所需的持续时间 (秒)
MOVEMENT_THRESHOLD = 0.004      # 判定为有效抬升/下降的最小归一化Y轴变化 (经验值)
HISTORY_LENGTH = 15             # 用于平滑运动分析的历史帧数

# --- 状态机变量 ---
state = "INACTIVE"  # INACTIVE (未激活), WAITING_ACTIVATION, ACTIVE (已激活), WAITING_CONFIRM
start_time = None           # 用于计时3秒持续时间
last_y_position = []        # 存储手腕关键点(0)的历史Y坐标 (归一化)
current_hand_shape = "NONE" # 当前静态手势 (用于绘制)
thumb_up_start_time = None  # 大拇指点赞向上开始时间
thumb_down_start_time = None # 大拇指点赞向下开始时间

# --- 辅助函数：判断手势形状 ---

def is_open_palm(landmarks):
    """判断是否为激活/请求手势：五指张开"""
    # 检查所有手指尖是否都远离手掌中心
    tips = [4, 8, 12, 16, 20]  # 所有指尖
    palm_center = landmarks[9]  # 手掌中心
    
    distances = []
    for tip in tips:
        tip_point = landmarks[tip]
        distance = np.sqrt((tip_point.x - palm_center.x)**2 + (tip_point.y - palm_center.y)**2)
        distances.append(distance)
    
    # 如果所有距离都大于阈值，则认为是张开手掌
    avg_distance = np.mean(distances)
    return avg_distance > 0.12

def is_ok_sign(landmarks):
    """判断是否为确认/锁定手势：拇指和食指形成圆圈 (O形)"""
    thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP] # 4
    index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP] # 8
    
    # 使用归一化坐标计算距离
    distance = np.sqrt((thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2)
    return distance < 0.05

def is_thumb_up(landmarks):
    """判断是否为大拇指点赞向上手势"""
    # 大拇指点赞：大拇指向上伸直，其他四指握拳
    thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]  # 4
    thumb_ip = landmarks[mp_hands.HandLandmark.THUMB_IP]    # 3
    thumb_mcp = landmarks[mp_hands.HandLandmark.THUMB_MCP]  # 2
    
    # 检查其他四指是否握拳
    index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]  # 8
    middle_tip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]  # 12
    ring_tip = landmarks[mp_hands.HandLandmark.RING_FINGER_TIP]  # 16
    pinky_tip = landmarks[mp_hands.HandLandmark.PINKY_TIP]  # 20
    
    wrist = landmarks[mp_hands.HandLandmark.WRIST]  # 0
    
    # 判断其他四指是否弯曲（握拳状态）
    fingers_bent = (
        index_tip.y > landmarks[mp_hands.HandLandmark.INDEX_FINGER_PIP].y and
        middle_tip.y > landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y and
        ring_tip.y > landmarks[mp_hands.HandLandmark.RING_FINGER_PIP].y and
        pinky_tip.y > landmarks[mp_hands.HandLandmark.PINKY_PIP].y
    )
    
    # 判断大拇指是否向上伸直（大拇指尖在手腕上方）
    thumb_up = thumb_tip.y < wrist.y - 0.05
    
    return fingers_bent and thumb_up

def is_thumb_down(landmarks):
    """判断是否为大拇指向下手势"""
    # 大拇指点赞向下：大拇指向下伸直，其他四指握拳
    thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]  # 4
    index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]  # 8
    middle_tip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]  # 12
    ring_tip = landmarks[mp_hands.HandLandmark.RING_FINGER_TIP]  # 16
    pinky_tip = landmarks[mp_hands.HandLandmark.PINKY_TIP]  # 20
    
    wrist = landmarks[mp_hands.HandLandmark.WRIST]  # 0
    
    # 判断其他四指是否弯曲（握拳状态）
    fingers_bent = (
        index_tip.y > landmarks[mp_hands.HandLandmark.INDEX_FINGER_PIP].y and
        middle_tip.y > landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y and
        ring_tip.y > landmarks[mp_hands.HandLandmark.RING_FINGER_PIP].y and
        pinky_tip.y > landmarks[mp_hands.HandLandmark.PINKY_PIP].y
    )
    
    # 判断大拇指向下伸直（大拇指尖在手腕下方）
    thumb_down = thumb_tip.y > wrist.y + 0.08
    
    return fingers_bent and thumb_down

def is_valid_up_down_gesture(landmarks):
    """判断是否是可以识别上升下降的手势"""
    # 简单检查：只要不是大拇指点赞手势，就允许识别上升下降
    return not (is_thumb_up(landmarks) or is_thumb_down(landmarks))

def get_gesture_type(landmarks):
    """获取手势类型"""
    if is_open_palm(landmarks):
        return "open_palm", "OPEN PALM (激活)"
    elif is_ok_sign(landmarks):
        return "ok_sign", "OK SIGN (锁定)"
    elif is_thumb_up(landmarks):
        return "thumb_up", "THUMB UP (顺时针)"
    elif is_thumb_down(landmarks):
        return "thumb_down", "THUMB DOWN (逆时针)"
    else:
        return "unknown", "错误手势，无法控制百叶"

# --- 主循环 ---

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 60)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue

    image = cv2.flip(image, 1) # 翻转图像
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    results = hands.process(image_rgb)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    hand_detected = False
    
    # 每次循环开始时，重置手势显示
    current_hand_shape = "NONE"
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            hand_detected = True
            mp_drawing.draw_landmarks(image_bgr, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # 获取手腕关键点 (0) 的归一化 Y 坐标
            current_y = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y
            
            # --- 手势识别 ---
            gesture_type, gesture_name = get_gesture_type(hand_landmarks.landmark)
            current_hand_shape = gesture_name
            
            # 状态机逻辑
            current_time = time.time()
            
            if state == "INACTIVE":
                if gesture_type == "open_palm":
                    state = "WAITING_ACTIVATION"
                    start_time = current_time
                    print("-> 计时开始：等待激活手势...")
                
            elif state == "WAITING_ACTIVATION":
                if gesture_type == "open_palm":
                    if current_time - start_time >= CONTROL_MODE_WAIT_TIME:
                        state = "ACTIVE"
                        print("-> 状态切换为 ACTIVE。灯光闪烁，可以进行控制。")
                        last_y_position = [] # 激活后清空历史位置，准备追踪新运动
                        thumb_up_start_time = None
                        thumb_down_start_time = None
                    else:
                        # 显示剩余时间
                        remaining = CONTROL_MODE_WAIT_TIME - (current_time - start_time)
                        print(f"-> 激活手势持续中... ({remaining:.1f}s 剩余)")
                else:
                    state = "INACTIVE"
                    print("-> 激活手势中断，返回 INACTIVE。")

            elif state == "ACTIVE":
                
                # 1. 确认/锁定手势 (ACTIVE 状态下，锁定优先)
                if gesture_type == "ok_sign":
                    state = "WAITING_CONFIRM"
                    start_time = current_time
                    print("-> 计时开始：等待确认手势...")
                    # 重置其他状态
                    last_y_position = []
                    thumb_up_start_time = None
                    thumb_down_start_time = None
                
                # 2. 大拇指点赞向上 - 顺时针旋转
                elif gesture_type == "thumb_up":
                    if thumb_up_start_time is None:
                        thumb_up_start_time = current_time
                        print("-> 检测到大拇指点赞向上，开始计时...")
                    elif current_time - thumb_up_start_time >= CONTROL_MODE_WAIT_TIME:
                        thumb_duration = current_time - thumb_up_start_time
                        print(f"-> 百叶顺时针旋转 (已持续: {thumb_duration:.1f}秒)")
                    # 重置其他状态
                    last_y_position = []
                    thumb_down_start_time = None
                
                # 3. 大拇指点赞向下 - 逆时针旋转
                elif gesture_type == "thumb_down":
                    if thumb_down_start_time is None:
                        thumb_down_start_time = current_time
                        print("-> 检测到大拇指点赞向下，开始计时...")
                    elif current_time - thumb_down_start_time >= CONTROL_MODE_WAIT_TIME:
                        thumb_duration = current_time - thumb_down_start_time
                        print(f"-> 百叶逆时针旋转 (已持续: {thumb_duration:.1f}秒)")
                    # 重置其他状态
                    last_y_position = []
                    thumb_up_start_time = None
                
                # 4. 上升和下降手势 (对于张开手掌或OK手势)
                elif is_valid_up_down_gesture(hand_landmarks.landmark):
                    # 收集Y坐标历史数据
                    last_y_position.append(current_y)
                    if len(last_y_position) > HISTORY_LENGTH:
                        last_y_position.pop(0) 

                    if len(last_y_position) == HISTORY_LENGTH:
                        # 计算运动趋势
                        y_diff_total = last_y_position[-1] - last_y_position[0]
                        
                        if abs(y_diff_total) > MOVEMENT_THRESHOLD:
                            amplitude_norm = abs(y_diff_total) 
                            
                            if y_diff_total < 0:
                                # Y 减小 => 手向上移动 (上升)
                                action = "上升"
                                print(f"[ACTIVE] -> 动作: {action}。变化幅度 (归一化): +{amplitude_norm:.4f}")
                                last_y_position = [] # 动作触发后清空历史
                            else:
                                # Y 增大 => 手向下移动 (下降)
                                action = "下降"
                                print(f"[ACTIVE] -> 动作: {action}。变化幅度 (归一化): -{amplitude_norm:.4f}")
                                last_y_position = [] # 动作触发后清空历史
                    # 重置大拇指手势状态
                    thumb_up_start_time = None
                    thumb_down_start_time = None
                
                # 5. 错误手势处理
                elif gesture_type == "unknown":
                    print("-> 错误手势，无法控制百叶")
                    # 重置所有状态
                    last_y_position = []
                    thumb_up_start_time = None
                    thumb_down_start_time = None

            elif state == "WAITING_CONFIRM":
                if gesture_type == "ok_sign":
                    if current_time - start_time >= CONTROL_MODE_WAIT_TIME:
                        state = "INACTIVE" 
                        print("-> 确认/锁定手势持续 3 秒。百叶停止运动，配置保存！返回 INACTIVE。")
                        # 重置所有状态
                        last_y_position = []
                        thumb_up_start_time = None
                        thumb_down_start_time = None
                    else:
                        # 显示剩余时间
                        remaining = CONTROL_MODE_WAIT_TIME - (current_time - start_time)
                        print(f"-> 确认手势持续中... ({remaining:.1f}s 剩余)")
                else:
                    state = "ACTIVE"
                    print("-> 确认手势中断，返回 ACTIVE。")
            
            break # 只处理检测到的第一只手

    else:
        # 没有检测到手部
        if state == "WAITING_ACTIVATION" or state == "WAITING_CONFIRM":
            state_before = state
            state = "INACTIVE"
            print(f"-> {state_before} 状态下，手部丢失，返回 INACTIVE。")
        elif state == "ACTIVE":
            # 重置所有状态
            if thumb_up_start_time or thumb_down_start_time:
                print(f"-> 手部丢失，大拇指手势结束")
            last_y_position = []
            thumb_up_start_time = None
            thumb_down_start_time = None


    # --- 绘制状态和提示信息 ---
    cv2.putText(image_bgr, f"State: {state}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(image_bgr, f"Gesture: {current_hand_shape}", (10, 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    
    # 绘制计时器
    if state.startswith("WAITING"):
        elapsed = current_time - start_time
        remaining = max(0, CONTROL_MODE_WAIT_TIME - elapsed)
        cv2.putText(image_bgr, f"HOLD TIME: {remaining:.1f}s", (350, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    # 绘制大拇指手势状态
    if thumb_up_start_time:
        thumb_duration = current_time - thumb_up_start_time
        cv2.putText(image_bgr, f"THUMB UP: {thumb_duration:.1f}s", (10, 110), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        if thumb_duration >= CONTROL_MODE_WAIT_TIME:
            cv2.putText(image_bgr, "百叶顺时针旋转", (10, 140), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    
    if thumb_down_start_time:
        thumb_duration = current_time - thumb_down_start_time
        cv2.putText(image_bgr, f"THUMB DOWN: {thumb_duration:.1f}s", (10, 110), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        if thumb_duration >= CONTROL_MODE_WAIT_TIME:
            cv2.putText(image_bgr, "百叶逆时针旋转", (10, 140), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)


    # 显示图像
    cv2.imshow('Mediapipe Hand Controller', image_bgr)
    
    if cv2.waitKey(5) & 0xFF == 27: # 按 ESC 键退出
        break

hands.close()
cap.release()
cv2.destroyAllWindows()