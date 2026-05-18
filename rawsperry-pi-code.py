#!/usr/bin/env python3
"""
树莓派 TCP 服务器 + 摄像头手势识别 - 防干扰版
防止垂直移动被水平移动意外中断
"""

import socket
import sys
import cv2
import numpy as np
import threading
import queue
import time
import select

# ========== 服务器配置 ==========
HOST = '0.0.0.0'          # 监听所有接口
PORT = 8888               # 通信端口

# ========== 手势识别配置 ==========
# HSV红色范围
RED_LOWER1 = np.array([0, 120, 70])
RED_UPPER1 = np.array([10, 255, 255])
RED_LOWER2 = np.array([170, 120, 70])
RED_UPPER2 = np.array([180, 255, 255])

# 识别参数
MOVE_THRESHOLD = 15       # 移动检测阈值（像素）
MIN_CONTOUR_AREA = 500    # 最小轮廓面积
VERTICAL_DURATION = 1.5   # 垂直移动最小持续时间（秒）
STILL_DURATION = 2.0      # 静止不动最小持续时间（秒）
DETECTION_WIDTH = 640     # 摄像头检测宽度
DETECTION_HEIGHT = 480    # 摄像头检测高度
STABILITY_FRAMES = 3      # 稳定性检查帧数

# 冷却时间配置
VERTICAL_COOLDOWN = 3.0   # 垂直移动之间冷却时间
HORIZONTAL_COOLDOWN = 1.0  # 水平移动之间冷却时间
CROSS_COOLDOWN = 2.5      # 交叉移动之间冷却时间
STILL_COOLDOWN = 3.0      # 静止暂停冷却时间

# 水平移动连续触发配置
HORIZONTAL_INTERVAL = 1.0  # 水平移动触发间隔（秒）

# 防干扰配置
LOCK_DURATION = 2.0        # 锁定持续时间（秒）
IGNORE_OTHER_DIRECTION = True  # 是否忽略其他方向移动

class GestureDetector:
    def __init__(self, command_queue):
        self.command_queue = command_queue
        
        # 向下移动检测
        self.prev_center_y_down = None
        self.downward_start_time = None
        self.downward_ongoing = False
        self.downward_history = []
        self.downward_locked = False  # 向下移动锁定
        
        # 向上移动检测
        self.prev_center_y_up = None
        self.upward_start_time = None
        self.upward_ongoing = False
        self.upward_history = []
        self.upward_locked = False    # 向上移动锁定
        
        # 向左移动检测（关闭）
        self.prev_center_x_left = None
        self.leftward_start_time = None
        self.leftward_ongoing = False
        self.leftward_history = []
        self.last_left_trigger = 0
        
        # 向右移动检测（打开）
        self.prev_center_x_right = None
        self.rightward_start_time = None
        self.rightward_ongoing = False
        self.rightward_history = []
        self.last_right_trigger = 0
        
        # 静止检测
        self.prev_center_still = None
        self.still_start_time = None
        self.still_ongoing = False
        
        # 全局锁定状态
        self.global_lock = False
        self.lock_start_time = 0
        self.lock_reason = None  # 锁定原因
        
        # 触发控制
        self.last_trigger_time = 0
        self.last_trigger_cmd = None
        self.running = True
        
    def detect_red_object(self, frame):
        """检测红色物体并返回其中心坐标"""
        # 转换为HSV颜色空间
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # 创建红色掩码
        mask1 = cv2.inRange(hsv, RED_LOWER1, RED_UPPER1)
        mask2 = cv2.inRange(hsv, RED_LOWER2, RED_UPPER2)
        red_mask = cv2.bitwise_or(mask1, mask2)
        
        # 形态学操作去除噪声
        kernel = np.ones((5, 5), np.uint8)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
        
        # 查找轮廓
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None, red_mask
        
        # 找到最大的轮廓
        largest_contour = max(contours, key=cv2.contourArea)
        
        # 检查轮廓面积是否足够大
        if cv2.contourArea(largest_contour) < MIN_CONTOUR_AREA:
            return None, red_mask
        
        # 计算轮廓的中心点
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            center_x = int(M["m10"] / M["m00"])
            center_y = int(M["m01"] / M["m00"])
            return (center_x, center_y), red_mask
        
        return None, red_mask
    
    def check_global_lock(self):
        """检查全局锁定状态"""
        if not self.global_lock:
            return False
        
        current_time = time.time()
        lock_duration = current_time - self.lock_start_time
        
        # 如果锁定时间超过设定值，解除锁定
        if lock_duration >= LOCK_DURATION:
            self.global_lock = False
            self.lock_reason = None
            print(f"🔓 全局锁定解除 (持续 {lock_duration:.1f} 秒)")
            return False
        
        # 仍在锁定中
        return True
    
    def set_global_lock(self, reason):
        """设置全局锁定"""
        if not self.global_lock:
            self.global_lock = True
            self.lock_start_time = time.time()
            self.lock_reason = reason
            print(f"🔒 全局锁定: {reason}")
    
    def get_cooldown_time(self, new_cmd):
        """根据命令类型获取冷却时间"""
        if self.last_trigger_cmd is None:
            return 0
        
        vertical_cmds = ['U', 'D']
        horizontal_cmds = ['C', 'O']
        
        last_is_vertical = self.last_trigger_cmd in vertical_cmds
        last_is_horizontal = self.last_trigger_cmd in horizontal_cmds
        new_is_vertical = new_cmd in vertical_cmds
        new_is_horizontal = new_cmd in horizontal_cmds
        
        # 静止命令
        if new_cmd == 'S' or self.last_trigger_cmd == 'S':
            return STILL_COOLDOWN
        
        # 垂直到垂直
        if last_is_vertical and new_is_vertical:
            return VERTICAL_COOLDOWN
        
        # 水平到水平
        if last_is_horizontal and new_is_horizontal:
            return HORIZONTAL_COOLDOWN
        
        # 交叉情况（垂直↔水平）
        return CROSS_COOLDOWN
    
    def can_trigger(self, cmd):
        """检查是否可以触发命令"""
        current_time = time.time()
        time_since_last = current_time - self.last_trigger_time
        
        # 检查全局锁定
        if self.check_global_lock():
            print(f"⛔ 全局锁定中，无法触发 '{cmd}' 指令")
            return False
        
        # 特殊处理水平移动连续触发
        if cmd in ['C', 'O']:
            # 检查是否正在连续移动
            if (cmd == 'C' and self.leftward_ongoing) or (cmd == 'O' and self.rightward_ongoing):
                # 获取该命令的最后触发时间
                last_specific = self.last_left_trigger if cmd == 'C' else self.last_right_trigger
                time_since_specific = current_time - last_specific
                
                # 如果距离上次特定触发超过1秒，允许触发
                if time_since_specific >= HORIZONTAL_INTERVAL:
                    # 检查是否有垂直移动锁定
                    if (self.downward_ongoing or self.upward_ongoing) and IGNORE_OTHER_DIRECTION:
                        print(f"⏸️  垂直移动中，忽略水平移动 '{cmd}'")
                        return False
                    return True
        
        # 常规冷却检查
        cooldown_needed = self.get_cooldown_time(cmd)
        return time_since_last >= cooldown_needed
    
    def send_command(self, cmd):
        """发送命令并更新状态"""
        current_time = time.time()
        
        if self.can_trigger(cmd):
            self.command_queue.put(cmd)
            self.last_trigger_time = current_time
            self.last_trigger_cmd = cmd
            
            # 更新特定命令的最后触发时间
            if cmd == 'C':
                self.last_left_trigger = current_time
            elif cmd == 'O':
                self.last_right_trigger = current_time
            elif cmd in ['U', 'D']:
                # 垂直移动后设置全局锁定
                self.set_global_lock(f"垂直移动 {cmd}")
            
            print(f"🎯 发送指令: '{cmd}'")
            return True
        
        # 计算剩余冷却时间
        cooldown_needed = self.get_cooldown_time(cmd)
        cooldown_remaining = cooldown_needed - (current_time - self.last_trigger_time)
        
        if cooldown_remaining > 0:
            print(f"⏳ 指令 '{cmd}' 冷却中，剩余 {cooldown_remaining:.1f} 秒")
        
        return False
    
    def detect_downward_movement(self, center_y):
        """检测向下移动并计时"""
        if self.prev_center_y_down is None:
            self.prev_center_y_down = center_y
            return False, 0.0
        
        movement = center_y - self.prev_center_y_down
        self.prev_center_y_down = center_y
        
        # 检查全局锁定和方向锁定
        if self.global_lock and self.lock_reason != "垂直移动 D":
            # 如果被其他方向锁定，忽略移动
            return False, 0.0
        
        self.downward_history.append(movement)
        if len(self.downward_history) > STABILITY_FRAMES * 2:
            self.downward_history.pop(0)
        
        # 检查是否稳定向下移动
        is_stable_downward = False
        if len(self.downward_history) >= STABILITY_FRAMES:
            recent_movements = self.downward_history[-STABILITY_FRAMES:]
            if all(m > MOVE_THRESHOLD for m in recent_movements):
                is_stable_downward = True
        
        if is_stable_downward:
            if not self.downward_ongoing:
                # 检查是否有水平移动在进行
                if (self.leftward_ongoing or self.rightward_ongoing) and IGNORE_OTHER_DIRECTION:
                    print("⏸️  水平移动中，忽略垂直移动检测")
                    return False, 0.0
                
                self.downward_start_time = time.time()
                self.downward_ongoing = True
                print(f"📏 开始检测向下移动...")
            
            duration = time.time() - self.downward_start_time
            return True, duration
        else:
            if self.downward_ongoing:
                print(f"⚠️  向下移动中断")
                self.downward_ongoing = False
                self.downward_start_time = None
            
            return False, 0.0
    
    def detect_upward_movement(self, center_y):
        """检测向上移动并计时"""
        if self.prev_center_y_up is None:
            self.prev_center_y_up = center_y
            return False, 0.0
        
        movement = center_y - self.prev_center_y_up
        self.prev_center_y_up = center_y
        
        # 检查全局锁定和方向锁定
        if self.global_lock and self.lock_reason != "垂直移动 U":
            # 如果被其他方向锁定，忽略移动
            return False, 0.0
        
        self.upward_history.append(movement)
        if len(self.upward_history) > STABILITY_FRAMES * 2:
            self.upward_history.pop(0)
        
        is_stable_upward = False
        if len(self.upward_history) >= STABILITY_FRAMES:
            recent_movements = self.upward_history[-STABILITY_FRAMES:]
            if all(m < -MOVE_THRESHOLD for m in recent_movements):
                is_stable_upward = True
        
        if is_stable_upward:
            if not self.upward_ongoing:
                # 检查是否有水平移动在进行
                if (self.leftward_ongoing or self.rightward_ongoing) and IGNORE_OTHER_DIRECTION:
                    print("⏸️  水平移动中，忽略垂直移动检测")
                    return False, 0.0
                
                self.upward_start_time = time.time()
                self.upward_ongoing = True
                print(f"📏 开始检测向上移动...")
            
            duration = time.time() - self.upward_start_time
            return True, duration
        else:
            if self.upward_ongoing:
                print(f"⚠️  向上移动中断")
                self.upward_ongoing = False
                self.upward_start_time = None
            
            return False, 0.0
    
    def detect_leftward_movement(self, center_x):
        """检测向左移动（关闭）"""
        if self.prev_center_x_left is None:
            self.prev_center_x_left = center_x
            return False, 0.0
        
        movement = self.prev_center_x_left - center_x
        self.prev_center_x_left = center_x
        
        # 检查全局锁定
        if self.global_lock:
            # 如果被锁定，忽略移动
            return False, 0.0
        
        self.leftward_history.append(movement)
        if len(self.leftward_history) > STABILITY_FRAMES * 2:
            self.leftward_history.pop(0)
        
        # 检查是否稳定向左移动
        is_stable_leftward = False
        if len(self.leftward_history) >= STABILITY_FRAMES:
            recent_movements = self.leftward_history[-STABILITY_FRAMES:]
            if all(m > MOVE_THRESHOLD for m in recent_movements):
                is_stable_leftward = True
        
        if is_stable_leftward:
            if not self.leftward_ongoing:
                # 检查是否有垂直移动在进行
                if (self.downward_ongoing or self.upward_ongoing) and IGNORE_OTHER_DIRECTION:
                    print("⏸️  垂直移动中，忽略水平移动检测")
                    return False, 0.0
                
                self.leftward_start_time = time.time()
                self.leftward_ongoing = True
                print(f"📏 开始检测向左移动（连续触发模式）...")
            
            duration = time.time() - self.leftward_start_time
            return True, duration
        else:
            if self.leftward_ongoing:
                print(f"⚠️  向左移动停止")
                self.leftward_ongoing = False
                self.leftward_start_time = None
            
            return False, 0.0
    
    def detect_rightward_movement(self, center_x):
        """检测向右移动（打开）"""
        if self.prev_center_x_right is None:
            self.prev_center_x_right = center_x
            return False, 0.0
        
        movement = center_x - self.prev_center_x_right
        self.prev_center_x_right = center_x
        
        # 检查全局锁定
        if self.global_lock:
            # 如果被锁定，忽略移动
            return False, 0.0
        
        self.rightward_history.append(movement)
        if len(self.rightward_history) > STABILITY_FRAMES * 2:
            self.rightward_history.pop(0)
        
        is_stable_rightward = False
        if len(self.rightward_history) >= STABILITY_FRAMES:
            recent_movements = self.rightward_history[-STABILITY_FRAMES:]
            if all(m > MOVE_THRESHOLD for m in recent_movements):
                is_stable_rightward = True
        
        if is_stable_rightward:
            if not self.rightward_ongoing:
                # 检查是否有垂直移动在进行
                if (self.downward_ongoing or self.upward_ongoing) and IGNORE_OTHER_DIRECTION:
                    print("⏸️  垂直移动中，忽略水平移动检测")
                    return False, 0.0
                
                self.rightward_start_time = time.time()
                self.rightward_ongoing = True
                print(f"📏 开始检测向右移动（连续触发模式）...")
            
            duration = time.time() - self.rightward_start_time
            return True, duration
        else:
            if self.rightward_ongoing:
                print(f"⚠️  向右移动停止")
                self.rightward_ongoing = False
                self.rightward_start_time = None
            
            return False, 0.0
    
    def detect_still_movement(self, center):
        """检测静止不动动作"""
        center_x, center_y = center
        current_time = time.time()
        
        if self.prev_center_still is None:
            self.prev_center_still = (center_x, center_y)
            return False, 0.0
        
        prev_x, prev_y = self.prev_center_still
        distance = np.sqrt((center_x - prev_x)**2 + (center_y - prev_y)**2)
        
        # 检查是否移动（超过阈值）
        if distance > 10:  # 静止检测阈值较小
            self.still_ongoing = False
            self.still_start_time = None
            self.prev_center_still = (center_x, center_y)
            return False, 0.0
        else:
            if not self.still_ongoing:
                self.still_start_time = current_time
                self.still_ongoing = True
                print(f"📏 开始检测静止不动...")
            
            self.prev_center_still = (center_x, center_y)
            
            if self.still_ongoing:
                duration = current_time - self.still_start_time
                return True, duration
            
            return False, 0.0
    
    def handle_continuous_horizontal(self):
        """处理水平移动的连续触发"""
        current_time = time.time()
        
        # 向左移动连续触发
        if self.leftward_ongoing:
            if current_time - self.last_left_trigger >= HORIZONTAL_INTERVAL:
                self.send_command('C')
        
        # 向右移动连续触发
        if self.rightward_ongoing:
            if current_time - self.last_right_trigger >= HORIZONTAL_INTERVAL:
                self.send_command('O')
    
    def run_detection(self):
        """运行手势检测主循环"""
        print("📷 正在启动摄像头...")
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ 无法打开摄像头")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, DETECTION_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, DETECTION_HEIGHT)
        
        print("✅ 摄像头启动成功")
        print("🎯 开始检测红色物体手势...")
        print("   - 持续向下移动1.5秒 → 百叶下行 (D)")
        print("   - 持续向上移动1.5秒 → 百叶上行 (U)")
        print("   - 向左移动 → 百叶关闭，持续移动时每隔1秒触发一次 (C)")
        print("   - 向右移动 → 百叶打开，持续移动时每隔1秒触发一次 (O)")
        print("   - 静止不动2秒 → 百叶暂停 (S)")
        print("   - 防干扰功能: 垂直移动中忽略水平移动，水平移动中忽略垂直移动")
        print("   - 冷却时间策略:")
        print("     * 上下移动之间: 3秒")
        print("     * 左右移动之间: 1秒")
        print("     * 垂直和水平移动之间: 2.5秒")
        
        last_frame_time = time.time()
        frame_count = 0
        fps_start_time = time.time()
        
        while self.running:
            current_time = time.time()
            
            # 处理水平移动的连续触发
            self.handle_continuous_horizontal()
            
            # 读取摄像头帧
            ret, frame = cap.read()
            if not ret:
                # 如果读取失败，等待一段时间后重试
                print("⚠️  读取摄像头帧失败，重试...")
                time.sleep(0.1)
                continue
            
            # 计算实际帧率
            frame_interval = current_time - last_frame_time
            last_frame_time = current_time
            frame_count += 1
            
            # 显示帧率信息
            if frame_count % 30 == 0:
                fps = frame_count / (current_time - fps_start_time)
                print(f"📊 当前帧率: {fps:.1f} FPS")
                frame_count = 0
                fps_start_time = current_time
            
            # 如果帧率过低，跳过处理
            if frame_interval > 0.2:  # 低于5fps
                print(f"⚠️  帧率过低: {1/frame_interval:.1f}fps")
                continue
            
            frame = cv2.flip(frame, 1)
            center, mask = self.detect_red_object(frame)
            
            if center is not None:
                center_x, center_y = center
                
                # 检测各种移动
                is_downward, down_duration = self.detect_downward_movement(center_y)
                is_upward, up_duration = self.detect_upward_movement(center_y)
                is_leftward, left_duration = self.detect_leftward_movement(center_x)
                is_rightward, right_duration = self.detect_rightward_movement(center_x)
                is_still, still_duration = self.detect_still_movement(center)
                
                # 处理垂直移动触发
                if self.downward_ongoing and down_duration >= VERTICAL_DURATION:
                    self.send_command('D')
                    self.downward_ongoing = False
                    self.downward_start_time = None
                
                if self.upward_ongoing and up_duration >= VERTICAL_DURATION:
                    self.send_command('U')
                    self.upward_ongoing = False
                    self.upward_start_time = None
                
                # 处理静止触发
                if self.still_ongoing and still_duration >= STILL_DURATION:
                    self.send_command('S')
                    self.still_ongoing = False
                    self.still_start_time = None
                
                # 绘制界面信息
                cv2.circle(frame, (center_x, center_y), 10, (0, 255, 0), -1)
                cv2.circle(frame, (center_x, center_y), 30, (0, 255, 0), 2)
                cv2.putText(frame, f"Red Object: ({center_x}, {center_y})", 
                           (center_x - 100, center_y - 40), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.6, (0, 255, 0), 2)
                
                y_offset = 30
                
                # 全局锁定状态显示
                if self.global_lock:
                    lock_duration = current_time - self.lock_start_time
                    lock_remaining = max(0, LOCK_DURATION - lock_duration)
                    cv2.putText(frame, f"🔒 全局锁定中: {self.lock_reason} ({lock_remaining:.1f}s)", 
                               (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 100), 2)
                    y_offset += 30
                
                # 冷却时间显示
                time_since_last = current_time - self.last_trigger_time
                if self.last_trigger_cmd:
                    cooldown_needed = self.get_cooldown_time(self.last_trigger_cmd)
                    if time_since_last < cooldown_needed:
                        cooldown_remaining = cooldown_needed - time_since_last
                        cv2.putText(frame, f"Cooldown: {cooldown_remaining:.1f}s", 
                                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                        y_offset += 30
                
                # 状态显示
                if self.downward_ongoing:
                    cv2.putText(frame, f"Downward: {down_duration:.1f}s / {VERTICAL_DURATION}s", 
                               (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                    y_offset += 30
                
                if self.upward_ongoing:
                    cv2.putText(frame, f"Upward: {up_duration:.1f}s / {VERTICAL_DURATION}s", 
                               (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)
                    y_offset += 30
                
                if self.leftward_ongoing:
                    cv2.putText(frame, f"Leftward (Continuous)", 
                               (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 150, 100), 2)
                    y_offset += 30
                
                if self.rightward_ongoing:
                    cv2.putText(frame, f"Rightward (Continuous)", 
                               (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 150, 255), 2)
                    y_offset += 30
                
                if self.still_ongoing:
                    cv2.putText(frame, f"Still: {still_duration:.1f}s / {STILL_DURATION}s", 
                               (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 255), 2)
                    y_offset += 30
                
                # 显示方向箭头
                if is_downward:
                    cv2.arrowedLine(frame, (center_x, center_y - 50), 
                                  (center_x, center_y + 20), (0, 255, 255), 3, tipLength=0.3)
                if is_upward:
                    cv2.arrowedLine(frame, (center_x, center_y + 50), 
                                  (center_x, center_y - 20), (255, 255, 0), 3, tipLength=0.3)
                if is_leftward:
                    cv2.arrowedLine(frame, (center_x + 50, center_y), 
                                  (center_x - 20, center_y), (255, 150, 100), 3, tipLength=0.3)
                if is_rightward:
                    cv2.arrowedLine(frame, (center_x - 50, center_y), 
                                  (center_x + 20, center_y), (100, 150, 255), 3, tipLength=0.3)
            else:
                # 没有检测到红色物体时重置检测状态
                if self.still_ongoing:
                    self.still_ongoing = False
                    self.still_start_time = None
            
            # 显示帧
            cv2.imshow('Gesture Control - Anti-Interference', frame)
            
            # 按'q'退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("🛑 摄像头检测已停止")
                self.running = False
                break
        
        cap.release()
        cv2.destroyAllWindows()
    
    def stop(self):
        """停止检测"""
        self.running = False

def run_server():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    command_queue = queue.Queue()
    
    try:
        server_socket.bind((HOST, PORT))
        server_socket.listen(1)
        print(f"✅ 服务器已启动，正在监听端口 {PORT}...")
        print("📡 等待ESP32连接...")
    except Exception as e:
        print(f"❌ 启动服务器失败: {e}")
        sys.exit(1)
    
    # 启动手势检测线程
    detector = GestureDetector(command_queue)
    detection_thread = threading.Thread(target=detector.run_detection, daemon=True)
    detection_thread.start()
    print("✅ 手势检测线程已启动")
    
    conn, addr = None, None
    try:
        conn, addr = server_socket.accept()
        print(f"✅ ESP32 已连接！客户端地址: {addr}")
        
        print("\n" + "="*60)
        print("智能百叶窗手势控制系统 - 防干扰版")
        print("="*60)
        print("控制方式:")
        print("  1. 手势控制：")
        print("     - 红色物体持续向下移动1.5秒 → 百叶下行 (D)")
        print("     - 红色物体持续向上移动1.5秒 → 百叶上行 (U)")
        print("     - 红色物体向左移动 → 百叶关闭，持续移动时每隔1秒触发一次 (C)")
        print("     - 红色物体向右移动 → 百叶打开，持续移动时每隔1秒触发一次 (O)")
        print("     - 红色物体静止不动2秒 → 百叶暂停 (S)")
        print("  2. 防干扰功能：")
        print("     - 垂直移动中自动忽略水平移动检测")
        print("     - 水平移动中自动忽略垂直移动检测")
        print("     - 垂直移动后锁定2秒防止意外切换")
        print("  3. 冷却时间策略：")
        print("     - 上下移动之间: 3秒")
        print("     - 左右移动之间: 1秒")
        print("     - 垂直和水平移动之间: 2.5秒")
        print("="*60 + "\n")
        
        conn.settimeout(0.5)
        
        while True:
            # 发送队列中的指令
            try:
                while True:
                    gesture_command = command_queue.get_nowait()
                    if gesture_command:
                        print(f"🎯 手势识别触发: 发送指令 '{gesture_command}'")
                        try:
                            conn.sendall(gesture_command.encode())
                            print("  指令已送出")
                        except BrokenPipeError:
                            print("❌ 连接已断开，无法发送指令")
                            break
            except queue.Empty:
                pass
            
            # 接收ESP32的消息
            try:
                data = conn.recv(1024).decode('utf-8').strip()
                if data:
                    print(f"📨 ESP32消息: {data}")
            except socket.timeout:
                pass
            except Exception as e:
                print(f"接收数据异常: {e}")
                break
            
            # 获取用户输入
            try:
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    command = sys.stdin.readline().strip().upper()
                    
                    if not command:
                        continue
                    
                    if command == 'Q':
                        print("正在退出...")
                        if conn:
                            conn.sendall(b'Bye')
                        break
                    elif command in ['D', 'U', 'C', 'O', 'S']:
                        print(f"🚀 手动发送指令 '{command}'...")
                        try:
                            conn.sendall(command.encode())
                            print("  指令已送出")
                        except BrokenPipeError:
                            print("❌ 连接已断开，无法发送指令")
                            break
                    else:
                        print(f"⚠️  未知指令 '{command}'，忽略")
            except (KeyboardInterrupt, EOFError):
                print("\n收到中断信号，准备退出...")
                break
    
    except KeyboardInterrupt:
        print("\n\n🛑 用户中断请求")
    finally:
        detector.stop()
        print("🛑 手势检测已停止")
        
        if conn:
            conn.close()
            print("🔒 连接已关闭")
        server_socket.close()
        print("🛑 服务器已停止")

if __name__ == "__main__":
    # 检查OpenCV是否可用
    try:
        import cv2
        import numpy
        print("✅ OpenCV 和 NumPy 已安装")
    except ImportError:
        print("❌ 缺少依赖库，请运行以下命令安装:")
        print("sudo apt-get update")
        print("sudo apt-get install python3-opencv python3-numpy")
        sys.exit(1)
    
    run_server()