#api $env:ROBOFLOW_API_KEY="PelbWk51UTuksJHp1jox"
#调用& E:\python\python.exe c:/Users/DELL/Desktop/Prograss/SEU/yolo/behavior_video_test.py
from inference import get_model
import supervision as sv
import cv2

MODEL_ID = "skeleton_new-zayvf/2"
model = get_model(model_id=MODEL_ID)

cap = cv2.VideoCapture(0)  # 0: 默认摄像头
# 定义目标分辨率
TARGET_WIDTH = 1280
TARGET_HEIGHT = 720

# 尝试设置宽度 (CAP_PROP_FRAME_WIDTH 对应属性 3)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, TARGET_WIDTH)

# 尝试设置高度 (CAP_PROP_FRAME_HEIGHT 对应属性 4)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, TARGET_HEIGHT)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

bounding_box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

# --- 3. 主循环：逐帧处理 ---
while cap.isOpened():
    # 逐帧读取
    success, frame = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue
        

    results = model.infer(frame)[0]

    detections = sv.Detections.from_inference(results)


    # --- 5. 标注图像 ---
    annotated_frame = bounding_box_annotator.annotate(
        scene=frame, detections=detections)
    annotated_frame = label_annotator.annotate(
        scene=annotated_frame, detections=detections)

    # --- 6. 显示实时结果 ---
    cv2.imshow("Roboflow Real-Time Inference", annotated_frame)
    
    # 按 'q' 或 ESC 键退出循环
    if cv2.waitKey(1) & 0xFF == ord('q') or cv2.waitKey(1) & 0xFF == 27:
        break

# --- 7. 清理资源 ---
cap.release()
cv2.destroyAllWindows()