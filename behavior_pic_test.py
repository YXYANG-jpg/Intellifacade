#api $env:ROBOFLOW_API_KEY="PelbWk51UTuksJHp1jox"
#调用& E:\python\python.exe c:/Users/DELL/Desktop/Prograss/SEU/yolo/behavior_pic_test.py
from inference import get_model
import supervision as sv
import cv2

# define the image url to use for inference
image_file =r"C:\Users\DELL\Desktop\Prograss\SEU\yolo\unnamed.jpg"
image = cv2.imread(image_file)

# load a pre-trained yolov8n model
model = get_model(model_id="skeleton_new-zayvf/2")

# run inference on our chosen image, image can be a url, a numpy array, a PIL image, etc.
results = model.infer(image)[0]

# load the results into the supervision Detections api
detections = sv.Detections.from_inference(results)

if len(detections) > 0:
        # 打印当前帧检测到的对象总数
        print(f"\n--- 帧检测结果 ({len(detections)} 个对象) ---")

        # 尝试从 results 对象中安全获取类别名称
        class_names = getattr(results, 'class_names', None) 

        for i in range(len(detections)):
            xyxy = detections.xyxy[i]
            confidence = detections.confidence[i]
            class_id = detections.class_id[i]

            # 使用类别名称（如果可用），否则使用 ID
            name = class_names[class_id] if class_names and class_id < len(class_names) else f"ID:{class_id}"

            # 打印详细结果
            print(f"  对象 {i+1}: 类别: {name}, 置信度: {confidence:.2f}, 坐标(x1,y1,x2,y2): [{xyxy[0]:.0f}, {xyxy[1]:.0f}, {xyxy[2]:.0f}, {xyxy[3]:.0f}]")

# create supervision annotators
bounding_box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

# annotate the image with our inference results
annotated_image = bounding_box_annotator.annotate(
    scene=image, detections=detections)
annotated_image = label_annotator.annotate(
    scene=annotated_image, detections=detections)

# display the image
sv.plot_image(annotated_image)