import cv2
import math
import os
from PIL import Image
import onnxruntime
import numpy as np

class YOLOv8:
    def __init__(self, model_path):
        self.session = onnxruntime.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]

MODEL_URL = "https://drive.google.com/file/d/19ZhROPCegDeVDFPnlG72GWKGOE0FTJlG/view?usp=sharing"  # REPLACE WITH YOUR ACTUAL URL
model_path = "best.onnx"
if not os.path.exists(model_path):
    print("Downloading model...")
    r = requests.get(MODEL_URL)
    with open(model_path, 'wb') as f:
        f.write(r.content)

# Initialize detector with original variable names
yolov8_detector = YOLOv8(model_path)
classNames = [chr(i) for i in range(65, 91)] 
detection_sign = '' 

def process_detection(img, conf_threshold=0.45):
    try:
        resized_img = cv2.resize(img, (640, 640))
        pil_img = Image.fromarray(cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB))
        detections = yolov8_detector(pil_img, size=640, conf_thres=conf_threshold, iou_thres=0.5)
        
        predicted_sign = None
        for detection in detections:
            bbox = detection['bbox']          
            conf = detection['score']         
            cls = detection['class_id']       
            
            if conf > conf_threshold:
                x1, y1, x2, y2 = map(int, bbox)  
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

                predicted_sign = classNames[cls]
                label = f'{predicted_sign} {conf:.2f}'
                t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
                c2 = x1 + t_size[0], y1 - t_size[1] - 3
                cv2.rectangle(img, (x1, y1), c2, [255, 0, 255], -1, cv2.LINE_AA)
                cv2.putText(img, label, (x1, y1 - 2), 0, 1, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)

        return predicted_sign, img
    except Exception as e:
        print(f"Detection error: {e}")
        return None, img

def image_detection(image_path):
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Image not found")
            
        predicted_sign, processed_img = process_detection(img, 0.45)
        processed_image_path = os.path.join('static/files', 'processed_' + os.path.basename(image_path))
        
        os.makedirs(os.path.dirname(processed_image_path), exist_ok=True)
        cv2.imwrite(processed_image_path, processed_img)
        return predicted_sign, processed_image_path
    except Exception as e:
        print(f"Image processing error: {e}")
        return None, None

def video_detection(video_path):
    try:
        cap = cv2.VideoCapture(video_path)
        global detection_sign
        
        while True:
            success, img = cap.read()
            if not success:
                break
                
            detection_sign, processed_img = process_detection(img, 0.45)
            yield processed_img, detection_sign
            
    except Exception as e:
        print(f"Video processing error: {e}")
    finally:
        if 'cap' in locals():
            cap.release()


'''import cv2
import math
import os
from PIL import Image
from yolo_onnx.yolov8_onnx import YOLOv8 

yolov8_detector = YOLOv8('best.onnx')
classNames = [chr(i) for i in range(65, 91)] 
detection_sign='' 
def process_detection(img, conf_threshold=0.45):
    resized_img = cv2.resize(img, (640, 640))
    pil_img = Image.fromarray(cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB))
    detections = yolov8_detector(pil_img, size=640, conf_thres=conf_threshold, iou_thres=0.5)
    predicted_sign = None
    for detection in detections:
        bbox = detection['bbox']          
        conf = detection['score']         
        cls = detection['class_id']       
        
        if conf > conf_threshold:
            x1, y1, x2, y2 = map(int, bbox)  
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            predicted_sign = classNames[cls]
            label = f'{predicted_sign} {conf:.2f}'
            t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
            c2 = x1 + t_size[0], y1 - t_size[1] - 3
            cv2.rectangle(img, (x1, y1), c2, [255, 0, 255], -1, cv2.LINE_AA)
            cv2.putText(img, label, (x1, y1 - 2), 0, 1, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)

    return predicted_sign, img

def image_detection(image_path):
    img = cv2.imread(image_path)
    predicted_sign, processed_img = process_detection(img, 0.45)
    processed_image_path = os.path.join('static/files', 'processed_' + os.path.basename(image_path))
    cv2.imwrite(processed_image_path, processed_img)
    return predicted_sign, processed_image_path

def video_detection(video_path):
    cap = cv2.VideoCapture(video_path)
    global detection_sign
    while True:
        success, img = cap.read()
        if not success:
            break  # Exit loop if no more frames
        detection_sign, processed_img = process_detection(img, 0.45)
        
        # Yield processed frame and detected sign
        yield processed_img, detection_sign
'''