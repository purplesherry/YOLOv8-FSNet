#import cv2
from ultralytics import YOLO
from yeelight import Bulb

img_path = r'D:/dachuang/ultralytics/assets/test1.jpg'
model = YOLO('D:/dachuang/ultralytics/runs/detect/train/weights/best.pt')

results = model.predict(source=img_path)

# 把tensor转为numpy格式
boxes = results[0].boxes.cpu().numpy()

# 输出模型中有哪些类别
#print(results[0])

# 访问 boxes 属性，它包含了检测到的边界框，对应的类别得分，及对应的类别
loc, scores, classes = [], [], []

# # 遍历每个检测结果
for box in boxes:
    loc.append(box.xyxy[0].tolist())
    scores.append(float(box.conf))
    classes.append(results[0].names[int(box.cls)])


if(classes[0]=="motobicycle"):
    if(scores[0]>=0.7):
        bulb = Bulb("192.168.184.73")
        bulb.turn_off()
