import cv2
from ultralytics import YOLO
from yeelight import Bulb


# 加载YOLOv8模型
model = YOLO('D:/dachuang/ultralytics/runs/detect/train/weights/best.pt')
cap = cv2.VideoCapture(1)

bulb = Bulb("192.168.43.56")
# 遍历视频帧
FLAG = 0
while cap.isOpened():
    # 从视频中读取一帧
    success, frame = cap.read()
    bulb.turn_on()
    if success:
        # 在该帧上运行YOLOv8推理
        results = model.predict(frame)
        if results: # 确保results不为空
            boxes = results[0].boxes.cpu().numpy()
            loc, scores, classes = [], [], []

            for box in boxes:
                loc.append(box.xyxy[0].tolist())
                scores.append(float(box.conf))
                classes.append(results[0].names[int(box.cls)])
            # 遍历每个检测到的边界框

            if classes:
                if (classes[0] == "motobicycle") and (scores[0] >= 0.7) and FLAG == 0:
                    #print(1)
                    bulb.turn_on()
                    FLAG = 1
                if (classes[0] != "motobicycle") and FLAG == 1:
                    bulb.turn_off()
                    FLAG = 0


            # 在帧上可视化结果
            annotated_frame = results[0].plot()



        # 显示带注释的帧
        cv2.imshow("YOLOv8推理", annotated_frame)

        # 如果按下'q'则中断循环
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # 如果视频结束则中断循环
        break

# 释放视频捕获对象并关闭显示窗口
cap.release()
cv2.destroyAllWindows()