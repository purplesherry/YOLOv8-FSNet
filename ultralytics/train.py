#coding:utf-8
from ultralytics import YOLO

# 加载预训练模型
# 添加注意力机制，SEAtt_yolov8.yaml 默认使用的是n。
# SEAtt_yolov8s.yaml，则使用的是s，模型。
model = YOLO("D:/dachuang/ultralytics/ultralytics/cfg/models/v8/SEAtt_yolov8.yaml").load('yolov8n.pt')

# Use the model
if __name__ == '__main__':
    # Use the model
    results = model.train(data='datasets/TomatoData/data.yaml', epochs=250, batch=4)  # 训练模型
    # 将模型转为onnx格式
    # success = model.export(format='onnx')
