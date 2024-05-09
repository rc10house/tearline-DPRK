from ultralytics import YOLO
import os
import sys


model = YOLO('yolov8l-obb.pt')

results = model.train(data='./dota.yaml',
        resume=False,
        epochs=300,
        imgsz=1024,
        batch=4,
        device=[0,1],
        project="./",
        name='yolov8_obb2',
        save=True,
        save_period=10,
        cache=True,
        )
