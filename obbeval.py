from ultralytics import YOLO

model = YOLO('best.pt')

results = model.predict('./AI_Counting2_3_30_21.jpg',
        device=0,
        imgsz=(4874, 3528),
        classes=[0],
        save=True,
        show_labels=True,
        retina_masks=True,
        augment=True,
        conf=0.3,
        agnostic_nms=True,
        line_width=4
        )

total = sum(r.__len__() for r in results)
print("detections: " + str(total))

for r in results:
    print(r.__len__())
