from ultralytics import YOLO
import os

# load model
model = YOLO('./best.pt')

image_paths = []
paths = os.listdir("../splits")
for p in paths:
    image_paths.append(os.path.join("../splits", p))

results = model.predict(image_paths,
                        imgsz=2048,
                        classes=[1, 2, 3],
                        save=True,
                        show_labels=True,
                        show_boxes=True,
                        retina_masks=True,
                        augment=True,
                        conf=0.01,
                        agnostic_nms=True,
                        visualize=True
                        )


# process results list
for result in results:
    boxes = result.boxes
    masks = result.masks
    keypoints = result.keypoints
    probs = result.probs
