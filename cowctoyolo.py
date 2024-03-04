import os
import cv2

def convert_to_yolo_format(annotation_path, output_dir, image_dir, padding_factor):
    with open(annotation_path, 'r') as f:
        annotations = f.readlines()

    for line in annotations:
        parts = line.strip().split()
        img_path = parts[0]
        img_name = os.path.basename(img_path)
        img_id = img_name.split('.')[0]
        img = cv2.imread(os.path.join(image_dir, img_name))
        img_height, img_width, _ = img.shape

        output_path = os.path.join(output_dir, img_id + ".txt")

        with open(output_path, 'w') as out_file:
            for i in range(1, len(parts), 5):
                x_min = float(parts[i])
                y_min = float(parts[i+1])
                x_max = float(parts[i+2])
                y_max = float(parts[i+3])

                #Padding dimensions: 0.1 = 10% smaller on each side
                width_pad = (x_max - x_min) * padding_factor
                height_pad = (y_max - y_min) * padding_factor

                #Cut given bounding box down to adjust for COWC's bigger grids.
                #Can be adjusted but not sure of doing this better without making new annotations manually
                x_min = max(0, x_min + width_pad)
                y_min = max(0, y_min + height_pad)
                x_max = min(img_width, x_max - width_pad)
                y_max = min(img_height, y_max - height_pad)

                #Convert bounding box to YOLOv8
                x_center = (x_min + x_max) / 2 / img_width
                y_center = (y_min + y_max) / 2 / img_height
                box_width = (x_max - x_min) / img_width
                box_height = (y_max - y_min) / img_height

                #Annotation to file
                class_id = 3 or 1
                out_file.write(f"{class_id} {x_center} {y_center} {box_width} {box_height}\n")

        print(f"Annotation for image {img_id} saved to {output_path}")

annotation_path = '64x64_15cm_24px-exc_v5-marg-32_expanded\toronto_train_label.txt'
output_dir = 'output/yolov8_annotations_padded'
image_dir = '64x64_15cm_24px-exc_v5-marg-32_expanded\Toronto_ISPRS\train'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

convert_to_yolo_format(annotation_path, output_dir, image_dir, padding_factor=0.1)