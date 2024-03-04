import os
import cv2

def convert_to_yolo_format(annotation_path, output_dir, image_dir):
    with open(annotation_path, 'r') as f:
        annotations = f.readlines()

    for line in annotations:
        img_path, class_id = line.strip().split('\t')
        img_name = os.path.basename(img_path)
        img_id = img_name.split('.')[0]
        img = cv2.imread(os.path.join(image_dir, img_name))

        output_path = os.path.join(output_dir, img_id + ".txt")

        with open(output_path, 'w') as out_file:
            # Write class ID
            out_file.write(f"{class_id}\n")

            # Convert bounding box coordinates to YOLO format
            # In this example, assume there is only one object in the image
            # If multiple objects are present, iterate over them and write their coordinates
            # This is going the be using the whole image as the bounding box.
            x_center = 0.5
            y_center = 0.5
            box_width = 0.75
            box_height = 0.75
            out_file.write(f"{x_center} {y_center} {box_width} {box_height}\n")

annotation_path = '64x64_15cm_24px-exc_v5-marg-32_expanded\toronto_train_label.txt'
output_dir = 'output/yolov8_annotations_padded'
image_dir = '64x64_15cm_24px-exc_v5-marg-32_expanded\Toronto_ISPRS\train'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

convert_to_yolo_format(annotation_path, output_dir, image_dir)