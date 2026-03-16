import os
import cv2

def preprocess_images(input_folder, output_folder, size=(224,224)):

    for root, dirs, files in os.walk(input_folder):
        for file in files:
            input_path = os.path.join(root, file)
            img = cv2.imread(input_path)
            if img is None:
                continue
            img = cv2.resize(img, size)
            class_name = os.path.basename(root)
            save_dir = os.path.join(output_folder, class_name)
            os.makedirs(save_dir, exist_ok=True)
            output_path = os.path.join(save_dir, file)
            cv2.imwrite(output_path, img)