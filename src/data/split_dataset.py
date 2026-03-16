import os
import random
import shutil

def split_dataset(source_folder, dest_folder):
    train_ratio = 0.7
    val_ratio = 0.15

    for class_name in os.listdir(source_folder):
        class_path = os.path.join(source_folder, class_name)
        images = os.listdir(class_path)
        random.shuffle(images)
        train_end = int(len(images) * train_ratio)
        val_end = int(len(images) * (train_ratio + val_ratio))

        splits = {
            "train": images[:train_end],
            "val": images[train_end:val_end],
            "test": images[val_end:]
        }

        for split_name, split_images in splits.items():
            for img in split_images:
                src = os.path.join(class_path, img)
                dst = os.path.join(dest_folder, split_name, class_name)
                os.makedirs(dst, exist_ok=True)
                shutil.copy(src, os.path.join(dst, img))