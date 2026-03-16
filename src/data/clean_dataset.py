import os
from PIL import Image

def clean_dataset(input_folder):
    removed = 0

    for root, dirs, files in os.walk(input_folder):
        for file in files:
            path = os.path.join(root, file)
            try:
                img = Image.open(path)
                img.verify()
            except:
                os.remove(path)
                removed += 1

    print(f"Imagenes corruptas eliminadas: {removed}")