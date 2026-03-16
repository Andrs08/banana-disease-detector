from clean_dataset import clean_dataset
from preprocess_images import preprocess_images
from split_dataset import split_dataset


RAW_FOLDER = "data/raw"
PROCESSED_FOLDER = "data/processed"
DATASET_FOLDER = "data/dataset"


def run_pipeline():

    print("Paso 1: limpieza de dataset")
    clean_dataset(RAW_FOLDER)

    print("Paso 2: preprocesamiento")
    preprocess_images(RAW_FOLDER, PROCESSED_FOLDER)

    print("Paso 3: dividir dataset")
    split_dataset(PROCESSED_FOLDER, DATASET_FOLDER)


if __name__ == "__main__":
    run_pipeline()