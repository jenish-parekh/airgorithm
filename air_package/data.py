# Description: This script cleans the raw data by removing files with no labels, files containing '20220424' in the filename, and bounding boxes smaller than the defined threshold.

# Package data.py
from google.cloud import storage
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import time
import os
import pandas as pd
import cv2
import shutil


# define variables & path
IMAGE_HEIGHT = 1024
IMAGE_WEIDTH = 1240
RESIZE_RATIO = 0.625
NEW_IMAGE_SIZE = (int(IMAGE_HEIGHT * RESIZE_RATIO), int(IMAGE_WEIDTH * RESIZE_RATIO))
BB_SMALL_SIZE = 0.0025
PATH_PROJECT = '/home/jenish/code/jenish-parekh/airgorithm'
IMAGE_PATH = os.path.join(PATH_PROJECT, 'raw_data/train/images/')
LABEL_PATH = os.path.join(PATH_PROJECT, 'raw_data/train/labels/')
PREPROCESSED_PATH = os.path.join(PATH_PROJECT, "preprocessed_data")
DATASETS = ['train', 'valid', 'test']

def copy_raw_data():
    """Copies raw_data to preprocessed_data before applying transformations."""
    if os.path.exists(PREPROCESSED_PATH):
        print("⚠️ preprocessed_data already exists. Removing old copy...")
        shutil.rmtree(PREPROCESSED_PATH)  # Supprime l'ancienne version pour éviter les conflits

    print("📂 Copying raw_data to preprocessed_data...")
    shutil.copytree(os.path.join(PATH_PROJECT, "raw_data"), PREPROCESSED_PATH)
    print("✅ raw_data copied to preprocessed_data.")

# drop the files with 'None' damage type
def drop_no_label():
    """
    Automatically deletes image and label files with no content
    in the train, val, and test datasets
    """
    for dataset_type in DATASETS:
        IMAGE_PATH = os.path.join(PREPROCESSED_PATH, f'{dataset_type}/images/')
        LABEL_PATH = os.path.join(PREPROCESSED_PATH, f'{dataset_type}/labels/')

        # Vérifier que les dossiers existent avant de continuer
        if not os.path.exists(LABEL_PATH) or not os.path.exists(IMAGE_PATH):
            print(f"⚠️ Dossier {dataset_type} introuvable. Vérifiez le chemin.")
            continue

        files = os.listdir(LABEL_PATH)
        for file in files:
            label_file = os.path.join(LABEL_PATH, file)
            image_file = os.path.join(IMAGE_PATH, file.replace('.txt', '.jpg'))

            with open(label_file, 'r') as f:
                lines = f.readlines()
                if len(lines) == 0:  # Fichier label vide
                    os.remove(label_file)
                    if os.path.exists(image_file):
                        os.remove(image_file)

        print(f"✅ files with 'None' damage type cleaned.")


# drop the files in image_path and label_path that contains string 20220424

def drop_file_20220424():
    files = os.listdir(LABEL_PATH)
    for file in files:
        if '20220424' in file:
            os.remove(os.path.join(LABEL_PATH, file))
            os.remove(os.path.join(IMAGE_PATH, file.replace('.txt', '.jpg')))
    print("✅ files containing '20220424' cleaned")

def drop_file_20220424():
    for dataset_type in DATASETS:
        IMAGE_PATH = os.path.join(PREPROCESSED_PATH, f'{dataset_type}/images/')
        LABEL_PATH = os.path.join(PREPROCESSED_PATH, f'{dataset_type}/labels/')

        if not os.path.exists(LABEL_PATH) or not os.path.exists(IMAGE_PATH):
            print(f"⚠️ Dossier {dataset_type} introuvable. Vérifiez le chemin.")
            continue

        files = os.listdir(LABEL_PATH)
        for file in files:
            if '20220424' in file:
                os.remove(os.path.join(LABEL_PATH, file))
                image_file = os.path.join(IMAGE_PATH, file.replace('.txt', '.jpg'))
                if os.path.exists(image_file):
                    os.remove(image_file)

        print(f"✅ Files containing '20220424' cleaned in {dataset_type}.")

# create a list of bounding boxes
def get_bounding_boxes():
    bounding_boxes = []

    for dataset_type in DATASETS:
        LABEL_PATH = os.path.join(PREPROCESSED_PATH, f'{dataset_type}/labels/')

        if not os.path.exists(LABEL_PATH):
            print(f"⚠️ Dossier {dataset_type} introuvable. Vérifiez le chemin.")
            continue

        files = os.listdir(LABEL_PATH)

        for file in files:
            with open(os.path.join(LABEL_PATH, file), 'r') as f:
                lines = f.readlines()

                if len(lines) == 0:
                    bounding_boxes.append([dataset_type, file, 'none', 0, 0, 0, 0])
                else:
                    for line in lines:
                        parts = line.split()
                        image_name = file
                        damage_type = parts[0]
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        bounding_boxes.append([dataset_type, image_name, damage_type, x_center, y_center, width, height])

    print(f"✅ Bounding boses extraction completed. {len(bounding_boxes)} objets trouvés.")
    return bounding_boxes

# Create a pandas dataframe from the bounding boxes

def create_dataframe(img_path=IMAGE_PATH):
    bounding_boxes = get_bounding_boxes()
    bounding_boxes_df = pd.DataFrame(bounding_boxes, columns=['label_name', 'damage_type', 'x_center', 'y_center', 'width', 'height'])
    bounding_boxes_df['area'] = bounding_boxes_df['width'] * bounding_boxes_df['height']
    bounding_boxes_df['image_path'] = img_path + bounding_boxes_df['label_name'].str.replace('.txt', '.jpg')
    return bounding_boxes_df

def create_dataframe():
    bounding_boxes = get_bounding_boxes()

    # Créer un DataFrame avec toutes les bounding boxes
    bounding_boxes_df = pd.DataFrame(bounding_boxes, columns=['dataset', 'label_name', 'damage_type', 'x_center', 'y_center', 'width', 'height'])

    # Calculer l'aire des bounding boxes
    bounding_boxes_df['area'] = bounding_boxes_df['width'] * bounding_boxes_df['height']

    # Générer les chemins d'image en fonction du dataset
    bounding_boxes_df['image_path'] = bounding_boxes_df.apply(
        lambda row: os.path.join(PATH_PROJECT, f'raw_data/{row["dataset"]}/images/', row['label_name'].replace('.txt', '.jpg')),
        axis=1
    )

    print(f"✅ DataFrame created with {len(bounding_boxes_df)} labels issued from {len(bounding_boxes_df['dataset'].unique())} datasets.")
    return bounding_boxes_df


# drop the files with area less than bb_small_size

def drop_small_bb():
    """Remove images and labels files if bounding boxes smaller than the defined threshold."""
    df = create_dataframe()
    small_bb = df[df['area'] < BB_SMALL_SIZE]
    for index, row in small_bb.iterrows():
        try:
            image_name = row['label_name'].replace('.txt', '.jpg')  # Correctly construct image name
            image_path = os.path.join(IMAGE_PATH, image_name)
            label_path = os.path.join(LABEL_PATH, row['label_name'])

            os.remove(image_path)
            os.remove(label_path)
        except FileNotFoundError:
            print(f"Warning: Could not find file: {image_path} or {label_path}")
        except Exception as e:
            print(f"An error occurred while processing {image_path} or {label_path}: {e}")
    print("✅ images with small bounding boxes cleaned")

def drop_small_bb():
    """Remove images and labels files if bounding boxes are smaller than the defined threshold for all dataset splits."""
    df = create_dataframe()
    small_bb = df[df['area'] < BB_SMALL_SIZE]

    for index, row in small_bb.iterrows():
        dataset_type = row['dataset']
        image_name = row['label_name'].replace('.txt', '.jpg')
        image_path = os.path.join(PATH_PROJECT, f'raw_data/{dataset_type}/images/', image_name)
        label_path = os.path.join(PATH_PROJECT, f'raw_data/{dataset_type}/labels/', row['label_name'])

        try:
            if os.path.exists(image_path):
                os.remove(image_path)
            if os.path.exists(label_path):
                os.remove(label_path)
        except FileNotFoundError:
            print(f"Warning: Could not find file: {image_path} or {label_path}")
        except Exception as e:
            print(f"An error occurred while processing {image_path} or {label_path}: {e}")

    print(f"✅ Images with small bounding boxes cleaned from {len(small_bb['dataset'].unique())} datasets.")

# Resize images

def resize_image():
    """Resize all images in train, val, and test datasets to the defined resolution."""
    for dataset_type in DATASETS:
        IMAGE_PATH = os.path.join(PREPROCESSED_PATH, f'{dataset_type}/images/')

        if not os.path.exists(IMAGE_PATH):
            print(f"⚠️ Dossier {dataset_type} introuvable. Vérifiez le chemin.")
            continue

        for image in os.listdir(IMAGE_PATH):
            image_path = os.path.join(IMAGE_PATH, image)

            try:
                img = cv2.imread(image_path)
                if img is None:
                    print(f"⚠️ Impossible de lire l'image: {image_path}")
                    continue

                img_resized = cv2.resize(img, NEW_IMAGE_SIZE)
                cv2.imwrite(image_path, img_resized)
            except Exception as e:
                print(f"❌ Erreur lors du redimensionnement de {image_path}: {e}")

        print(f"✅ Images resized in {dataset_type}.")



##############################################
## - Use only if you need to check the data ##
##############################################

# Check the image size of first 10 images

def check_image_size():
    """Check and display the size of the first 10 images in each dataset (train, val, test)."""
    for dataset_type in DATASETS:
        IMAGE_PATH = os.path.join(PREPROCESSED_PATH, f'{dataset_type}/images/')

        if not os.path.exists(IMAGE_PATH):
            print(f"⚠️ Dossier {dataset_type} introuvable. Vérifiez le chemin.")
            continue

        images = os.listdir(IMAGE_PATH)[:10]  # Prendre seulement les 10 premières images

        if not images:
            print(f"⚠️ Aucun fichier image trouvé dans {dataset_type}.")
            continue

        print(f"\n📏 Vérification des tailles des images dans {dataset_type}:")

        for image in images:
            image_path = os.path.join(IMAGE_PATH, image)
            img = cv2.imread(image_path)

            if img is None:
                print(f"⚠️ Impossible de lire l'image: {image_path}")
                continue

            print(f"{image}: {img.shape}")

    print("\n✅ Images size verified.")

##############################################
## - Use only if you need to check the data ##
##############################################

# display the bounding boxes on the first 50 images

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches

def display_bounding_boxes():
    """Display bounding boxes for images from train, val, and test datasets."""
    df = create_dataframe()

    if df.empty:
        print("⚠️ Aucune bounding box trouvée. Vérifiez vos fichiers d'annotations.")
        return

    # Boucle sur chaque dataset
    for dataset_type in DATASETS:
        dataset_df = df[df['dataset'] == dataset_type]

        if dataset_df.empty:
            print(f"⚠️ Aucune bounding box trouvée pour {dataset_type}.")
            continue

        print(f"\n📸 Affichage des bounding boxes pour {dataset_type} (max 50 images)")

        top_50 = dataset_df.head(50)  # Limite à 50 images

        for i, row in top_50.iterrows():
            image_path = row['image_path']

            if not os.path.exists(image_path):
                print(f"⚠️ Image introuvable : {image_path}")
                continue

            try:
                img = mpimg.imread(image_path)
                h, w = img.shape[:2]

                fig, ax = plt.subplots()
                ax.imshow(img)

                rect = patches.Rectangle(
                    ((row['x_center'] - row['width'] / 2) * w, (row['y_center'] - row['height'] / 2) * h),
                    row['width'] * w,
                    row['height'] * h,
                    linewidth=2,
                    edgecolor='r',
                    facecolor='none'
                )
                ax.add_patch(rect)
                ax.set_title(f"{dataset_type} - {row['label_name']} ({row['damage_type']})")

                plt.show()

            except Exception as e:
                print(f"❌ Erreur lors de l'affichage de {image_path}: {e}")

    print("\n✅ Display of bounding boxes completed.")

def main():
    # load_data_from_gcp() dans son propre script
    copy_raw_data()  # Copie les fichiers bruts vers preprocessed_data avant transformation
    drop_no_label()
    drop_file_20220424()
    drop_small_bb()
    resize_image()
    print("✅ Processing completed on preprocessed_data.")

if __name__ == "__main__":
    main()
