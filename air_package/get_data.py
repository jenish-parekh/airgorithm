
# Fonction load_data_from_gcp()
from google.cloud import storage
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import time
import os


def load_data_from_gcp(bucket_name="roboflow-airgorithm-dataset",
                       dataset_path="new_splits",
                       local_download_dir=Path("./raw_data")):
    # Initialiser le client Google Cloud Storage
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    # Récupérer la liste des fichiers disponibles
    print("📂 Récupération de la liste des fichiers...")
    blobs = list(bucket.list_blobs(prefix=dataset_path))

    # Créer le dossier de téléchargement si nécessaire
    local_download_dir.mkdir(parents=True, exist_ok=True)

    def download_blob(blob, max_retries=3):
        if blob.name.endswith("/"):
            return None

        relative_path = local_download_dir / blob.name.replace(f"{dataset_path}/", "")

        if relative_path.exists() and relative_path.stat().st_size == blob.size:
            return None

        relative_path.parent.mkdir(parents=True, exist_ok=True)

        for attempt in range(max_retries):
            try:
                blob.download_to_filename(str(relative_path))
                return True
            except Exception as e:
                print(f"⚠️ Erreur téléchargement ({attempt+1}/{max_retries}): {blob.name} - {e}")
                time.sleep(2**attempt)
        return False

    total_files = len(blobs)
    print(f"🚀 Téléchargement de {total_files} fichiers en parallèle...")

    with ThreadPoolExecutor(max_workers=12) as executor:
        futures = {executor.submit(download_blob, blob): blob for blob in blobs}

        with tqdm(total=total_files, desc="Téléchargement en cours", unit="file") as pbar:
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    pbar.update(1)

    print("✅ Téléchargement terminé. Les fichiers sont disponibles dans :", local_download_dir)

load_data_from_gcp()
