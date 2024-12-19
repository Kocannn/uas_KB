import os
import numpy as np
import pandas as pd
from PIL import Image
from skimage import measure
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
import cv2
from tqdm import tqdm
import logging
from datetime import datetime

class ImageFeatureExtractor:
    def __init__(self, image_folder, output_file):
        """
        Inisialisasi feature extractor untuk dataset gambar.
        
        Parameters:
        image_folder (str): Path folder yang berisi gambar
        output_file (str): Path file output CSV
        """
        self.image_folder = image_folder
        self.output_file = output_file
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging untuk mencatat proses"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler()
            ]
        )
    
    def extract_features(self, image):
        """
        Mengekstrak fitur dari satu gambar.
        
        Parameters:
        image: Array gambar dalam format grayscale
        
        Returns:
        dict: Dictionary berisi fitur-fitur gambar
        """
        # Threshold gambar
        thresh = threshold_otsu(image)
        binary = image > thresh
        
        # Analisis region
        regions = measure.regionprops(binary.astype(int))
        
        if len(regions) > 0:
            region = regions[0]  # Ambil region terbesar
            
            # Hitung roundness
            roundness = 4 * np.pi * region.area / (region.perimeter ** 2) if region.perimeter > 0 else 0
            
            # Hitung aspect ratio
            aspect_ratio = region.major_axis_length / region.minor_axis_length if region.minor_axis_length > 0 else 0
            
            return {
                'Area': region.area,
                'MajorAxisLength': region.major_axis_length,
                'MinorAxisLength': region.minor_axis_length,
                'Eccentricity': region.eccentricity,
                'ConvexArea': region.convex_area,
                'EquivDiameter': region.equivalent_diameter,
                'Extent': region.extent,
                'Perimeter': region.perimeter,
                'Roundness': roundness,
                'AspectRatio': aspect_ratio
            }
        return None

    def process_dataset(self):
        """
        Memproses seluruh dataset dan mengekstrak fitur.
        
        Returns:
        DataFrame: Data fitur dari seluruh gambar
        """
        features_list = []
        id_counter = 1
        
        # Dapatkan semua subfolder (kelas)
        classes = [d for d in os.listdir(self.image_folder) 
                  if os.path.isdir(os.path.join(self.image_folder, d))]
        
        logging.info(f"Memulai ekstraksi fitur dari {len(classes)} kelas")
        
        for class_name in classes:
            class_path = os.path.join(self.image_folder, class_name)
            logging.info(f"Memproses kelas: {class_name}")
            
            for image_name in tqdm(os.listdir(class_path)):
                try:
                    # Baca dan pra-proses gambar
                    image_path = os.path.join(class_path, image_name)
                    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                    
                    if image is None:
                        continue
                    
                    # Ekstrak fitur
                    features = self.extract_features(image)
                    
                    if features is not None:
                        # Tambahkan id dan kelas
                        features['id'] = id_counter
                        features['Class'] = class_name
                        features_list.append(features)
                        id_counter += 1
                        
                except Exception as e:
                    logging.error(f"Error memproses {image_path}: {str(e)}")
                    continue
        
        # Buat DataFrame
        df = pd.DataFrame(features_list)
        
        # Atur urutan kolom
        column_order = ['id', 'Area', 'MajorAxisLength', 'MinorAxisLength', 
                       'Eccentricity', 'ConvexArea', 'EquivDiameter', 
                       'Extent', 'Perimeter', 'Roundness', 'AspectRatio', 'Class']
        df = df[column_order]
        
        # Simpan ke CSV
        df.to_csv(self.output_file, index=False)
        logging.info(f"Dataset tersimpan di: {self.output_file}")
        
        # Tampilkan statistik dasar
        logging.info("\nStatistik Dataset:")
        logging.info(f"Total gambar: {len(df)}")
        logging.info(f"Total kelas: {len(df['Class'].unique())}")
        logging.info("\nDistribusi kelas:")
        for class_name, count in df['Class'].value_counts().items():
            logging.info(f"{class_name}: {count} gambar")
        
        return df

# Contoh penggunaan
if __name__ == "__main__":
    # Konfigurasi
    image_folder = r"G:\Coding\python\UTS KB\tugas2\Rice_Image_Dataset"
    output_file = r"dataset_features.csv"
    
    # Inisialisasi dan jalankan ekstraksi fitur
    extractor = ImageFeatureExtractor(
        image_folder=image_folder,
        output_file=output_file
    )
    
    # Proses dataset
    df = extractor.process_dataset()