import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import logging
from datetime import datetime

class DataPreprocessor:
    def __init__(self, input_file, output_file):
        """
        Inisialisasi preprocessor untuk dataset.
        
        Parameters:
        input_file (str): Path ke file CSV input
        output_file (str): Path untuk menyimpan hasil preprocessing
        """
        self.input_file = input_file
        self.output_file = output_file
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging untuk mencatat proses"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
    
    def handle_missing_values(self, df, strategy='median', threshold=0.5):
        """
        Menangani missing values dalam dataset.
        
        Parameters:
        df: DataFrame input
        strategy: Strategi imputation ('mean', 'median', 'most_frequent')
        threshold: Threshold untuk menghapus kolom dengan terlalu banyak missing values
        """
        logging.info("Menangani missing values...")
        
        # Hitung persentase missing values per kolom
        missing_percentages = df.isnull().mean()
        
        # Hapus kolom dengan missing values melebihi threshold
        columns_to_drop = missing_percentages[missing_percentages > threshold].index
        df = df.drop(columns=columns_to_drop)
        
        # Imputation untuk kolom numerik
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) > 0:
            imputer = SimpleImputer(strategy=strategy)
            df[numeric_columns] = imputer.fit_transform(df[numeric_columns])
        
        # Imputation untuk kolom kategorikal
        categorical_columns = df.select_dtypes(exclude=[np.number]).columns
        if len(categorical_columns) > 0:
            imputer = SimpleImputer(strategy='most_frequent')
            df[categorical_columns] = imputer.fit_transform(df[categorical_columns])
        
        return df
    
    def encode_categorical(self, df, encode_target=True):
        """
        Encoding variabel kategorikal.
        
        Parameters:
        df: DataFrame input
        encode_target: Apakah encode kolom target (Class)
        """
        logging.info("Melakukan encoding variabel kategorikal...")
        
        # Simpan kolom target
        if 'Class' in df.columns and encode_target:
            target = df['Class']
            df = df.drop('Class', axis=1)
        
        # Identifikasi kolom kategorikal (kecuali 'id')
        categorical_columns = df.select_dtypes(include=['object']).columns
        categorical_columns = [col for col in categorical_columns if col != 'id']
        
        # Label Encoding untuk kolom dengan 2 unique values
        label_encoders = {}
        for col in categorical_columns:
            if df[col].nunique() == 2:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
                label_encoders[col] = le
        
        # One-Hot Encoding untuk kolom lainnya
        df = pd.get_dummies(df, columns=[col for col in categorical_columns if col not in label_encoders])
        
        # Encode target jika diperlukan
        if 'Class' in locals() and encode_target:
            le = LabelEncoder()
            target = le.fit_transform(target)
            df['Class'] = target
            label_encoders['Class'] = le
        
        return df, label_encoders
    
    def scale_features(self, df, method='standard'):
        """
        Scaling fitur numerik.
        
        Parameters:
        df: DataFrame input
        method: Metode scaling ('standard' atau 'minmax')
        """
        logging.info(f"Melakukan {method} scaling...")
        
        # Pisahkan kolom yang tidak perlu di-scale
        non_scaling_cols = ['id', 'Class']
        features_to_scale = [col for col in df.columns if col not in non_scaling_cols]
        
        if method == 'minmax':
            scaler = MinMaxScaler()
        else:
            scaler = StandardScaler()
        
        df[features_to_scale] = scaler.fit_transform(df[features_to_scale])
        return df, scaler
    
    def select_features(self, df, n_features=10):
        """
        Seleksi fitur menggunakan SelectKBest.
        
        Parameters:
        df: DataFrame input
        n_features: Jumlah fitur yang akan dipilih
        """
        logging.info("Melakukan seleksi fitur...")
        
        if 'Class' not in df.columns:
            logging.warning("Kolom Class tidak ditemukan. Melewati seleksi fitur.")
            return df, None
        
        # Pisahkan fitur dan target
        X = df.drop(['Class', 'id'] if 'id' in df.columns else ['Class'], axis=1)
        y = df['Class']
        
        # Pilih fitur terbaik
        selector = SelectKBest(f_classif, k=min(n_features, len(X.columns)))
        X_selected = selector.fit_transform(X, y)
        
        # Buat DataFrame baru dengan fitur terpilih
        selected_features = X.columns[selector.get_support()].tolist()
        df_selected = pd.DataFrame(X_selected, columns=selected_features)
        
        # Tambahkan kembali kolom yang dipisahkan
        if 'id' in df.columns:
            df_selected['id'] = df['id']
        df_selected['Class'] = y
        
        return df_selected, selector
    
    def handle_outliers(self, df, method='zscore', threshold=3):
        """
        Menangani outlier dalam dataset.
        
        Parameters:
        df: DataFrame input
        method: Metode deteksi outlier ('zscore' atau 'iqr')
        threshold: Threshold untuk menentukan outlier
        """
        logging.info("Menangani outliers...")
        
        # Pisahkan kolom numerik
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col not in ['id', 'Class']]
        
        if method == 'zscore':
            # Deteksi outlier menggunakan Z-score
            z_scores = np.abs((df[numeric_cols] - df[numeric_cols].mean()) / df[numeric_cols].std())
            df = df[(z_scores < threshold).all(axis=1)]
        else:
            # Deteksi outlier menggunakan IQR
            Q1 = df[numeric_cols].quantile(0.25)
            Q3 = df[numeric_cols].quantile(0.75)
            IQR = Q3 - Q1
            df = df[~((df[numeric_cols] < (Q1 - 1.5 * IQR)) | 
                     (df[numeric_cols] > (Q3 + 1.5 * IQR))).any(axis=1)]
        
        return df
    
    def balance_dataset(self, df, method='smote'):
        """
        Menyeimbangkan dataset menggunakan SMOTE atau undersampling.
        
        Parameters:
        df: DataFrame input
        method: Metode balancing ('smote' atau 'undersample')
        """
        logging.info(f"Menyeimbangkan dataset menggunakan {method}...")
        
        if 'Class' not in df.columns:
            logging.warning("Kolom Class tidak ditemukan. Melewati balance dataset.")
            return df
        
        # Pisahkan fitur dan target
        X = df.drop(['Class', 'id'] if 'id' in df.columns else ['Class'], axis=1)
        y = df['Class']
        
        if method == 'smote':
            # Oversampling menggunakan SMOTE
            sampler = SMOTE(random_state=42)
        else:
            # Undersampling
            sampler = RandomUnderSampler(random_state=42)
        
        X_resampled, y_resampled = sampler.fit_resample(X, y)
        
        # Buat DataFrame baru dengan data yang sudah diseimbangkan
        df_balanced = pd.DataFrame(X_resampled, columns=X.columns)
        df_balanced['Class'] = y_resampled
        
        return df_balanced
    
    def preprocess_data(self, handle_missing=True, encode_cat=True, scale=True,
                       select_feat=True, handle_out=True, balance=True):
        """
        Melakukan seluruh proses preprocessing.
        """
        # Baca dataset
        df = pd.read_csv(self.input_file)
        logging.info(f"Dataset awal shape: {df.shape}")
        
        # Handling missing values
        if handle_missing:
            df = self.handle_missing_values(df)
            logging.info(f"Shape setelah handling missing values: {df.shape}")
        
        # Encoding categorical variables
        if encode_cat:
            df, encoders = self.encode_categorical(df)
            logging.info(f"Shape setelah encoding: {df.shape}")
        
        # Handle outliers
        if handle_out:
            df = self.handle_outliers(df)
            logging.info(f"Shape setelah handling outliers: {df.shape}")
        
        # Feature scaling
        if scale:
            df, scaler = self.scale_features(df)
            logging.info(f"Shape setelah scaling: {df.shape}")
        
        # Feature selection
        if select_feat:
            df, selector = self.select_features(df)
            logging.info(f"Shape setelah feature selection: {df.shape}")
        
        # Balance dataset
        if balance:
            df = self.balance_dataset(df)
            logging.info(f"Shape setelah balancing: {df.shape}")
        
        # Simpan hasil
        df.to_csv(self.output_file, index=False)
        logging.info(f"Dataset hasil preprocessing disimpan di: {self.output_file}")
        
        return df

# Contoh penggunaan
if __name__ == "__main__":
    # Inisialisasi preprocessor
    preprocessor = DataPreprocessor(
        input_file="dataset_features.csv",
        output_file="preprocessed_dataset.csv"
    )
    
    # Lakukan preprocessing
    df_preprocessed = preprocessor.preprocess_data(
        handle_missing=True,
        encode_cat=True,
        scale=True,
        select_feat=True,
        handle_out=True,
        balance=True
    )