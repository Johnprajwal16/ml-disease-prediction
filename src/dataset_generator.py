import os
from data_ingestion import load_raw_data
from feature_engineering import create_feature_variants
from config import PROCESSED_DATA_PATH

def generate_datasets():
    os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
    df = load_raw_data()
    datasets = create_feature_variants(df)
    for name, data in datasets.items():
        data.to_csv(f'{PROCESSED_DATA_PATH}{name}.csv', index=False)

if __name__ == '__main__':
    generate_datasets()
