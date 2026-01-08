import pandas as pd
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

def create_feature_variants(df):
    datasets = {}
    datasets['v1_base'] = df.copy()
    df_v3 = df.copy()
    df_v3['BMI_Glucose'] = df['BMI'] * df['Glucose']
    datasets['v3_interaction'] = df_v3
    scaler = StandardScaler()
    for col in df.columns[:-1]:
        temp = df.copy()
        temp[col] = scaler.fit_transform(df[[col]])
        datasets[f'scaled_{col}'] = temp
    return datasets
