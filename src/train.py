import pandas as pd
import joblib, os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from config import TARGET_COLUMN, MODEL_PATH, RANDOM_STATE

def train_model(path):
    df = pd.read_csv(path)
    X = df.drop(TARGET_COLUMN, axis=1)
    y = df[TARGET_COLUMN]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    os.makedirs(MODEL_PATH, exist_ok=True)
    joblib.dump(model, f'{MODEL_PATH}rf_model.pkl')

if __name__ == '__main__':
    train_model('data/processed/v3_interaction.csv')
