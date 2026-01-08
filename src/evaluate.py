import pandas as pd, joblib, json
from sklearn.metrics import accuracy_score
from config import TARGET_COLUMN

def evaluate(path):
    df = pd.read_csv(path)
    X = df.drop(TARGET_COLUMN, axis=1)
    y = df[TARGET_COLUMN]
    model = joblib.load('models/rf_model.pkl')
    preds = model.predict(X)
    with open('reports/metrics.json', 'w') as f:
        json.dump({'accuracy': accuracy_score(y, preds)}, f)

if __name__ == '__main__':
    evaluate('data/processed/v3_interaction.csv')
