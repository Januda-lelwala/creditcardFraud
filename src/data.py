import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_and_preprocess(csv_path='creditcard.csv'):
    df = pd.read_csv(csv_path)

    print(f"Dataset: {df.shape[0]:,} rows, {df.shape[1]} columns")
    print(f"Fraud: {df['Class'].sum():,} ({df['Class'].mean()*100:.4f}%)")

    scaler = StandardScaler()
    df['scaled_amount'] = scaler.fit_transform(df[['Amount']])
    df['scaled_time']   = scaler.fit_transform(df[['Time']])

    features = [c for c in df.columns if c not in ['Time', 'Amount', 'Class']]
    X = df[features].values
    y = df['Class'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Train: {X_train.shape[0]:,} samples  fraud={y_train.sum()}")
    print(f"Test : {X_test.shape[0]:,} samples  fraud={y_test.sum()}")

    return X_train, X_test, y_train, y_test
