
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

def load_data():
    file_path = Path(__file__).resolve().parent.parent / "data" / "breast_cancer.csv"

    df = pd.read_csv(file_path)

    X = df.drop(columns=["target"])
    y = df["target"]

    X_dev, X_test, y_dev, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    X_train, X_cal, y_train, y_cal = train_test_split(
        X_dev, y_dev, test_size=0.25, random_state=42, stratify=y_dev
    )

    return X_train, X_cal, X_test, y_train, y_cal, y_test


if __name__ == "__main__":
    X_train, X_cal, X_test, y_train, y_cal, y_test = load_data()

    print("Training samples:", len(X_train))
    print("Calibration samples:", len(X_cal))
    print("Testing samples:", len(X_test))
