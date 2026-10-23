
import pandas as pd
from pathlib import Path
from sklearn.datasets import load_breast_cancer

# Load the built-in dataset
dataset = load_breast_cancer(as_frame=True)

# Combine features and target
df = dataset.frame

# Create the data folder if it doesn't exist
data_dir = Path(__file__).resolve().parent.parent / "data"
data_dir.mkdir(exist_ok=True)

# Save as CSV
file_path = data_dir / "breast_cancer.csv"
df.to_csv(file_path, index=False)

print("Dataset saved to:", file_path)
print("Dataset shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
