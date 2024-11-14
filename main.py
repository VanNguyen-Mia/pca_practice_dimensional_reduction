import os
from src.main import load_data, preprocess, apply_pca

# Define dataset path and save path
dataset_path = "./dataset/Breast_GSE45827.csv"  # Replace with actual path
save_path = "./results"

# Step 1: Load Data
data = load_data(dataset_path)

# Step 2: Preprocess Data
data_cleaned = preprocess(data)

# Step 3: Apply PCA and save the output
pca_csv_path, pca_image_path = apply_pca(data_cleaned, save_path, variance_threshold=0.95)
print("PCA transformation completed and results saved.")
