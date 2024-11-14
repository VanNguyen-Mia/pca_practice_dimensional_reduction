import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder


def load_data(path:str) -> pd.DataFrame:
    """
    Load the dataset from a specified path

    Args:
        path (str): path to the dataset folder

    Returns:
        pd.DataFrame: DataFrame containing the dataset
    """

    data = pd.read_csv(path)
    return data

def preprocess(data: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the data by cleaning and standardizing it

    Args:
        data (pd.DataFrame): the raw dataset

    Returns:
        pd.DataFrame: cleaned data
    """
    data_cleaned = data.dropna()
    return data_cleaned

def apply_pca(data_cleaned, save_path, variance_threshold: float = 0.95) -> str:
    """
    Perform PCA, save the transformed dataset, and generate a scatter plot

    Args:
        data_cleaned (_type_): cleaned dataset
        save_path (_type_): path of the folder to save the transformed data and images

    Returns:
        str: paths to the transformed dataset and images
    """
    # Standardise the features
    features = data_cleaned.drop(columns = ['samples', 'type'])
    target = data_cleaned['type']
    features_scaled = StandardScaler().fit_transform(features)

    # Determine the optimal number of components
    pca_temp = PCA().fit(features_scaled)
    cumulative_variance = pca_temp.explained_variance_ratio_.cumsum()
    n_components = next(i for i, cumulative in enumerate(cumulative_variance) if cumulative >= variance_threshold) + 1

    # Apply PCA transformation
    pca = PCA(n_components=n_components)
    features_pca = pca.fit_transform(features_scaled)

    # Create a DataFrame with the transformed components and target
    df_pca = pd.DataFrame(features_pca, columns=[f"PC{i+1}" for i in range(n_components)])
    df_pca['type'] = target.reset_index(drop=True)

    # Convert categorical target labels to numeric
    df_pca['type_encoded'] = LabelEncoder().fit_transform(df_pca['type'])

    # Create directories for saving dataset and images if they don't exist
    dataset_folder = os.path.join(save_path, "dataset")
    images_folder = os.path.join(save_path, "images")
    os.makedirs(dataset_folder, exist_ok=True)
    os.makedirs(images_folder, exist_ok=True)

    # Save the PCA-transformed data to CSV
    pca_csv_path = os.path.join(dataset_folder, "pca_transformed_data.csv")
    df_pca.to_csv(pca_csv_path, index=False)

    # Generate and save scatter plot for the first two principal components
    plt.figure(figsize=(10, 8))
    plt.scatter(df_pca["PC1"], df_pca["PC2"], c=df_pca['type_encoded'], cmap="viridis", edgecolor="k", s=40)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("PCA 2D Scatter Plot")
    plt.colorbar(label="Cancer Type")
    pca_image_path = os.path.join(images_folder, "pca_scatter_plot.png")
    plt.savefig(pca_image_path)

    return pca_csv_path, pca_image_path



