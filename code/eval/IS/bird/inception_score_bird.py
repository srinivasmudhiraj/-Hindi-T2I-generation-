import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from scipy.stats import entropy
from PIL import Image
import pandas as pd

def load_and_preprocess_image(image_path, target_size=(299, 299)):
    """Load and preprocess a single image for InceptionV3."""
    try:
        img = Image.open(image_path).convert('RGB')
        img = img.resize(target_size)
        img = np.array(img)  # Convert image to numpy array
        img = preprocess_input(img)  # Preprocess as per InceptionV3 requirements
    except Exception as e:
        print(f"Skipping {image_path}: {e}")
        return None
    return img


def get_all_images(image_dir):
    """Recursively load images from all subdirectories."""
    images = []
    image_files = []
    for root, dirs, files in os.walk(image_dir):  # Walk through directories recursively
        for img_file in files:
            img_path = os.path.join(root, img_file)
            img_data = load_and_preprocess_image(img_path)
            if img_data is not None:
                images.append(img_data)
                image_files.append(img_path)
    
    if not images:
        raise ValueError("No images were loaded. Ensure your directory is correct.")
    
    # Convert list of images into a numpy array
    images = np.array(images)
    return images
    
    
def get_inception_model():
    """Load pre-trained InceptionV3 model."""
    model = InceptionV3(weights='imagenet')
    return model


def compute_inception_probabilities(model, image_dir):
    """Compute the Inception probabilities for all images in a directory and subdirectories."""
    print("Loading images...")
    images = get_all_images(image_dir)

    # Perform prediction
    print("Performing predictions...")
    predictions = model.predict(images, batch_size=32)
    return predictions
    
    


def calculate_inception_score(predictions, splits=10):
    """
    Computes Inception Score over predictions.

    Args:
        predictions: Predicted probabilities from InceptionV3.
        splits (int): Number of splits to divide data into for score computation.

    Returns:
        mean_score: Average Inception Score across splits.
        std_score: Standard deviation across splits.
    """
    # Ensure predictions are numpy arrays
    predictions = np.array(predictions)

    # Number of images
    num_images = predictions.shape[0]
    # Split data into chunks
    split_size = num_images // splits
    scores = []

    for i in range(splits):
        # Select data split
        start_idx = i * split_size
        end_idx = (i + 1) * split_size if i != splits - 1 else num_images
        preds = predictions[start_idx:end_idx]

        # Compute marginal class probabilities
        p_y = np.mean(preds, axis=0)

        # Compute KL divergence for each image
        kl_divergence = np.array([entropy(pred, p_y) for pred in preds])
        # Compute average KL divergence for this split
        split_score = np.exp(np.mean(kl_divergence))
        scores.append(split_score)

    # Calculate mean and std over splits
    mean_score = np.mean(scores)
    std_score = np.std(scores)
    
    return mean_score, std_score
    
    
def compute_IS(image_dir, writer=None, epoch=None):
    """
    Compute the Inception Score for the given images in a directory or subdirectories.
    
    Args:
        image_dir: Directory containing the images (and subfolders).
        writer: TensorBoard writer (optional).
        epoch: Current epoch number (optional).
    
    Returns:
        Mean Inception Score and standard deviation.
    """
    # Load InceptionV3 Model
    model = get_inception_model()
    
    # Compute predictions for all images in directory (and subdirs)
    print("Computing probabilities from the Inception model...")
    predictions = compute_inception_probabilities(model, image_dir)
    
    # Calculate Inception Score
    print("Calculating Inception Score...")
    mean_score, std_score = calculate_inception_score(predictions)
    
    print(f"Inception Score: Mean = {mean_score:.4f}, Std = {std_score:.4f}")
   # save_to_excel(mean_score, std_score, epoch)
    
    # If you have TensorBoard writer support
    if writer:
        writer.add_scalar('inception_score/mean', mean_score, epoch)
        writer.add_scalar('inception_score/std', std_score, epoch)
    
    return mean_score, std_score
    
    
def save_to_excel(mean_score, std_score, epoch, file_path="epoch_scores.xlsx"):
    """
    Save the calculated Inception score to an Excel file across epochs.

    Args:
        mean_score: Calculated mean score.
        std_score: Calculated standard deviation score.
        epoch: The current epoch number.
        file_path: Path to save the results Excel file.
    """
    # Create a DataFrame for every new epoch
    data = {"Epoch": [epoch], "Mean_Score": [mean_score], "Std_Score": [std_score]}
    df = pd.DataFrame(data)

    # If file exists, append new data; otherwise, create a new file
    if os.path.exists(file_path):
        with pd.ExcelWriter(file_path, mode="a", engine="openpyxl") as writer:
            df.to_excel(writer, index=False, header=False, startrow=writer.sheets['Sheet2'].max_row)
    else:
        df.to_excel(file_path, index=False)

    print(f"Results for epoch {epoch} saved to {file_path}") 
    
    return mean_score, std_score
