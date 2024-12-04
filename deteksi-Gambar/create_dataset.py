import numpy as np
import pickle
import os
import cv2

# Paths to CIFAR-10 data batches and meta file
data_dir = 'C:\\Users\\lenovo\\deteksi-warna-main'  # Adjust this to your data folder path
meta_file = os.path.join(data_dir, 'C:\\Users\\lenovo\\deteksi-warna-main\\cifar-10-batches-py\\batches.meta')
batch_files = [os.path.join(data_dir, f'C:\\Users\\lenovo\\deteksi-warna-main\\cifar-10-batches-py\\data_batch_{i}') for i in range(1, 6)]  # data_batch_1 to data_batch_5
test_file = os.path.join(data_dir, 'C:\\Users\\lenovo\\deteksi-warna-main\\cifar-10-batches-py\\test_batch')

# Load CIFAR-10 label names from the meta file
def load_label_names(meta_path):
    with open(meta_path, 'rb') as file:
        meta = pickle.load(file, encoding='bytes')
        return [label.decode('utf-8') for label in meta[b'label_names']]

# Load data batch file
def load_data_batch(batch_path):
    with open(batch_path, 'rb') as file:
        batch = pickle.load(file, encoding='bytes')
        images = batch[b'data']
        labels = batch[b'labels']
        return images, labels

# Convert CIFAR-10 data to images and save them in class folders
def create_image_dataset(data_batches, label_names, output_dir='dataset'):
    os.makedirs(output_dir, exist_ok=True)
    
    # Loop through each batch file
    for batch_file in data_batches:
        images, labels = load_data_batch(batch_file)
        images = images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)  # Reshape to (num_samples, 32, 32, 3)

        # Loop through each image in the batch
        for i in range(len(images)):
            label = labels[i]
            label_name = label_names[label]
            img = images[i]
            
            # Create folder for each label if it does not exist
            label_dir = os.path.join(output_dir, label_name)
            os.makedirs(label_dir, exist_ok=True)
            
            # Save the image
            img_filename = os.path.join(label_dir, f'{label_name}_{i}.png')
            cv2.imwrite(img_filename, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

# Get label names
label_names = load_label_names(meta_file)

# Create dataset from all batches
create_image_dataset(batch_files, label_names, output_dir='CIFAR10_dataset/train')
create_image_dataset([test_file], label_names, output_dir='CIFAR10_dataset/test')

print("Dataset has been successfully created in the 'CIFAR10_dataset' directory.")
