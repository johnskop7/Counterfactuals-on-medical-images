# transforms.py

import os
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset, random_split
from PIL import Image
import pydicom
from pydicom.pixels import apply_voi_lut
import pandas as pd


class RSNA_Pneumonia_Dataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the CSV file with annotations.
            root_dir (string): Directory with all the DICOM images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        dicom_path = os.path.join(self.root_dir, self.data_frame.iloc[idx]['patientId'] + '.dcm')
        dicom_image = pydicom.dcmread(dicom_path)
        image = apply_voi_lut(dicom_image.pixel_array, dicom_image)  # Apply VOI LUT for correct windowing
        image = Image.fromarray(image).convert('RGB')  # Convert to RGB for compatibility

        if self.transform:
            image = self.transform(image)

        label = self.data_frame.iloc[idx]['class']
        label = int(label == 'Lung Opacity')  # Convert label to binary

        return image, label
    
def get_transforms(image_size):
    """
    Returns training, validation, and test transforms based on the specified image size.
    """
    train_transforms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])

    test_transforms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    
    return train_transforms, val_transforms, test_transforms

def create_datasets(data_dir, image_size):
    """
    Splits the dataset into training, validation, and testing datasets with specified image size.
    """
    # Get transforms with the specified image size
    train_transforms, val_transforms, test_transforms = get_transforms(image_size)

    # Load the full dataset with training transformations
    full_dataset = datasets.ImageFolder(os.path.join(data_dir), transform=train_transforms)

    # Determine sizes of splits
    train_size = int(0.8 * len(full_dataset))
    validation_size = int(0.05 * len(full_dataset))
    test_size = len(full_dataset) - train_size - validation_size

    # Split dataset
    train_dataset, temp_dataset = random_split(full_dataset, [train_size, len(full_dataset) - train_size])
    val_dataset, test_dataset = random_split(temp_dataset, [validation_size, test_size])

    # Apply different transforms to validation and test datasets
    val_dataset.dataset.transform = val_transforms
    test_dataset.dataset.transform = test_transforms

    return train_dataset, val_dataset, test_dataset

def create_datasets_for_chestxrays(data_dir, image_size):
    transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor()
    ])

    # Paths to the CSV files
    train_csv_file = '/content/drive/MyDrive/models/chest_xray_normal_pneumonia/normal_pneumonia_256_resolution/train_dataset.csv'
    val_csv_file = '/content/drive/MyDrive/models/chest_xray_normal_pneumonia/normal_pneumonia_256_resolution/val_dataset.csv'
    test_csv_file = '/content/drive/MyDrive/models/chest_xray_normal_pneumonia/normal_pneumonia_256_resolution/test_dataset.csv'


    # Creating dataset instances for each dataset type
    train_dataset = RSNA_Pneumonia_Dataset(csv_file=train_csv_file, root_dir=data_dir, transform=transform)
    val_dataset = RSNA_Pneumonia_Dataset(csv_file=val_csv_file, root_dir=data_dir, transform=transform)
    test_dataset = RSNA_Pneumonia_Dataset(csv_file=test_csv_file, root_dir=data_dir, transform=transform)
    return train_dataset, val_dataset, test_dataset