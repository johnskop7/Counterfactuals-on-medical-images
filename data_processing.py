# data_processing.py

import argparse
from torch.utils.data import DataLoader
import os
import sys

# # Get the current directory of the script
# current_dir = os.path.dirname(os.path.abspath(__file__))

# # Add the current directory to the system path
# sys.path.append(current_dir)
from transform_functions import create_datasets,create_datasets_for_chestxrays


def load_data(dataset_name, batch_size,image_size, data_dir=None):
    if dataset_name == "oct":
        train_dataset, val_dataset, test_dataset =  create_datasets(data_dir, image_size )

    elif dataset_name == "chest_xray":
        
        train_dataset, val_dataset, test_dataset =  create_datasets_for_chestxrays(data_dir, image_size )

    elif dataset_name == "brain_mri":
        train_dataset, val_dataset, test_dataset =  create_datasets(data_dir, image_size)


    else:
        raise ValueError("Unsupported dataset name. Choose from 'oct', 'chest_xray', or 'brain_mri'.")
    
    # Define data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader,val_loader,test_loader


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Load dataset and create data loaders.")
#     parser.add_argument("--dataset_name", type=str, required=True, choices=["oct", "chest_xray", "brain_mri"], help="Name of the dataset to load.")
#     parser.add_argument("--batch_size", type=int, default=32, help="Batch size for DataLoader.")
#     parser.add_argument("--image_size", type=int, default=256, help="Size to which images will be resized.")
#     parser.add_argument("--data_dir", type=str, help="Directory that contains the image_data")
#     args = parser.parse_args()

#     # Load data
#     train_loader, val_loader, test_loader = load_data(
#         args.dataset_name, 
#         args.batch_size, 
#         args.image_size,  
#         args.data_dir
#     )

#     # Print a sample output to verify
#     print(f"Loaded {args.dataset_name} dataset with {len(train_loader)} batches for training.")
#     print(f"Validation set has {len(val_loader)} batches, and test set has {len(test_loader)} batches.")