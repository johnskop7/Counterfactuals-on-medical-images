# store_activations.py

import torch
import pickle
import numpy as np
from tqdm import tqdm

def store_class_activations(cnn, data_loader, device, save_path):
    """
    Stores the class activations of a CNN model for each predicted class.

    Args:
        cnn (torch.nn.Module): The CNN model.
        data_loader (torch.utils.data.DataLoader): DataLoader for the dataset.
        device (torch.device): Device to run the model on (e.g., 'cuda' or 'cpu').
        save_path (str): Path to save the activations as a pickle file.

    Returns:
        None
    """
    cnn.eval()
    predicted_class_activations = {}  # Dictionary to store activations for each class

    with torch.no_grad():
        for inputs, _ in tqdm(data_loader, desc='Storing features'):
            inputs = inputs.to(device)
            logits, features = cnn(inputs)
            probs = torch.softmax(logits, dim=1)
            _, predicted_labels = torch.max(probs, dim=1)

            # Store activations based on the predicted class
            for feature, predicted_label in zip(features, predicted_labels):
                predicted_label_item = predicted_label.item()
                if predicted_label_item not in predicted_class_activations:
                    predicted_class_activations[predicted_label_item] = []
                predicted_class_activations[predicted_label_item].append(feature.cpu().numpy())

    # Convert lists to numpy arrays
    for class_id in predicted_class_activations.keys():
        predicted_class_activations[class_id] = np.stack(predicted_class_activations[class_id])

    # Save activations to the specified path
    with open(save_path, 'wb') as handle:
        pickle.dump(predicted_class_activations, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Class activations saved to {save_path}")
