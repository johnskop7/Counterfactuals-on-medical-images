#connected_components_utils.py
import cv2
from skimage.morphology import opening, disk
import numpy as np
import matplotlib.pyplot as plt
import torch
from utils import compute_class_probabilities


def find_and_plot_connected_components(difference_image, synthetic_image_initial_np, pixel_threshold, radius, connectivity):
    # Threshold the difference image
    _, thresholded_diff = cv2.threshold(difference_image, pixel_threshold, 255, cv2.THRESH_BINARY)

    # Perform opening on the thresholded difference image
    opened_mask = opening(thresholded_diff, disk(radius=radius))

    # Find connected components
    num_labels, labels_im, statistics, centroids = cv2.connectedComponentsWithStats(opened_mask, connectivity=connectivity)

    # Create an output image to visualize the components
    output_image = np.zeros((*labels_im.shape, 3), dtype=np.uint8)

    # Create an image with only the connected components
    components_image = np.zeros_like(output_image)

    # Assign random colors to each component and count components
    component_colors = []
    for label in range(1, num_labels):
        mask = (labels_im == label)
        color = np.random.randint(0, 255, size=3)
        output_image[mask] = color
        components_image[mask] = color
        component_colors.append(color)

    # Apply Gaussian smoothing to each connected component
    smoothed_components_image = np.zeros_like(output_image)
    for label in range(1, num_labels):
        mask = (labels_im == label).astype(np.uint8)
        smoothed_mask = cv2.GaussianBlur(mask, (5, 5), 0)
        color = component_colors[label - 1]
        for i in range(3):  # Apply color to each channel
            smoothed_components_image[:, :, i] += smoothed_mask * color[i]

    # Normalize the synthetic image for blending
    synthetic_image_initial_normalized = (synthetic_image_initial_np * 255).astype(np.uint8)

    # Blend the original image with the output image containing smoothed components
    blended_image = cv2.addWeighted(synthetic_image_initial_normalized, 0.5, smoothed_components_image, 0.5, 0)

    # Plot the images
    fig, axes = plt.subplots(1, 3, figsize=(20, 10))
    axes[0].imshow(synthetic_image_initial_normalized)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    axes[1].imshow(blended_image)
    axes[1].set_title('Connected Components Overlay')
    axes[1].axis('off')

    axes[2].imshow(smoothed_components_image)
    axes[2].set_title(f'Smoothed Connected Components (Count: {num_labels - 1})')
    axes[2].axis('off')

    # Plot labels on the smoothed_components_image
    for label in range(1, num_labels):
        centroid = centroids[label]
        axes[2].text(centroid[0], centroid[1], str(label), color='white', fontsize=12, ha='center', va='center')

    plt.show()

    # Print the number of connected components
    print(f'Number of connected components: {num_labels - 1}')

    return labels_im, statistics, centroids


def choose_connected_components(method, n=None, manual_labels=None, labels_im=None, statistics=None, centroids=None, synthetic_image_initial_np=None, counterfactual_image_np=None):
    if labels_im is None or statistics is None or centroids is None:
        raise ValueError("Connected components data must be provided.")

    # Extract the areas of the connected components (excluding the background)
    areas = statistics[1:, cv2.CC_STAT_AREA]  # Skip the background label

    if method == 1:
        # Select the n largest components based on area
        sorted_indices = np.argsort(areas)[::-1]
        selected_indices = sorted_indices[:n]
        selected_labels = [index + 1 for index in selected_indices]  # Adjust for background label
    elif method == 2:
        # Manually select components by labels
        selected_labels = manual_labels
    else:
        raise ValueError("Invalid method. Use 1 for the n largest components or 2 for manual selection.")

    # Create a new binary mask with only the selected components
    selected_mask = np.zeros_like(labels_im, dtype=np.uint8)
    for label in selected_labels:
        selected_mask[labels_im == label] = 1

    # Apply Gaussian smoothing to the selected mask
    smoothed_mask = cv2.GaussianBlur(selected_mask.astype(np.uint8), (5, 5), 0)

    # Plot the smoothed binary mask
    plt.figure(figsize=(5, 5))
    plt.imshow(smoothed_mask, cmap='gray')
    plt.title("Smoothed Mask of Selected Connected Components")
    plt.axis('off')
    plt.show()

    # Normalize the synthetic image for blending
    synthetic_image_initial_normalized = (synthetic_image_initial_np * 255).astype(np.uint8)
    counterfactual_image_normalized = (counterfactual_image_np * 255).astype(np.uint8)

    # Create a 3-channel version of the smoothed mask for blending
    smoothed_mask_3ch = cv2.merge([smoothed_mask] * 3)

    # Blend the original image with the counterfactual image using the smoothed mask
    blended_image = (synthetic_image_initial_normalized * (1 - smoothed_mask_3ch) +
                     counterfactual_image_normalized * smoothed_mask_3ch).astype(np.uint8)

    # Plot the images
    fig, axes = plt.subplots(1, 3, figsize=(20, 10))
    axes[0].imshow(synthetic_image_initial_normalized)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    axes[1].imshow(smoothed_mask, cmap='gray')
    axes[1].set_title('Smoothed Mask of Selected Components')
    axes[1].axis('off')

    axes[2].imshow(blended_image)
    axes[2].set_title('Blended Image with Smoothed Mask')
    axes[2].axis('off')

    plt.show()

    # Print the number of selected connected components
    print(f'Number of selected connected components: {len(selected_labels)}')

    return blended_image

def classification_probs_original_blended(blended_image ,model,device):
  blended_image_tensor = torch.tensor(blended_image).permute(2, 0, 1).float() / 255.0
  #probs_original = compute_class_probabilities(synthetic_image_initial,model,device)
  probs_blended  = compute_class_probabilities(blended_image_tensor,model,device)
  #print("The probability scores of the CNN for the initial image are:",probs_original[0].detach().cpu().numpy())
  print("The probability scores of the CNN for the masked counterfactual image are:",probs_blended[0].detach().cpu().numpy())
