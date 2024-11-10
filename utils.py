# utils.py
import torch
import imageio
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim



def plot_image(tensor):
    """Plot a tensor as an image."""
    img = tensor.detach().cpu().squeeze().numpy()
    img = np.clip(img, 0, 1)
    plt.imshow(np.transpose(img, (1, 2, 0)))
    plt.axis('off')
    plt.show()

def compute_class_probabilities(image, model, device):
    # Check if the tensor is 3D and convert to 4D if needed
    if image.dim() == 3:
        image = image.unsqueeze(0)

    image = image.to(device)
    logits, features = model(image)
    probs = torch.softmax(logits, dim=1)
    _, predicted_labels = torch.max(probs, dim=1)
    return probs


def convert_tensor_to_png(image_tensor, filename='output_image.png'):
    # Normalize the tensor to [0, 255]
    tensor = image_tensor.mul(255).byte()

    # Check the number of channels in the tensor
    if tensor.shape[0] == 3:  # RGB
        # Convert to NumPy array and rearrange dimensions to [H, W, C]
        array = tensor.numpy()
        array = np.transpose(array, (1, 2, 0))
        # Create a PIL image with RGB mode
        image = Image.fromarray(array, 'RGB')
    elif tensor.shape[0] == 1:  # Grayscale
        # Remove the channel dimension and convert to [H, W]
        array = tensor.squeeze(0).numpy()
        # Create a PIL image with grayscale mode
        image = Image.fromarray(array, 'L')  # 'L' is for grayscale
    else:
        raise ValueError("Unsupported number of channels. Expected 1 (grayscale) or 3 (RGB).")

    # Save the image to the specified filename
    image.save(filename)
    return image

def apply_gaussian_blur(tensor_image, kernel_size=3, sigma=0.5):
    transform = transforms.GaussianBlur(kernel_size=kernel_size, sigma=sigma)
    blurred_image = transform(tensor_image)
    return blurred_image


def save_correctly_classified_images(model,class_index,number_of_images,test_loader,device):
    model.eval()
    correctly_classified_images = []
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Processing batches"):
            if len(correctly_classified_images) >= number_of_images:
                break

            images, labels = images.to(device), labels.to(device)
            model.to(device)
            outputs, _ = model(images)
            _, preds = torch.max(outputs, 1)
            correctly_classified_indices = (preds == labels).nonzero(as_tuple=False).squeeze()

            # Handle the case where there's only one correctly classified image in the batch
            if correctly_classified_indices.dim() == 0:
                correctly_classified_indices = correctly_classified_indices.unsqueeze(0)

            for idx in correctly_classified_indices:
                if len(correctly_classified_images) >= number_of_images:  # Stop after 400 images
                    break

                true_label = labels[idx].item()
                if true_label == class_index:
                    correctly_classified_image = images[idx].cpu()
                    predicted_label = preds[idx].item()
                    correctly_classified_images.append((correctly_classified_image, true_label, predicted_label))

    print(f"Total correctly classified images saved: {len(correctly_classified_images)}")
    return correctly_classified_images


def generate_and_plot_synthetic_image(latent_vector_path,cnn,device,gan):
  #load the computed latent vector of the image and convert it to tensor
  ws = np.load(latent_vector_path)['w']
  ws = torch.tensor(ws, device=device)
  #Generate the image and normalize it
  synthetic_image_initial = gan.synthesis(ws,noise_mode = 'const',force_fp32=True)
  synthetic_image_initial = (synthetic_image_initial * 0.5 + 0.5).clamp(0, 1) #normalize the image from [-1,1] to [0,1]
  #Plot the generated and the original image along with the respective probability scores
  plot_image(synthetic_image_initial)
  probs = compute_class_probabilities(synthetic_image_initial,cnn,device)
  print('The class probabilities of the GAN approximated image are:',probs[0].detach().cpu().numpy())
  print("---------------------------------------------------------------------------------")
  return synthetic_image_initial,ws


def tensor_to_pil(tensor):
    unloader = transforms.ToPILImage()
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)
    return image


def update(frame, ax, number_of_classes=2):
    """
    Updates the plot with the current frame and probabilities.

    Args:
        frame (tuple): Contains the image (PIL format) and its associated probabilities.
        ax (matplotlib.axes.Axes): Axis to plot on.
        number_of_classes (int): Number of classes in the dataset (default: 2).
    """
    image, probs = frame
    ax.clear()
    ax.imshow(image)

    # Display probabilities based on number of classes
    if number_of_classes == 2:
        prob1, prob2 = probs[0][0].detach().cpu().numpy(), probs[0][1].detach().cpu().numpy()
        prob_text = f'Prob 1: {prob1:.2f}, Prob 2: {prob2:.2f}'
    elif number_of_classes == 4:
        prob1, prob2, prob3, prob4 = (probs[0][i].detach().cpu().numpy() for i in range(4))
        prob_text = f'Prob 1: {prob1:.2f}, Prob 2: {prob2:.2f}, Prob 3: {prob3:.2f}, Prob 4: {prob4:.2f}'

    # Display probability text below the image
    ax.text(0.5, -0.1, prob_text, ha='center', va='top', transform=ax.transAxes, fontsize=10)
    ax.axis("off")  # Hide axes for a cleaner view
    return ax

def latent_vector_optimization(learning_rate,epochs,generator,initial_latent_vector,device,classifier,ideal_xp):
  # Explanation latent input (to optimize...) in W space
  w_e = initial_latent_vector.clone().detach().requires_grad_(True)

  criterion = nn.MSELoss()
  optimizer = optim.Adam([w_e], lr = learning_rate)

  images_for_gif = []

  for i in range(epochs):
      optimizer.zero_grad()

      # Generate image from w instead of z
      # Ensure gan.generator can accept w directly or modify accordingly
      image = generator.synthesis(w_e, noise_mode='const')  # Adjust according to your generator's API

      image = (image * 0.5 + 0.5).clamp(0,1)  # Normalize the image to [0, 1]
      #image = resize(image)

      logits, x_e = classifier(image)
      loss = criterion(x_e[0], ideal_xp)

      loss.backward()
      optimizer.step()

      if i % (epochs/50) == 0:
          pil_image = tensor_to_pil(image)
          probs = torch.softmax(logits, dim=1)
          images_for_gif.append((pil_image, probs))
          print("Loss:", loss.item())
          print('Epoch:', i)
  return w_e,images_for_gif



def plot_images_difference(initial_latent_vector, counterfactual_latent_vector, generator, save_path=None):
  #Generate the counterfactual image
  counterfactual_image = (generator.synthesis(counterfactual_latent_vector, noise_mode='const'))
  counterfactual_image = (counterfactual_image * 0.5 + 0.5).clamp(0,1)  # Normalize the image to [0, 1]
  counterfactual_image = counterfactual_image.detach().cpu().squeeze(0)

  #Generate again the initial synthetic image
  synthetic_image_initial = (generator.synthesis(initial_latent_vector, noise_mode='const'))
  synthetic_image_initial = (synthetic_image_initial * 0.5 + 0.5).clamp(0,1)  # Normalize the image to [0, 1]
  synthetic_image_initial = synthetic_image_initial.detach().cpu().squeeze(0)

  # Calculate the absolute difference between the images after detachment and squeezing
  difference = torch.abs(counterfactual_image - synthetic_image_initial)
  difference = difference.sum(dim=0)  # Sum over the color channel
  difference = difference / difference.max()  # Normalize the difference

  # Convert tensors to NumPy arrays for plotting
  synthetic_image_initial_np = synthetic_image_initial.permute(1, 2, 0).numpy()
  counterfactual_image_np = counterfactual_image.permute(1, 2, 0).numpy()
  difference_np = difference.numpy()
  #The values of the difference image should be between [0,255]
  difference_np = (difference_np * 255).astype(np.uint8)

  fig, axes = plt.subplots(1, 3, figsize=(20, 10))
  axes[0].imshow(synthetic_image_initial_np)
  axes[0].set_title('Initial Synthetic Image')
  axes[0].axis('off')

  axes[1].imshow(counterfactual_image_np)
  axes[1].set_title('Counterfactual Synthetic Image')
  axes[1].axis('off')

  # Ensure difference_np is suitable for imshow by checking if it needs reshaping
  if difference_np.ndim == 3 and difference_np.shape[0] == 1:
      difference_np = difference_np.squeeze(0)

  axes[2].imshow(difference_np,cmap='magma')
  axes[2].set_title('Differences Highlighted')
  axes[2].axis('off')

  plt.show()

  # Save arrays if save_path is provided
  if save_path:
      np.savez(save_path,
                difference=difference_np,
                synthetic_image_initial=synthetic_image_initial_np,
                counterfactual_image=counterfactual_image_np)
      print(f"Data saved at {save_path}")

  return difference_np,synthetic_image_initial_np,counterfactual_image_np,counterfactual_image





