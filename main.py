# main.py

import argparse
import torch
from data_processing import load_data  # Load the dataset
from cnn_models import load_cnn_model  # Load the CNN model
from store_activations import store_class_activations  # Store class activations
from utils import convert_tensor_to_png,save_correctly_classified_images, plot_image,compute_class_probabilities

def main(args):
    
    # Step 1: Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Step 2: Load the dataset
    train_loader, val_loader, test_loader = load_data(
        dataset_name=args.dataset_name,
        batch_size=args.batch_size,
        image_size=args.image_size,
        data_dir=args.data_dir
    )

    # Step 3: Initialize the model
    cnn_model = load_cnn_model(model_type=args.model_type, model_weights=args.model_weights)
    cnn_model = cnn_model.to(device)

    # Step 4: Save a number of correclty classified images
    correctly_classified_images = save_correctly_classified_images(
        model = cnn_model,
        class_index = args.original_class,
        number_of_images = args.number_of_images,
        test_loader = test_loader,
        device = device
    )

    # Step 5: Pick an image to implement PIECE 
    image_index = args.selected_image  # Replace with the desired image index

    correctly_classified_image, true_label, predicted_label = correctly_classified_images[image_index]
    print(f"True label: {true_label}, Predicted label: {predicted_label}")
    plot_image(correctly_classified_image)
    probs = compute_class_probabilities(correctly_classified_image,cnn_model,device)
    print('The class probabilities of the image are:',probs[0].detach().cpu().numpy())

    convert_tensor_to_png(correctly_classified_image,args.output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load dataset, initialize model, and store class activations.")
    parser.add_argument("--dataset_name", type=str, required=True, choices=["oct", "chest_xray", "brain_mri"], help="Dataset to load")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for DataLoader")
    parser.add_argument("--image_size", type=int, default=256, help="Size to which images will be resized")
    parser.add_argument("--data_dir", type=str, help="Path to the image data directory in Google Drive")
    parser.add_argument("--model_type", type=str, required=True, choices=["resnet18", "alexnet", "resnet50"], help="Type of CNN model to load")
    parser.add_argument("--model_weights", type=str, required=True, help="Path to pretrained model weights")
    parser.add_argument("--original_class", type=int, default=1, help="Original class of the selected image as predicted by the CNN")
    parser.add_argument("--number_of_images",type=int,default=50,help="The number of correctly classified images we want to save for the PIECE method")
    parser.add_argument("--selected_image",type=int,default=10,help="Index of the selected image to apply our method"   )
    parser.add_argument("--output_path",type=str,required=True,help="Path where the selected image will be saved")
    args = parser.parse_args()
    
    main(args)
