#connected_comp
import argparse
import numpy as np
from connected_components_utils import classification_probs_original_blended,find_and_plot_connected_components,choose_connected_components
from cnn_models import load_cnn_model
import torch

def connected_comp(args):
    # Load the saved .npz file
    data = np.load(args.info_path)

    # Access each array
    difference_np = data['difference']
    synthetic_image_initial_np = data['synthetic_image_initial']
    counterfactual_image_np = data['counterfactual_image']

    # Compute connected components and plot the
    labels_im, statistics, centroids = find_and_plot_connected_components(
        difference_image=difference_np,
        synthetic_image_initial_np=synthetic_image_initial_np,
        pixel_threshold=args.pixel_threshold,
        radius=args.radius,
        connectivity=args.connectivity
    )

    # Use the computed connected components to choose and mask the image
    blended_image = choose_connected_components(
    method=args.method,  # or 2 for manual selection
    n=args.n,  # Number of largest components to select
    manual_labels=args.list,  # Or specify labels manually if method=2
    labels_im=labels_im,
    statistics=statistics,
    centroids=centroids,
    synthetic_image_initial_np=synthetic_image_initial_np,
    counterfactual_image_np=counterfactual_image_np
  )

    model = load_cnn_model(args.model_type, args.model_weights)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    model.to(device)
    classification_probs_original_blended(blended_image = blended_image  ,model = model,device = device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load dataset, initialize model, and store class activations.")
    parser.add_argument("--info_path",type=str,required=True,help="The path to the necessary information needed for computing the connected components.")
    parser.add_argument("--pixel_threshold",type=int,default=20,help="Pixel threshold for the creation of the binary image")
    parser.add_argument("--radius",type=float,default=1.0,help="The radius of the disk used for the creation of the connected components")
    parser.add_argument("--connectivity",type=int,default=8,help="The type of connectivity(it can be either 4 or 8)")
    parser.add_argument("--method",type=int,default=1,help="Method for choosing connected components: Method 1 chooses the nth biggest ccs and Method 2 selects manually ccs")
    parser.add_argument("--n",type=int,default=20,help="Number of largest components to select if method 1 is chosen")
    parser.add_argument("--model_type", type=str, required=True, choices=["resnet18", "alexnet", "resnet50"], help="Type of CNN model to load.")
    parser.add_argument("--model_weights", type=str, help="Path to pretrained model weights.")
    parser.add_argument('-l', '--list', nargs='+',type=int, help='List of manually selected connected components', required=True)

    args = parser.parse_args()
    
    connected_comp(args)