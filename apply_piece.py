# implement_method.py
import argparse
import pickle 
import torch
from IPython.display import HTML
from functools import partial
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation,PillowWriter,FFMpegWriter
from cnn_models import load_cnn_model
from utils import generate_and_plot_synthetic_image,latent_vector_optimization,update,plot_images_difference
from piece_utils import acquire_feature_probabilities,filter_df_of_exceptional_noise,modifying_exceptional_features

def implement_method(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    cnn_model = load_cnn_model(model_type=args.model_type, model_weights=args.model_weights,num_classes=args.num_classes)
    cnn_model = cnn_model.to(device)

    gan_network = args.gan_model
    with open(gan_network, 'rb') as f:
        gan = pickle.load(f)['G_ema'].cuda()

    synthetic_image_initial,ws = generate_and_plot_synthetic_image(args.latent_vector_path,cnn_model,device,gan)

    class_activations = args.class_activations
    counterfactual_class = args.counterfactual_class
  

    #### ========== First step of PIECE Algorithm ========== ####
    # Step 1: Acquire the probability of each features,
    # and identify the excpetional ones (i.e., those with a probability lower than alpha)
    df = acquire_feature_probabilities( counterfactual_class, cnn_model ,original_query_img=synthetic_image_initial, alpha=args.alpha ,class_activations_path=class_activations)

    # Step 2: Filter out exceptional features which we want to change, and change them to their expected values in the counterfactual class
    df = filter_df_of_exceptional_noise(df, counterfactual_class, cnn_model, alpha=args.alpha)

    # Sort by least probable to the most probable
    df = df.sort_values('Probability of Event')

    print('Features Changed to Expected Value:',df.shape[0],'out of 2048')

    query_x =  cnn_model(synthetic_image_initial)[1][0]

    # Get x' -- The Ideal Explanation
    ideal_xp = modifying_exceptional_features(df, counterfactual_class, query_x)
    ideal_xp = ideal_xp.clone().detach().float().requires_grad_(False)


    initial_latent_vector = ws
    w_e,images_for_gif = latent_vector_optimization(args.lr,args.epochs,gan,initial_latent_vector,device,cnn_model,ideal_xp)

    # Assuming `images_for_gif` and other variables are defined
    fig, ax = plt.subplots()

    # Use partial to pass `ax` and `number_of_classes` as fixed arguments to `update`
    update_with_ax = partial(update, ax=ax, number_of_classes=args.num_classes)

    # Set up FuncAnimation with partial to pass ax to update
    anim = FuncAnimation(fig, update_with_ax, frames=images_for_gif, blit=False)
    HTML(anim.to_jshtml())


    # Save the animation as an MP4 video
    anim.save(args.video_path, writer=FFMpegWriter(fps=10))

    print(f"Video saved at {args.video_path}")

    initial_latent_vector = ws
    counterfactual_latent_vector = w_e
    difference_np,synthetic_image_initial_np,counterfactual_image_np,counterfactual_image = plot_images_difference(initial_latent_vector, counterfactual_latent_vector,gan,args.info_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load dataset, initialize model, and store class activations.")
    parser.add_argument("--gan_model",type=str,required=True,help="The path to the pretrained gan model's weights.")
    parser.add_argument("--latent_vector_path",type=str,required=True,help="Path to the saved latent vector of the selected image.")
    parser.add_argument("--model_type", type=str, required=True, choices=["resnet18", "alexnet", "resnet50"], help="Type of CNN model to load")
    parser.add_argument("--model_weights", type=str, required=True, help="Path to pretrained model weights")
    parser.add_argument("--counterfactual_class",type=int,required=True,help="Index of the counterfactual class")
    parser.add_argument("--class_activations",type=str,required=True,help="Path to the class activations pickle file")
    parser.add_argument("--alpha",type=int,default=0.1,help="Alpha parameter used in the PIECE methodology")
    parser.add_argument("--lr",type=int,default=0.01,help="Learning rate for the latent vector optimization")
    parser.add_argument("--epochs",type=int,default=200,help="Number of epochs for the latent vecttor optimization")
    parser.add_argument("--video_path",type=str,required=True,help="Path where the transition video will be saved")
    parser.add_argument("--info_path",type=str,help="The path where the necessary info for the connected compononents' procedure will be saved")
    parser.add_argument("--num_classes",type=int,default=2,help="Number of classes of the dataset")
    args = parser.parse_args()
    
    implement_method(args)