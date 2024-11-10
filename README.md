# Counterfactuals-on-medical-images
A method dedicated to the generation of medical image counterfactuals based on the PIECE algorithm .
For the generation of a counterfactual image certain steps should be followed :
1) Pick a certain medical dataset , a cnn model pretrained on that dataset and an image on which the method will be applied
2) Using a GAN trained on that dataset find the latent representation z so that GAN(z) = initial image
3) Select the counterfactual class and apply PIECE on the GAN approximated image to find the exceptional features
4) Generate the counterfacual image through optimization on the latent vector z
5) Use some connected components of the difference between the initial and the counterfactual image in order to assess the model's decision boundary     


Currently we have applied our framework on three different datasets :
* Retinal OCT Images (optical coherence tomography) with four different classes (choroidal neovascularization (CNV),Diabetic macular edema (DME),Multiple drusen (arrowheads) present in early AMD and Normal retina)
* Lung Opacity dataset from the RSNA pneumonia detection challenge with two classes (Lung Opacity and Normal)
* Brain MRI dataset which initially had four classes (Mild ,Moderate, Very Moderate and No dementia ) but we then reduced them to two (Dementia and No Dementia)


The first two datasets are available in kaggle. If you want to use them in google colab through the kaggle API you can run the following commands :
```!kaggle competitions download -c rsna-pneumonia-detection-challenge```
```!kaggle datasets download -d tobiassilalahi/oct-images-normalize-zip```
```!unzip \*.zip && rm *.zip```

The Brain MRI images are available [here](https://drive.google.com/drive/folders/1-4DDyMcHBT_NGrNEZcw8Bb7eovnvilDT?usp=drive_link) 

You can also find [here](https://drive.google.com/drive/folders/1VW6pL0TVRiZyDBFlYknk1dhtFgJdaOzQ?usp=drive_link) a folder that contains the weights of the pretrained models along with other information needed for running the experiments . You can check the README files of each folder to find out about training details, class indexes etc. Here is how you can use the scripts to run the experiments:

1. Pick an image from one of the above dataset and save it in .png format. If you are using google colab you can do that by running the script main.py as follows:
```%run main.py --num_classes 2 --selected_image 49 --dataset_name chest_xray --image_size 256 --data_dir "path/to/dataset" --model_type resnet50 --model_weights=path/to/model/weights.pth  --original_class 1 --output_path "path/to/output_image.png"```
With this script you are also indicating the dimensions of the image(for lung opacity dataset we have ran experiments for both 128 and 256 dimensions, for brain mri only for 128 and for oct images only for 256), the original class that you want your image to belong (check the documentation to find out which index corresponds to each class) , the number of classes of each dataset (it is required) , the model type (resnet50,resnet18 or alexnet, although the results are mostly for resnet50 CNN models) along with the respective weights . The selected_image argument is just an index to a list of saved images. The sript picks an image and saves it in .png format in the path that you have dictated.

2. You should also clone the official github repo of the StyleGAN2-ada model with the following command : ```!git clone https://github.com/dvschultz/stylegan2-ada-pytorch``` .
3. We are going to use the script pbaylies_projector.py from the StyleGAN2-ada repo to compute the latent representation z that approximates our initial image. Run the script like this :
```!python pbaylies_projector.py --outdir=out --target-image='/path/to/saved/output_image.png' --network=$gan_network --seed=5 --num-steps=1000 --save-video=False``` . In the --target-image argument we put the path of the image that we saved and in the --network argument the path to the weights of our pretrained gan netwrok . This script saves the computed latent_vector in the path defined in --outdir in .npz foramt.
4.Then by running the implement_method.py you can apply the PIECE algorithm , find the exceptional features , compute the counterfactual latent vectro z' through optimization and generate the counterfactual image.
```%run implement_method.py --num_classes 2 --gan_model=$gan_network --latent_vector_path "projected_w.npz" --model_type resnet50 --model_weights=$cnn_weights --class_activations=$class_activations --counterfactual_class 0 --video_path "/content/drive/MyDrive/counterfactual_transition.mp4" --info_path "/content/info.npz"``` . In order to run this script you should also indicate the index of the counterfactual class , the path to the saved class activations , the path where the transition video from the initial to the counterfacatual image will be saved and the path where some necessary info for the next step will be stored.

5. Finally you can apply the method of the connected components by running the script connected_comp.py .
```%run connected_comp.py --info_path "/content/info.npz" --method 2 --n 2 --l 21 --pixel_threshold 10 --model_type resnet50 --model_weights=$cnn_weights ``` . You should indicate the path where the info from step 4 where saved along with some other important arguments the meaning of which you can find in the script. 

