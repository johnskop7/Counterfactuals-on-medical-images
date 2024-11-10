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
With this script you are also indicating the dimensions of the image(for lung opacity dataset we have ran expeeriments for both 128 and 256 dimensions, for brain mri only for 128 and for oct images only for 256), the original class that you want your image to belong (check the documentation to find out which index corresponds to each class) , the number of classes of each dataset (it is required) , the model type (resnet50,resnet18 or alexnet, although the results are mostly for resnet50 CNN models) along with the respective weigths . The selected_image argument is just an index to a list of saved images. 

