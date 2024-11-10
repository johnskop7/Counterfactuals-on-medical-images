# Counterfactuals-on-medical-images
A method dedicated to the generation of medical image counterfactuals based on the PIECE algorithm .
For the generation of a counterfactual image certain steps should be followed :
1) Pick a certain medical dataset , a cnn model pretrained on that dataset and an image on which the method will be applied
2) Using a GAN trained on that dataset find the latent representation z so that GAN(z) = initial image
3) Select the counterfactual class and apply PIECE on the GAN approximated image to find the exceptional features
4) Generate the counterfacual image through optimization on the latent vector z
5) Use some connected componenents of the difference between the initial and the counterfactual image in order to assess the model's decision boundary     
