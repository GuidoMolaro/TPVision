# README
## _Tumor recognition via Tensorflow and Keras models_

This Python project classifies brain MRIs into 4 labels:
* Glioma
* Meningioma
* Pituary Tumor
* No Tumor

The dataset can be found [here](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset). A smaller and less specific dataset (divided into Tumor and No Tumor) can be found [here](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection). This is useful if you are using a less powerful computer and it's taking a long time to build the model with the original dataset. To run it with the original dataset, extract the folders into the same location you're running your code (or change the dataset path in the code to where the files are). You should have a Training and a Testing folder to facilitate the division of the datasets. If you're using the small dataset, divide the Yes/No folders and fill the Training/Testing folders like so:
* Training
    * Yes
    * No

Keras builds all the necessary datasets within the first 30 lines of code. The code also stores both the model and the learning progression.

