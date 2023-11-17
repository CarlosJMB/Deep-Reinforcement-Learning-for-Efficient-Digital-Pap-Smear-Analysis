# Deep Reinforcement Learning for Efficient Digital Pap Smear Analysis

## Overview
This paper proposes an automated screening for pap smear tests using Deep Reinforcement Learning and deep learning techniques. The environment is generated using images retrieved from Mendeley Repository and using Open-AI-GYM API for the structure. The cell classifier model is  implemented using a pre-trained convolutional neural network, ResNet50,  this will allow us to obtain better results and a CNN with more accurate predictions when classifying malignant cells that could lead to cervical cancer.

## Requirements

* Keras

Moreover, it is necessary to use and consider the following versions of these libraries

| Library  | Version |
| ---------| ------- |
| pytorch  | 1.10.1 |
| tensoflow| 2.9.2  |
| opencv   | 4.6.6  |
| gym | 0.26.2|
| numpy | 1.26.1|

**Consideration:** pytorch==1.10.1  for CUDA 11.3 (the version depends on your CUDA version.)

Make sure that other common libraries such as numpy works correctly with the other library versions.

## Dataset
The following drive link contains both the classification and detection trained model and the datasets used for training the two stages. 

**Drive link:**  https://drive.google.com/drive/folders/1fz9-srsO7EBUKztheth8f1W_g1DAySB_?usp=sharing

**Content:** 

* DB_3
   Database classified into two categories: 
   1. Cell images
   2. Test background images with a total of 1600 training  images and 400 images for validation.

* DB_4
   Isolated cell database. Images corresponding to liquid-based Pap smears.
   Database classified into four categories:
  
  1. High squamous intra-epithelial lesion
  2. Low squamous intra-epithelial lesion
  3. Negative for intra-epithelial malignancy
  4. Squamous cell carcinoma

Both folders contain your images for both training and validation with a total of 4000 training images and 800 validation images.

## CNNs Usage
###Cell Detection Model 

In order to run the model, the following information should be taken into account:

* The first stage model was trained using the **FP_Cells.py** file. In the file, are the steps to train other models or use the already trained model. Consider that the dataset that the model uses is in the folder **DB_3** in the Google Drive Link. 

### Cell Classifier Model

In order to run the model, the following information should be taken into account:

* The second  model was trained using the following files:
  1. **dataset.py**: This file defines the paths where the images are retrieved. The dataset used in this file is contained within the folder named **DB_4**. 
  
  2. **utils.py**. This file generates the result images and saves the model. It is important to add the path where the user wants to save the model and the images generated from the results. In our case, the folder `outputs50` is the current path, as can be seen in this method extracted from said file:
     ````
      torch.save({
                'epoch': epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                }, 'outputs50/model.path')
     ````
  3. **model50.py**: This file generate the ResNet50 architecture.
  4. **train.py**: This file calls the previously mentioned files for starting the training.
  5. **inference.py**: This file classifies the images into four different categories. The dataset used in this file is in the folder named DB_4 and its folder is named validation. Additionally, the code in the file assesses the model using some metrics such as Precision, Recall, and F1 score. 

#### Train a model 
 ````
  $ python train.py --epochs "# of epochs"
  ````
#### Running and testing the model.

````
  $ python inference.py
````

## Environment

### PPO folder
In order to understand the usage of the environment is important to know the following information about the files in the folder.
####Folrder information

1.  **base** folder which contains the images for testing the environment and agents.
2.  **Cells_colab0.992_epo56** folder which contains the Cell Detection Model, which is used for defining the reward function.
3.  **env.py** file is where the environment is constructed and it follows the structure defined for GYM-API
4.  **runPPO.py** file contains the code for running the agents.
5.  **test_env.py** file contains the code for testing the behavior of the environment without using any trained agent.
6.  **tracking.py** file contains the code for tracking the movements of the agents during the testing stage. 
7.  **trainingPPO.py** file contains the code for training the agents.

### Usage
Before running the code, make sure all the directions and needed models are correctly located. 

 ````
  $ python runPPO.py 
  ````
### Results


### Images

Image:

![](https://pandao.github.io/editor.md/examples/images/4.jpg)

