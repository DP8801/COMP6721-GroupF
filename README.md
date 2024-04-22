# COMP-6721-Group-F
# Project: Geo-Spatial Image Classification using CNN models
# Team Details
  1. Aarya Parikh (40262787)
  2. Dev Pandya (40268577)
  3. Priyansh Bhuva (40269498)
  4. Harshvardhan Rao (40268567)
# High Level Description/ presenetation of project
This project aims to develop a robust model to accurately classify diverse land features in satellite imagery, with a specific focus on enhancing disaster response capabilities for events such as wildfires and melting ice, amid the escalating challenges posed by climate change. Leveraging Convolutional Neural Network (CNN) architectures such as ResNet18, VGG16, and AlexNet, coupled with fine-tuning hyperparameters, we tackle various challenges including data imbalances, image anomalies, and computational costs. Notably, ResNet18 showcased superior performance on RESISC25 dataset whereas AlexNet emerged as the top performer on the SeaICE dataset and VGG outperformed in Wildfire dataset. To further enhance performance, we implemented techniques such as transfer learning and early stopping strategies. Through this endeavor, we strive to contribute to the advancement of environmental monitoring and disaster management strategies, thereby mitigating the adverse impacts of climate change on our ecosystems and communities.

# Requirements 
PIL 9.2.0

cv2 4.7.0

matplotlib 3.5.2

numpy 1.21.5

sklearn 1.0.2

torch 1.8.1+cu111

torchvision 0.9.1+cu101

cuda V11.1

shutil 11.0.0

# Instruction on how to train/validate the model
To train and validate the model, we have created multiple .ipynb files for each dataset and each model. The models are saved in the respective folders.
To train the model, change the location of the dataset in the code or link the kaggle account in your .ipynb files. The dataset can be downloaded automatically from the kaggle. Run all the blocks under the training section. The models weigh would be saved in the current directory.

# Instructions on how to run the test sample on the pretrained model
User can also run the validation models by running the blocks under validation section in the .ipynb files. Dataloader should be preloaded with the test dataset. The validation accuracy, precision, recall and f1 score would be printed and T-SNE plots would be generated.

# Description on how to obtain the Dataset from an available download link
  1. [Dataset 1](https://www.kaggle.com/datasets/abdelghaniaaba/wildfire-prediction-dataset/data) (Wildfire)
  2. [Dataset 2](https://www.kaggle.com/datasets/aaryaankurparikh/nwpu-resisc25) (Resisc45)
  3. [Dataset 3](https://www.kaggle.com/datasets/aaryaankurparikh/seaice) (SeaIce)

# Presentation
[Presentation](https://docs.google.com/presentation/d/1nLjfl3HO8f1kChaua5jbcPII9rXpSQtK/edit#slide=id.p1)
