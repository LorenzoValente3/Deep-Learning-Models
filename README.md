# Deep Learning Models


Collection of different Deep Learning models suggested in research papers, using [Keras](https://keras.io/).
The implemented models are applied to the two different datasets. 
Models listed here are some cases simplified versions of the ones ultimately described in papers.

## Table of contents
- [Installation](#installation)

- [About the project](#about-the-project)
    - [Datasets](#datasets)

- [implementations](#implementations)
    - [Datasets](#datasets)
    

## Installation
    $ git clone https://github.com/LorenzoValente3/Deep-Learning-Models.git
    $ cd Deep-Learning-Models/
    $ sudo pip3 install -r requirements.txt'

## About the Project
The project is divided into two folders. 
Each of them contains the programs tailored to the different datasets it uses. 

### Datasets
#### MNIST
The [MNIST database](https://en.wikipedia.org/wiki/MNIST_database) is a handwritten digits dataset. 
The [class](./models_using_MNIST/MNIST_dataset.py) considered in this project includes 60.000 training samples and 10.000 test samples. 
Each image is represented by 28x28 pixels, each value ranges from 0 to 255 and has a grayscale value.

#### Polynomial
The Polynomial database is a bi-dimensional contour plots dataset. 
The [class](./GANs_using_Polyomials) considered in this project includes 20000 samples showing polynomial up to a maximum degree (the fifth degree is taken into account as the maximum one here) in two variables.
Each image is represented by 40x40 pixels for 1-channel. 
An image of the dataset is shown below as an example.
The dataset _file.npy_ can be downloaded [here](https://drive.google.com/drive/folders/13HlpRhNTrz7WK0NQnrNoA7BQTlPOXb3u?usp=sharing). 

<img src="GANs_using_Polyomials/images/DCGAN/contour.png" align="center" width="auto" height="auto"/>


## Implementations
### MNIST - Autoencoder