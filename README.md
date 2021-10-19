# Deep Learning Models


Collection of different Deep Learning models suggested in research papers, using [Keras](https://keras.io/).
The implemented models are applied to the two different datasets. 
Models listed here are some cases simplified versions of the ones ultimately described in papers.

## Table of contents
- [Installation](#installation)

- [About the project](#about-the-project)
    - [Datasets](#datasets)

- [implementations](#implementations)
    

## Installation
    $ git clone https://github.com/LorenzoValente3/Deep-Learning-Models.git
    $ cd Deep-Learning-Models/
    $ sudo pip3 install -r requirements.txt'

## About the Project
The project is divided into two folders. 
Each of them contains the programs tailored to the different datasets it uses. 

### Datasets
#### MNIST
The [MNIST database](https://en.wikipedia.org/wiki/MNIST_database) is a handwritten digits dataset. The [class](models using MNIST/MNIST_dataset.py) considered here includes 60.000 training samples and 10.000 test samples. Each image is represented by 28x28 pixels, each value ranges from 0 to 255 and has a grayscale value.

#### Polynomial
Polynomial data class: Generation of 2-D images (40x40 for 1-channel) showing polynomial up to a maximum degree in two variables.


## Implementations
### MNIST - Autoencoder