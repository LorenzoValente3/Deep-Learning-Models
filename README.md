# Deep Learning Models

Collection of different Deep Learning models suggested in research papers, using [Keras](https://keras.io/).
The implemented models are applied to the two different [datasets](#datasets). 
Models listed here are some cases simplified versions of the ones ultimately described in papers.

## Table of contents
- [Installation](#installation)

- [About the project](#about-the-project)
    - [Datasets](#datasets)

- [Implementations](#implementations)
    - [Autoencoder - MNIST](#autoencoder-mnist)
    - [Variational Autoencoder - MNIST](#variational-autoencoder-mnist)
    - [DCGAN - MNIST](#dcgan-mnist)
    - [DCGAN - Polynomial](#dcgan-polynomial)
    - [WGAN - Polynomial](#wgan-polynomial)
    

## Installation
    $ git clone https://github.com/LorenzoValente3/Deep-Learning-Models.git
    $ cd Deep-Learning-Models/
    $ sudo pip3 install -r requirements.txt

## About the Project
The project is divided into two folders. 
Each of them contains the model implementations tailored to the different datasets they use. 

### Datasets
#### 1. MNIST
The [MNIST database](https://en.wikipedia.org/wiki/MNIST_database) is a handwritten digits dataset. 
The [[class]](./models_using_MNIST/MNIST_dataset.py) implemented in this project includes 60.000 training samples and 10.000 test samples. 
Each image is represented by 28x28 pixels, each value ranges from 0 to 255 and has a grayscale value.

#### 2. Polynomial
The Polynomial database is a bi-dimensional contour plots dataset. 
The [[class]](./GANs_using_Polynomials/POLY_dataset.py) implemented in this project includes 20000 samples showing polynomial up to a maximum degree (the fifth degree is taken into account as the maximum one here) in two variables.
Each image is represented by 40x40 pixels for 1-channel. 
An image of the dataset is shown below as an example.
The dataset _file.npy_ can be downloaded [here](https://drive.google.com/drive/folders/13HlpRhNTrz7WK0NQnrNoA7BQTlPOXb3u?usp=sharing). 

<p align="center">
    <img src="GANs_using_Polynomials/images/DCGAN/contour.png" width="400"\>
</p>

## Implementations 
### Autoencoder-MNIST
Implementation of a simple _Autoencoder_ for the MNIST data and an autoencoder that is able to _classify_ data in its latent dimension is built as well.

The design implemented here uses an architecture in which a _bottleneck_ in the network is imposed, forcing a compressed knowledge representation of the original input. In the absence of structure in the data (i.e. correlations between input features), compression and subsequent reconstruction would be very difficult. 
However, if some sort of structure exists in the data, this structure can be learned and therefore leveraged when forcing the input through the bottleneck.

#### Results 
Distribution of labaled data in its two-latent dimension space as well as plots of model score losses are shown below, for both models with and without the classifier.
Looking at the distribution of images in the latent space, it is clearly visible a linear behaviour. That happens because we have two dimensions to express a handwritten digit, it could happen that the heigth increases and the width increases as well, linearly as displayed. 
Model losses score converges at high epochs.

| Autoencoder without classifier                                                                               | Autoencoder with classifier  |
| ------------------------------                                                                              | -------------------------   |
|<p float="center"> <img src="/models_using_MNIST/images/AE/Latent dimension of Autoencoder without classifier.png" width="400" />                                   |           <img src="/models_using_MNIST/images/AE/Latent dimension of Autoencoder with classifier.png" width="400" />                                                                                                                                           |
|  <img src="/models_using_MNIST/images/AE/Accuracy of Autoencoder without classifier.png" width="400" />    |<img src="/models_using_MNIST/images/AE/Accuracy of Autoencoder with classifier.png" width="400" />                                                                                             |
</p>

#### Run Example
```
$ cd models_using_MNIST/
$ ipython AE.ipynb
```
[[Code]](models_using_MNIST/AE.ipynb)

### Variational Autoencoder-MNIST
Implementation of _Variational Autoencoder_ with factorized gaussian posteriors, <img src="https://render.githubusercontent.com/render/math?math=q_{\phi}(z|x ) = \mathcal{N}(z, \mu(x),diag(\sigma^{2}))"> and standard normal latent variables <img src="https://render.githubusercontent.com/render/math?math=p(z) =\mathcal{N}(0, \pmb I )">
The variational autoencoder able to _classify_ data in its data is built as well.

In contrast with the standard _Autoencoder_, the final part of the *encoder* structure bottleneck has two additional Dense layers: _mean_ and _variance_.
These two layers are used for the _sampling trick_ implementation, which help us to impose multi-gaussian distribution on the latent space.  


We have 'encoded_mean' and 'encoded_var' which we use for the sampling trick which help us to impose multi-gaussian distribution on the latent space.
We build the encoder in perfect analogy with the previous Autoencoder.
In this case we don't just have a latent dimension of two, but two Dense layers each one of dimension two (--> so for 2-dim latent space we need 2D mean and 2D var). 
A Lambda Layer is created, it takes both of the previous layers and measures them to the latent space dimension, via the self.sampling (reparametrization trick).
The sampling creates a structure that is a mixture of multiple gaussian distribution. in the end we spread out the standard normal distribution and shift the original mean.

Decoder is a perfect analogy with the standard autoencoder.

#### Results 
To visualize the results in the latent space we build a model for the encoder structure

In this case the distribution of images in latent space are no more linear but a point clouds. This happens because we impose a gaussian mixture model on the latent space, essentially we expect k-differnt point clouds, that are one digit each.

To generate a new sample we take a point in the latent space and the recostruction with the decoder will give us something consistent with the orginal dataset and finally generate new handwritten digit of that digit.




| VAE without classifier                                                                            | VAE with classifier         |
| ------------------------------                                                                      | -------------------------   |
|<p align="center"> <img src="/models_using_MNIST/images/VAE/Latent dimension of Variational Autoencoder without classifier.png" width="400" />                                                                                                     |           <img src="/models_using_MNIST/images/VAE/Latent dimension of Variational Autoencoder with classifier.png" width="400" />                                                                                                                                   |
|  <img src="/models_using_MNIST/images/VAE/Accuracy of Variational Autoencoder without classifier.png" width="400" />                                                                                                     |<img src="/models_using_MNIST/images/VAE/Accuracy of Variational Autoencoder with classifier.png" width="400" />                                                                         |
</p>


#### Run Example
```
$ cd models_using_MNIST/
$ ipython VAE.ipynb
```
[[Code]](models_using_MNIST/VAE.ipynb) [[Paper]](https://arxiv.org/abs/1312.6114)

### DCGAN-MNIST
Implementation of _Deep Convolutional Generative Adversarial Network_ with a custom training loop that aims at generating MNIST samples.
To address this task GPU is used. 

#### Results 
Below generated images over 500 epochs embedded in a _gif_ are reported and distribution of Generator and Discriminator losses over each epochs are displayed in the plot.


<p align="center">
 <img src="/models_using_MNIST/images/GAN/dcgan.gif" width="300" />      


#### Run Example
```
$ cd models_using_MNIST/
$ ipython DCGAN_mnist.ipynb
```
[[Code]](models_using_MNIST/DCGAN_mnist.ipynb) [[Paper]](https://arxiv.org/abs/1511.06434)

### DCGAN-Polynomial
Implementation of _Deep Convolutional Generative Adversarial Network_.

#### Run Example
```
$ cd GANs_using_Polynomials/
$ ipython DCGAN_poly.ipynb
```

[[Code]](GANs_using_Polynomials/DCGAN_poly.ipynb) 

### WGAN-Polynomial
Implementation of _Wasserstein Generative Adversarial Network_.

#### Run Example
```
$ cd GANs_using_Polynomials/
$ ipython WGAN_poly.ipynb
```

[[Code]](GANs_using_Polynomials/WGAN_poly.ipynb) [[Paper]](https://arxiv.org/abs/1701.07875)