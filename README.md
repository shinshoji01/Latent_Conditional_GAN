# Latent Conditional GAN
This repository is to introduce PyTorch implementation of our paper: ["LCGAN: Conditional GAN with Multiple Discrete Classes"](https://www.jstage.jst.go.jp/article/pjsai/JSAI2020/0/JSAI2020_2K4ES202/_article/-char/ja/)

---
## Introduction

This paper introduces Latent Conditional GAN. This is Conditional GAN whose label is generated by the latent code of Variational Auto-Encoder(VAE). Thanks to this kind of label representation, comparing to the one-hot vector, the model has advantages as follows:
- Reduction of the label dimension
- Inclusion of the class relevance
- Representing the continuous label by discrete classes
- *not depending on the structure of GANs and VAEs

I conducted the experimentation with CelebA dataset, which has plenty of annotations. As a result, this model could generate data that changes continuously in regard to the given class vector with a lower dimension.

---
## Notebook Structure
The whole picture of my model is shown below.

<img width="743" alt="Screen Shot 2020-07-28 at 0 37 02" src="https://user-images.githubusercontent.com/28431328/88561521-7c600600-d06a-11ea-9126-c434364f06eb.png">

This procedure is divided into **3 notebooks** and they are colored separately. I'm gonna show the results and concise explanations, please visit the notebooks for the detail and the implementation.

#### VAE
The reconstructed images are shown below, where the upper images represent the input image and the others are reconstructed by the model. The reconstructed image seems to be distinguishable among the classes.

image_VAE

#### Dimension Reduction

image_DR
gif_DR

#### LCGAN

image_LCGAN
gif_LCGAN

---
## Git LFS (large file storage)
Since this repository contains the parameters of VAE and LCGAN. I used Git LFS to store a large file. The codes below are the recipe for this.

```bash
brew update
brew install git-lfs
```
- then, navigate to this repository.
```bash
git lfs install
git lfs fetch --all
git lfs pull
```

[Japanese explanation](https://www.slideshare.net/hibiki443/git-git-lfs-60951449)

---
## Citation

---
## Coming soon
Some are not explained which include
- How to download and use CelebA dataset

---
## Notice
ここに掲載した著作物の利用に関する注意 本著作物の著作権は人工知能学会に帰属しま
す。本著作物は著作権者である人工知能学会の許可のもとに掲載するものです。ご利用に当
たっては「著作権法」に従うことをお願いいたします。 

Notice for the use of this material. The copyright of this material is retained by
the Japanese Society for Artificial Intelligence (JSAI). This material is published
here with the agreement of JSAI. Please be complied with Copyright Law of Japan
if any users wish to reproduce, make derivative work, distribute or make available
to the public any part or whole thereof.
All Rights Reserved, Copyright (C) The Japanese Society for Artificial Intelligence. 
