# Modeling of Generated Data from Text Documents

<p align="justify">Deep generative modelling of text documents using GANs and NLP.</p>

## Motivation

<p align="justify">In this modern day and age, a large amount of data is collected everyday. With this ever increasing data, it has become more and more difficult to manually process the data and get the desired information. Topic modelling provides us with a method to organise, understand and summarise large amounts of data. It does this by finding various patterns in the given data. With this project, our main goal was to create a GAN model that can successfully classify documents based on generated text.</p>

## Dataset

<p align="justify">The dataset used is the “20newsgroups” which comprises of 18828 newsgroups posts on 20 topics. All the messages on the newgroups were merged under their respective titles and all the titles from the 20 different topics were merged into a single file. This file was used as the dataset for the model. The dataset consists of two columns: Label and Document.</p>

## Data preprocessing

<p align="justify">Since the data contained a lot of commonly used words like prepositions and articles, these words had to be removed. After that, the text was converted to lowercase for better efficiency. The text was then tokenized, stemmed, lemmatized. The final step of the preprocessing was converting the result of all the previous steps into a vector and arguments were passed into the program.</p>

## Model

<p align="justify">Initially the vectors are masked, gradients are defined and then gradient norm scaling is done. After this, generator and discriminator are defined.</p>

<p align="justify">The generator contains two fully connected ReLU layers and a final sigmoid layer. The discriminator has one leaky ReLU layer and it linearly maps input vector to input space.</p>

<p align="centre">
<img src="https://user-images.githubusercontent.com/76239328/168326987-1f4afa71-4f38-4e70-aea4-6fb10d686201.png"/><br>
<i>GAN Layout</i>
</p>

## Training

<p align="justify">For training, the input dataset and model output directories are passed to the train function. The training data is first divided into mini batches and these mini batches are used for training. Two copies of the generator are created with one network taking the real sample as input from a mini batch and the other taking the generated samples as input. The update to the discriminator and generator are done separately and at each update, we generate a new noise vector to pass to the generator, and a new noise mask for the denoising autoencoder (the same noise mask is used for each input in the batch). The training was carried on for 10000 steps. </p>

## Training validation and evaluation

<p align="centre">
<img src="https://user-images.githubusercontent.com/76239328/168326653-a0770186-8131-4315-b332-2d60d66e2f4e.png"/><br>
<i>Discriminator and Generator Losses</i><br>
</p>

<p align="centre">
<img src="https://user-images.githubusercontent.com/76239328/168326762-b1868d36-6d80-42f1-9106-c05a7914013f.png"/><br>
<i>Validation accuracy</i>
</p>

<p align="justify">The final validation accuracy at the end of the training was 59.6%.
The testing accuracy was 57%.</p>

## Future improvements
* One of the improvements that could be made is to select a better dataset to improve the accuracy. 
* Instead of using batch normalisation, spectral normalisation can be used which can stabilise the training of our discriminator.

## Team members
* [Aryan Rajput](https://github.com/AryanRajput2083)
* [Milly Sharma](https://github.com/milly710)
* [Brahatesh Vasantha](https://github.com/brahatesh)
* [Deepak Sharma](https://github.com/deep0505sharma)
* [Shreya Attri](https://github.com/Shreya003)

## References
* https://www.ibm.com/cloud/learn/data-modeling#:~:text=Data%20modeling%20is%20the%20process,between%20data%20points%20and%20structures
* https://arxiv.org/abs/1612.09122
* https://arxiv.org/abs/1511.06434
* https://arxiv.org/abs/1609.03126
* https://youtube.com/playlist?list=PLhhyoLH6IjfwIp8bZnzX8QR30TRcHO8Va
