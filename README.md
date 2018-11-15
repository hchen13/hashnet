# Image Matching with Deep Learning

## Abstract 

*Retrieving similar images of a given query image from a large corpus is a demanding functionality of our company and it is challenging. We adopted a deep learning based encoding and retrieving method to achieve such task. The base idea is that deep neural networks is able to learn rich image representations. The method then learns binary codes by introducing an extra hidden layer for representing the images. After that, image retrieval is performed on the binary codes to accelerate the process.*

## Introduction

The method is originally proposed by [Kevin *et al.*](https://www.iis.sinica.edu.tw/~kevinlin311.tw/cvprw15.pdf) The theory is based on the findings of deep convolutional neural networks (CNNs) over the years: A typical CNN is consists with several convolution layers followed by a number of fully connected layers with ReLu activation. And a typical way of interpreting this architecture is that the convolution layers gradually learns low-level image features, such as edges, corners, colors, etc. to mid-level feature, such as shapes and clusters. As the layers approach the top, the features learned are increasingly abstract and more representative. The fully connected layers then summarize the correlations between the features that convolution layers learned to represent images and the output labels. Thus, in a classification task, the fully connected layers effectively learn features that make an image a specific class. In other words, images of an identical class should have similar outputs from the convolution layers. Image retrieval can then be achieved by comparing the simiarities of the feature vectors in the corpus. 

However, the outputs of the last layer of typical CNN architectures such as VGG16 and Inception are usually 3-dimensional tensors, flattening them as feature vectors and using them directly for similarity calculations in a large corpus is impractical. 

This work utilizes a supervised learning approach to solve the problem by training a neural network to produce binary hash codes that can represent images, and use the binary codes as the first phase of image retrieval to shortlist the candidates for further rich-feature-vector comparison. The unusual part of this approach is that instead of directly training to solve the problem itself, the training process is actually aiming at the goal of image classification. 

## Method

The complete process of this image retrieval method consits of 3 steps: 

1. extracting binary code and feature vector
2. phase 1 of searching: shortlist the candidates using binary code
3. phase 2 of searching: image retrieval by vector similarity comparison

### Extracting Binary Code and Feature Vector

This step is achieved by pre-training the neural network using domain-specific image data samples. And the architecture of the neural network is the combination of a pre-trained topless VGG16 model concatenated with several fully-connected layers with ReLu activation function except one using sigmoid activation, as shown in the figure below. 

The output of the topless VGG16 model induced by an image is served as the real-value feature vector since CNNs are empirically considered good at learning representative features. 

The reason sigmoid function is used in a particular layer is that it outputs values in the range between 0 and 1, which can be binarized by having all values less than 0.5 to be 0, otherwise 1, so that this layer is responsible for producing representative binary codes. 

The underlying assumption is that for image classification task, the second to last layer outputs a vetor that contains very important class information so that the last layer can successfully classify images. With this in mind, the second to last layer is designed to use sigmoid function, so that it can be trained to summarize representative binary codes. 

### Phase 1: Shortlisting 

With the above neural network trained, the image corpus can then be pre-processed, each of which is represented with a binary vector and a real-value vector. Then all the binary vectors can be represented as matrix $B$ and feature vectors as $F$, where $B_i$ and $F_i$ are the vectors of the $i$-th image. 

Assuming for a given target image, the neural network produces vector $b_t$ and $f_t$ representing the binary and real-value vectors respectively. 

Calculating the hamming distance between $B$ and $b_t$ yields a distance vector whose length equals to the size of the corpus. The binary code $B_i$ with minimum hamming distance against $b_t$ does not necessarily guarantee the corresponding is the most similar image since the binarization sacrifices significant amount of information. Therefore, the compensation is to shortlist a number of candidates with top-$k$ minimum distances.

### Phase 2: Retrieval Through Similarity Comparison

After the first phase, the corpus has been filtered down to a limited number of candidates, which made vector similarity calculation affordable. There are more than one method of calculating similarities between vectors, such as [Euclidean distance](https://en.wikipedia.org/wiki/Euclidean_distance) and [cosine similarities](https://en.wikipedia.org/wiki/Cosine_similarity). This project uses cosine similarity since a similarity score metric between 0 and 1 is asked to be returned along with the matched image. 

#### Euclidean distance between a vector and a matrix

Suppose a vector $v$ and a matrix $M$ where $v \in R^{1 \times n}, M \in R^{m \times n}$. The Euclidean distance is calculated as the Euclidean distance between $v$ and each row of $M$:
$$
d = \begin{bmatrix}
\sqrt{\sum_{i=1}^{n}(v_i-M_{1i})^2} \\
\sqrt{\sum_{i=1}^{n}(v_i-M_{2i})^2} \\
\vdots \\
\sqrt{\sum_{i=1}^{n}(v_i-M_{mi})^2} \\
\end {bmatrix}
$$

#### Cosine similarity between a vector and a matrix

Suppose a vector $v$ and a matrix $M$ where $v \in R^{1 \times n}, M \in R^{m \times n}$. The cosine similarity is as the cosine similarity between the $v$ and each row of $M$. However, by applying vectorization, the calculation can be optimized:
$$
s = {v \cdot M^T \over |v| \times |M|}
$$