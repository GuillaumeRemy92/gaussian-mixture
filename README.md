# Denoising Gaussian mixture models via Wasserstein gradient flow

Gaussian mixture models are one of the most popular data fitting models as they appear in a wide variety
of context. The model consists of a set of points which are each sample from an unknown distribution plus an i.i.d. Gaussian random variable. The goal of the denoising problem is given the observed data points to "remove" the Gaussian noise and recover the underlying distribution. This problem is typically tackle using the well known Expectation-Maximization (EM) algorithm. We propose here a numerical study of an alternative method, a space-time discretization of the Wasserstein gradient flow (WGF), and compare its performance to the EM algorithm. Below are figures correponding to the denoising using WGF of three initial distributions: i) points uniformly sampled on the unit circle, ii) points uniformly sampled on a T-shaped figure, iii) points sampled from 5 point masses in the plane. More comming very soon. 

![alt text](https://github.com/GuillaumeRemy92/gaussian-mixture/blob/main/Figure1-gmm.png?raw=true)
