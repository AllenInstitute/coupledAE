# A coupled autoencoder approach for multi-modal analysis of cell types

Rohan Gala, Nathan Gouwens, Zizhen Yao, Agata Budzillo, Osnat Penn,Bosiljka Tasic, Gabe Murphy, Hongkui Zeng, Uygar Sümbül

**Abstract**

Recent developments in high throughput profiling of individual neurons have spurred data driven exploration of the idea that there exist natural groupings of neurons referred to as cell types. The promise of this idea is that the immense complexity of brain circuits can be reduced, and effectively studied by means of interactions between cell types. While clustering of neuron populations based on a particular data modality can be used to define cell types, such definitions are often inconsistent across different characterization modalities. We pose this issue of cross-modal alignment as an optimization problem, and develop an approach based on coupled training of autoencoders as a framework for such analyses. We apply this framework to a Patch-seq dataset consisting of transcriptomic and electrophysiological profiles for the same set of neurons to study consistency of representations across modalities, and evaluate cross-modal data prediction ability. We explore the problem where only a subset of the neurons is characterized with more than one modality, and demonstrate that representations learned by coupled autoencoders can be used to identify cell types that are sampled only by a single modality.

**About this repository**

This repository contains codes and data files for the analysis presented in the NeurIPS 2019 proceedings. 

Files include
 - Codes to test representations with different coupling functions
 - Codes to obtain representations with the transcriptomic and electrophysiological data
 - Codes to reproduce main results of the paper

**Link to poster**


**Link to 3d representations**