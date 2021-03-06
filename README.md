# A coupled autoencoder approach for multi-modal analysis of cell types

Rohan Gala, Nathan Gouwens, Zizhen Yao, Agata Budzillo, Osnat Penn,Bosiljka Tasic, Gabe Murphy, Hongkui Zeng, Uygar Sümbül

**Abstract**

Recent developments in high throughput profiling of individual neurons have spurred data driven exploration of the idea that there exist natural groupings of neurons referred to as cell types. The promise of this idea is that the immense complexity of brain circuits can be reduced, and effectively studied by means of interactions between cell types. While clustering of neuron populations based on a particular data modality can be used to define cell types, such definitions are often inconsistent across different characterization modalities. We pose this issue of cross-modal alignment as an optimization problem and develop an approach based on coupled training of autoencoders as a framework for such analyses. We apply this framework to a Patch-seq dataset consisting of transcriptomic and electrophysiological profiles for the same set of neurons to study consistency of representations across modalities, and evaluate cross-modal data prediction ability. We explore the problem where only a subset of neurons is characterized with more than one modality, and demonstrate that representations learned by coupled autoencoders can be used to identify types sampled only by a single modality.

**About this repository**

This repository contains codes for the analysis presented in the NeurIPS 2019 proceedings. 

**Patch-seq dataset representations**

3D latent space representations of 1,518 cells in the Transcriptomic and electrophysiology dataset, colored by the ground truth cell type annotation obtained using autoencoders in the 
 - Uncoupled setting: Coupling strenth = 0.0

![$\lambda$=0.0 Transcriptomics representations](./docs/T_z_0-0.gif)
![$\lambda$=0.0 Electrophysiology representations](./docs/E_z_0-0.gif)

 - Coupled setting: Coupling strength = 1.0

![$\lambda$=1.0 Transcriptomics representations](./docs/T_z_1-0.gif)
![$\lambda$=1.0 Electrophysiology representations](./docs/E_z_1-0.gif)

**Link to poster**

[Poster](https://github.com/AllenInstitute/coupledAE/blob/master/docs/Gala_NeurIPS_2019_poster.pdf)