# Deep Fusion Generative Adversarial Networks for Text-to-Image Synthesis

A PyTorch implementation of 
[Deep Fusion GAN](https://arxiv.org/abs/2008.05865) 
by *Ming Tao, Hao Tang, Songsong Wu, Nicu 
Sebe, Xiaoyuan Jing, Fei Wu, Bingkun Bao*.

## Deep Fusion GAN architecture

<p align="center">
 <img src="./images/model.png" alt="Drawing", width=75%, height="100%">
</p>

<div align="center">
  <b>The architecture of the proposed DF-GAN for text-to-image synthesis. DF-GAN generates high-resolution images directly by one pair of generator and discriminator and fuses the text information and visual feature maps through multiple Deep text-image Fusion Blocks (DFBlock) in UPBlocks. Armed with Matching-Aware Gradient Penalty (MA-GP) and one-way output, our model can synthesize more realistic and text-matching images..</b>
</div>