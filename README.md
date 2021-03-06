# W-GAN
Pytorch implementation of [Wasserstein GAN](https://arxiv.org/abs/1701.07875) by Martin Arjovsky et al.
## Examples
W-GAN (G,D : DCGAN)
for Randomly choosen celebrity (10 picture)

`lr = 1E-4` , `n_critic = 5` , `latent_dim = 100` using RMSprop

Note that the images were resized to (64,64).

## Training (4500 epochs)
![samples_imgs](https://github.com/Oowkrap/W-GAN/blob/master/imgs/sample_imgs.png)

## Loss
![loss](https://github.com/Oowkrap/W-GAN/blob/master/imgs/loss.png)


# W-GAN - GP
Pytorch implementation of [Improved Training of Wasserstein GANs](https://arxiv.org/abs/1704.00028) by Gulrajani et al.
## Examples

## Training

## Loss




## Sources and inspiration
https://github.com/martinarjovsky/WassersteinGAN

https://github.com/eriklindernoren/PyTorch-GAN

https://github.com/EmilienDupont/wgan-gp
