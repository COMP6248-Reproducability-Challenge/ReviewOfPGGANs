# Review of ‘Progressive Growing Of GANs for Improved Quality, Stability, and Variation’
Authors: Jay Santokhi, Adrian Wilczynski & Percy Jacks<br />
Original paper can be found [here](https://openreview.net/forum?id=Hk99zCeAb)
<br />
<br />

## Abstract
Karras et al. (2018) introduced a new methodology for training GANs through progressively growing the Generator and Discriminator starting at a low resolution and gradually increasing, through the addition of new network layers. Their work on Progressively Grown GANs (PGGAN) with the addition of two normalisation techniques (Equalised Learning Rate and Pixelwise Feature Vector Normalisation) claimed to improve stability, reduce training time and improve generated image quality over current state of the art GANs. This review paper has found that this methodology and its claimed benefits are mostly valid, however, as well as requir- ing considerable computational assets it is also ineffective for simple datasets.
<br />
<br />

## Repo file structure
* GAN - contains code for running the basic DCGAN and LSGAN networks on various datasets.
* PGGAN - contains code for running the Progressive Growing GAN networks on celeba (can be trivially modified for other datasets)
* Results - contains log files and videos of the output of various GANs

```
.
├── GAN - code for running the regular GAN
├── PGGAN - code for running the progressive growing GAN
├── Results - training videos, loss graphs and training logs
│   ├── CIFAR-10
│   |   └── LOGs
│   ├── CelebA
│   |   └── LOGs
│   ├── Fashion-MNIST
│   |   └── LOGs
│   ├── MNIST
│   |   └── LOGs
│   └── POKEMON
│       └── LOGs
├── README.md
└── ReviewPaper.pdf
```
<br />

## Explanation/Methodology
DCGAN and LSGAN were both trained on a full dataset, with a batch size of 64 images and trained for 375 epochs.

PGGAN was trained differently than this.
* Each resolution has an increasing number of batch-epochs to train on.
* For each batch-epoch, a random selection of images of batch size was chosen.
* One batch-epoch is defined as a single batch trained for a single cycle.

For example:<br />
Using a batch-epoch of 4k, with a batch size of 64, means training 4000 random batches from the dataset.<br />
Using this method, less time and less data an be used for training while not noticeably hindering performance.
<br />

## Results of Networks
Results using the CelebA dataset, trained on DCGAN, LSGAN and PGGAN.<br />
(Higher resolution videos can be found in 'Results' folder)

**DCGAN and LSGAN**<br />
Training epochs: 375<br />
Sample size: 100k images<br />
Latent noise vector size: 200<br />
Batch size: 64<br />
Training time: 6h 15min and 6h 10min<br />
GPU: GTX1060 (3GB VRAM)

![DCGAN](/Results/CelebA/dcgan_celeba32.gif) ![LSGAN](/Results/CelebA/lsgan_celeba32.gif)

**PGGAN**<br />
Batch-epochs: [4k, 8k, 16k, 32k]<br />
Sample size: 10K images<br />
Latent noise vector size: 64<br />
Batch size: 64<br />
Training time: 4h<br />
GPU: Google Colab GPU

![PGGAN](/Results/CelebA/pggan_celeba32.gif)

