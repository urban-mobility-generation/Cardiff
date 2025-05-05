# Cardiff

This repo contains PyTorch model definitions, training, and sampling code for our paper:
> Leveraging the Spatial Hierarchy: Coarse-to-fine Trajectory Generation via Cascaded Hybrid Diffusion

## Setup

First, download and set up the repo:

```
git clone https://github.com/urban-mobility-generation/Cardiff.git
cd Cardiff
```

We provide an [`environment.yml`](environment.yml) file 
that can be used to create a Conda environment. 

## Sampling 

- **Pre-trained checkpoints** are stored in [`saved_models`](saved_models)

- We provided a jupyter notebook [`inference_con.ipynb`](inference_con.ipynb) for quick sampling test.

## Training Cardiff

We provide a training script for DiT in [`train.py`](train.py). 

## BibTeX

This project is under review, and the preprint version is not online yet.
Please drop me a message if you want to use our code or if you have any questions.