# Cardiff

This repo contains PyTorch model definitions, training, and sampling code for our paper https://www.arxiv.org/abs/2507.13366:
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

We provide a training script for Cardiff in [`train.py`](train.py). 

## BibTeX

```
@article{guo2025leveraging,
  title={Leveraging the Spatial Hierarchy: Coarse-to-fine Trajectory Generation via Cascaded Hybrid Diffusion},
  author={Guo, Baoshen and Hong, Zhiqing and Li, Junyi and Wang, Shenhao and Zhao, Jinhua},
  journal={arXiv preprint arXiv:2507.13366},
  year={2025}
}

```


Please drop me an email if you have any questions .