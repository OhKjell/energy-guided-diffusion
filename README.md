# 🥚 EGG: Energy Guided Diffusion for optimizing neurally exciting images

This is the official implementation of the paper "Energy Guided Diffusion for Generating Neurally Exciting Images".

<p align="center"><img src="./assets/teaser.png" width="100%" alt="EGG diffusion generated neurally exciting images" /></p>

> [**Energy Guided Diffusion for Generating Neurally Exciting Images**](https://www.biorxiv.org/content/10.1101/2023.05.18.541176v1), \
> Pawel A. Pierzchlewicz, Konstantin F. Willeke, Arne F. Nix, Pavithra Elumalai, Kelli Restivo, Tori Shinn, Cate Nealley, Gabrielle Rodriguez, Saumil Patel, Katrin Franke, Andreas S. Tolias, and Fabian H. Sinz

*This repository is based on the [guided-diffusion](https://github.com/openai/guided-diffusion) repository.*

# Installation

## Package Requirements
Some packages need to be downloaded manually.
Run the following commands to download the required packages:
```bash
./download_requirements.sh
```

You can install the remaining packages by running:
```bash
pip install -e .
```

## Pre-trained model weights
To run EGG you need to download the pre-trained weights of the ADM model.
The experiments use a model pretrained by OpenAI on 256x256 ImageNet images.

| Model                   | Weights |
|-------------------------| --- |
| ImageNet 256x256 uncond | [256x256_diffusion_uncond.pt](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt) |

place the weights in the `models` folder.

# Usage
Here is a minimal example for running the EGG diffusion on a pretrained model.

```python
from functools import partial

from egg.models import models
from egg.diffusion import EGG

# Setup the parameters
energy_scale = 5
num_samples = 1
num_steps = 50
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def energy_fn(pred_x_0, unit_idx=0):
    """
    Energy function for optimizing MEIs, i.e. images that maximally excite a given unit.

    :param pred_x_0: the predicted "clean" image
    :param unit_idx: the index of the unit to optimize
    :return: the neural of the predicted image for the given unit
    """
    return dict(train=models['task_driven']['train'](pred_x_0)[..., unit_idx])


diffusion = EGG(
    diffusion_artefact='./models/256x256_diffusion_uncond.pt',
    num_steps=num_steps
)
samples = diffusion.sample(
    energy_fn=partial(energy_fn, unit_idx=0),
    energy_scale=energy_scale,
    num_samples=1,
    device=device,
)
```