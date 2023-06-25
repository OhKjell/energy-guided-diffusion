from functools import partial
import torch
import torch.nn.functional as F
from egg.models import models
from egg.diffusion import EGG
import matplotlib.pyplot as plt
import numpy as np

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
    x = F.interpolate(
        pred_x_0.clone(), size=(100, 100), mode="bilinear", align_corners=False
    ).mean(1, keepdim=True) # resize to 100x100 and convert to grayscale
    
    return dict(train=models['task_driven']['train'](x)[..., unit_idx])


diffusion = EGG(
    diffusion_artefact='./models/256x256_diffusion_uncond.pt',
    num_steps=num_steps
)

samples = diffusion.sample(
    energy_fn=partial(energy_fn, unit_idx=0),
    energy_scale=energy_scale
)
*_, sample = samples
print(sample["sample"].shape)
#image = np.transpose(sample["sample"][0], (1,2,0))
image = np.transpose(sample["sample"][0].cpu().numpy(), (1, 2, 0))


#image_tensor = sample["sample"][0].cpu()
#image_tensor = torch.transpose(image_tensor, 0, 2).numpy()
#image = np.transpose(image_tensor, (1, 2, 0))
#image = image / 255.0
image = image.clip(0, 1)

#plt.imshow(image)
#plt.show()
plt.imshow(tar.cpu().detach().squeeze(), cmap="gray", vmin=-1.7, vmax=1.7)
plt.axis("off")
plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
plt.margins(0, 0)
plt.gca().xaxis.set_major_locator(plt.NullLocator())
plt.gca().yaxis.set_major_locator(plt.NullLocator())
plt.savefig(
    "output.png", transparent=True, bbox_inches="tight", pad_inches=0
)
 