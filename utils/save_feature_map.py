import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision.utils import make_grid

# Get the color map by name:
cm = plt.get_cmap('terrain')


@torch.no_grad()
def save_feature_map(tensor, fp, **kwargs): # [c, h, w]
    tensor = tensor.unsqueeze(1)
    grid = make_grid(tensor, **kwargs).permute(1, 2, 0).to('cpu').numpy()
    colored_ndarr = (cm(grid[:, :, 0])[:, :, :3] * 255).astype(np.uint8)
    im = Image.fromarray(colored_ndarr)
    im.save(fp)
