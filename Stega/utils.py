import torch
import numpy as np
import matplotlib.pyplot as plt
import random
from torchvision import datasets, transforms
from PIL import Image
from pathlib import Path



def denormalize(tensor):
    denorm = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                                  std=[1/0.229, 1/0.224, 1/0.225])
    
    # REMOVE TO NORMALIZE! SKIPPING TO TEST IF IMPROVES MODEL TRAINING
    return denorm(tensor)
    # return tensor


def plot_images_comparison(cover, cover_x, secret, secret_x, show_image=True):
    # Denomralize and plot images for visual comparison.

    # for i, x in enumerate(X):
    fig, ax = plt.subplots(2, 2)
    # Note: permute() will change shape of image to suit matplotlib 
    # (PyTorch default is [C, H, W] but Matplotlib is [H, W, C])
    cover_denorm = denormalize(cover).permute(1, 2, 0)
    ax[0,0].imshow(cover_denorm) 
    ax[0,0].set_title(f"Original Cover")
    ax[0,0].axis("off")

    cover_x_denorm = denormalize(cover_x).permute(1, 2, 0)
    ax[0,1].imshow(cover_x_denorm) 
    ax[0,1].set_title(f"Reconstructed Cover")
    ax[0,1].axis("off")


    secret_denorm = denormalize(secret).permute(1, 2, 0)
    ax[1,0].imshow(secret_denorm) 
    ax[1,0].set_title(f"Original Secret")
    ax[1,0].axis("off")

    secret_x_denorm = denormalize(secret_x).permute(1, 2, 0)
    ax[1,1].imshow(secret_x_denorm) 
    ax[1,1].set_title(f"Recovered Secret")
    ax[1,1].axis("off")

    fig.suptitle(f"Image Comparion", fontsize=16)

    if show_image:
        plt.show()
    
    return fig



def save_model(model, model_name, save_path="saved_models"):
    save_path = Path(save_path)
    if not save_path.is_dir():
        save_path.mkdir()

    # Create model save path
    # if not model_name.endswith(".pth") and model_name.endswith(".pt"):
    #     model_name = f"{model_name}.pth"
    # assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
    
    model_save_path = save_path / model_name

    # Save the model
    torch.save(obj=model, f=model_save_path)
    # print(f"Model state saved to: {model_save_path}")


