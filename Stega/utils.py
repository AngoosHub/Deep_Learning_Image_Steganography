import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import pyplot
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


def test_plot(cover, cover_x, secret, secret_x, show_image=True):
    
    # cover = cover.clamp(min=0, max=1)
    # cover_x = cover_x.clamp(min=0, max=1)
    # secret = secret.clamp(min=0, max=1)
    # secret_x = secret_x.clamp(min=0, max=1)
    print(f"cover min: {torch.min(cover)} max: {torch.max(cover)}")
    print(f"cover_x min: {torch.min(cover_x)} max: {torch.max(cover_x)}")
    print(f"secret min: {torch.min(secret)} max: {torch.max(secret)}")
    print(f"secret_x min: {torch.min(secret_x)} max: {torch.max(secret_x)}")

    fig = pyplot.figure(figsize=(10, 20))

    gs0 = gridspec.GridSpec(2, 1)
    gs00 = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs0[0], wspace=0)
    gs01 = gridspec.GridSpecFromSubplotSpec(2, 5, subplot_spec=gs0[1], wspace=0)

    cover_denorm = denormalize(cover).permute(1, 2, 0)
    cover_x_denorm = denormalize(cover_x).permute(1, 2, 0)
    secret_denorm = denormalize(secret).permute(1, 2, 0)
    secret_x_denorm = denormalize(secret_x).permute(1, 2, 0)

    print(f"cover min: {torch.min(cover_denorm)} max: {torch.max(cover_denorm)}")
    print(f"cover_x min: {torch.min(cover_x_denorm)} max: {torch.max(cover_x_denorm)}")
    print(f"secret min: {torch.min(secret_denorm)} max: {torch.max(secret_denorm)}")
    print(f"secret_x min: {torch.min(secret_x_denorm)} max: {torch.max(secret_x_denorm)}")

    # plt.subplots_adjust(wspace=0, hspace=0)

    # Image Comparison Section
    gs00_ax00 = fig.add_subplot(gs00[0, 0])
    gs00_ax00.imshow(cover_denorm) 
    gs00_ax00.set_title(f"Original Cover")
    gs00_ax00.axis("off")

    gs00_ax01 = fig.add_subplot(gs00[0, 1])
    gs00_ax01.imshow(cover_x_denorm) 
    gs00_ax01.set_title(f"Reconstructed Cover")
    gs00_ax01.axis("off")

    gs00_ax10 = fig.add_subplot(gs00[1, 0])
    gs00_ax10.imshow(secret_denorm) 
    gs00_ax10.set_title(f"Original Secret")
    gs00_ax10.axis("off")

    gs00_ax11 = fig.add_subplot(gs00[1, 1])
    gs00_ax11.imshow(secret_x_denorm) 
    gs00_ax11.set_title(f"Recovered Secret")
    gs00_ax11.axis("off")


    # Residual Error Section
    cover_error = np.abs(cover_x_denorm - cover_denorm)
    secret_error = np.abs(secret_x_denorm - secret_denorm)
    

    gs01_ax00 = fig.add_subplot(gs01[0, 0])
    gs01_ax00.imshow(cover_denorm) 
    gs01_ax00.set_title(f"Cover")
    gs01_ax00.axis("off")

    gs01_ax01 = fig.add_subplot(gs01[0, 1])
    gs01_ax01.imshow(cover_x_denorm) 
    gs01_ax01.set_title(f"Re-Cover")
    gs01_ax01.axis("off")

    gs01_ax02 = fig.add_subplot(gs01[0, 2])
    gs01_ax02.imshow(np.multiply(cover_error, 1.0))
    gs01_ax02.set_title(f"Residual Error x1")
    gs01_ax02.axis("off")

    gs01_ax03 = fig.add_subplot(gs01[0, 3])
    gs01_ax03.imshow(np.multiply(cover_error, 3.0))
    gs01_ax03.set_title(f"Residual Error x3")
    gs01_ax03.axis("off")

    gs01_ax04 = fig.add_subplot(gs01[0, 4])
    gs01_ax04.imshow(np.multiply(cover_error, 5.0))
    gs01_ax04.set_title(f"Residual Error x5")
    gs01_ax04.axis("off")

    gs01_ax10 = fig.add_subplot(gs01[1, 0])
    gs01_ax10.imshow(secret_denorm) 
    gs01_ax10.set_title(f"Secret")
    gs01_ax10.axis("off")

    gs01_ax11 = fig.add_subplot(gs01[1, 1])
    gs01_ax11.imshow(secret_x_denorm) 
    gs01_ax11.set_title(f"Re-Secret")
    gs01_ax11.axis("off")

    gs01_ax12 = fig.add_subplot(gs01[1, 2])
    gs01_ax12.imshow(np.multiply(secret_error, 1.0))
    gs01_ax12.set_title(f"Residual Error x1")
    gs01_ax12.axis("off")

    gs01_ax13 = fig.add_subplot(gs01[1, 3])
    gs01_ax13.imshow(np.multiply(secret_error, 3.0))
    gs01_ax13.set_title(f"Residual Error x3")
    gs01_ax13.axis("off")

    gs01_ax14 = fig.add_subplot(gs01[1, 4])
    gs01_ax14.imshow(np.multiply(secret_error, 5.0))
    gs01_ax14.set_title(f"Residual Error x5")
    gs01_ax14.axis("off")


    fig.suptitle(f"Image Comparison", fontsize=16)

    if show_image:
        plt.show()
    
    return fig



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


