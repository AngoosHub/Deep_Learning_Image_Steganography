import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import pyplot
import random
from torchvision import datasets, transforms
from PIL import Image
from pathlib import Path
import wandb
import time


class utils():

    @staticmethod
    def denormalize(tensor):
        denorm = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                                    std=[1/0.229, 1/0.224, 1/0.225])
        
        # REMOVE TO NORMALIZE! SKIPPING TO TEST IF IMPROVES MODEL TRAINING
        return denorm(tensor)
        # return tensor


    @staticmethod
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

        cover_denorm = utils.denormalize(cover).permute(1, 2, 0)
        cover_x_denorm = utils.denormalize(cover_x).permute(1, 2, 0)
        secret_denorm = utils.denormalize(secret).permute(1, 2, 0)
        secret_x_denorm = utils.denormalize(secret_x).permute(1, 2, 0)

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


    @staticmethod
    def plot_images_comparison(cover, cover_x, secret, secret_x, show_image=True):
        # Denomralize and plot images for visual comparison.

        # for i, x in enumerate(X):
        fig, ax = plt.subplots(2, 2)
        # Note: permute() will change shape of image to suit matplotlib 
        # (PyTorch default is [C, H, W] but Matplotlib is [H, W, C])
        cover_denorm = utils.denormalize(cover).permute(1, 2, 0)
        ax[0,0].imshow(cover_denorm) 
        ax[0,0].set_title(f"Original Cover")
        ax[0,0].axis("off")

        cover_x_denorm = utils.denormalize(cover_x).permute(1, 2, 0)
        ax[0,1].imshow(cover_x_denorm) 
        ax[0,1].set_title(f"Reconstructed Cover")
        ax[0,1].axis("off")


        secret_denorm = utils.denormalize(secret).permute(1, 2, 0)
        ax[1,0].imshow(secret_denorm) 
        ax[1,0].set_title(f"Original Secret")
        ax[1,0].axis("off")

        secret_x_denorm = utils.denormalize(secret_x).permute(1, 2, 0)
        ax[1,1].imshow(secret_x_denorm) 
        ax[1,1].set_title(f"Recovered Secret")
        ax[1,1].axis("off")

        fig.suptitle(f"Image Comparion", fontsize=16)

        if show_image:
            plt.show()
        
        return fig


    @staticmethod
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


    @staticmethod
    def get_single_batch_into_image(train_dataloader):
        # 1. Get a batch of images and labels from the DataLoader
        img_batch, label_batch = next(iter(train_dataloader))

        # 2. Get a single image from the batch and unsqueeze the image so its shape fits the model
        img_cover, label_single = img_batch[0].unsqueeze(dim=0), label_batch[0]
        print(f"Single image shape: {img_cover.shape}\n")
        img_secret, label_single = img_batch[1].unsqueeze(dim=0), label_batch[1]

        return img_cover, img_secret


    @staticmethod
    def test_plot_single_batch(img_cover, img_secret, test_model, device, SKIP_WANDB):

        cuda_cover = img_cover.to(device)
        cuda_secret = img_secret.to(device)

        # print(img_cover.is_cuda)
        # print(img_secret.is_cuda)
        # print(cuda_cover.is_cuda)
        # print(cuda_secret.is_cuda)

        # 3. Perform a forward pass on a single image
        test_model.eval()
        with torch.inference_mode():
            modified_cover, recovered_secret = test_model(cuda_cover, cuda_secret)
            
        # 4. Print out what's happening and convert model logits -> pred probs -> pred label
        # print(f"Output logits:\n{modified_cover}\n")
        # print(f"Output cover MSE loss:\n{torch.nn.functional.mse_loss(modified_cover, cuda_cover)}\n")
        # print(f"Output secret MSE loss:\n{torch.nn.functional.mse_loss(recovered_secret, cuda_secret)}\n")
        # print(f"Actual label:\n{label_single}")


        cover = cuda_cover.cpu().squeeze(0)
        cover_x = modified_cover.cpu().squeeze(0)
        secret = cuda_secret.cpu().squeeze(0)
        secret_x = recovered_secret.cpu().squeeze(0)

        # fig = utils.plot_images_comparison(cover, cover_x, secret, secret_x, show_image=False)
        fig = utils.test_plot(cover, cover_x, secret, secret_x, show_image=False)
        if not SKIP_WANDB:
            wandb.log({"Image": fig})



    @staticmethod
    def save_checkpoint(model, optimizer, optimizer_reveal, epoch_idx, batch_idx, SAVE_EPOCH_PROGRESS, EPOCHS):
        # Check if save model flag is active
            if SAVE_EPOCH_PROGRESS:
                timestamp = f'{time.strftime("%Y%m%d-%H%M%S")}'
                save_path = Path("saved_models") / timestamp
                if not save_path.is_dir():
                    save_path.mkdir(parents=True)
                    
                # model_name = f'{timestamp}/Test_Model_Epoch_{epoch}.pth'
                if epoch_idx == (EPOCHS):
                    model_name = f'{timestamp}/Test_Model_Epoch_{epoch_idx}_FINAL.pth'
                else:
                    model_name = f'{timestamp}/Test_Model_Epoch_{epoch_idx}.pth'

                
                save_state = {'epoch': epoch_idx,
                            'batch': batch_idx,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'optimizer_reveal_state_dict': optimizer_reveal.state_dict()}

                utils.save_model(model=save_state, model_name=model_name)

                # Try plotting a batch of image fed to model.
                global img_cover, img_secret
                utils.test_plot_single_batch(img_cover, img_secret, model)

                global WANDB_VAR
                if WANDB_VAR != False:
                    artifact_path = Path("saved_models/latest")
                    if not artifact_path.is_dir():
                        artifact_path.mkdir(parents=True)
                    
                    # latest_model = "latest/model.pth"
                    latest_model_path = Path("saved_models") / model_name
                    # utils.save_model(model=save_state, model_name=latest_model)
                    artifact = wandb.Artifact(name='model', type='model')
                    artifact.add_file(latest_model_path)
                    WANDB_VAR.log_artifact(artifact)



    @staticmethod
    def save_checkpoint_detector(model, optimizer, device, epoch_idx, batch_idx, SAVE_EPOCH_PROGRESS, EPOCHS):
        # Check if save model flag is active
            if SAVE_EPOCH_PROGRESS:
                timestamp = f'{time.strftime("%Y%m%d-%H%M%S")}'
                save_path = Path("saved_models") / timestamp
                if not save_path.is_dir():
                    save_path.mkdir(parents=True)
                    
                # model_name = f'{timestamp}/Test_Model_Epoch_{epoch}.pth'
                if epoch_idx == (EPOCHS):
                    model_name = f'{timestamp}/Test_Model_Detector_Epoch_{epoch_idx}_FINAL.pth'
                else:
                    model_name = f'{timestamp}/Test_Model_Detector_Epoch_{epoch_idx}.pth'

                
                save_state = {'epoch': epoch_idx,
                            'batch': batch_idx,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict()}

                utils.save_model(model=save_state, model_name=model_name)

