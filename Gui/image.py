
import torch
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import pyplot
import numpy as np
import time

from torch.utils.data import IterableDataset
import queue
from torchvision import transforms
from torchvision.utils import save_image



device = "cuda" if torch.cuda.is_available() else "cpu"
old_model_path = Path("Deep_Learning_Image_Steganography/Trained_Models/Steganography/Old_Model_V1.pth")
model_V3_path = Path("Deep_Learning_Image_Steganography/Trained_Models/Steganography/Stega_Model_V3_(detector_V1).pth")
MODEL_PATH = model_V3_path
COVER_SAVE_PATH = "Deep_Learning_Image_Steganography/Saved_Images/Modified_Cover"
SECRET_SAVE_PATH = "Deep_Learning_Image_Steganography/Saved_Images/Revealed_Secret"


class ImageDataset(IterableDataset):
    '''
    Wraps the user input cover and secret images as an Iterable Dataset to feed as input into steganography model.
    '''
    def __init__(self, image_queue):
      self.queue = image_queue

    def read_next_image(self):
        while self.queue.qsize() > 0:
            # you can add transform here
            yield self.queue.get()
        return None

    def __iter__(self):
        return self.read_next_image()


class StegaImageProcessing():
    '''
    A Utility class that handling various images processing, model output processing, matplotlib plotting functions
    to support visuals displayed in GUI class.
    '''

    @staticmethod
    def hide_image(model, cover_o, secret_o):

        buffer = queue.Queue()
        # new_input = Image.open(image_path)

        # Populate queue with cover and secret
        buffer.put(StegaImageProcessing.normalize_and_transform(cover_o))
        buffer.put(StegaImageProcessing.normalize_and_transform(secret_o))

        dataset = ImageDataset(buffer)
        dataloader = torch.utils.data.DataLoader(dataset=dataset, 
                                                batch_size=2,
                                                shuffle=False)

        for data in dataloader:
            a, b = data.split(2//2,dim=0)
            cuda_cover = a.to(device)
            cuda_secret = b.to(device)

            model.eval()
            with torch.inference_mode():
                modified_cover, recovered_secret = model(cuda_cover, cuda_secret)
            
            cover = cuda_cover.cpu().squeeze(0)
            cover_x = modified_cover.cpu().squeeze(0)
            secret = cuda_secret.cpu().squeeze(0)
            secret_x = recovered_secret.cpu().squeeze(0)

            # Save modified cover image
            if not Path(COVER_SAVE_PATH).is_dir():
                Path(COVER_SAVE_PATH).mkdir(parents=True, exist_ok=True)
            # img = Image.fromarray(modified_cover.cpu().detach().numpy()[0])
            timestamp = f'{time.strftime("%Y%m%d-%H%M%S")}'
            # img.save(COVER_SAVE_PATH / f"Stega_{timestamp}.png")
            save_image(StegaImageProcessing.denormalize(modified_cover.cpu()), f'{COVER_SAVE_PATH}/Stega_{timestamp}.png')
            save_image(StegaImageProcessing.denormalize(recovered_secret.cpu()), f'{COVER_SAVE_PATH}/Stega_r_{timestamp}.png')

            save_image(StegaImageProcessing.denormalize(cuda_cover.cpu()), f'{COVER_SAVE_PATH}/Stega_o_{timestamp}.png')
            save_image(StegaImageProcessing.denormalize(cuda_secret.cpu()), f'{COVER_SAVE_PATH}/Stega_r_o_{timestamp}.png')


            # plot_images_comparison(cover, cover_x, secret, secret_x, show_image=True)
            StegaImageProcessing.test_plot(cover, cover_x, secret, secret_x, show_image=True)

            break

        return "Done"


    @staticmethod
    def normalize_and_transform(image):
        data_transforms = transforms.Compose([
            transforms.Resize(size=(224)),
            transforms.CenterCrop(size=(224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
        image_transform = data_transforms(image)
        return image_transform
        

    @staticmethod
    def denormalize_and_toPIL(tensor):
        pil_transfrom = transforms.ToPILImage()
        tensor_denorm = StegaImageProcessing.denormalize(tensor)
        image = pil_transfrom(tensor_denorm)
        return image


    @staticmethod
    def denormalize(tensor):
        denorm = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                                    std=[1/0.229, 1/0.224, 1/0.225])
        return denorm(tensor)


    @staticmethod
    def plot_images_comparison(cover, cover_x, secret, secret_x, show_image=True):
        # Denomralize and plot images for visual comparison.

        # for i, x in enumerate(X):
        fig, ax = plt.subplots(2, 2)
        # Note: permute() will change shape of image to suit matplotlib 
        # (PyTorch default is [C, H, W] but Matplotlib is [H, W, C])
        cover_denorm = StegaImageProcessing.denormalize(cover).permute(1, 2, 0)
        ax[0,0].imshow(cover_denorm) 
        ax[0,0].set_title(f"Original Cover")
        ax[0,0].axis("off")

        cover_x_denorm = StegaImageProcessing.denormalize(cover_x).permute(1, 2, 0)
        ax[0,1].imshow(cover_x_denorm) 
        ax[0,1].set_title(f"Reconstructed Cover")
        ax[0,1].axis("off")


        secret_denorm = StegaImageProcessing.denormalize(secret).permute(1, 2, 0)
        ax[1,0].imshow(secret_denorm) 
        ax[1,0].set_title(f"Original Secret")
        ax[1,0].axis("off")

        secret_x_denorm = StegaImageProcessing.denormalize(secret_x).permute(1, 2, 0)
        ax[1,1].imshow(secret_x_denorm) 
        ax[1,1].set_title(f"Recovered Secret")
        ax[1,1].axis("off")

        fig.suptitle(f"Image Comparion", fontsize=16)

        if show_image:
            plt.show()
        
        return fig


    @staticmethod
    def get_model(model_path = MODEL_PATH):
        checkpoint = torch.load(model_path)

        if model_path.resolve() == old_model_path.resolve():
            model = CombinedNetwork_Old()
        else:
            model = CombinedNetwork()
        # optimizer = torch.optim.Adam(test_model.parameters(), lr=LEARNING_RATE)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # epoch_idx = checkpoint['epoch']
        # batch_idx = checkpoint['batch']
        return model.to(device)



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

        cover_denorm = StegaImageProcessing.denormalize(cover).permute(1, 2, 0)
        cover_x_denorm = StegaImageProcessing.denormalize(cover_x).permute(1, 2, 0)
        secret_denorm = StegaImageProcessing.denormalize(secret).permute(1, 2, 0)
        secret_x_denorm = StegaImageProcessing.denormalize(secret_x).permute(1, 2, 0)

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
    def reveal_image(model, cover_o, secret_o):

        buffer = queue.Queue()
        # new_input = Image.open(image_path)

        # Populate queue with cover and secret
        buffer.put(StegaImageProcessing.normalize_and_transform(cover_o))
        buffer.put(StegaImageProcessing.normalize_and_transform(secret_o))

        dataset = ImageDataset(buffer)
        dataloader = torch.utils.data.DataLoader(dataset=dataset, 
                                                batch_size=2,
                                                shuffle=False)

        for data in dataloader:
            a, b = data.split(2//2,dim=0)
            cuda_cover = a.to(device)
            cuda_secret = b.to(device)

            model.eval()
            with torch.inference_mode():
                recovered_secret = model.reveal_only(secret_tensor=cuda_secret)
            
            # cover = cuda_cover.cpu().squeeze(0)
            # cover_x = modified_cover.cpu().squeeze(0)
            secret = cuda_secret.cpu().squeeze(0)
            secret_x = recovered_secret.cpu().squeeze(0)

            # Save modified cover image
            if not Path(SECRET_SAVE_PATH).is_dir():
                Path(SECRET_SAVE_PATH).mkdir(parents=True, exist_ok=True)
            # img = Image.fromarray(modified_cover.cpu().detach().numpy()[0])
            timestamp = f'{time.strftime("%Y%m%d-%H%M%S")}'
            # img.save(COVER_SAVE_PATH / f"Stega_{timestamp}.png")
            save_image(StegaImageProcessing.denormalize(recovered_secret.cpu()), f'{SECRET_SAVE_PATH}/Secret_{timestamp}.png')

            StegaImageProcessing.test_plot_reveal(secret, secret_x, show_image=True)
            break

        return "Done"


    @staticmethod
    def test_plot_reveal(secret, secret_x, show_image=True):
        
        fig = pyplot.figure(figsize=(10, 20))

        gs0 = gridspec.GridSpec(2, 1)
        # gs00 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs0[0], wspace=0)
        gs00 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs0[0], wspace=0)
        gs01 = gridspec.GridSpecFromSubplotSpec(1, 5, subplot_spec=gs0[1], wspace=0)

        secret_denorm = StegaImageProcessing.denormalize(secret).permute(1, 2, 0)
        secret_x_denorm = StegaImageProcessing.denormalize(secret_x).permute(1, 2, 0)


        gs00_ax00 = fig.add_subplot(gs00[0, 0])
        gs00_ax00.imshow(secret_x_denorm)   
        gs00_ax00.set_title(f"Recovered Secret")
        gs00_ax00.axis("off")

        # # Image Comparison Section
        # gs00_ax00 = fig.add_subplot(gs00[0, 0])
        # gs00_ax00.imshow(secret_denorm)   
        # gs00_ax00.set_title(f"Original Secret")
        # gs00_ax00.axis("off")

        # gs00_ax01 = fig.add_subplot(gs00[0, 1])
        # gs00_ax01.imshow(secret_x_denorm) 
        # gs00_ax01.set_title(f"Recovered Secret")
        # gs00_ax01.axis("off")


        # # Residual Error Section
        # secret_error = np.abs(secret_x_denorm - secret_denorm)

        # gs01_ax00 = fig.add_subplot(gs01[0, 0])
        # gs01_ax00.imshow(secret_denorm) 
        # gs01_ax00.set_title(f"Secret")
        # gs01_ax00.axis("off")

        # gs01_ax01 = fig.add_subplot(gs01[0, 1])
        # gs01_ax01.imshow(secret_x_denorm)
        # gs01_ax01.set_title(f"Re-Secret")
        # gs01_ax01.axis("off")

        # gs01_ax02 = fig.add_subplot(gs01[0, 2])
        # gs01_ax02.imshow(np.multiply(secret_error, 1.0))
        # gs01_ax02.set_title(f"Residual Error x1")
        # gs01_ax02.axis("off")

        # gs01_ax03 = fig.add_subplot(gs01[0, 3])
        # gs01_ax03.imshow(np.multiply(secret_error, 3.0))
        # gs01_ax03.set_title(f"Residual Error x3")
        # gs01_ax03.axis("off")

        # gs01_ax04 = fig.add_subplot(gs01[0, 4])
        # gs01_ax04.imshow(np.multiply(secret_error, 5.0))
        # gs01_ax04.set_title(f"Residual Error x5")
        # gs01_ax04.axis("off")


        # fig.suptitle(f"Image Comparison", fontsize=16)

        if show_image:
            plt.show()
    

import sys
sys.path.insert(0, 'Deep_Learning_Image_Steganography/Stega')
from models import CombinedNetwork
from models_old import CombinedNetwork_Old

cover = Image.open(Path("Deep_Learning_Image_Steganography/Gui/assets/HideImagePage/image_1.png")).convert('RGB')
secret = Image.open(Path("Deep_Learning_Image_Steganography/Gui/assets/HideImagePage/image_2.png")).convert('RGB')
model = StegaImageProcessing.get_model()

# def hide_image_command():
#     hide_image(model=model, cover=cover, secret=secret)


# def image_errors(cover, cover_x, secret, secret_x):
#     # Get absolute difference between original and recontructed images.
#     cover_error, secret_error = np.abs(secret_x - secret), np.abs(cover_x - cover)

#     # # Plot distribution of errors in cover and secret images.
#     # pixel_histogram(diff_S, diff_C)


def visualize_secret_prep(model, cover_o, secret_o):

    buffer = queue.Queue()
    # new_input = Image.open(image_path)

    # Populate queue with cover and secret
    buffer.put(StegaImageProcessing.normalize_and_transform(cover_o))
    buffer.put(StegaImageProcessing.normalize_and_transform(secret_o))

    dataset = ImageDataset(buffer)
    dataloader = torch.utils.data.DataLoader(dataset=dataset, 
                                            batch_size=2,
                                            shuffle=False)

    for data in dataloader:
        a, b = data.split(2//2,dim=0)
        cuda_cover = a.to(device)
        cuda_secret = b.to(device)

        model.eval()
        with torch.inference_mode():
            # modified_cover, recovered_secret = model(cuda_cover, cuda_secret)
            prepped_secrets, conv3x3, con4x4, conv5x5 = model.secret_prep(cuda_secret)
            
        
        # cover = cuda_cover.cpu().squeeze(0)
        # cover_x = modified_cover.cpu().squeeze(0)
        secret = cuda_secret.cpu().squeeze(0)
        secret_x = prepped_secrets.cpu().squeeze(0)

        break


    fig = pyplot.figure(figsize=(10, 20))

    gs0 = gridspec.GridSpec(2, 1)
    gs00 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs0[0], wspace=0)
    gs01 = gridspec.GridSpecFromSubplotSpec(2, 5, subplot_spec=gs0[1], wspace=0)

    # cover_denorm = StegaImageProcessing.denormalize(cover).permute(1, 2, 0)
    # cover_x_denorm = StegaImageProcessing.denormalize(cover_x).permute(1, 2, 0)
    secret_denorm = StegaImageProcessing.denormalize(secret).permute(1, 2, 0)
    secret_x_denorm = secret_x.permute(1, 2, 0)


    # plt.subplots_adjust(wspace=0, hspace=0)

    # Image Comparison Section
    gs00_ax00 = fig.add_subplot(gs00[0, 0])
    gs00_ax00.imshow(secret_denorm) 
    gs00_ax00.set_title(f"Original Secret")
    gs00_ax00.axis("off")

    # gs00_ax01 = fig.add_subplot(gs00[0, 1])
    # gs00_ax01.imshow(cover_x_denorm) 
    # gs00_ax01.set_title(f"Reconstructed Cover")
    # gs00_ax01.axis("off")

    # gs00_ax10 = fig.add_subplot(gs00[1, 0])
    # gs00_ax10.imshow(secret_denorm) 
    # gs00_ax10.set_title(f"Original Secret")
    # gs00_ax10.axis("off")

    # gs00_ax11 = fig.add_subplot(gs00[1, 1])
    # gs00_ax11.imshow(secret_x_denorm) 
    # gs00_ax11.set_title(f"Recovered Secret")
    # gs00_ax11.axis("off")


    # Visualize A few tensors from Prep


    denorm = transforms.Normalize(mean=[-0.456/0.224],
                                    std=[1/0.224])
    
    visual1 = denorm(secret_x_denorm[:, :, 1:2])
    visual2 = denorm(secret_x_denorm[:, :, 15:16])
    visual3 = denorm(secret_x_denorm[:, :, 45:46])
    visual4 = denorm(secret_x_denorm[:, :, 55:56])
    visual5 = denorm(secret_x_denorm[:, :, 75:76])
    visual6 = denorm(secret_x_denorm[:, :, 95:96])
    visual7 = denorm(secret_x_denorm[:, :, 105:106])
    visual8 = denorm(secret_x_denorm[:, :, 125:126])
    visual9 = denorm(secret_x_denorm[:, :, 135:136])
    visual10 = denorm(secret_x_denorm[:, :, 145:146])

    # cover_error = np.abs(cover_x_denorm - cover_denorm)
    # secret_error = np.abs(secret_x_denorm - secret_denorm)
    

    gs01_ax00 = fig.add_subplot(gs01[0, 0])
    gs01_ax00.imshow(visual1, cmap='gray') 
    gs01_ax00.set_title(f"visual1")
    gs01_ax00.axis("off")

    gs01_ax01 = fig.add_subplot(gs01[0, 1])
    gs01_ax01.imshow(visual2, cmap='gray') 
    gs01_ax01.set_title(f"visual2")
    gs01_ax01.axis("off")

    gs01_ax02 = fig.add_subplot(gs01[0, 2])
    gs01_ax02.imshow(visual3, cmap='gray')
    gs01_ax02.set_title(f"visual3")
    gs01_ax02.axis("off")

    gs01_ax03 = fig.add_subplot(gs01[0, 3])
    gs01_ax03.imshow(visual4, cmap='gray')
    gs01_ax03.set_title(f"visual4")
    gs01_ax03.axis("off")

    gs01_ax04 = fig.add_subplot(gs01[0, 4])
    gs01_ax04.imshow(visual5, cmap='gray')
    gs01_ax04.set_title(f"visual5")
    gs01_ax04.axis("off")

    gs01_ax10 = fig.add_subplot(gs01[1, 0])
    gs01_ax10.imshow(visual6, cmap='gray') 
    gs01_ax10.set_title(f"visual6")
    gs01_ax10.axis("off")

    gs01_ax11 = fig.add_subplot(gs01[1, 1])
    gs01_ax11.imshow(visual7, cmap='gray') 
    gs01_ax11.set_title(f"visual7")
    gs01_ax11.axis("off")

    gs01_ax12 = fig.add_subplot(gs01[1, 2])
    gs01_ax12.imshow(visual8, cmap='gray')
    gs01_ax12.set_title(f"visual8")
    gs01_ax12.axis("off")

    gs01_ax13 = fig.add_subplot(gs01[1, 3])
    gs01_ax13.imshow(visual9, cmap='gray')
    gs01_ax13.set_title(f"visual9")
    gs01_ax13.axis("off")

    gs01_ax14 = fig.add_subplot(gs01[1, 4])
    gs01_ax14.imshow(visual10, cmap='gray')
    gs01_ax14.set_title(f"visual10")
    gs01_ax14.axis("off")


    fig.suptitle(f"Prep Secret Visualization", fontsize=16)

    plt.show()
        

from ignite.metrics import Loss, SSIM
from ignite.engine import *
from ignite.handlers import *
from ignite.metrics import *
from ignite.utils import *
from ignite.contrib.metrics.regression import *
from ignite.contrib.metrics import *
from skimage.metrics import peak_signal_noise_ratio

def calculate_pnsr(target, preds):
    psnr = PSNR(data_range=1.0)
    default_evaluator = Engine(eval_step)
    psnr.attach(default_evaluator, 'psnr')
    state = default_evaluator.run([[preds, target]])
    return state.metrics['psnr']


def get_pnsr_from_images():

    cover_c_o = Image.open(Path("pnsr_c_o.png"))
    secret_r_o = Image.open(Path("pnsr_r_o.png"))

    # data_transforms = transforms.Compose([
    #         transforms.Resize(size=(224)),
    #         transforms.CenterCrop(size=(224, 224)),
    #         transforms.PILToTensor(),
    #         ])

    cover_x = Image.open(Path("pnsr_c.png"))
    secret_x = Image.open(Path("pnsr_r.png"))

    transform = transforms.Compose([transforms.PILToTensor()])

    # print(f"Cover PNSR = {calculate_pnsr(data_transforms(cover_c1), transform(cover_x))}")
    # print(f"Secret PNSR = {calculate_pnsr(data_transforms(secret_r1), transform(secret_x))}")
    print(f"Cover PSNR = {calculate_pnsr(transform(cover_c_o), transform(cover_x))}")
    print(f"Secret PSNR = {calculate_pnsr(transform(secret_r_o), transform(secret_x))}")


#     c1 = cover.resize((224,224))


#     PSNR += peak_signal_noise_ratio(Iclean[i,:,:,:], Img[i,:,:,:], data_range=data_range)


# For default evaluator in MSE_and_SSIM_and_Detector_loss
def eval_step(engine, batch):
    return batch



if __name__ == "__main__":
    cover = Image.open(Path("Deep_Learning_Image_Steganography/Gui/assets/HideImagePage/image_1.png")).convert('RGB')
    secret = Image.open(Path("Deep_Learning_Image_Steganography/Gui/assets/HideImagePage/image_2.png")).convert('RGB')
    # secret = Image.open(Path("image4.JPEG"))

    # with Image.open(relative_to_assets_page1("image_1.png")).convert('RGB') as image1:
    #     # image1.show()

    #     image_tensor = normalize_and_transform(image1)
    #     image_denorm = denormalize_and_toPIL(image_tensor)
    #     image_denorm.show()

    get_pnsr_from_images()

    # model = StegaImageProcessing.get_model()
    # visualize_secret_prep(model, cover, secret)
    print("Done")
    # StegaImageProcessing.hide_image(model, cover, secret)




# if __name__ == "__main__":
#     fig = pyplot.figure(figsize=(24, 12))

#     gs0 = gridspec.GridSpec(1, 2)
#     gs00 = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs0[0])
#     gs01 = gridspec.GridSpecFromSubplotSpec(2, 5, subplot_spec=gs0[1])

#     # gs00_ax00 = fig.add_subplot(gs00[0, 0])
#     # gs00_ax01 = fig.add_subplot(gs00[0, 1])
#     # gs00_ax10 = fig.add_subplot(gs00[1, 0])
#     # gs00_ax11 = fig.add_subplot(gs00[1, 1])

#     # gs01_ax00 = fig.add_subplot(gs01[0, 0])
#     # gs01_ax01 = fig.add_subplot(gs01[0, 1])
#     # gs01_ax02 = fig.add_subplot(gs01[0, 2])
#     # gs01_ax03 = fig.add_subplot(gs01[0, 3])
#     # gs01_ax04 = fig.add_subplot(gs01[0, 4])
#     # gs01_ax10 = fig.add_subplot(gs01[1, 0])
#     # gs01_ax11 = fig.add_subplot(gs01[1, 1])
#     # gs01_ax12 = fig.add_subplot(gs01[1, 2])
#     # gs01_ax13 = fig.add_subplot(gs01[1, 3])
#     # gs01_ax14 = fig.add_subplot(gs01[1, 4])

#     # for i in range(2):
#     #     for j in range(2):
#     #         ax00 = fig.add_subplot(gs00[i, j])
#     #         ax00.text(0.5, 0.5, '0_{}_{}'.format(i, j), ha='center')
#     #         ax00.set_xticks([])
#     #         ax00.set_yticks([])
    
#     # for i in range(2):
#     #     for j in range(5):
#     #         ax01 = fig.add_subplot(gs01[i, j])
#     #         ax01.text(0.5, 0.5, '1_{}_{}'.format(i, j), ha='center')
#     #         ax01.set_xticks([])
#     #         ax01.set_yticks([])

#     plt.show()
