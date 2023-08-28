
import torch
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import pyplot

from torch.utils.data import IterableDataset
import queue
from torchvision import transforms



device = "cuda" if torch.cuda.is_available() else "cpu"



class MyDataset(IterableDataset):
    def __init__(self, image_queue):
      self.queue = image_queue

    def read_next_image(self):
        while self.queue.qsize() > 0:
            # you can add transform here
            yield self.queue.get()
        return None

    def __iter__(self):
        return self.read_next_image()


def hide_image(model, cover_o, secret_o):

    buffer = queue.Queue()
    # new_input = Image.open(image_path)

    # Populate queue with cover and secret
    buffer.put(normalize_and_transform(cover_o))
    buffer.put(normalize_and_transform(secret_o))

    dataset = MyDataset(buffer)
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

        # plot_images_comparison(cover, cover_x, secret, secret_x, show_image=True)
        test_plot(cover, cover_x, secret, secret_x, show_image=True)
        break

    return "Done"


def normalize_and_transform(image):
    data_transforms = transforms.Compose([
        transforms.Resize(size=(224)),
        transforms.CenterCrop(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
    image_transform = data_transforms(image)
    return image_transform
    

def denormalize_and_toPIL(tensor):
    pil_transfrom = transforms.ToPILImage()
    tensor_denorm = denormalize(tensor)
    image = pil_transfrom(tensor_denorm)
    return image



def denormalize(tensor):
    denorm = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                                  std=[1/0.229, 1/0.224, 1/0.225])
    return denorm(tensor)


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



def get_model():
    model_path = Path("saved_models/20230711-063730/Test_Model_Epoch_2.pth")
    checkpoint = torch.load(model_path)
    model = CombinedNetwork()
    # optimizer = torch.optim.Adam(test_model.parameters(), lr=LEARNING_RATE)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # epoch_idx = checkpoint['epoch']
    # batch_idx = checkpoint['batch']
    return model.to(device)




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
    gs01_ax00 = fig.add_subplot(gs01[0, 0])
    gs01_ax00.imshow(cover_denorm) 
    gs01_ax00.set_title(f"Cover")
    gs01_ax00.axis("off")

    gs01_ax01 = fig.add_subplot(gs01[0, 1])
    gs01_ax01.imshow(cover_x_denorm) 
    gs01_ax01.set_title(f"Re-Cover")
    gs01_ax01.axis("off")

    gs01_ax02 = fig.add_subplot(gs01[0, 2])
    gs01_ax03 = fig.add_subplot(gs01[0, 3])
    gs01_ax04 = fig.add_subplot(gs01[0, 4])

    gs01_ax10 = fig.add_subplot(gs01[1, 0])
    gs01_ax10.imshow(secret_denorm) 
    gs01_ax10.set_title(f"Secret")
    gs01_ax10.axis("off")

    gs01_ax11 = fig.add_subplot(gs01[1, 1])
    gs01_ax11.imshow(secret_x_denorm) 
    gs01_ax11.set_title(f"Re-Secret")
    gs01_ax11.axis("off")

    gs01_ax12 = fig.add_subplot(gs01[1, 2])
    gs01_ax13 = fig.add_subplot(gs01[1, 3])
    gs01_ax14 = fig.add_subplot(gs01[1, 4])

    # for i in range(2):
    #     for j in range(2):
    #         ax00 = fig.add_subplot(gs00[i, j])
    #         ax00.text(0.5, 0.5, '0_{}_{}'.format(i, j), ha='center')
    #         ax00.set_xticks([])
    #         ax00.set_yticks([])
    
    # for i in range(2):
    #     for j in range(5):
    #         ax01 = fig.add_subplot(gs01[i, j])
    #         ax01.text(0.5, 0.5, '1_{}_{}'.format(i, j), ha='center')
    #         ax01.set_xticks([])
    #         ax01.set_yticks([])


    # # Denomralize and plot images for visual comparison.

    # # for i, x in enumerate(X):
    # fig, ax = plt.subplots(2, 2)
    # # Note: permute() will change shape of image to suit matplotlib 
    # # (PyTorch default is [C, H, W] but Matplotlib is [H, W, C])
    # cover_denorm = denormalize(cover).permute(1, 2, 0)
    # ax[0,0].imshow(cover_denorm) 
    # ax[0,0].set_title(f"Original Cover")
    # ax[0,0].axis("off")

    # cover_x_denorm = denormalize(cover_x).permute(1, 2, 0)
    # ax[0,1].imshow(cover_x_denorm) 
    # ax[0,1].set_title(f"Reconstructed Cover")
    # ax[0,1].axis("off")


    # secret_denorm = denormalize(secret).permute(1, 2, 0)
    # ax[1,0].imshow(secret_denorm) 
    # ax[1,0].set_title(f"Original Secret")
    # ax[1,0].axis("off")

    # secret_x_denorm = denormalize(secret_x).permute(1, 2, 0)
    # ax[1,1].imshow(secret_x_denorm) 
    # ax[1,1].set_title(f"Recovered Secret")
    # ax[1,1].axis("off")

    fig.suptitle(f"Image Comparison", fontsize=16)

    if show_image:
        plt.show()
    
    return fig


def reveal_image(model, cover_o, secret_o):

    buffer = queue.Queue()
    # new_input = Image.open(image_path)

    # Populate queue with cover and secret
    buffer.put(normalize_and_transform(cover_o))
    buffer.put(normalize_and_transform(secret_o))

    dataset = MyDataset(buffer)
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

        test_plot_reveal(secret, secret_x, show_image=True)
        break

    return "Done"

def test_plot_reveal(secret, secret_x, show_image=True):
    
    fig = pyplot.figure(figsize=(10, 20))

    gs0 = gridspec.GridSpec(2, 1)
    gs00 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs0[0], wspace=0)
    gs01 = gridspec.GridSpecFromSubplotSpec(1, 5, subplot_spec=gs0[1], wspace=0)

    secret_denorm = denormalize(secret).permute(1, 2, 0)
    secret_x_denorm = denormalize(secret_x).permute(1, 2, 0)


    # Image Comparison Section
    gs00_ax00 = fig.add_subplot(gs00[0, 0])
    gs00_ax00.imshow(secret_denorm)   
    gs00_ax00.set_title(f"Original Secret")
    gs00_ax00.axis("off")

    gs00_ax01 = fig.add_subplot(gs00[0, 1])
    gs00_ax01.imshow(secret_x_denorm) 
    gs00_ax01.set_title(f"Recovered Secret")
    gs00_ax01.axis("off")


    # Residual Error Section
    gs01_ax00 = fig.add_subplot(gs01[0, 0])
    gs01_ax00.imshow(secret_denorm) 
    gs01_ax00.set_title(f"Secret")
    gs01_ax00.axis("off")

    gs01_ax01 = fig.add_subplot(gs01[0, 1])
    gs01_ax01.imshow(secret_x_denorm)
    gs01_ax01.set_title(f"Re-Secret")
    gs01_ax01.axis("off")

    gs01_ax02 = fig.add_subplot(gs01[0, 2])
    gs01_ax03 = fig.add_subplot(gs01[0, 3])
    gs01_ax04 = fig.add_subplot(gs01[0, 4])


    fig.suptitle(f"Image Comparison", fontsize=16)

    if show_image:
        plt.show()
    

import sys
sys.path.insert(0, 'Deep_Learning_Image_Steganography/Stega')
from models import CombinedNetwork

cover = Image.open(Path("Deep_Learning_Image_Steganography/Gui/assets/page1/image_1.png")).convert('RGB')
secret = Image.open(Path("Deep_Learning_Image_Steganography/Gui/assets/page1/image_2.png")).convert('RGB')
model = get_model()

def hide_image_command():
    hide_image(model=model, cover=cover, secret=secret)



if __name__ == "__main__":
    cover = Image.open(Path("Deep_Learning_Image_Steganography/Gui/assets/page1/image_1.png")).convert('RGB')
    secret = Image.open(Path("Deep_Learning_Image_Steganography/Gui/assets/page1/image_2.png")).convert('RGB')

    # with Image.open(relative_to_assets_page1("image_1.png")).convert('RGB') as image1:
    #     # image1.show()

    #     image_tensor = normalize_and_transform(image1)
    #     image_denorm = denormalize_and_toPIL(image_tensor)
    #     image_denorm.show()

    model = get_model()
    reveal_image(model, cover, secret)




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
