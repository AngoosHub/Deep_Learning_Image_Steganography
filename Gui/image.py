
import torch
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

from torch.utils.data import IterableDataset
import queue
# import torchvision.transforms.functional as TF
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


def hide_image(model, cover, secret):

    buffer = queue.Queue()
    # new_input = Image.open(image_path)

    # Populate queue with cover and secret
    buffer.put(normalize_and_transform(cover))
    buffer.put(normalize_and_transform(secret))

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

        plot_images_comparison(cover, cover_x, secret, secret_x, show_image=True)

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





def relative_to_assets_page1(path: str) -> Path:
    return ASSETS_PATH_PAGE1 / Path(path)


import sys
sys.path.insert(0, 'Deep_Learning_Image_Steganography/Stega')
from models import CombinedNetwork

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


if __name__ == "__main__":
    ASSETS_PATH_PAGE1 = Path("Deep_Learning_Image_Steganography/Gui/assets/page1")
    cover = Image.open(relative_to_assets_page1("image_1.png")).convert('RGB')
    secret = Image.open(relative_to_assets_page1("image_2.png")).convert('RGB')

    # with Image.open(relative_to_assets_page1("image_1.png")).convert('RGB') as image1:
    #     # image1.show()

    #     image_tensor = normalize_and_transform(image1)
    #     image_denorm = denormalize_and_toPIL(image_tensor)
    #     image_denorm.show()

    model = get_model()
    hide_image(model, cover, secret)

