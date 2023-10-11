import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path

# from image import *

# Path to Imagenet training images. Place in "normal" folder to indicate original images.
DATA_PATH = Path("data/stega_dataset/original/normal")
# Path to save normal training images into.
DATA_PATH_NORMAL = Path("data/stega_dataset/train/normal")
# Path to save modified training images into.
DATA_PATH_MODIFIED = Path("data/stega_dataset/train/modified")
# Path to move validation images into.
DATA_PATH_VAL = Path("data/stega_dataset/val")
# Path to move test images into.
DATA_PATH_TEST = Path("data/stega_dataset/test")
# Path to model to generate steganography images
old_model_path = Path("saved_models/old/Test_Model_Epoch_2.pth")
model_V3_path = Path("saved_models/latest/Stega_Model_V3_(detector_V1).pth")
MODEL_PATH = model_V3_path

device = "cuda" if torch.cuda.is_available() else "cpu"

import sys
sys.path.insert(0, 'Deep_Learning_Image_Steganography/Stega')
from models import CombinedNetwork
from models_old import CombinedNetwork_Old



class SteganographyDatasetBuilder:
    def __init__(self) -> None:
        pass


    def create_stega_database(self):
        
        model = self.get_model()
        

        data_path = DATA_PATH
        val_normal_path = DATA_PATH_VAL / "normal"
        val_modified_path = DATA_PATH_VAL / "modified"
        test_normal_path = DATA_PATH_TEST / "normal"
        test_modified_path = DATA_PATH_TEST / "modified"

        # Create dataset directory tree if does not exists.
        if not DATA_PATH_NORMAL.is_dir():
            DATA_PATH_NORMAL.mkdir(parents=True, exist_ok=True)
        if not DATA_PATH_MODIFIED.is_dir():
            DATA_PATH_MODIFIED.mkdir(parents=True, exist_ok=True)
        if not DATA_PATH_VAL.is_dir():
            DATA_PATH_VAL.mkdir(parents=True, exist_ok=True)
        if not DATA_PATH_TEST.is_dir():
            DATA_PATH_TEST.mkdir(parents=True, exist_ok=True)
        if not val_normal_path.is_dir():
            val_normal_path.mkdir(parents=True, exist_ok=True)
        if not val_modified_path.is_dir():
            val_modified_path.mkdir(parents=True, exist_ok=True)
        if not test_normal_path.is_dir():
            test_normal_path.mkdir(parents=True, exist_ok=True)
        if not test_modified_path.is_dir():
            test_modified_path.mkdir(parents=True, exist_ok=True)

        normal_transform = transforms.Compose([
            transforms.Resize(size=(224)),
            transforms.CenterCrop(size=(224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])

        secret_transform = transforms.Compose([
            transforms.Resize(size=(224)),
            transforms.CenterCrop(size=(224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
        
        normal_dataset = datasets.ImageFolder(root=data_path,
                                            transform=normal_transform)
        
        secret_dataset = datasets.ImageFolder(root=data_path,
                                            transform=secret_transform)
        
        normal_dataloader = DataLoader(dataset=normal_dataset, 
                                    batch_size=1, 
                                    num_workers=1,
                                    shuffle=False,
                                    drop_last=True)
        
        secret_dataloader = DataLoader(dataset=secret_dataset, 
                                    batch_size=1, 
                                    num_workers=1,
                                    shuffle=True,
                                    drop_last=True)
        

        denorm = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                                    std=[1/0.229, 1/0.224, 1/0.225])
        pil_transfrom = transforms.ToPILImage()
        
        model.eval()
        image_num = 1
        secret_iter = iter(secret_dataloader)
        
        len(normal_dataloader)
        # Split into 90% / 5% / 5% distribution
        val_start = int(0.9 * len(normal_dataloader))
        test_start = int(0.95 * len(normal_dataloader))

        for index, (data, label) in enumerate(normal_dataloader):
            
            cuda_cover = data.to(device)
            cuda_secret = next(secret_iter).to(device)

            with torch.inference_mode():
                modified_cover, recovered_secret = model(cuda_cover, cuda_secret)
            
            # cover = cuda_cover.cpu().squeeze(0)
            cover_x = modified_cover.cpu().squeeze(0)
            # secret = cuda_secret.cpu().squeeze(0)
            # secret_x = recovered_secret.cpu().squeeze(0)

            # plot_images_comparison(cover, cover_x, secret, secret_x, show_image=True)
            # test_plot(cover, cover_x, secret, secret_x, show_image=True)

            # cover_denorm = denormalize_and_toPIL(cover)
            # cover_x_denorm = denormalize_and_toPIL(cover_x)
            tensor_denorm = denorm(cover_x).clamp(min=0, max=1)
            cover_x_denorm = pil_transfrom(tensor_denorm)
            # secret_denorm = denormalize_and_toPIL(secret)
            # secret_x_denorm = denormalize_and_toPIL(secret_x)

            tensor_original_denorm = denorm(data.squeeze(0)).clamp(min=0, max=1)
            cover_original_denorm = pil_transfrom(tensor_original_denorm)

            image_num_str = str(image_num).zfill(8)

            if image_num > val_start:
                cover_original_denorm.save(val_normal_path / f"Normal_{image_num_str}.JPEG")
                cover_x_denorm.save(val_modified_path / f"Stega_{image_num_str}.JPEG")
            elif image_num > test_start:
                cover_original_denorm.save(test_normal_path / f"Normal_{image_num_str}.JPEG")
                cover_x_denorm.save(test_modified_path / f"Stega_{image_num_str}.JPEG")
            else:
                cover_original_denorm.save(DATA_PATH_NORMAL / f"Normal_{image_num_str}.JPEG")
                cover_x_denorm.save(DATA_PATH_MODIFIED / f"Stega_{image_num_str}.JPEG")

            image_num += 1


        print("Done")

    
    def get_model(self, model_path = MODEL_PATH):
        # model_path = Path("saved_models/20230711-063730/Test_Model_Epoch_2.pth")
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



if __name__ == "__main__":
    stega_ds_builder = SteganographyDatasetBuilder()
    stega_ds_builder.create_stega_database()


