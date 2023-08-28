import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from ignite.metrics import Loss, SSIM
import os
from pathlib import Path
import time
from models import *
import utils
import numpy as np
import wandb
import random


# from collections import OrderedDict

# import torch
# from torch import nn, optim

from ignite.engine import *
from ignite.handlers import *
from ignite.metrics import *
from ignite.utils import *
from ignite.contrib.metrics.regression import *
from ignite.contrib.metrics import *

SKIP_WANDB = False # True to skip wandb login and save time in testing
SAVE_EPOCH_PROGRESS = True # False to to skip saving models when testing
SKIP_WANDB_SAVE_IMAGE_OF_FIRST_FEW_BATCHES = False # True to skip saving image showing early learning progress
# WANDB_VAR

LEARNING_RATE = 0.001 # 0.0001 slower
BATCH_SIZE = 8 # above 16 runs out of memory
EPOCHS = 1 # 3 Epochs takes about 7 hours
BETA = 0.75
RESUME_EPOCH = 1 # Set to resume checkpoint's epoch number (ignored if = 1)
RESUME_PATH = "saved_models/latest/model.pth" # Path to model.pth file to resume training

# Normalize input images using imageNet mean and standard deviation
NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

# Set cpu number for loading data
NUM_CPU = 1
if os.cpu_count() > 35:
    NUM_CPU = 10 # 10 used for each train, val, and test dataloader.

# Set torch seed
torch.manual_seed(42)


def train(load_model=False, load_path=""):

    batch_idx = 1
    epoch_idx = 1

    # Set device type
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    data_path = Path("data/")
    if not os.path.isdir('data/'):
        data_path = Path("../data/")
    train_dir = data_path / "train"
    val_dir = data_path / "val"
    test_dir = data_path / "test"
    # train_dir = data_path / "temptrain"
    # val_dir = data_path / "tempval"
    # test_dir = data_path / "temptest"
    # current_working_dir = os.getcwd()
    # train_dir = current_working_dir / "train"
    # val_dir = current_working_dir / "val"

    # Resume training from checkpoint
    if RESUME_EPOCH > 1:
        resume_epoch = RESUME_EPOCH - 1
        # resume(model, RESUME_PATH)

    start_time = time.perf_counter()

    train_dataloader = get_train_dataloader(train_dir, BATCH_SIZE, NUM_CPU, NORMALIZE)
    val_dataloader = get_val_dataloader(val_dir, BATCH_SIZE, NUM_CPU, NORMALIZE)
    test_dataloader = get_test_dataloader(test_dir, BATCH_SIZE, 1, NORMALIZE)


    my_custom_loss = MSE_and_SSIM_loss()
    test_model = CombinedNetwork()
    
    # optimizer = torch.optim.Adam(test_model.net_prep.parameters(), lr=LEARNING_RATE)
    optimizer = torch.optim.AdamW(list(test_model.net_prep.parameters()) + list(test_model.net_hide.parameters()), lr=LEARNING_RATE)
    optimizer_reveal = torch.optim.AdamW(test_model.net_reveal.parameters(), lr=LEARNING_RATE)

    if load_model:
        checkpoint = torch.load(load_path)
        test_model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        optimizer_reveal.load_state_dict(checkpoint['optimizer_reveal_state_dict'])
        epoch_idx = checkpoint['epoch']
        batch_idx = checkpoint['batch']
    
    test_model.to(device)

    test_model.apply(init_weights)


    if not SKIP_WANDB:
        # watch our model and custom loss function
        wandb.watch(test_model, my_custom_loss, log="all", log_freq=100)

    for n in range(EPOCHS):
        train_step(
            model=test_model,
            dataloader=train_dataloader,
            loss=my_custom_loss,
            optimizer=optimizer,
            optimizer=optimizer_reveal,
            device=device,
            batch_size=BATCH_SIZE,
            batch_idx=batch_idx,
            epoch_idx=epoch_idx,
            epoch_total=EPOCHS,
            val_dataloader=val_dataloader)
        
        # Uncomment for wandb logging of average epoch error.
        # train_loss_epoch, cover_loss_epoch, secret_loss_epoch, cover_loss_mse_epoch, secret_loss_mse_epoch, cover_loss_ssim_epoch, secret_loss_ssim_epoch = train_step(
        #     model=test_model,
        #     dataloader=train_dataloader,
        #     loss=my_custom_loss,
        #     optimizer=optimizer,
        #     device=device,
        #     batch_size=BATCH_SIZE,
        #     batch_idx=batch_idx,
        #     epoch_idx=epoch_idx,
        #     epoch_total=EPOCHS,
        #     val_dataloader=val_dataloader)
        
        # Uncomment for wandb logging of average epoch error.
        # train_loss = np.mean(train_loss_epoch)
        # cover_loss = np.mean(cover_loss_epoch)
        # secret_loss = np.mean(secret_loss_epoch)
        # cover_loss_mse = np.mean(cover_loss_mse_epoch)
        # secret_loss_mse = np.mean(secret_loss_mse_epoch)
        # cover_loss_ssim = np.mean(cover_loss_ssim_epoch)
        # secret_loss_ssim = np.mean(secret_loss_ssim_epoch)

        # Uncomment for wandb logging of average epoch error.
        # wandb.log({'train/epoch': batch_idx,
        #         'train/loss': train_loss,
        #         'train/cover_loss': cover_loss,
        #         'train/secret_loss': secret_loss,
        #         'train/cover_mse': cover_loss_mse,
        #         'train/secret_mse': secret_loss_mse,
        #         'train/cover_ssim': cover_loss_ssim,
        #         'train/secret_ssim': secret_loss_ssim})
        
        # print(
        #     f"Epoch: [{epoch_idx}/{EPOCHS}] | "
        #     f"train_loss: {np.mean(train_loss_epoch):.4f} | "
        #     f"cover_loss: {np.mean(cover_loss_epoch):.4f} | "
        #     f"secret_loss: {np.mean(secret_loss_epoch):.4f}"
        # )

        # Save checkpoint after each epoch
        if SAVE_EPOCH_PROGRESS:
            save_checkpoint(test_model, optimizer, optimizer_reveal, device, epoch_idx, batch_idx)

        epoch_idx += 1
    
        
    end_time = time.perf_counter()
    print(f"\nEND: {end_time - start_time} seconds.")    



def get_train_dataloader(train_dir, batch_size, num_cpu, normalize):
    '''
        Training set transform with data augmentation:
            1. Resize the images to smallest edge is 256 (to handle non-square images).
            2. Randomly crops image into size of 224x224.
            3. Randomly flip the images on the horizontal (50% chance).
            4. Performs Trivial Aguments, state of the art random image augmentations.
            5. Converts image to tensor, all pixel values from 0 to 255 to be between 0.0 and 1.0.
            6. Normalize pixel values using ImageNet mean & standard deviation.
    '''
    train_transform = transforms.Compose([
        transforms.Resize(size=(256)),
        transforms.RandomCrop(size=(224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        # transforms.TrivialAugmentWide(num_magnitude_bins=31),
        transforms.ToTensor(),
        normalize
        ])

    # Use ImageFolder to create dataset(s)
    train_dataset = datasets.ImageFolder(root=train_dir, # target folder of images
                                         transform=train_transform) # transforms to perform on dataset
    
    # print(f"Train data:\n{train_dataset}\n")
    # # print(f"Train data:\n{train_dataset}\nValidation data:\n{val_dataset}")
    # img, label = train_dataset[0][0], train_dataset[0][1]
    # print(f"Image tensor:\n{img}")
    # print(f"Image shape: {img.shape}")
    # print(f"Image datatype: {img.dtype}")
    # print(f"Image label: {label}")
    # print(f"Label datatype: {type(label)}")

    # # print(train_dataloader) 
    # # print(val_dataloader)
    
    # Turn train and val Datasets into DataLoaders
    train_dataloader = DataLoader(dataset=train_dataset, 
                                  batch_size=batch_size,
                                  num_workers=num_cpu, # number of subprocesses to use for data loading
                                  shuffle=True, # shuffle data
                                  pin_memory=True, # 
                                  drop_last=True) # Drops last batch if remaining images less than full batch size, prevents tensor dimension errors
    
    return train_dataloader


def get_val_dataloader(val_dir, batch_size, num_cpu, normalize):
    '''
        Validation set transform:
            1. Resize the images to smallest edge is 256 (to handle non-square images).
            2. Center crops image into size of 224x224 for consistency.
            3. Converts image to tensor, all pixel values from 0 to 255 to be between 0.0 and 1.0.
            4. Normalize pixel values using ImageNet mean & standard deviation.
    '''
    val_transform = transforms.Compose([
        transforms.Resize(size=(256)),
        transforms.CenterCrop(size=(224, 224)),
        # transforms.RandomCrop(size=(224, 224)),
        # transforms.RandomHorizontalFlip(p=0.5),
        # transforms.TrivialAugmentWide(num_magnitude_bins=31),
        transforms.ToTensor(),
        normalize
        ])
    
    val_dataset = datasets.ImageFolder(root=val_dir,
                                       transform=val_transform)
    
    val_dataloader = DataLoader(dataset=val_dataset, 
                                batch_size=batch_size, 
                                num_workers=num_cpu, 
                                shuffle=True, # Controls shuffle on valset (random is seeded)
                                pin_memory=True,
                                drop_last=True)
    
    return val_dataloader


def get_test_dataloader(test_dir, batch_size, num_cpu, normalize):
    '''
        Test set transform:
            We want to minimize changes to original image while resize into 224x224.
            1. Resize the images to smallest edge is 224.
            2. Center crops image into size of 224x224.
            3. Converts image to tensor, all pixel values from 0 to 255 to be between 0.0 and 1.0.
            4. Normalize pixel values using ImageNet mean & standard deviation.
    '''
    test_transform = transforms.Compose([
        transforms.Resize(size=(224)),
        transforms.CenterCrop(size=(224, 224)),
        transforms.ToTensor(),
        normalize
        ])
    
    test_dataset = datasets.ImageFolder(root=test_dir,
                                       transform=test_transform)
    
    test_dataloader = DataLoader(dataset=test_dataset, 
                                 batch_size=batch_size, 
                                 num_workers=num_cpu,
                                 worker_init_fn=seed_worker,
                                 generator=g,
                                 shuffle=False, # Controls shuffle on testset (random is seeded)
                                 pin_memory=True,
                                 drop_last=True)
    
    return test_dataloader


def get_single_batch_into_image(train_dataloader):
    # 1. Get a batch of images and labels from the DataLoader
    img_batch, label_batch = next(iter(train_dataloader))

    # 2. Get a single image from the batch and unsqueeze the image so its shape fits the model
    img_cover, label_single = img_batch[0].unsqueeze(dim=0), label_batch[0]
    print(f"Single image shape: {img_cover.shape}\n")
    img_secret, label_single = img_batch[1].unsqueeze(dim=0), label_batch[1]

    return img_cover, img_secret

def test_plot_single_batch(img_cover, img_secret, test_model, device):

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

    fig = utils.plot_images_comparison(cover, cover_x, secret, secret_x, show_image=False)
    if not SKIP_WANDB:
        wandb.log({"Image": fig})


def custom_loss(cover, secret, cover_original, secret_original):

    cover_loss = torch.nn.functional.mse_loss(input=cover, target=cover_original)
    secret_loss = torch.nn.functional.mse_loss(input=secret, target=secret_original)

    combined_loss = cover_loss + BETA * secret_loss

    return combined_loss, cover_loss, secret_loss


# For default evaluator in 
def eval_step(engine, batch):
    return batch

class MSE_and_SSIM_loss(nn.Module):
    
    def __init__(self):
        super(MSE_and_SSIM_loss, self).__init__()
        self.loss_mse = torch.nn.MSELoss()
        self.default_evaluator = Engine(eval_step)
        self.metric = SSIM(data_range=1.0)
        self.metric.attach(self.default_evaluator, 'ssim')



    def forward(self, cover, secret, cover_original, secret_original):        
        cover_mse = self.loss_mse(input=cover, target=cover_original)
        secret_mse = self.loss_mse(input=secret, target=secret_original)

        cover_ssim = self.default_evaluator.run([[cover, cover_original]])
        secret_ssim = self.default_evaluator.run([[secret, secret_original]])
        
        # print(f"Cover SSIM: {1 - cover_ssim.metrics['ssim']}")
        # print(f"Secret SSIM: {1 - secret_ssim.metrics['ssim']}")

        # print(f"Cover MSE: {cover_loss}")
        # print(f"Secret MSE: {secret_loss}")


        cover_loss = cover_mse + (1 - cover_ssim.metrics['ssim'])
        secret_loss = secret_mse + (1 - secret_ssim.metrics['ssim'])

        # print(f"Cover Loss: {cover_loss}")
        # print(f"Secret Loss: {secret_loss}")
        
        combined_loss = cover_loss + BETA * secret_loss

        return combined_loss, cover_loss, (BETA * secret_loss), cover_mse, secret_mse, cover_ssim, secret_ssim



# class ContrastiveLoss(nn.Module):

#     def __init__(self, margin=1.25):  # margin=2
#         super(ContrastiveLoss, self).__init__()
#         self.margin = margin

#     def forward(self, output1, output2, label):
#         label = label.to(torch.float32)
#         euclidean_distance = F.pairwise_distance(output1, output2)
#         loss_contrastive = torch.mean(
#             (1 - label) * torch.pow(euclidean_distance, 2) +
#             label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
#         )

#         return loss_contrastive


def train_step(model: torch.nn.Module, 
               dataloader: torch.utils.data.DataLoader, 
               loss: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               optimizer_reveal: torch.optim.Optimizer,
               device,
               batch_size,
               batch_idx,
               epoch_idx,
               epoch_total,
               val_dataloader: torch.utils.data.DataLoader):
                  
    model.train()
    train_loss = []
    cover_loss = []
    secret_loss = []
    cover_loss_mse = []
    secret_loss_mse = []
    cover_loss_ssim = []
    secret_loss_ssim = []


    for index, (data, label) in enumerate(dataloader):

        # Split batch into cover and secret
        a, b = data.split(batch_size//2,dim=0)

        # Send data to GPU device if using GPU
        covers = a.to(device)
        secrets = b.to(device)

        # Forward
        model_covers, model_secrets = model(covers, secrets)

        # Calculate Loss
        combined_loss,  c_loss, s_loss, c_mse, s_mse, c_ssim, s_ssim  = loss(model_covers, model_secrets, covers, secrets)
        # combined_loss = loss(model_covers, covers)

        # combined_loss_1, c_loss_1, s_loss_1 = custom_loss(model_covers, model_secrets, covers, secrets)
        
        # print(f"Cutsom Loss: \n"
        #   f"Combined: {combined_loss}, Cover: {c_loss}, Secret: {s_loss}")

        # print(f"Custom Loss 1: \n"
        #   f"Combined: {combined_loss_1}, Cover: {c_loss_1}, Secret: {s_loss_1}")

        # Uncomment for wandb logging of average epoch error.
        # train_loss.append(combined_loss.item()) # loss.item() to get number from 1 element tensor
        # cover_loss.append(c_loss.item())
        # secret_loss.append(s_loss.item())
        # cover_loss_mse.append(c_mse.item())
        # secret_loss_mse.append(s_mse.item())
        # cover_loss_ssim.append(1-c_ssim.metrics['ssim'])
        # secret_loss_ssim.append(1-s_ssim.metrics['ssim'])

        # During training, we zero out gradient in optimizer, 
        # so that our backpropagation does not include information from previous passes.
        optimizer_reveal.zero_grad()

        # Backpropagate and optimize Reveal optimizer
        s_loss.backward(retain_graph=True)
        optimizer_reveal.step()

        # Backpropagate and optimize Prep & Hide optimizer
        optimizer.zero_grad()
        combined_loss.backward()
        optimizer.step()

        # Prints batch progress
        print(f'Epoch: Epoch: [{epoch_idx}/{epoch_total}] | '
              f'Batch {index}/{len(dataloader)}:  combined_loss = {combined_loss.item():.4f}, cover_loss = {c_loss.item():.4f}, secret_loss = {s_loss.item():.4f}')
        # print(f'Batch {index+1}/{len(dataloader)} | combined_loss = {combined_loss.item():.4f} | cover_loss = | secret_loss = ')

        # log progress every 100 batches
        if not SKIP_WANDB and batch_idx % 100 == 0:
            wandb.log({#'train/batch': batch_idx,
                    'train/batch_loss': combined_loss.item(),
                    'train/batch_cover_loss': c_loss.item(),
                    'train/batch_secret_loss': s_loss.item(),
                    # 'train/batch_cover_mse': c_mse.item(),
                    # 'train/batch_secret_mse': s_mse.item(),
                    # 'train/batch_cover_ssim': 1-c_ssim.metrics['ssim'],
                    # 'train/batch_secret_ssim': 1-s_ssim.metrics['ssim']
                    })
            
            validation_step(model, 
                            val_dataloader, 
                            loss,
                            device,
                            batch_size,
                            batch_idx)
            
        
        batch_idx += 1

        break

    # Uncomment for wandb logging of average epoch error.
    # return train_loss, cover_loss, secret_loss, cover_loss_mse,secret_loss_mse, cover_loss_ssim, secret_loss_ssim


def validation_step(model: torch.nn.Module, 
                    dataloader: torch.utils.data.DataLoader, 
                    loss: torch.nn.Module,
                    device,
                    batch_size,
                    batch_idx):
    
    model.eval()

    with torch.inference_mode():
        img_batch, label_batch = next(iter(dataloader))
        a, b = img_batch.split(batch_size//2,dim=0)
        covers = a.to(device)
        secrets = b.to(device)

        model_covers, model_secrets = model(covers, secrets)
        combined_loss,  c_loss, s_loss, c_mse, s_mse, c_ssim, s_ssim  = loss(model_covers, model_secrets, covers, secrets)
    
    # log progress every 100 batches
        if not SKIP_WANDB and batch_idx % 100 == 0:
            wandb.log({#'train/batch': batch_idx,
                    'train/batch_loss_val': combined_loss.item(),
                    'train/batch_cover_loss_val': c_loss.item(),
                    'train/batch_secret_loss_val': s_loss.item(),
                    # 'train/batch_cover_mse_val': c_mse.item(),
                    # 'train/batch_secret_mse_val': s_mse.item(),
                    # 'train/batch_cover_ssim_val': 1-c_ssim.metrics['ssim'],
                    # 'train/batch_secret_ssim_val': 1-s_ssim.metrics['ssim']
                    })
            
            if not SKIP_WANDB_SAVE_IMAGE_OF_FIRST_FEW_BATCHES:
                if (batch_idx % 100 == 0 and batch_idx <= 1000):
                    # Try plotting a batch of image fed to model.
                    test_plot_single_batch(img_cover, img_secret, model, device)

    # Turn training back on after eval finished
    model.train()


def save_checkpoint(model, optimizer, optimizer_reveal, device, epoch_idx, batch_idx):
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
            test_plot_single_batch(img_cover, img_secret, model, device)

            global WANDB_VAR
            if WANDB_VAR != False:
                artifact_path = Path("saved_models/latest")
                if not artifact_path.is_dir():
                    artifact_path.mkdir(parents=True)
                
                latest_model = "latest/model.pth"
                latest_model_path = Path("saved_models") / latest_model
                utils.save_model(model=save_state, model_name=latest_model)
                artifact = wandb.Artifact(name='model', type='model')
                artifact.add_file(latest_model_path)
                WANDB_VAR.log_artifact(artifact)



def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight)
    # for l in m.children(): init_weights(l)


            
# Seed Pytorch dataloader random seed for workers.
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(0)

if __name__ == "__main__":

    global WANDB_VAR
    data_path = Path("data/")
    if not os.path.isdir('data/'):
        data_path = Path("../data/")
    test_plot_dataloader = get_test_dataloader((data_path / "test"), 2, 1, NORMALIZE)
    # test_plot_dataloader = get_test_dataloader((data_path / "test"), 2, 1)
    global img_cover, img_secret
    img_cover, img_secret = get_single_batch_into_image(test_plot_dataloader)

    if SKIP_WANDB:
        WANDB_VAR = False
        train()
    else:
        # start a new wandb run to track this script, # set the wandb project where this run will be logged
        with wandb.init(project="test_model", config={
            "learning_rate": LEARNING_RATE,
            "architecture": "CNN",
            "dataset": "ImageNet",
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            }) as wandb_run:
            # load_path = Path("saved_models") / "20230707-131247" / "Test_Model_Epoch_5_FINAL.pth"
            # train(load_model=True, load_path=load_path)
            WANDB_VAR = wandb_run
            train()
