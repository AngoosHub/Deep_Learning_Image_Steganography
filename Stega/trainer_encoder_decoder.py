import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from ignite.metrics import Loss, SSIM
import os
from pathlib import Path
import time
from models import *
from utils import utils
import mse_ssim_detector_loss
import dataset_loader
import numpy as np
import wandb
import random
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# abstract base class work
from abc import ABC, abstractmethod



class TrainerBase(ABC):

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def train_step(self):
        pass

    @abstractmethod
    def validation_step(self):
        pass



class TrainerEncoderDecoder(TrainerBase):

    def __init__(self,
                 LEARNING_RATE,
                 BATCH_SIZE,
                 EPOCHS,
                 BETA,
                 NORMALIZE,
                 NUM_CPU,
                 DETECTOR_PATH,
                 SKIP_WANDB,
                 SAVE_EPOCH_PROGRESS,
                 SAVE_PROGRESS_EVERY_10000_BATCHES,
                 SKIP_WANDB_SAVE_IMAGE_OF_FIRST_FEW_BATCHES,
                 device,
                 detector_model) -> None:
        
        self.LEARNING_RATE = LEARNING_RATE # 0.001
        self.BATCH_SIZE = BATCH_SIZE # lower to reduce memory usage
        self.EPOCHS = EPOCHS # ILSVRC2017 Training Set has about half million images. Model training usually stablizes within 1 Epoch.
        self.BETA = BETA # Loss function parameter. Controls Secret and Detector contribution to loss.
        self.NORMALIZE = NORMALIZE # Normalize input images using imageNet mean and standard deviation
        self.NUM_CPU = NUM_CPU # Set cpu number for loading data
        self.DETECTOR_PATH = DETECTOR_PATH # Trained Detector model path to load from

        self.SKIP_WANDB = SKIP_WANDB # True to skip wandb login and save time in testing
        self.SAVE_EPOCH_PROGRESS = SAVE_EPOCH_PROGRESS # False to to skip saving models when testing
        self.SAVE_PROGRESS_EVERY_10000_BATCHES = SAVE_PROGRESS_EVERY_10000_BATCHES
        self.SKIP_WANDB_SAVE_IMAGE_OF_FIRST_FEW_BATCHES = SKIP_WANDB_SAVE_IMAGE_OF_FIRST_FEW_BATCHES # True to skip saving image showing early learning progress

        self.device = device # GPU device to use
        self.detector_model = detector_model


    def train(self):

        batch_idx = 0
        epoch_idx = 1

        data_path = Path("data/")
        train_dir = data_path / "train"
        val_dir = data_path / "val"
        test_dir = data_path / "test"

        
        train_dataloader = dataset_loader.DatasetLoader.get_train_dataloader(train_dir, self.BATCH_SIZE, self.NUM_CPU, self.NORMALIZE)
        val_dataloader = dataset_loader.DatasetLoader.get_val_dataloader(val_dir, self.BATCH_SIZE, self.NUM_CPU, self.NORMALIZE)
        test_dataloader = dataset_loader.DatasetLoader.get_test_dataloader(test_dir, self.BATCH_SIZE, 1, self.NORMALIZE)
        self.img_cover, self.img_secret = utils.get_single_batch_into_image(test_dataloader)


        my_custom_loss = mse_ssim_detector_loss.MSE_and_SSIM_and_Detector_loss(detector_model=self.detector_model, BETA=self.BETA)
        test_model = CombinedNetwork()
        
        # optimizer = torch.optim.Adam(test_model.net_prep.parameters(), lr=LEARNING_RATE)
        optimizer = torch.optim.AdamW(list(test_model.net_prep.parameters()) + list(test_model.net_hide.parameters()), lr=self.LEARNING_RATE)
        optimizer_reveal = torch.optim.AdamW(test_model.net_reveal.parameters(), lr=self.LEARNING_RATE)


        # if load_model:
        #     checkpoint = torch.load(load_path)
        #     test_model.load_state_dict(checkpoint['model_state_dict'])
        #     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        #     optimizer_reveal.load_state_dict(checkpoint['optimizer_reveal_state_dict'])
        #     epoch_idx = checkpoint['epoch']
        #     batch_idx = checkpoint['batch']
        
        test_model.to(self.device)

        test_model.apply(self.init_weights)


        # Commented because slowing down training from uploading too much model details on gradients.
        # if not SKIP_WANDB:
        #     # watch our model and custom loss function, logs the model gradients.
        #     wandb.watch(test_model, my_custom_loss, log="all", log_freq=5000)

        for n in range(self.EPOCHS):
            self.train_step(
                model=test_model,
                dataloader=train_dataloader,
                loss=my_custom_loss,
                optimizer=optimizer,
                optimizer_reveal=optimizer_reveal,
                batch_size=self.BATCH_SIZE,
                batch_idx=batch_idx,
                epoch_idx=epoch_idx,
                epoch_total=self.EPOCHS,
                val_dataloader=val_dataloader)
            
            # Save checkpoint after each epoch
            if self.SAVE_EPOCH_PROGRESS:
                utils.save_checkpoint(test_model, optimizer, optimizer_reveal, epoch_idx, batch_idx, self.SAVE_EPOCH_PROGRESS, self.EPOCHS)

            epoch_idx += 1
            
            # Uncomment for wandb logging of average epoch error.
            # train_loss_epoch, cover_loss_epoch, secret_loss_epoch, cover_loss_mse_epoch, secret_loss_mse_epoch, cover_loss_ssim_epoch, secret_loss_ssim_epoch = train_step(
            #     model=test_model,
            #     dataloader=train_dataloader,
            #     loss=my_custom_loss,
            #     optimizer=optimizer,
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

    

    def train_step(self,
                   model: torch.nn.Module, 
                   dataloader: torch.utils.data.DataLoader, 
                   loss: torch.nn.Module,
                   optimizer: torch.optim.Optimizer,
                   optimizer_reveal: torch.optim.Optimizer,
                   batch_size,
                   batch_idx,
                   epoch_idx,
                   epoch_total,
                   val_dataloader: torch.utils.data.DataLoader):

        # # Uncomment for wandb logging of average epoch error.   
        # train_loss = []
        # cover_loss = []
        # secret_loss = []
        # cover_loss_mse = []
        # secret_loss_mse = []
        # cover_loss_ssim = []
        # secret_loss_ssim = []

        iter_val_dataloader = iter(val_dataloader)

        for index, (data, label) in enumerate(dataloader):
            model.train()
            # Split batch into cover and secret
            a, b = data.split(batch_size//2,dim=0)

            # Send data to GPU device if using GPU
            covers = a.to(self.device)
            secrets = b.to(self.device)

            # Forward
            model_covers, model_secrets = model(covers, secrets)

            # Calculate Loss
            combined_loss,  c_loss, s_loss, c_mse, s_mse, c_ssim, s_ssim, combined_loss_log, c_loss_log, s_loss_log, avg_bce_diff = loss(model_covers, model_secrets, covers, secrets)
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
            if not self.SKIP_WANDB and batch_idx % 100 == 0:
                wandb.log({#'train/batch': batch_idx,
                        'train/batch_loss': combined_loss_log.item(),
                        'train/batch_cover_loss': c_loss_log.item(),
                        'train/batch_secret_loss': s_loss_log.item(),
                        'train/batch_avg_bce_diff': avg_bce_diff.item()*100,
                        'train/batch_cover_mse': c_mse.item(),
                        'train/batch_secret_mse': s_mse.item(),
                        'train/batch_cover_ssim': c_ssim,
                        'train/batch_secret_ssim': s_ssim,
                        })
                
                try:
                    self.validation_step(model, 
                                         iter_val_dataloader, 
                                         loss,
                                         batch_size,
                                         batch_idx)
                
                except StopIteration:
                    iter_val_dataloader = iter(val_dataloader)
                    self.validation_step(model, 
                                         iter_val_dataloader, 
                                         loss,
                                         batch_size,
                                         batch_idx)
                
                # if not SKIP_WANDB_SAVE_IMAGE_OF_FIRST_FEW_BATCHES:
                #     if (batch_idx % 100 == 0 and batch_idx <= 1000):
                #         validation_step(model, 
                #                         val_dataloader, 
                #                         loss,
                #                         batch_size,
                #                         batch_idx)
                
            if self.SAVE_PROGRESS_EVERY_10000_BATCHES and batch_idx % 10000 == 0 and batch_idx > 5000:
                utils.save_checkpoint(model, optimizer, optimizer_reveal, epoch_idx, batch_idx, self.SAVE_EPOCH_PROGRESS, self.EPOCHS)

            batch_idx += 1

        # Uncomment for wandb logging of average epoch error.
        # return train_loss, cover_loss, secret_loss, cover_loss_mse,secret_loss_mse, cover_loss_ssim, secret_loss_ssim



    def validation_step(self,
                        model: torch.nn.Module, 
                        dataloader, 
                        loss: torch.nn.Module,
                        batch_size,
                        batch_idx):
        
        model.eval()

        with torch.inference_mode():
            # img_batch, label_batch = next(iter(dataloader))
            img_batch, label_batch = next(dataloader)
            a, b = img_batch.split(batch_size//2,dim=0)
            covers = a.to(self.device)
            secrets = b.to(self.device)

            model_covers, model_secrets = model(covers, secrets)
            combined_loss,  c_loss, s_loss, c_mse, s_mse, c_ssim, s_ssim, combined_loss_log, c_loss_log, s_loss_log, avg_bce_diff = loss(model_covers, model_secrets, covers, secrets)
        
        # log progress every 100 batches
            if not self.SKIP_WANDB and batch_idx % 100 == 0:
                wandb.log({#'train/batch': batch_idx,
                        'train/batch_loss_val': combined_loss_log.item(),
                        'train/batch_cover_loss_val': c_loss_log.item(),
                        'train/batch_secret_loss_val': s_loss_log.item(),
                        'train/batch_avg_bce_diff_val': avg_bce_diff.item()*100,
                        'train/batch_cover_mse_val': c_mse.item(),
                        'train/batch_secret_mse_val': s_mse.item(),
                        'train/batch_cover_ssim_val': c_ssim,
                        'train/batch_secret_ssim_val': s_ssim,
                        })
                
                if not self.SKIP_WANDB_SAVE_IMAGE_OF_FIRST_FEW_BATCHES:
                    if (batch_idx % 100 == 0 and batch_idx <= 1000):
                        # Try plotting a batch of image fed to model.
                        utils.test_plot_single_batch(self.img_cover, self.img_secret, model, self.device, self.SKIP_WANDB)
                    elif (batch_idx % 500 == 0 and batch_idx <= 3000):
                        # Try plotting a batch of image fed to model.
                        utils.test_plot_single_batch(self.img_cover, self.img_secret, model, self.device, self.SKIP_WANDB)
                    elif (batch_idx % 1000 == 0 and batch_idx > 3000):
                        utils.test_plot_single_batch(self.img_cover, self.img_secret, model, self.device, self.SKIP_WANDB)

        # Turn training back on after eval finished
        model.train()

    def init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)
        # for l in m.children(): init_weights(l)



