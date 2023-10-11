import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from ignite.metrics import Loss, SSIM
import os
from pathlib import Path
import time
from models import *
import utils
import dataset_loader
import numpy as np
import wandb
import random
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from trainer_encoder_decoder import TrainerBase


class TrainerDetector(TrainerBase):

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
                 device) -> None:
        
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


        my_custom_loss = nn.BCELoss() # Replace with Binary Cross Entropy for detector loss function
        test_model = DetectNetwork()
        
        # optimizer = torch.optim.AdamW(list(test_model.net_prep.parameters()) + list(test_model.net_hide.parameters()), lr=self.LEARNING_RATE)
        # optimizer_reveal = torch.optim.AdamW(test_model.net_reveal.parameters(), lr=self.LEARNING_RATE)
        optimizer = torch.optim.AdamW(test_model.parameters(), lr=self.LEARNING_RATE)
        
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
                # optimizer_reveal=optimizer_reveal,
                batch_size=self.BATCH_SIZE,
                batch_idx=batch_idx,
                epoch_idx=epoch_idx,
                epoch_total=self.EPOCHS,
                val_dataloader=val_dataloader)
            
            # Save checkpoint after each epoch
            if self.SAVE_EPOCH_PROGRESS:
                utils.save_checkpoint_detector(test_model, optimizer, epoch_idx, batch_idx, self.SAVE_EPOCH_PROGRESS, self.EPOCHS)

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
                #    optimizer_reveal: torch.optim.Optimizer,
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
            covers = data.to(self.device)
            cover_labels = label.to(self.device)
            cover_labels = cover_labels.unsqueeze(1).float()
            
            # Forward pass
            label_pred = model(covers)

            # Calculate Loss
            bce_loss = loss(label_pred, cover_labels)
            model_acc = accuracy_fn(y_pred=torch.round(label_pred), y_true=cover_labels)

            print(label_pred)
            print(cover_labels)

            # Zero out Optimizer gradients before calculate loss
            optimizer.zero_grad()

            # backpropagate loss
            bce_loss.backward()

            # Optimizer step
            optimizer.step()


            # Prints batch progress
            print(f'Epoch: Epoch: [{epoch_idx}/{epoch_total}] | '
                f'Batch {index}/{len(dataloader)}:  BCE_loss = {bce_loss:.4f}, Accuracy = {model_acc}')
            
            # log progress every 100 batches
            if not self.SKIP_WANDB and batch_idx % 10 == 0:
                wandb.log({#'train/batch': batch_idx,
                        'train/BCE_loss': bce_loss,
                        'train/Accuracy': model_acc,
                        # 'train/batch_cover_loss': c_loss.item(),
                        # 'train/batch_secret_loss': s_loss.item(),
                        })
                
                try:
                    self.validation_step(model, 
                                         iter_val_dataloader, 
                                         loss,
                                         self.device,
                                         batch_size,
                                         batch_idx)
                
                except StopIteration:
                    iter_val_dataloader = iter(val_dataloader)
                    self.validation_step(model, 
                                         iter_val_dataloader, 
                                         loss,
                                         self.device,
                                         batch_size,
                                         batch_idx)
                
                # if not SKIP_WANDB_SAVE_IMAGE_OF_FIRST_FEW_BATCHES:
                #     if (batch_idx % 100 == 0 and batch_idx <= 1000):
                #         validation_step(model, 
                #                         val_dataloader, 
                #                         loss,
                #                         device,
                #                         batch_size,
                #                         batch_idx)
                
            
            batch_idx += 1



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
            covers = img_batch.to(self.device)
            cover_labels = label_batch.to(self.device)
            cover_labels = cover_labels.unsqueeze(1).float()

            label_pred = model(covers)
            bce_loss = loss(label_pred, cover_labels)
            model_acc = accuracy_fn(y_pred=torch.round(label_pred), y_true=cover_labels)

            
            # a, b = img_batch.split(batch_size//2,dim=0)
            # covers = a.to(device)
            # secrets = b.to(device)

            # model_covers, model_secrets = model(covers, secrets)
            # combined_loss,  c_loss, s_loss, c_mse, s_mse, c_ssim, s_ssim  = loss(model_covers, model_secrets, covers, secrets)
        
        # log progress every 100 batches
            if not self.SKIP_WANDB and batch_idx % 10 == 0:
                wandb.log({#'train/batch': batch_idx,
                        'train/BCE_loss_val': bce_loss,
                        'train/Accuracy_val': model_acc,
                        # 'train/batch_cover_loss_val': c_loss.item(),
                        # 'train/batch_secret_loss_val': s_loss.item(),
                        })
                
                # if not SKIP_WANDB_SAVE_IMAGE_OF_FIRST_FEW_BATCHES:
                #     if (batch_idx % 100 == 0 and batch_idx <= 1000):
                #         # Try plotting a batch of image fed to model.
                #         test_plot_single_batch(img_cover, img_secret, model, device)
                #     elif batch_idx % 5000 == 0:
                #         test_plot_single_batch(img_cover, img_secret, model, device)

        # Turn training back on after eval finished
        model.train()




    def init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)
        # for l in m.children(): init_weights(l)



# Calculate binary classification accuracy
def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item() # torch.eq() calculates where two tensors are equal
    acc = (correct / len(y_pred)) * 100 
    return acc
