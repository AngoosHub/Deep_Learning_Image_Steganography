import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import random
import numpy as np



class DatasetLoader():
    def __init__(self) -> None:
        pass

    @staticmethod
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


    @staticmethod
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


    @staticmethod
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


# Seed Pytorch dataloader random seed for workers.
def seed_worker(worker_num):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(0)