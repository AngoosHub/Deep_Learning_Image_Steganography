import random
import os
from pathlib import Path
import shutil



''' 
Run once at the start to reproduce train-val-test split for the dataset. 
Extract the val.zip from Kaggle ImageNet Object Localization Challenge and 
    rename the extracted folder to "train" to save time in moving images.
'''

class DatasetBuilder:
    def __init__(self, data_path = "data/", image_name_file = "val.txt" , train_dir = "train", 
                 val_dir = "val", test_dir = "test"):
        self.data_path = Path(data_path)
        self.image_name_file = self.data_path / image_name_file

        # PyTorch ImageFolder expects a root directory with subdirectories for images, so we will move images to a subdirectory to satisfy it.
        self.imagefolder_train = self.data_path / train_dir
        self.imagefolder_val = self.data_path / val_dir
        self.imagefolder_test = self.data_path / test_dir

        # The new subdir paths
        self.train_path = self.data_path / train_dir / train_dir
        self.val_path = self.data_path / val_dir / val_dir
        self.test_path = self.data_path / test_dir / test_dir

    def split_dataset(self):
        # Get image names
        with open(self.image_name_file) as file:
            image_names = [line.rstrip().split()[0] for line in file]

        # Shuffle images
        random.seed(42)
        random.shuffle(image_names)

        # Split into 80% / 10% / 10% distribution
        train_split = int(0.8 * len(image_names))
        test_split = int(0.9 * len(image_names))
        train_image_names = image_names[:train_split] # slice array from beginning to 80% of array length
        val_image_names = image_names[train_split:test_split] # slice array from 80% to 90% of array length
        test_image_names = image_names[test_split:] # slice array from 90% to end

        print(f"{len(val_image_names)} images to move from train to val.")
        print(f"{len(test_image_names)} images to move from train to test.")

        return train_image_names, val_image_names, test_image_names

        # train_first_last = [train_image_names[0], train_image_names[-1]]
        # val_first_last  = [val_image_names[0], val_image_names[-1]]
        # test_first_last  = [test_image_names[0], test_image_names[-1]]
        # print(f"Train: {str(train_first_last)}")
        # print(f"Val: {str(val_first_last)}")
        # print(f"Test: {str(test_first_last)}")

    def build_dataset(self):
        train_image_names, val_image_names, test_image_names = self.split_dataset()
        if not self.imagefolder_train.exists():
            print("Error, data directory has no train subdirectory.")
            exit()

        # Make directories if they do not exist.
        if not self.imagefolder_val.is_dir():
            self.imagefolder_val.mkdir(parents=True, exist_ok=True)
        if not self.imagefolder_test.is_dir():
            self.imagefolder_test.mkdir(parents=True, exist_ok=True)
        if not self.val_path.is_dir():
            self.val_path.mkdir(parents=True, exist_ok=True)
        if not self.test_path.is_dir():
            self.test_path.mkdir(parents=True, exist_ok=True)
        if not self.train_path.is_dir():
            self.train_path.mkdir(parents=True, exist_ok=True)

        val_image_to_move = []
        for img in val_image_names:
            img_path = f"{img}.JPEG"
            if os.path.isfile(self.imagefolder_train / img_path):
                val_image_to_move.append(img_path)


        test_image_to_move = []
        for img in test_image_names:
            img_path = f"{img}.JPEG"
            if os.path.isfile(self.imagefolder_train / img_path):
                test_image_to_move.append(img_path)

        
        train_image_to_move = []
        for img in train_image_names:
            img_path = f"{img}.JPEG"
            if os.path.isfile(self.imagefolder_train / img_path) and not os.path.isfile(self.train_path / img_path):
                train_image_to_move.append(img_path)


        for img_path in val_image_to_move:
            try:
                (self.imagefolder_train / img_path).rename(self.val_path / img_path)
            except FileNotFoundError:
                print(f"ERROR - Image {img_path} does not exist in training dataset.")
            
        
        for img_path in test_image_to_move:
            try:
                (self.imagefolder_train / img_path).rename(self.test_path / img_path)
            except FileNotFoundError:
                print(f"ERROR - Image {img_path} does not exist in training dataset.")

        for img_path in train_image_to_move:
            try:
                (self.imagefolder_train / img_path).rename(self.train_path / img_path)
            except FileNotFoundError:
                print(f"ERROR - Image {img_path} does not exist in training dataset.")
        
        

        print("Move Complete.")
    


# ds_builder = DatasetBuilder(image_name_file="valtemp.txt", train_dir="temptrain", 
#                             val_dir="tempval", test_dir="temptest")
# ds_builder = DatasetBuilder()
# ds_builder.build_dataset()

# print("Move complete.")
