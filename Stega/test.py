
import unittest
import os
import torch
from pathlib import Path
from dataset_builder import DatasetBuilder
from models import *
from trainer import *

'''

class TestDatasetBuilder(unittest.TestCase):

    def setUp(self):
        # self.ds_builder = DatasetBuilder(image_name_file="valtemp.txt", train_dir="temptrain", 
        #                     val_dir="tempval", test_dir="temptest")
        self.ds_builder = DatasetBuilder()

    def test_split_dataset(self):
        # Expected split locations for newly downloaded dataset from kaggle, with random shuffle seed = 42
        expected_train_first_last = ["ILSVRC2012_val_00009927", "ILSVRC2012_val_00041715"]
        expected_val_first_last = ["ILSVRC2012_val_00003078", "ILSVRC2012_val_00031343"]
        expected_test_first_last = ["ILSVRC2012_val_00025279", "ILSVRC2012_val_00041906"]

        # Get the split results.
        train_image_names, val_image_names, test_image_names = self.ds_builder.split_dataset()
        train_first_last = [train_image_names[0], train_image_names[-1]]
        val_first_last  = [val_image_names[0], val_image_names[-1]]
        test_first_last  = [test_image_names[0], test_image_names[-1]]

        self.assertListEqual(expected_train_first_last, train_first_last)
        self.assertListEqual(expected_val_first_last, val_first_last)
        self.assertListEqual(expected_test_first_last, test_first_last)

    def test_build_dataset(self):
        # Builds dataset
        self.ds_builder.build_dataset()
        
        # Counts images in train/val/test directory after building dataset.
        total_train_images = len(list(self.ds_builder.train_path.rglob('*.JPEG')))
        total_val_images = len(list(self.ds_builder.val_path.rglob('*.JPEG')))
        total_test_images = len(list(self.ds_builder.test_path.rglob('*.JPEG')))

        self.assertTrue(total_train_images == 40000)
        self.assertTrue(total_val_images == 5000)
        self.assertTrue(total_test_images == 5000)

        # total_train_images = len([name for name in os.listdir("./"+self.ds_builder.train_path.as_posix()) if os.path.isfile(name)])
        # total_val_images = len([name for name in os.listdir(self.ds_builder.val_path.as_posix()) if os.path.isfile(name)])
        # total_test_images = len([name for name in os.listdir(self.ds_builder.test_path.as_posix()) if os.path.isfile(name)])

    def test_verify_directory_structure(self):
        data_path = Path("data/")
        train_dir, val_dir, test_dir = "train", "val", "test"

        imagefolder_train = data_path / train_dir
        imagefolder_val = data_path / val_dir
        imagefolder_test = data_path / test_dir
        train_path = data_path / train_dir / train_dir
        val_path = data_path / val_dir / val_dir
        test_path = data_path / test_dir / test_dir


        self.assertTrue(imagefolder_train.is_dir())
        self.assertTrue(imagefolder_val.is_dir())
        self.assertTrue(imagefolder_test.is_dir())
        self.assertTrue(train_path.is_dir())
        self.assertTrue(val_path.is_dir())
        self.assertTrue(test_path.is_dir())

        

class TestPrepareNetwork(unittest.TestCase):

    def setUp(self):
        self.prep_net = PrepareNetwork()
        self.test_tensor = torch.rand((1, 3, 224, 224))

    def test_forward_helper(self):

        x3, x4, x5, concat_tensor, x3_concat, x4_concat, x5_concat, concat_final = self.prep_net.forward_helper(self.test_tensor, is_unittest=True)

        self.assertTupleEqual(x3.size(), (1, 50, 224, 224))
        self.assertTupleEqual(x4.size(), (1, 50, 224, 224))
        self.assertTupleEqual(x5.size(), (1, 50, 224, 224))
        self.assertTupleEqual(concat_tensor.size(), (1, 150, 224, 224))
        self.assertTupleEqual(x3_concat.size(), (1, 50, 224, 224))
        self.assertTupleEqual(x4_concat.size(), (1, 50, 224, 224))
        self.assertTupleEqual(x5_concat.size(), (1, 50, 224, 224))
        self.assertTupleEqual(concat_final.size(), (1, 150, 224, 224))
        self.assertTrue(concat_final.requires_grad)
    
    def test_forward_helper_operator_fusion(self):

        concat_tensor, final_concat = self.prep_net.forward_helper_operator_fusion(self.test_tensor)

        self.assertTupleEqual(concat_tensor.size(), (1, 150, 224, 224))
        self.assertTupleEqual(final_concat.size(), (1, 150, 224, 224))




class TestHidingNetwork(unittest.TestCase):

    def setUp(self):
        self.hide_net = HidingNetwork()
        self.test_tensor = torch.rand((1, 153, 224, 224))

    def test_forward_helper(self):

        x3, x4, x5, concat_tensor, x3_concat, x4_concat, x5_concat, concat_final, tensor_final, tensor_noise = self.hide_net.forward_helper(self.test_tensor, is_unittest=True)

        self.assertTupleEqual(x3.size(), (1, 50, 224, 224))
        self.assertTupleEqual(x4.size(), (1, 50, 224, 224))
        self.assertTupleEqual(x5.size(), (1, 50, 224, 224))
        self.assertTupleEqual(concat_tensor.size(), (1, 150, 224, 224))
        self.assertTupleEqual(x3_concat.size(), (1, 50, 224, 224))
        self.assertTupleEqual(x4_concat.size(), (1, 50, 224, 224))
        self.assertTupleEqual(x5_concat.size(), (1, 50, 224, 224))
        self.assertTupleEqual(concat_final.size(), (1, 150, 224, 224))

        self.assertTupleEqual(tensor_final.size(), (1, 3, 224, 224))
        self.assertTupleEqual(tensor_noise.size(), (1, 3, 224, 224))
        self.assertTrue(tensor_final.requires_grad)
        self.assertTrue(tensor_noise.requires_grad)
    
    def test_forward_helper_operator_fusion(self):

        concat_tensor, tensor_final, tensor_noise = self.hide_net.forward_helper_operator_fusion(self.test_tensor)

        self.assertTupleEqual(concat_tensor.size(), (1, 150, 224, 224))
        self.assertTupleEqual(tensor_final.size(), (1, 3, 224, 224))
        self.assertTupleEqual(tensor_noise.size(), (1, 3, 224, 224))
        self.assertTrue(tensor_final.requires_grad)
        self.assertTrue(tensor_noise.requires_grad)



class TestRevealNetwork(unittest.TestCase):

    def setUp(self):
        self.reveal_net = RevealNetwork()
        self.test_tensor = torch.rand((1, 3, 224, 224))

    def test_forward_helper(self):

        x3, x4, x5, concat_tensor, x3_concat, x4_concat, x5_concat, concat_final, tensor_final = self.reveal_net.forward_helper(self.test_tensor, is_unittest=True)

        self.assertTupleEqual(x3.size(), (1, 50, 224, 224))
        self.assertTupleEqual(x4.size(), (1, 50, 224, 224))
        self.assertTupleEqual(x5.size(), (1, 50, 224, 224))
        self.assertTupleEqual(concat_tensor.size(), (1, 150, 224, 224))
        self.assertTupleEqual(x3_concat.size(), (1, 50, 224, 224))
        self.assertTupleEqual(x4_concat.size(), (1, 50, 224, 224))
        self.assertTupleEqual(x5_concat.size(), (1, 50, 224, 224))
        self.assertTupleEqual(concat_final.size(), (1, 150, 224, 224))
        self.assertTupleEqual(tensor_final.size(), (1, 3, 224, 224))
        self.assertTrue(tensor_final.requires_grad)
    
    def test_forward_helper_operator_fusion(self):

        concat_tensor, tensor_final = self.reveal_net.forward_helper_operator_fusion(self.test_tensor)

        self.assertTupleEqual(concat_tensor.size(), (1, 150, 224, 224))
        self.assertTupleEqual(tensor_final.size(), (1, 3, 224, 224))
        self.assertTrue(tensor_final.requires_grad)

'''


class TestCombinedNetwork(unittest.TestCase):

    def setUp(self):
        self.combined_net = CombinedNetwork()
        self.cover_tensor = torch.rand((1, 3, 224, 224))
        self.secret_tensor = torch.rand((1, 3, 224, 224))

    def test_forward_helper(self):

        prepped_secrets, prepped_data, modified_cover, modified_cover_noisy, recovered_secret = self.combined_net.forward_helper(self.cover_tensor, self.secret_tensor, 
                                                                                                                                 is_unittest=True)

        self.assertTupleEqual(prepped_secrets.size(), (1, 150, 224, 224))
        self.assertTupleEqual(prepped_data.size(), (1, 153, 224, 224))
        self.assertTupleEqual(modified_cover.size(), (1, 3, 224, 224))
        self.assertTupleEqual(modified_cover_noisy.size(), (1, 3, 224, 224))
        self.assertTupleEqual(recovered_secret.size(), (1, 3, 224, 224))

        self.assertTrue(prepped_secrets.requires_grad)
        self.assertTrue(prepped_data.requires_grad)
        self.assertTrue(modified_cover.requires_grad)
        self.assertTrue(modified_cover_noisy.requires_grad)
        self.assertTrue(recovered_secret.requires_grad)
    
    def test_forward_helper_operator_fusion(self):

        modified_cover, recovered_secret = self.combined_net.forward_helper_operator_fusion(self.cover_tensor, self.secret_tensor)

        self.assertTupleEqual(modified_cover.size(), (1, 3, 224, 224))
        self.assertTupleEqual(recovered_secret.size(), (1, 3, 224, 224))
        self.assertTrue(modified_cover.requires_grad)
        self.assertTrue(recovered_secret.requires_grad)



class TestTrainer(unittest.TestCase):

    def setUp(self):
        self.batch_idx = 1
        self.epoch_idx = 1

        torch.manual_seed(42)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.data_path = Path("data/")
        self.train_dir = self.data_path / "temptrain"
        self.val_dir = self.data_path / "tempval"
        self.test_dir = self.data_path / "temptest"

    def test_get_train_dataloader(self):

        train_dataloader = get_train_dataloader(self.train_dir, BATCH_SIZE, NUM_CPU, NORMALIZE)

        # prepped_secrets, prepped_data, modified_cover, modified_cover_noisy, recovered_secret = self.combined_net.forward_helper(self.cover_tensor, self.secret_tensor, 
        #                                                                                                                          is_unittest=True)

        # self.assertTupleEqual(prepped_secrets.size(), (1, 150, 224, 224))
        # self.assertTupleEqual(prepped_data.size(), (1, 153, 224, 224))
        # self.assertTupleEqual(modified_cover.size(), (1, 3, 224, 224))
        # self.assertTupleEqual(modified_cover_noisy.size(), (1, 3, 224, 224))
        # self.assertTupleEqual(recovered_secret.size(), (1, 3, 224, 224))

        # self.assertTrue(prepped_secrets.requires_grad)
        # self.assertTrue(prepped_data.requires_grad)
        # self.assertTrue(modified_cover.requires_grad)
        # self.assertTrue(modified_cover_noisy.requires_grad)
        # self.assertTrue(recovered_secret.requires_grad)




if __name__ == '__main__':
    unittest.main()

    # suite =  unittest.TestSuite()
    # suite.addTest(TestModels('test_prepare_network_forward'))
    # unittest.TextTestRunner().run(suite())
