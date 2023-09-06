import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
from pathlib import Path
import torch.nn as nn


class PrepareNetwork(nn.Module):
    '''
    Prepare Network:
        Takes in secret image as input.
        Outputs secret image with extracted features.
    
    Architecture:
        Part A:
            1. 3x3 Block, 4x4 Block, 5x5 Block in parallel
            2. Concat output of 3 blocks together
        Part B:
            1. 3x3 Block, 4x4 Block, 5x5 Block in parallel
            2. Concat output of 3 blocks together
        Part C:
            1. 3x3 Block, 4x4 Block, 5x5 Block in parallel
            2. Concat output of 3 blocks together
    
    3x3 Conv Blocks consists of the following layers:
        1. Convolution with 3x3 kernel
        2. Batch normalization
        3. PReLu activation
        4. Convolution with 3x3 kernel
        5. Batch normalization
        6. PReLu activation
    
    4x4 Conv Blocks consists of the following layers:
        1. Convolution with 4x4 kernel
        2. Batch normalization
        3. PReLu activation
        4. Convolution with 4x4 kernel
        5. Batch normalization
        6. PReLu activation
    
    5x5 Conv Blocks consists of the following layers:
        1. Convolution with 5x5 kernel
        2. Batch normalization
        3. PReLu activation
        4. Convolution with 5x5 kernel
        5. Batch normalization
        6. PReLu activation
    '''
    def __init__(self, inital_in_channels: int = 3, out_channels: int = 50):
        super().__init__()
        # First convolutional block uses 3x3 kernel with 50 channels.
        self.conv_block_3x3_a = nn.Sequential(
            nn.Conv2d(in_channels=inital_in_channels, # Inital input shape, should be 3 channels for RGB.
                      out_channels=out_channels, # Out channels shape, set to 50.
                      kernel_size=3, # 3x3 kernel for conv block.
                      stride=1, # default, kernel window slides 1 pixel when scanning over image.
                      padding="same"), # "same" (output has same shape as input)
                      # padding=1), # padding of 1 to maintain same shape of 224x224
            nn.BatchNorm2d(out_channels),
            nn.PReLU(out_channels),
            nn.Conv2d(in_channels=out_channels, # Takes previous Conv2d out_channels as input.
                      out_channels=out_channels, # Keep 50 for output channels.
                      kernel_size=3,
                      stride=1,
                      padding="same"),
                      # padding=1),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(out_channels),
        )
        self.conv_block_3x3_b = nn.Sequential(
            nn.Conv2d(in_channels=out_channels*3, # 3rd conv2d layer
                      out_channels=out_channels,
                      kernel_size=3,
                      stride=1,
                      padding="same"),
                      # padding=1),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(out_channels),
            nn.Conv2d(in_channels=out_channels, # 4th conv2d later
                      out_channels=out_channels,
                      kernel_size=3,
                      stride=1,
                      padding="same"),
                      # padding=1),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(out_channels),
        )
        self.conv_block_3x3_c = nn.Sequential(
            nn.Conv2d(in_channels=out_channels*3, # 5rd conv2d layer
                      out_channels=out_channels,
                      kernel_size=3,
                      stride=1,
                      padding="same"),
                      # padding=1),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(out_channels),
            nn.Conv2d(in_channels=out_channels, # 6th conv2d later
                      out_channels=out_channels,
                      kernel_size=3,
                      stride=1,
                      padding="same"),
                      # padding=1),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(out_channels),
        )
        # Second convolutional block uses 4x4 kernel with 50 channels.
        self.conv_block_4x4_a = nn.Sequential(
            nn.Conv2d(in_channels=inital_in_channels, # Inital input shape, should be 3 channels for RGB.
                      out_channels=out_channels, # Out channels shape, set to 50.
                      kernel_size=4, # 4x4 kernel for conv block.
                      stride=1,
                    #   padding="same"), # UserWarning: Using padding='same' with even kernel lengths and odd dilation may require a zero-padded copy of the input be created
                      padding=1), # alternate padding of 1 and 2 to maintain same shape of 224x224
            nn.BatchNorm2d(out_channels),
            nn.PReLU(out_channels),
            nn.Conv2d(in_channels=out_channels, # Takes previous Conv2d out_channels as input.
                      out_channels=out_channels, # Keep 50 for output channels.
                      kernel_size=4,
                      stride=1,
                    #   padding="same"),
                      padding=2), # alternate padding of 1 and 2 to maintain same shape of 224x224
            nn.BatchNorm2d(out_channels),
            nn.PReLU(out_channels),
        )
        self.conv_block_4x4_b = nn.Sequential(
            nn.Conv2d(in_channels=out_channels*3, # 3rd conv2d layer
                      out_channels=out_channels,
                      kernel_size=4,
                      stride=1,
                    #   padding="same"),
                      padding=1), # alternate padding of 1 and 2 to maintain same shape of 224x224
            nn.BatchNorm2d(out_channels),
            nn.PReLU(out_channels),
            nn.Conv2d(in_channels=out_channels, # 4th conv2d later
                      out_channels=out_channels,
                      kernel_size=4,
                      stride=1,
                    #   padding="same"),
                      padding=2), # alternate padding of 1 and 2 to maintain same shape of 224x224
            nn.BatchNorm2d(out_channels),
            nn.PReLU(out_channels),
        )
        self.conv_block_4x4_c = nn.Sequential(
            nn.Conv2d(in_channels=out_channels*3, # 5rd conv2d layer
                      out_channels=out_channels,
                      kernel_size=4,
                      stride=1,
                    #   padding="same"),
                      padding=1), # alternate padding of 1 and 2 to maintain same shape of 224x224
            nn.BatchNorm2d(out_channels),
            nn.PReLU(out_channels),
            nn.Conv2d(in_channels=out_channels, # 6th conv2d later
                      out_channels=out_channels,
                      kernel_size=4,
                      stride=1,
                    #   padding="same"),
                      padding=2), # alternate padding of 1 and 2 to maintain same shape of 224x224
            nn.BatchNorm2d(out_channels),
            nn.PReLU(out_channels),
        )
        # Third convolutional block uses 5x5 kernel with 50 channels.
        self.conv_block_5x5_a = nn.Sequential(
            nn.Conv2d(in_channels=inital_in_channels, # Inital input shape, should be 3 channels for RGB.
                      out_channels=out_channels, # Out channels shape, set to 50.
                      kernel_size=5, # 5x5 kernel for conv block.
                      stride=1,
                      padding="same"),
                      # padding=2), # padding of 2 to maintain same shape of 224x224
            nn.BatchNorm2d(out_channels),
            nn.PReLU(out_channels),
            nn.Conv2d(in_channels=out_channels, # Takes previous Conv2d out_channels as input.
                      out_channels=out_channels, # Keep 50 for output channels.
                      kernel_size=5,
                      stride=1,
                      padding="same"),
                      # padding=2),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(out_channels),
        )
        self.conv_block_5x5_b = nn.Sequential(
            nn.Conv2d(in_channels=out_channels*3, # 3rd conv2d layer
                      out_channels=out_channels,
                      kernel_size=5,
                      stride=1,
                      padding="same"),
                      # padding=1),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(out_channels),
            nn.Conv2d(in_channels=out_channels, # 4th conv2d later
                      out_channels=out_channels,
                      kernel_size=5,
                      stride=1,
                      padding="same"),
                      # padding=2),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(out_channels),
        )
        self.conv_block_5x5_c = nn.Sequential(
            nn.Conv2d(in_channels=out_channels*3, # 5rd conv2d layer
                      out_channels=out_channels,
                      kernel_size=5,
                      stride=1,
                      padding="same"),
                      # padding=1),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(out_channels),
            nn.Conv2d(in_channels=out_channels, # 6th conv2d later
                      out_channels=out_channels,
                      kernel_size=5,
                      stride=1,
                      padding="same"),
                      # padding=2),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(out_channels),
        )

        # ### After first concatenation ###
        # # Conv block takes 150 input channels with 3x3 kernels, outputs 50 channels.
        # self.conv_block_3x3_concat = nn.Sequential(
        #     nn.Conv2d(in_channels=out_channels*3, # Inital input shape, should be 150 channels.
        #               out_channels=out_channels, # Out channels shape, set to 50.
        #               kernel_size=3, # 3x3 kernel.
        #               stride=1,
        #               padding="same"),
        #               # padding=1), 
        #     nn.BatchNorm2d(out_channels),
        #     nn.PReLU(out_channels),
        # )
        # # Conv block takes 150 input channels with 4x4 kernels, outputs 50 channels. (2 layers due to 4x4 kernel imbalanced padding)
        # self.conv_block_4x4_concat = nn.Sequential(
        #     nn.Conv2d(in_channels=out_channels*3, # Inital input shape, should be 150 channels.
        #               out_channels=out_channels, # Out channels shape, set to 50.
        #               kernel_size=4, # 4x4 kernel.
        #               stride=1,
        #             #   padding="same"),
        #               padding=1), 
        #     nn.BatchNorm2d(out_channels),
        #     nn.PReLU(out_channels),
        #     nn.Conv2d(in_channels=out_channels, # Takes previous Conv2d out_channels as input.
        #               out_channels=out_channels, # Keep 50 for output channels.
        #               kernel_size=4,
        #               stride=1,
        #             #   padding="same"),
        #               padding=2),
        #     nn.BatchNorm2d(out_channels),
        #     nn.PReLU(out_channels),
        # )
        # # Conv block takes 150 input channels with 5x5 kernels, outputs 50 channels.
        # self.conv_block_5x5_concat = nn.Sequential(
        #     nn.Conv2d(in_channels=out_channels*3, # Inital input shape, should be 150 channels.
        #               out_channels=out_channels, # Out channels shape, set to 50.
        #               kernel_size=5, # 5x5 kernel.
        #               stride=1,
        #               padding="same"),
        #               # padding=1), 
        #     nn.BatchNorm2d(out_channels),
        #     nn.PReLU(out_channels),
        #     # nn.Conv2d(in_channels=out_channels, # Takes previous Conv2d out_channels as input.
        #     #           out_channels=out_channels, # Keep 50 for output channels.
        #     #           kernel_size=5,
        #     #           stride=1,
        #     #           padding="same"),
        #     #           # padding=1),
        #     # nn.#PReLU(),
        # )
    
    def forward(self, input_tensor: torch.Tensor):
        concat_final = self.forward_helper(input_tensor)
        return concat_final
    
    # Helper function that can be unit tested
    def forward_helper(self, input_tensor: torch.Tensor, is_unittest = False):
        # print(input_tensor.shape)
        x3_a = self.conv_block_3x3_a(input_tensor)
        # print(x3.shape)
        x4_a = self.conv_block_4x4_a(input_tensor)
        # print(x4.shape)
        x5_a = self.conv_block_5x5_a(input_tensor)
        # print(x5.shape)

        # Concat the three output tensors together.
        concat_tensor_a = torch.cat((x3_a, x4_a, x5_a), 1)

        x3_b = self.conv_block_3x3_b(concat_tensor_a)
        x4_b = self.conv_block_4x4_b(concat_tensor_a)
        x5_b = self.conv_block_5x5_b(concat_tensor_a)

        # Concat the three output tensors together.
        concat_tensor_b = torch.cat((x3_b, x4_b, x5_b), 1)

        x3_c = self.conv_block_3x3_c(concat_tensor_b)
        x4_c = self.conv_block_4x4_c(concat_tensor_b)
        x5_c = self.conv_block_5x5_c(concat_tensor_b)

        concat_final = torch.cat((x3_c, x4_c, x5_c), 1)

        # # print(concat_tensor.shape)
        # x3_concat = self.conv_block_3x3_concat(concat_tensor_b)
        # # print(x3_concat.shape)
        # x4_concat = self.conv_block_4x4_concat(concat_tensor_b)
        # # print(x4_concat.shape)
        # x5_concat = self.conv_block_5x5_concat(concat_tensor_b)
        # # print(x5_concat.shape)
        
        # # Concat the three output tensors together again.
        # concat_final = torch.cat((x3_concat, x4_concat, x5_concat), 1)
        # # print(concat_final.shape)

        # Return final for backpropagation. If Unittesting, return all tensors.
        if is_unittest:
            # return x3_a, x4_a, x5_a, concat_tensor_a, x3_concat, x4_concat, x5_concat, concat_final
            return x3_a, x4_a, x5_a, concat_tensor_a, x3_b, x4_b, x5_b, concat_final
            # return x3_a, x4_a, x5_a, concat_tensor_a, x3_b, x4_b, x5_b, concat_tensor_b, x3_c, x4_c, x5_c, concat_final
        else:
            return concat_final


    def forward_helper_operator_fusion(self, input_tensor: torch.Tensor):
        concat_tensor_a = torch.cat((self.conv_block_3x3_a(input_tensor), self.conv_block_4x4_a(input_tensor), self.conv_block_5x5_a(input_tensor)), 1)
        concat_tensor_b = torch.cat((self.conv_block_3x3_b(concat_tensor_a), self.conv_block_4x4_b(concat_tensor_a), self.conv_block_5x5_b(concat_tensor_a)), 1)
        concat_tensor_c = torch.cat((self.conv_block_3x3_c(concat_tensor_b), self.conv_block_4x4_c(concat_tensor_b), self.conv_block_5x5_c(concat_tensor_b)), 1)
        # tensor_final = self.conv_block_1x1_final(concat_tensor_c)
        
        return concat_tensor_a, concat_tensor_c
        # # operator fusion part 1
        # concat_tensor = torch.cat((self.conv_block_3x3(input_tensor), self.conv_block_4x4(input_tensor), self.conv_block_5x5(input_tensor)), 1)
        # # operator fusion part 2
        # final_concat = torch.cat((self.conv_block_3x3_concat(concat_tensor), self.conv_block_4x4_concat(concat_tensor), self.conv_block_5x5_concat(concat_tensor)), 1)
        # return concat_tensor, final_concat




class HidingNetwork(nn.Module):
    '''
    Hiding Network:
        Takes in cover image channels and prepare network output (secret image) concatenated as input.
        Adds noise to the output tensor (modified cover image), so network generalizes a hiding method beyond just LSB.

    Architecture:
        Part A:
            1. 3x3 Block, 4x4 Block, 5x5 Block in parallel
            2. Concat output of 3 blocks together
        Part B:
            1. 3x3 Block, 4x4 Block, 5x5 Block in parallel
            2. Concat output of 3 blocks together
        Part C:
            1. 3x3 Block, 4x4 Block, 5x5 Block in parallel
            2. Concat output of 3 blocks together
        Part D:
            1. Final Convolution layer outputting an image tensor
    
    3x3 Conv Blocks consists of the following layers:
        1. Convolution with 3x3 kernel
        2. Batch normalization
        3. PReLu activation
        4. Convolution with 3x3 kernel
        5. Batch normalization
        6. PReLu activation
    
    4x4 Conv Blocks consists of the following layers:
        1. Convolution with 4x4 kernel
        2. Batch normalization
        3. PReLu activation
        4. Convolution with 4x4 kernel
        5. Batch normalization
        6. PReLu activation
    
    5x5 Conv Blocks consists of the following layers:
        1. Convolution with 5x5 kernel
        2. Batch normalization
        3. PReLu activation
        4. Convolution with 5x5 kernel
        5. Batch normalization
        6. PReLu activation
    '''
    def __init__(self, inital_in_channels: int = 153, out_channels: int = 50, output_shape: int = 3):
        super().__init__()
        # First convolutional block uses 3x3 kernel with 50 channels.
        self.conv_block_3x3_a = nn.Sequential(
            nn.Conv2d(in_channels=inital_in_channels, # Inital input shape, should be 153 channels.
                      out_channels=out_channels, # Out channels shape, set to 50.
                      kernel_size=3, # 3x3 kernel for conv block.
                      stride=1, # default, kernel window slides 1 pixel when scanning over image.
                      padding="same"), # "same" (output has same shape as input)
                      # padding=1), # padding of 1 to maintain same shape of 224x224
            nn.BatchNorm2d(out_channels),
            nn.PReLU(out_channels),
            nn.Conv2d(in_channels=out_channels, # Takes previous Conv2d out_channels as input.
                      out_channels=out_channels, # Keep 50 for output channels.
                      kernel_size=3,
                      stride=1,
                      padding="same"),
                      # padding=1),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(out_channels),
        )
        self.conv_block_3x3_b = nn.Sequential(
            nn.Conv2d(in_channels=out_channels*3, # 3rd conv2d layer
                      out_channels=out_channels,
                      kernel_size=3,
                      stride=1,
                      padding="same"),
                      # padding=1),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(out_channels),
            nn.Conv2d(in_channels=out_channels, # 4th conv2d later
                      out_channels=out_channels,
                      kernel_size=3,
                      stride=1,
                      padding="same"),
                      # padding=1),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(out_channels),
        )
        self.conv_block_3x3_c = nn.Sequential(
            nn.Conv2d(in_channels=out_channels*3, # 5rd conv2d layer
                      out_channels=out_channels,
                      kernel_size=3,
                      stride=1,
                      padding="same"),
                      # padding=1),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(out_channels),
            nn.Conv2d(in_channels=out_channels, # 6th conv2d later
                      out_channels=out_channels,
                      kernel_size=3,
                      stride=1,
                      padding="same"),
                      # padding=1),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(out_channels),
        )
        # Second convolutional block uses 4x4 kernel with 50 channels.
        self.conv_block_4x4_a = nn.Sequential(
            nn.Conv2d(in_channels=inital_in_channels, # Inital input shape, should be 153 channels.
                      out_channels=out_channels, # Out channels shape, set to 50.
                      kernel_size=4, # 4x4 kernel for conv block.
                      stride=1,
                    #   padding="same"), # UserWarning: Using padding='same' with even kernel lengths and odd dilation may require a zero-padded copy of the input be created
                      padding=1), # alternate padding of 1 and 2 to maintain same shape of 224x224
            nn.BatchNorm2d(out_channels),
            nn.PReLU(out_channels),
            nn.Conv2d(in_channels=out_channels, # Takes previous Conv2d out_channels as input.
                      out_channels=out_channels, # Keep 50 for output channels.
                      kernel_size=4,
                      stride=1,
                    #   padding="same"),
                      padding=2), # alternate padding of 1 and 2 to maintain same shape of 224x224
            nn.BatchNorm2d(out_channels),
            nn.PReLU(out_channels),
        )
        self.conv_block_4x4_b = nn.Sequential(
            nn.Conv2d(in_channels=out_channels*3, # 3rd conv2d layer
                      out_channels=out_channels,
                      kernel_size=4,
                      stride=1,
                    #   padding="same"),
                      padding=1), # alternate padding of 1 and 2 to maintain same shape of 224x224
            nn.BatchNorm2d(out_channels),
            nn.PReLU(out_channels),
            nn.Conv2d(in_channels=out_channels, # 4th conv2d later
                      out_channels=out_channels,
                      kernel_size=4,
                      stride=1,
                    #   padding="same"),
                      padding=2), # alternate padding of 1 and 2 to maintain same shape of 224x224
            nn.BatchNorm2d(out_channels),
            nn.PReLU(out_channels),
        )
        self.conv_block_4x4_c = nn.Sequential(
            nn.Conv2d(in_channels=out_channels*3, # 5rd conv2d layer
                      out_channels=out_channels,
                      kernel_size=4,
                      stride=1,
                    #   padding="same"),
                      padding=1), # alternate padding of 1 and 2 to maintain same shape of 224x224
            nn.BatchNorm2d(out_channels),
            nn.PReLU(out_channels),
            nn.Conv2d(in_channels=out_channels, # 6th conv2d later
                      out_channels=out_channels,
                      kernel_size=4,
                      stride=1,
                    #   padding="same"),
                      padding=2), # alternate padding of 1 and 2 to maintain same shape of 224x224
            nn.BatchNorm2d(out_channels),
            nn.PReLU(out_channels),
        )
        # Third convolutional block uses 5x5 kernel with 50 channels.
        self.conv_block_5x5_a = nn.Sequential(
            nn.Conv2d(in_channels=inital_in_channels, # Inital input shape, should be 153 channels.
                      out_channels=out_channels, # Out channels shape, set to 50.
                      kernel_size=5, # 5x5 kernel for conv block.
                      stride=1,
                      padding="same"),
                      # padding=2), # padding of 2 to maintain same shape of 224x224
            nn.BatchNorm2d(out_channels),
            nn.PReLU(out_channels),
            nn.Conv2d(in_channels=out_channels, # Takes previous Conv2d out_channels as input.
                      out_channels=out_channels, # Keep 50 for output channels.
                      kernel_size=5,
                      stride=1,
                      padding="same"),
                      # padding=2),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(out_channels),
        )
        self.conv_block_5x5_b = nn.Sequential(
            nn.Conv2d(in_channels=out_channels*3, # 3rd conv2d layer
                      out_channels=out_channels,
                      kernel_size=5,
                      stride=1,
                      padding="same"),
                      # padding=1),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(out_channels),
            nn.Conv2d(in_channels=out_channels, # 4th conv2d later
                      out_channels=out_channels,
                      kernel_size=5,
                      stride=1,
                      padding="same"),
                      # padding=2),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(out_channels),
        )
        self.conv_block_5x5_c = nn.Sequential(
            nn.Conv2d(in_channels=out_channels*3, # 5rd conv2d layer
                      out_channels=out_channels,
                      kernel_size=5,
                      stride=1,
                      padding="same"),
                      # padding=1),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(out_channels),
            nn.Conv2d(in_channels=out_channels, # 6th conv2d later
                      out_channels=out_channels,
                      kernel_size=5,
                      stride=1,
                      padding="same"),
                      # padding=2),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(out_channels),
        )

        # ### After first concatenation ###
        # # Conv block takes 150 input channels with 3x3 kernels, outputs 50 channels.
        # self.conv_block_3x3_concat = nn.Sequential(
        #     nn.Conv2d(in_channels=out_channels*3, # Inital input shape, should be 150 channels.
        #               out_channels=out_channels, # Out channels shape, set to 50.
        #               kernel_size=3, # 3x3 kernel.
        #               stride=1,
        #               padding="same"),
        #               # padding=1), 
        #     nn.BatchNorm2d(out_channels),
        #     nn.PReLU(out_channels),
        # )
        # # Conv block takes 150 input channels with 4x4 kernels, outputs 50 channels. (2 layers due to 4x4 kernel imbalanced padding)
        # self.conv_block_4x4_concat = nn.Sequential(
        #     nn.Conv2d(in_channels=out_channels*3, # Inital input shape, should be 150 channels.
        #               out_channels=out_channels, # Out channels shape, set to 50.
        #               kernel_size=4, # 4x4 kernel.
        #               stride=1,
        #             #   padding="same"),
        #               padding=1), 
        #     nn.BatchNorm2d(out_channels),
        #     nn.PReLU(out_channels),
        #     nn.Conv2d(in_channels=out_channels, # Takes previous Conv2d out_channels as input.
        #               out_channels=out_channels, # Keep 50 for output channels.
        #               kernel_size=4,
        #               stride=1,
        #             #   padding="same"),
        #               padding=2),
        #     nn.BatchNorm2d(out_channels),
        #     nn.PReLU(out_channels),
        # )
        # # Conv block takes 150 input channels with 5x5 kernels, outputs 50 channels.
        # self.conv_block_5x5_concat = nn.Sequential(
        #     nn.Conv2d(in_channels=out_channels*3, # Inital input shape, should be 150 channels.
        #               out_channels=out_channels, # Out channels shape, set to 50.
        #               kernel_size=5, # 5x5 kernel.
        #               stride=1,
        #               padding="same"),
        #               # padding=1), 
        #     nn.BatchNorm2d(out_channels),
        #     nn.PReLU(out_channels),
        # )

        ### After second concatenation ###
        # Conv block takes 150 input channels with 1x1 kernels, outputs 3 channels for RGB image.
        self.conv_block_1x1_final = nn.Sequential(
            nn.Conv2d(in_channels=out_channels*3, # Inital input shape, should be 150 channels.
                      out_channels=output_shape, # Output shape, set to 3.
                      kernel_size=1, # 1x1 kernel.
                      stride=1,
                      padding="same"),
                      # padding=0), 
            nn.BatchNorm2d(output_shape),
            nn.PReLU(output_shape),
        )
    
    def forward(self, input_tensor: torch.Tensor):
        concat_final = self.forward_helper(input_tensor)
        return concat_final
    
    # Helper function that can be unit tested
    def forward_helper(self, input_tensor: torch.Tensor, is_unittest = False):
        # print(input_tensor.shape)
        x3_a = self.conv_block_3x3_a(input_tensor)
        # print(x3.shape)
        x4_a = self.conv_block_4x4_a(input_tensor)
        # print(x4.shape)
        x5_a = self.conv_block_5x5_a(input_tensor)
        # print(x5.shape)

        # Concat the three output tensors together.
        concat_tensor_a = torch.cat((x3_a, x4_a, x5_a), 1)

        x3_b = self.conv_block_3x3_b(concat_tensor_a)
        x4_b = self.conv_block_4x4_b(concat_tensor_a)
        x5_b = self.conv_block_5x5_b(concat_tensor_a)

        concat_tensor_b = torch.cat((x3_b, x4_b, x5_b), 1)

        x3_c = self.conv_block_3x3_c(concat_tensor_b)
        x4_c = self.conv_block_4x4_c(concat_tensor_b)
        x5_c = self.conv_block_5x5_c(concat_tensor_b)

        concat_tensor_c = torch.cat((x3_c, x4_c, x5_c), 1)

        # # print(concat_tensor.shape)
        # x3_concat = self.conv_block_3x3_concat(concat_tensor_c)
        # # print(x3_concat.shape)
        # x4_concat = self.conv_block_4x4_concat(concat_tensor_c)
        # # print(x4_concat.shape)
        # x5_concat = self.conv_block_5x5_concat(concat_tensor_c)
        # # print(x5_concat.shape)
        
        # # Concat the three output tensors together again.
        # concat_final = torch.cat((x3_concat, x4_concat, x5_concat), 1)
        # # print(concat_final.shape)

        # Output tensor into image shape.
        # tensor_final = self.conv_block_1x1_final(concat_final)
        tensor_final = self.conv_block_1x1_final(concat_tensor_c)
        # print(tensor_final.shape)

        # Add noise to tensor. Set autograd to true as torch.nn.init.normal_ creates tensors with autograd = false by default.
        if (tensor_final.is_cuda):
            device = "cuda" if torch.cuda.is_available() else "cpu"
            tensor_noise = tensor_final + torch.nn.init.normal_(torch.Tensor(tensor_final.size()), 0, 0.1).to(device)
        else:
            tensor_noise = tensor_final + torch.nn.init.normal_(torch.Tensor(tensor_final.size()), 0, 0.1)
        # tensor_noise.requires_grad_()


        # Return final for backpropagation. If Unittesting, return all tensors.
        if is_unittest:
            # return x3, x4, x5, concat_tensor, x3_concat, x4_concat, x5_concat, concat_final, tensor_final, tensor_noise
            return x3_a, x4_a, x5_a, concat_tensor_a, x3_b, x4_b, x5_b, concat_tensor_b, tensor_final, tensor_noise
            # return x3_a, x4_a, x5_a, concat_tensor_a, x3_b, x4_b, x5_b, concat_tensor_b, x3_c, x4_c, x5_c, concat_tensor_c, tensor_final, tensor_noise
        else:
            return tensor_final, tensor_noise


    def forward_helper_operator_fusion(self, input_tensor: torch.Tensor):
        concat_tensor_a = torch.cat((self.conv_block_3x3_a(input_tensor), self.conv_block_4x4_a(input_tensor), self.conv_block_5x5_a(input_tensor)), 1)
        concat_tensor_b = torch.cat((self.conv_block_3x3_b(concat_tensor_a), self.conv_block_4x4_b(concat_tensor_a), self.conv_block_5x5_b(concat_tensor_a)), 1)
        concat_tensor_c = torch.cat((self.conv_block_3x3_c(concat_tensor_b), self.conv_block_4x4_c(concat_tensor_b), self.conv_block_5x5_c(concat_tensor_b)), 1)
        tensor_final = self.conv_block_1x1_final(concat_tensor_c)
        
        return concat_tensor_a, tensor_final
    
        # # operator fusion part 1
        # concat_tensor = torch.cat((self.conv_block_3x3(input_tensor), self.conv_block_4x4(input_tensor), self.conv_block_5x5(input_tensor)), 1)
        # # operator fusion part 2
        # tensor_final = self.conv_block_1x1_final(torch.cat((self.conv_block_3x3_concat(concat_tensor), 
        #                                                     self.conv_block_4x4_concat(concat_tensor), 
        #                                                     self.conv_block_5x5_concat(concat_tensor)), 1))
        # tensor_noise = tensor_final + torch.nn.init.normal_(torch.Tensor(tensor_final.size()), 0, 0.1)
        
        # return concat_tensor, tensor_final, tensor_noise





class RevealNetwork(nn.Module):
    '''
    Reveal Network:
        Takes in modifed cover image as input.
        Reveals/extracts the embedded secret image as output.

    Architecture:
        Part A:
            1. 3x3 Block, 4x4 Block, 5x5 Block in parallel
            2. Concat output of 3 blocks together
        Part B:
            1. 3x3 Block, 4x4 Block, 5x5 Block in parallel
            2. Concat output of 3 blocks together
        Part C:
            1. 3x3 Block, 4x4 Block, 5x5 Block in parallel
            2. Concat output of 3 blocks together
        Part D:
            1. Final Convolution layer outputting an image tensor
    
    3x3 Conv Blocks consists of the following layers:
        1. Convolution with 3x3 kernel
        2. Batch normalization
        3. PReLu activation
        4. Convolution with 3x3 kernel
        5. Batch normalization
        6. PReLu activation
    
    4x4 Conv Blocks consists of the following layers:
        1. Convolution with 4x4 kernel
        2. Batch normalization
        3. PReLu activation
        4. Convolution with 4x4 kernel
        5. Batch normalization
        6. PReLu activation
    
    5x5 Conv Blocks consists of the following layers:
        1. Convolution with 5x5 kernel
        2. Batch normalization
        3. PReLu activation
        4. Convolution with 5x5 kernel
        5. Batch normalization
        6. PReLu activation
    '''
    def __init__(self, inital_in_channels: int = 3, out_channels: int = 50, output_shape: int = 3):
        super().__init__()
        # First convolutional block uses 3x3 kernel with 50 channels.
        self.conv_block_3x3_a = nn.Sequential(
            nn.Conv2d(in_channels=inital_in_channels, # Inital input shape, should be 3 channels for RGB.
                      out_channels=out_channels, # Out channels shape, set to 50.
                      kernel_size=3, # 3x3 kernel for conv block.
                      stride=1, # default, kernel window slides 1 pixel when scanning over image.
                      padding="same"), # "same" (output has same shape as input)
                      # padding=1), # padding of 1 to maintain same shape of 224x224
            nn.BatchNorm2d(out_channels),
            nn.PReLU(out_channels),
            nn.Conv2d(in_channels=out_channels, # Takes previous Conv2d out_channels as input.
                      out_channels=out_channels, # Keep 50 for output channels.
                      kernel_size=3,
                      stride=1,
                      padding="same"),
                      # padding=1),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(out_channels),
        )
        self.conv_block_3x3_b = nn.Sequential(
            nn.Conv2d(in_channels=out_channels*3, # 3rd conv2d layer
                      out_channels=out_channels,
                      kernel_size=3,
                      stride=1,
                      padding="same"),
                      # padding=1),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(out_channels),
            nn.Conv2d(in_channels=out_channels, # 4th conv2d later
                      out_channels=out_channels,
                      kernel_size=3,
                      stride=1,
                      padding="same"),
                      # padding=1),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(out_channels),
        )
        self.conv_block_3x3_c = nn.Sequential(
            nn.Conv2d(in_channels=out_channels*3, # 3rd conv2d layer
                      out_channels=out_channels,
                      kernel_size=3,
                      stride=1,
                      padding="same"),
                      # padding=1),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(out_channels),
            nn.Conv2d(in_channels=out_channels, # 4th conv2d later
                      out_channels=out_channels,
                      kernel_size=3,
                      stride=1,
                      padding="same"),
                      # padding=1),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(out_channels),
        )
        # Second convolutional block uses 4x4 kernel with 50 channels.
        self.conv_block_4x4_a = nn.Sequential(
            nn.Conv2d(in_channels=inital_in_channels, # Inital input shape, should be 3 channels for RGB.
                      out_channels=out_channels, # Out channels shape, set to 50.
                      kernel_size=4, # 4x4 kernel for conv block.
                      stride=1,
                    #   padding="same"), # UserWarning: Using padding='same' with even kernel lengths and odd dilation may require a zero-padded copy of the input be created
                      padding=1), # alternate padding of 1 and 2 to maintain same shape of 224x224
            nn.BatchNorm2d(out_channels),
            nn.PReLU(out_channels),
            nn.Conv2d(in_channels=out_channels, # Takes previous Conv2d out_channels as input.
                      out_channels=out_channels, # Keep 50 for output channels.
                      kernel_size=4,
                      stride=1,
                    #   padding="same"),
                      padding=2), # alternate padding of 1 and 2 to maintain same shape of 224x224
            nn.BatchNorm2d(out_channels),
            nn.PReLU(out_channels),
        )
        self.conv_block_4x4_b = nn.Sequential(
            nn.Conv2d(in_channels=out_channels*3, # 3rd conv2d layer
                      out_channels=out_channels,
                      kernel_size=4,
                      stride=1,
                    #   padding="same"),
                      padding=1), # alternate padding of 1 and 2 to maintain same shape of 224x224
            nn.BatchNorm2d(out_channels),
            nn.PReLU(out_channels),
            nn.Conv2d(in_channels=out_channels, # 4th conv2d later
                      out_channels=out_channels,
                      kernel_size=4,
                      stride=1,
                    #   padding="same"),
                      padding=2), # alternate padding of 1 and 2 to maintain same shape of 224x224
            nn.BatchNorm2d(out_channels),
            nn.PReLU(out_channels),
        )
        self.conv_block_4x4_c = nn.Sequential(
            nn.Conv2d(in_channels=out_channels*3, # 3rd conv2d layer
                      out_channels=out_channels,
                      kernel_size=4,
                      stride=1,
                    #   padding="same"),
                      padding=1), # alternate padding of 1 and 2 to maintain same shape of 224x224
            nn.BatchNorm2d(out_channels),
            nn.PReLU(out_channels),
            nn.Conv2d(in_channels=out_channels, # 4th conv2d later
                      out_channels=out_channels,
                      kernel_size=4,
                      stride=1,
                    #   padding="same"),
                      padding=2), # alternate padding of 1 and 2 to maintain same shape of 224x224
            nn.BatchNorm2d(out_channels),
            nn.PReLU(out_channels),
        )
        # Third convolutional block uses 5x5 kernel with 50 channels.
        self.conv_block_5x5_a = nn.Sequential(
            nn.Conv2d(in_channels=inital_in_channels, # Inital input shape, should be 3 channels for RGB.
                      out_channels=out_channels, # Out channels shape, set to 50.
                      kernel_size=5, # 5x5 kernel for conv block.
                      stride=1,
                      padding="same"),
                      # padding=2), # padding of 2 to maintain same shape of 224x224
            nn.BatchNorm2d(out_channels),
            nn.PReLU(out_channels),
            nn.Conv2d(in_channels=out_channels, # Takes previous Conv2d out_channels as input.
                      out_channels=out_channels, # Keep 50 for output channels.
                      kernel_size=5,
                      stride=1,
                      padding="same"),
                      # padding=2),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(out_channels),
        )
        self.conv_block_5x5_b = nn.Sequential(
            nn.Conv2d(in_channels=out_channels*3, # 3rd conv2d layer
                      out_channels=out_channels,
                      kernel_size=5,
                      stride=1,
                      padding="same"),
                      # padding=1),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(out_channels),
            nn.Conv2d(in_channels=out_channels, # 4th conv2d later
                      out_channels=out_channels,
                      kernel_size=5,
                      stride=1,
                      padding="same"),
                      # padding=2),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(out_channels),
        )
        self.conv_block_5x5_c = nn.Sequential(
            nn.Conv2d(in_channels=out_channels*3, # 3rd conv2d layer
                      out_channels=out_channels,
                      kernel_size=5,
                      stride=1,
                      padding="same"),
                      # padding=1),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(out_channels),
            nn.Conv2d(in_channels=out_channels, # 4th conv2d later
                      out_channels=out_channels,
                      kernel_size=5,
                      stride=1,
                      padding="same"),
                      # padding=2),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(out_channels),
        )

        # ### After first concatenation ###
        # # Conv block takes 150 input channels with 3x3 kernels, outputs 50 channels.
        # self.conv_block_3x3_concat = nn.Sequential(
        #     nn.Conv2d(in_channels=out_channels*3, # Inital input shape, should be 150 channels.
        #               out_channels=out_channels, # Out channels shape, set to 50.
        #               kernel_size=3, # 3x3 kernel.
        #               stride=1,
        #               padding="same"),
        #               # padding=1), 
        #     nn.BatchNorm2d(out_channels),
        #     nn.PReLU(out_channels),
        # )
        # # Conv block takes 150 input channels with 4x4 kernels, outputs 50 channels. (2 layers due to 4x4 kernel imbalanced padding)
        # self.conv_block_4x4_concat = nn.Sequential(
        #     nn.Conv2d(in_channels=out_channels*3, # Inital input shape, should be 150 channels.
        #               out_channels=out_channels, # Out channels shape, set to 50.
        #               kernel_size=4, # 4x4 kernel.
        #               stride=1,
        #             #   padding="same"),
        #               padding=1), 
        #     nn.BatchNorm2d(out_channels),
        #     nn.PReLU(out_channels),
        #     nn.Conv2d(in_channels=out_channels, # Takes previous Conv2d out_channels as input.
        #               out_channels=out_channels, # Keep 50 for output channels.
        #               kernel_size=4,
        #               stride=1,
        #             #   padding="same"),
        #               padding=2),
        #     nn.BatchNorm2d(out_channels),
        #     nn.PReLU(out_channels),
        # )
        # # Conv block takes 150 input channels with 5x5 kernels, outputs 50 channels.
        # self.conv_block_5x5_concat = nn.Sequential(
        #     nn.Conv2d(in_channels=out_channels*3, # Inital input shape, should be 150 channels.
        #               out_channels=out_channels, # Out channels shape, set to 50.
        #               kernel_size=5, # 5x5 kernel.
        #               stride=1,
        #               padding="same"),
        #               # padding=1), 
        #     nn.BatchNorm2d(out_channels),
        #     nn.PReLU(out_channels),
        # )

        ### After second concatenation ###
        # Conv block takes 150 input channels with 1x1 kernels, outputs 3 channels for RGB image.
        self.conv_block_1x1_final = nn.Sequential(
            nn.Conv2d(in_channels=out_channels*3, # Inital input shape, should be 150 channels.
                      out_channels=output_shape, # Output shape, set to 3.
                      kernel_size=1, # 1x1 kernel.
                      stride=1,
                      padding="same"),
                      # padding=0), 
            nn.BatchNorm2d(output_shape),
            nn.PReLU(output_shape),
        )
    
    def forward(self, input_tensor: torch.Tensor):
        concat_final = self.forward_helper(input_tensor)
        return concat_final
    
    # Helper function that can be unit tested
    def forward_helper(self, input_tensor: torch.Tensor, is_unittest = False):
        # print(input_tensor.shape)
        x3_a = self.conv_block_3x3_a(input_tensor)
        # print(x3.shape)
        x4_a = self.conv_block_4x4_a(input_tensor)
        # print(x4.shape)
        x5_a = self.conv_block_5x5_a(input_tensor)
        # print(x5.shape)

        # Concat the three output tensors together.
        concat_tensor_a = torch.cat((x3_a, x4_a, x5_a), 1)

        x3_b = self.conv_block_3x3_b(concat_tensor_a)
        x4_b = self.conv_block_4x4_b(concat_tensor_a)
        x5_b = self.conv_block_5x5_b(concat_tensor_a)

        concat_tensor_b = torch.cat((x3_b, x4_b, x5_b), 1)

        x3_c = self.conv_block_3x3_c(concat_tensor_b)
        x4_c = self.conv_block_4x4_c(concat_tensor_b)
        x5_c = self.conv_block_5x5_c(concat_tensor_b)

        concat_tensor_c = torch.cat((x3_c, x4_c, x5_c), 1)

        # # print(concat_tensor.shape)
        # x3_concat = self.conv_block_3x3_concat(concat_tensor)
        # # print(x3_concat.shape)
        # x4_concat = self.conv_block_4x4_concat(concat_tensor)
        # # print(x4_concat.shape)
        # x5_concat = self.conv_block_5x5_concat(concat_tensor)
        # # print(x5_concat.shape)
        
        # # Concat the three output tensors together again.
        # concat_final = torch.cat((x3_concat, x4_concat, x5_concat), 1)
        # # print(concat_final.shape)

        # Output tensor into image shape.
        tensor_final = self.conv_block_1x1_final(concat_tensor_c)
        # print(tensor_final.shape)

        # Return final for backpropagation. If Unittesting, return all tensors.
        if is_unittest:
            # return x3, x4, x5, concat_tensor, x3_concat, x4_concat, x5_concat, concat_final, tensor_final
            return x3_a, x4_a, x5_a, concat_tensor_a, x3_b, x4_b, x5_b, concat_tensor_b, tensor_final
            # return x3_a, x4_a, x5_a, concat_tensor_a, x3_b, x4_b, x5_b, concat_tensor_b, x3_c, x4_c, x5_c, concat_tensor_c, tensor_final
        else:
            return tensor_final


    def forward_helper_operator_fusion(self, input_tensor: torch.Tensor):
        # operator fusion part 1
        concat_tensor_a = torch.cat((self.conv_block_3x3_a(input_tensor), self.conv_block_4x4_a(input_tensor), self.conv_block_5x5_a(input_tensor)), 1)
        concat_tensor_b = torch.cat((self.conv_block_3x3_b(concat_tensor_a), self.conv_block_4x4_b(concat_tensor_a), self.conv_block_5x5_b(concat_tensor_a)), 1)
        concat_tensor_c = torch.cat((self.conv_block_3x3_c(concat_tensor_b), self.conv_block_4x4_c(concat_tensor_b), self.conv_block_5x5_c(concat_tensor_b)), 1)
        # operator fusion part 2
        tensor_final = self.conv_block_1x1_final(concat_tensor_c)
        
        # return concat_tensor, tensor_final
        return concat_tensor_a, tensor_final




class CombinedNetwork(nn.Module):
    '''
    Combined Network:
        Joins the all the networks together into a single neural network.
        Passes the output a network as input to the next network in sequence.
    '''
    def __init__(self):
        super().__init__()
        self.net_prep = PrepareNetwork()
        self.net_hide = HidingNetwork()
        self.net_reveal = RevealNetwork()

    
    def forward(self, cover_images: torch.Tensor, secret_images: torch.Tensor):
        modified_cover, recovered_secret,  = self.forward_helper(cover_images, secret_images)
        return modified_cover, recovered_secret
    

    # Helper function that can be unit tested
    def forward_helper(self, cover_images: torch.Tensor, secret_images: torch.Tensor, is_unittest = False):

        prepped_secrets = self.net_prep(secret_images)
        prepped_data = torch.cat((prepped_secrets, cover_images), 1)
        modified_cover, modified_cover_noisy = self.net_hide(prepped_data)
        recovered_secret = self.net_reveal(modified_cover_noisy)

        # Return final for backpropagation. If Unittesting, return all tensors.
        if is_unittest:
            return prepped_secrets, prepped_data, modified_cover, modified_cover_noisy, recovered_secret
        else:
            return modified_cover, recovered_secret


    def forward_helper_operator_fusion(self, cover_images: torch.Tensor, secret_images: torch.Tensor):
        modified_cover, modified_cover_noisy = self.net_hide(torch.cat((self.net_prep(secret_images), cover_images), 1))
        return modified_cover, self.net_reveal(modified_cover_noisy)
    
    def reveal_only(self, secret_tensor: torch.Tensor):
        recovered_secret = self.net_reveal(secret_tensor)
        return recovered_secret






class DetectNetwork(nn.Module):
    '''
    Detect Network:
        Takes in modifed cover image as input.
        Outputs prediction of image as steganography image.

    Each Convolutional Block consists of the following layers:
        1. Convolution with 3x3 kernel
        2. Batch normalization
        3. PReLu activation
        4. Average Pooling or Adaptive Average Pooling
    
    Architecture:
        Block 1
            - Input Shape: 3 x 224 x 224
            - Output Shape: 8 x 112 x 112
        Block 2
            - Input Shape: 8 x 112 x 112
            - Output Shape: 16 x 56 x 56
        Block 3
            - Input Shape: 16 x 56 x 56
            - Output Shape: 32 x 28 x 28
        Block 4
            - Input Shape: 32 x 28 x 28
            - Output Shape: 64 x 14 x 14
        Block 5
            - Input Shape: 64 x 28 x 28
            - Output Shape: 128 x 1
        Fully-connected and Sigmond layer for prediction.
        
    '''
    def __init__(self, inital_in_channels: int = 3, out_channels: int = 50, output_shape: int = 3):
        super().__init__()

        self.conv_block_3x3 = nn.Sequential(
            # in_channels is 3 for RGB
            # out_channels is 8 (8>16>32>64>128 is common CNN classifer filter sizes)
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=1), 
            nn.BatchNorm2d(8),
            nn.PReLU(8),
            # Average pool stride is 2 to reduce shape (W and H) by half 224x224 > 112x112
            nn.AvgPool2d(kernel_size=3, stride=2),

            # in_channels is 8
            # out_channels is 16
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1), 
            nn.BatchNorm2d(16),
            nn.PReLU(16),
            # Average pool (112x112 > 56x56)
            nn.AvgPool2d(kernel_size=3, stride=2),

            # in_channels is 16
            # out_channels is 32
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1), 
            nn.BatchNorm2d(32),
            nn.PReLU(32),
            # Average pool (56x56 > 28x28)
            nn.AvgPool2d(kernel_size=3, stride=2),

            # in_channels is 32
            # out_channels is 64
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1), 
            nn.BatchNorm2d(64),
            nn.PReLU(64),
            # Average pool (28x28 > 14x14)
            nn.AvgPool2d(kernel_size=3, stride=2),

            # in_channels is 64
            # out_channels is 128
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1), 
            nn.BatchNorm2d(128),
            nn.PReLU(128),
            # Global average pool
            nn.AdaptiveAvgPool2d((1, 1)),

            # Fully connected layer 128 to 1, and sigmoid for binary classification.
            nn.Flatten(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )
        
    
    def forward(self, input_tensor: torch.Tensor):
        concat_final = self.forward_helper(input_tensor)
        return concat_final
    
    # Helper function that can be unit tested
    def forward_helper(self, input_tensor: torch.Tensor, is_unittest = False):
        # print(input_tensor.shape)
        tensor_final = self.conv_block_3x3(input_tensor)
        print(tensor_final.shape)

        # Return final for backpropagation. If Unittesting, return all tensors.
        if is_unittest:
            # return x3, x4, x5, concat_tensor, x3_concat, x4_concat, x5_concat, concat_final, tensor_final
            return tensor_final
        else:
            return tensor_final


    # def forward_helper_operator_fusion(self, input_tensor: torch.Tensor):
    #     # operator fusion part 1
    #     concat_tensor_a = torch.cat((self.conv_block_3x3_a(input_tensor), self.conv_block_4x4_a(input_tensor), self.conv_block_5x5_a(input_tensor)), 1)
    #     concat_tensor_b = torch.cat((self.conv_block_3x3_b(concat_tensor_a), self.conv_block_4x4_b(concat_tensor_a), self.conv_block_5x5_b(concat_tensor_a)), 1)
    #     concat_tensor_c = torch.cat((self.conv_block_3x3_c(concat_tensor_b), self.conv_block_4x4_c(concat_tensor_b), self.conv_block_5x5_c(concat_tensor_b)), 1)
    #     # operator fusion part 2
    #     tensor_final = self.conv_block_1x1_final(concat_tensor_c)
        
    #     # return concat_tensor, tensor_final
    #     return concat_tensor_a, tensor_final



