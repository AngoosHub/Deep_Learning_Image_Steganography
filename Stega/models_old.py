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
    Part A:
        First convolutional block uses 3x3 kernel with 3 input channels (RGB) and 50 output channels. (4 layers)
        Second convolutional block uses 4x4 kernel with 3 input channels (RGB) and 50 output channels. (4 layers)
        Third convolutional block uses 5x5 kernel with 3 input channels (RGB) and 50 output channels. (4 layers)
    Concat the output channels from the first three convolutional blocks (50*3 = 150 channels)
    Part B:
        First convolutional block uses 3x3 kernel with 150 input channels and 50 output channels. (1 layer)
        Second convolutional block uses 4x4 kernel with 150 input channels and 50 output channels. (2 layer)
        Third convolutional block uses 5x5 kernel with 150 input channels and 50 output channels. (1 layer)
    Concat the output channels from the first three convolutional blocks (50*3 = 150 channels)
    '''
    def __init__(self, inital_in_channels: int = 3, out_channels: int = 50):
        super().__init__()
        # First convolutional block uses 3x3 kernel with 50 channels.
        self.conv_block_3x3 = nn.Sequential(
            nn.Conv2d(in_channels=inital_in_channels, # Inital input shape, should be 3 channels for RGB.
                      out_channels=out_channels, # Out channels shape, set to 50.
                      kernel_size=3, # 3x3 kernel for conv block.
                      stride=1, # default, kernel window slides 1 pixel when scanning over image.
                      padding="same"), # "same" (output has same shape as input)
                      # padding=1), # padding of 1 to maintain same shape of 224x224
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, # Takes previous Conv2d out_channels as input.
                      out_channels=out_channels, # Keep 50 for output channels.
                      kernel_size=3,
                      stride=1,
                      padding="same"),
                      # padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, # 3rd conv2d layer
                      out_channels=out_channels,
                      kernel_size=3,
                      stride=1,
                      padding="same"),
                      # padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, # 4th conv2d later
                      out_channels=out_channels,
                      kernel_size=3,
                      stride=1,
                      padding="same"),
                      # padding=1),
            nn.ReLU(),
        )
        # Second convolutional block uses 4x4 kernel with 50 channels.
        self.conv_block_4x4 = nn.Sequential(
            nn.Conv2d(in_channels=inital_in_channels, # Inital input shape, should be 3 channels for RGB.
                      out_channels=out_channels, # Out channels shape, set to 50.
                      kernel_size=4, # 4x4 kernel for conv block.
                      stride=1,
                    #   padding="same"), # UserWarning: Using padding='same' with even kernel lengths and odd dilation may require a zero-padded copy of the input be created
                      padding=1), # alternate padding of 1 and 2 to maintain same shape of 224x224
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, # Takes previous Conv2d out_channels as input.
                      out_channels=out_channels, # Keep 50 for output channels.
                      kernel_size=4,
                      stride=1,
                    #   padding="same"),
                      padding=2), # alternate padding of 1 and 2 to maintain same shape of 224x224
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, # 3rd conv2d layer
                      out_channels=out_channels,
                      kernel_size=4,
                      stride=1,
                    #   padding="same"),
                      padding=1), # alternate padding of 1 and 2 to maintain same shape of 224x224
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, # 4th conv2d later
                      out_channels=out_channels,
                      kernel_size=4,
                      stride=1,
                    #   padding="same"),
                      padding=2), # alternate padding of 1 and 2 to maintain same shape of 224x224
            nn.ReLU(),
        )
        # Third convolutional block uses 5x5 kernel with 50 channels.
        self.conv_block_5x5 = nn.Sequential(
            nn.Conv2d(in_channels=inital_in_channels, # Inital input shape, should be 3 channels for RGB.
                      out_channels=out_channels, # Out channels shape, set to 50.
                      kernel_size=5, # 5x5 kernel for conv block.
                      stride=1,
                      padding="same"),
                      # padding=2), # padding of 2 to maintain same shape of 224x224
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, # Takes previous Conv2d out_channels as input.
                      out_channels=out_channels, # Keep 50 for output channels.
                      kernel_size=5,
                      stride=1,
                      padding="same"),
                      # padding=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, # 3rd conv2d layer
                      out_channels=out_channels,
                      kernel_size=5,
                      stride=1,
                      padding="same"),
                      # padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, # 4th conv2d later
                      out_channels=out_channels,
                      kernel_size=5,
                      stride=1,
                      padding="same"),
                      # padding=2),
            nn.ReLU(),
        )

        ### After first concatenation ###
        # Conv block takes 150 input channels with 3x3 kernels, outputs 50 channels.
        self.conv_block_3x3_concat = nn.Sequential(
            nn.Conv2d(in_channels=out_channels*3, # Inital input shape, should be 150 channels.
                      out_channels=out_channels, # Out channels shape, set to 50.
                      kernel_size=3, # 3x3 kernel.
                      stride=1,
                      padding="same"),
                      # padding=1), 
            nn.ReLU(),
            # nn.Conv2d(in_channels=out_channels, # Takes previous Conv2d out_channels as input.
            #           out_channels=out_channels, # Keep 50 for output channels.
            #           kernel_size=3,
            #           stride=1,
            #           padding="same"),
            #           # padding=1),
            # nn.#PReLU(),
        )
        # Conv block takes 150 input channels with 4x4 kernels, outputs 50 channels. (2 layers due to 4x4 kernel imbalanced padding)
        self.conv_block_4x4_concat = nn.Sequential(
            nn.Conv2d(in_channels=out_channels*3, # Inital input shape, should be 150 channels.
                      out_channels=out_channels, # Out channels shape, set to 50.
                      kernel_size=4, # 4x4 kernel.
                      stride=1,
                    #   padding="same"),
                      padding=1), 
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, # Takes previous Conv2d out_channels as input.
                      out_channels=out_channels, # Keep 50 for output channels.
                      kernel_size=4,
                      stride=1,
                    #   padding="same"),
                      padding=2),
            nn.ReLU(),
        )
        # Conv block takes 150 input channels with 5x5 kernels, outputs 50 channels.
        self.conv_block_5x5_concat = nn.Sequential(
            nn.Conv2d(in_channels=out_channels*3, # Inital input shape, should be 150 channels.
                      out_channels=out_channels, # Out channels shape, set to 50.
                      kernel_size=5, # 5x5 kernel.
                      stride=1,
                      padding="same"),
                      # padding=1), 
            nn.ReLU(),
            # nn.Conv2d(in_channels=out_channels, # Takes previous Conv2d out_channels as input.
            #           out_channels=out_channels, # Keep 50 for output channels.
            #           kernel_size=5,
            #           stride=1,
            #           padding="same"),
            #           # padding=1),
            # nn.#PReLU(),
        )
    
    def forward(self, input_tensor: torch.Tensor):
        concat_final = self.forward_helper(input_tensor)
        return concat_final
    
    # Helper function that can be unit tested
    def forward_helper(self, input_tensor: torch.Tensor, is_unittest = False):
        # print(input_tensor.shape)
        x3 = self.conv_block_3x3(input_tensor)
        # print(x3.shape)
        x4 = self.conv_block_4x4(input_tensor)
        # print(x4.shape)
        x5 = self.conv_block_5x5(input_tensor)
        # print(x5.shape)

        # Concat the three output tensors together.
        concat_tensor = torch.cat((x3, x4, x5), 1)
        # print(concat_tensor.shape)
        x3_concat = self.conv_block_3x3_concat(concat_tensor)
        # print(x3_concat.shape)
        x4_concat = self.conv_block_4x4_concat(concat_tensor)
        # print(x4_concat.shape)
        x5_concat = self.conv_block_5x5_concat(concat_tensor)
        # print(x5_concat.shape)
        
        # Concat the three output tensors together again.
        concat_final = torch.cat((x3_concat, x4_concat, x5_concat), 1)
        # print(concat_final.shape)

        # Return final for backpropagation. If Unittesting, return all tensors.
        if is_unittest:
            return x3, x4, x5, concat_tensor, x3_concat, x4_concat, x5_concat, concat_final
        else:
            return concat_final


    def forward_helper_operator_fusion(self, input_tensor: torch.Tensor):
        # operator fusion part 1
        concat_tensor = torch.cat((self.conv_block_3x3(input_tensor), self.conv_block_4x4(input_tensor), self.conv_block_5x5(input_tensor)), 1)
        # operator fusion part 2
        final_concat = torch.cat((self.conv_block_3x3_concat(concat_tensor), self.conv_block_4x4_concat(concat_tensor), self.conv_block_5x5_concat(concat_tensor)), 1)
        return concat_tensor, final_concat




class HidingNetwork(nn.Module):
    '''
    Hiding Network:
        Takes in cover image channels and prepare network output (secret image) concatenated as input.
        Adds noise to the output tensor (modified cover image), so network generalizes a hiding method beyond just LSB.
    Part A:
        First convolutional block uses 3x3 kernel with 153 input channels and 50 output channels. (4 layers)
        Second convolutional block uses 4x4 kernel with 153 input channels and 50 output channels. (4 layers)
        Third convolutional block uses 5x5 kernel with 153 input channels and 50 output channels. (4 layers)
        Concat the output channels from the first three convolutional blocks (50*3 = 150 channels)
    Part B:
        First convolutional block uses 3x3 kernel with 150 input channels and 50 output channels. (1 layer)
        Second convolutional block uses 4x4 kernel with 150 input channels and 50 output channels. (2 layer)
        Third convolutional block uses 5x5 kernel with 150 input channels and 50 output channels. (1 layer)
        Concat the output channels from the first three convolutional blocks (50*3 = 150 channels)
    Part C:
        First convolutional block dimensionality reduction uses 1x1 kernel with 150 input channels and 3 output channels. (1 layer)
        (This outputs tensor back into image with 3 channel)
        Add the output image with a random noise tensor to add noise into the image.
    '''
    def __init__(self, inital_in_channels: int = 153, out_channels: int = 50, output_shape: int = 3):
        super().__init__()
        # First convolutional block uses 3x3 kernel with 50 channels.
        self.conv_block_3x3 = nn.Sequential(
            nn.Conv2d(in_channels=inital_in_channels, # Inital input shape, should be 153 channels.
                      out_channels=out_channels, # Out channels shape, set to 50.
                      kernel_size=3, # 3x3 kernel for conv block.
                      stride=1, # default, kernel window slides 1 pixel when scanning over image.
                      padding="same"), # "same" (output has same shape as input)
                      # padding=1), # padding of 1 to maintain same shape of 224x224
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, # Takes previous Conv2d out_channels as input.
                      out_channels=out_channels, # Keep 50 for output channels.
                      kernel_size=3,
                      stride=1,
                      padding="same"),
                      # padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, # 3rd conv2d layer
                      out_channels=out_channels,
                      kernel_size=3,
                      stride=1,
                      padding="same"),
                      # padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, # 4th conv2d later
                      out_channels=out_channels,
                      kernel_size=3,
                      stride=1,
                      padding="same"),
                      # padding=1),
            nn.ReLU(),
        )
        # Second convolutional block uses 4x4 kernel with 50 channels.
        self.conv_block_4x4 = nn.Sequential(
            nn.Conv2d(in_channels=inital_in_channels, # Inital input shape, should be 153 channels.
                      out_channels=out_channels, # Out channels shape, set to 50.
                      kernel_size=4, # 4x4 kernel for conv block.
                      stride=1,
                    #   padding="same"), # UserWarning: Using padding='same' with even kernel lengths and odd dilation may require a zero-padded copy of the input be created
                      padding=1), # alternate padding of 1 and 2 to maintain same shape of 224x224
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, # Takes previous Conv2d out_channels as input.
                      out_channels=out_channels, # Keep 50 for output channels.
                      kernel_size=4,
                      stride=1,
                    #   padding="same"),
                      padding=2), # alternate padding of 1 and 2 to maintain same shape of 224x224
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, # 3rd conv2d layer
                      out_channels=out_channels,
                      kernel_size=4,
                      stride=1,
                    #   padding="same"),
                      padding=1), # alternate padding of 1 and 2 to maintain same shape of 224x224
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, # 4th conv2d later
                      out_channels=out_channels,
                      kernel_size=4,
                      stride=1,
                    #   padding="same"),
                      padding=2), # alternate padding of 1 and 2 to maintain same shape of 224x224
            nn.ReLU(),
        )
        # Third convolutional block uses 5x5 kernel with 50 channels.
        self.conv_block_5x5 = nn.Sequential(
            nn.Conv2d(in_channels=inital_in_channels, # Inital input shape, should be 153 channels.
                      out_channels=out_channels, # Out channels shape, set to 50.
                      kernel_size=5, # 5x5 kernel for conv block.
                      stride=1,
                      padding="same"),
                      # padding=2), # padding of 2 to maintain same shape of 224x224
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, # Takes previous Conv2d out_channels as input.
                      out_channels=out_channels, # Keep 50 for output channels.
                      kernel_size=5,
                      stride=1,
                      padding="same"),
                      # padding=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, # 3rd conv2d layer
                      out_channels=out_channels,
                      kernel_size=5,
                      stride=1,
                      padding="same"),
                      # padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, # 4th conv2d later
                      out_channels=out_channels,
                      kernel_size=5,
                      stride=1,
                      padding="same"),
                      # padding=2),
            nn.ReLU(),
        )

        ### After first concatenation ###
        # Conv block takes 150 input channels with 3x3 kernels, outputs 50 channels.
        self.conv_block_3x3_concat = nn.Sequential(
            nn.Conv2d(in_channels=out_channels*3, # Inital input shape, should be 150 channels.
                      out_channels=out_channels, # Out channels shape, set to 50.
                      kernel_size=3, # 3x3 kernel.
                      stride=1,
                      padding="same"),
                      # padding=1), 
            nn.ReLU(),
        )
        # Conv block takes 150 input channels with 4x4 kernels, outputs 50 channels. (2 layers due to 4x4 kernel imbalanced padding)
        self.conv_block_4x4_concat = nn.Sequential(
            nn.Conv2d(in_channels=out_channels*3, # Inital input shape, should be 150 channels.
                      out_channels=out_channels, # Out channels shape, set to 50.
                      kernel_size=4, # 4x4 kernel.
                      stride=1,
                    #   padding="same"),
                      padding=1), 
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, # Takes previous Conv2d out_channels as input.
                      out_channels=out_channels, # Keep 50 for output channels.
                      kernel_size=4,
                      stride=1,
                    #   padding="same"),
                      padding=2),
            nn.ReLU(),
        )
        # Conv block takes 150 input channels with 5x5 kernels, outputs 50 channels.
        self.conv_block_5x5_concat = nn.Sequential(
            nn.Conv2d(in_channels=out_channels*3, # Inital input shape, should be 150 channels.
                      out_channels=out_channels, # Out channels shape, set to 50.
                      kernel_size=5, # 5x5 kernel.
                      stride=1,
                      padding="same"),
                      # padding=1), 
            nn.ReLU(),
        )

        ### After second concatenation ###
        # Conv block takes 150 input channels with 1x1 kernels, outputs 3 channels for RGB image.
        self.conv_block_3x3_final = nn.Sequential(
            nn.Conv2d(in_channels=out_channels*3, # Inital input shape, should be 150 channels.
                      out_channels=output_shape, # Output shape, set to 3.
                      kernel_size=1, # 1x1 kernel.
                      stride=1,
                      padding="same"),
                      # padding=0), 
            nn.ReLU(),
        )
    
    def forward(self, input_tensor: torch.Tensor):
        concat_final = self.forward_helper(input_tensor)
        return concat_final
    
    # Helper function that can be unit tested
    def forward_helper(self, input_tensor: torch.Tensor, is_unittest = False):
        # print(input_tensor.shape)
        x3 = self.conv_block_3x3(input_tensor)
        # print(x3.shape)
        x4 = self.conv_block_4x4(input_tensor)
        # print(x4.shape)
        x5 = self.conv_block_5x5(input_tensor)
        # print(x5.shape)

        # Concat the three output tensors together.
        concat_tensor = torch.cat((x3, x4, x5), 1)
        # print(concat_tensor.shape)
        x3_concat = self.conv_block_3x3_concat(concat_tensor)
        # print(x3_concat.shape)
        x4_concat = self.conv_block_4x4_concat(concat_tensor)
        # print(x4_concat.shape)
        x5_concat = self.conv_block_5x5_concat(concat_tensor)
        # print(x5_concat.shape)
        
        # Concat the three output tensors together again.
        concat_final = torch.cat((x3_concat, x4_concat, x5_concat), 1)
        # print(concat_final.shape)

        # Output tensor into image shape.
        tensor_final = self.conv_block_3x3_final(concat_final)
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
            return x3, x4, x5, concat_tensor, x3_concat, x4_concat, x5_concat, concat_final, tensor_final, tensor_noise
        else:
            return tensor_final, tensor_noise


    def forward_helper_operator_fusion(self, input_tensor: torch.Tensor):
        # operator fusion part 1
        concat_tensor = torch.cat((self.conv_block_3x3(input_tensor), self.conv_block_4x4(input_tensor), self.conv_block_5x5(input_tensor)), 1)
        # operator fusion part 2
        tensor_final = self.conv_block_3x3_final(torch.cat((self.conv_block_3x3_concat(concat_tensor), 
                                                            self.conv_block_4x4_concat(concat_tensor), 
                                                            self.conv_block_5x5_concat(concat_tensor)), 1))
        tensor_noise = tensor_final + torch.nn.init.normal_(torch.Tensor(tensor_final.size()), 0, 0.1)
        
        return concat_tensor, tensor_final, tensor_noise





class RevealNetwork(nn.Module):
    '''
    Reveal Network:
        Takes in modifed cover image as input.
        Reveals/extracts the embedded secret image as output.
    Part A:
        First convolutional block uses 3x3 kernel with 3 input channels (RGB) and 50 output channels. (4 layers)
        Second convolutional block uses 4x4 kernel with 3 input channels (RGB) and 50 output channels. (4 layers)
        Third convolutional block uses 5x5 kernel with 3 input channels (RGB) and 50 output channels. (4 layers)
        Concat the output channels from the first three convolutional blocks (50*3 = 150 channels)
    Part B:
        First convolutional block uses 3x3 kernel with 150 input channels and 50 output channels. (1 layer)
        Second convolutional block uses 4x4 kernel with 150 input channels and 50 output channels. (2 layer)
        Third convolutional block uses 5x5 kernel with 150 input channels and 50 output channels. (1 layer)
        Concat the output channels from the first three convolutional blocks (50*3 = 150 channels)
    Part C:
        First convolutional block dimensionality reduction uses 1x1 kernel with 150 input channels and 3 output channels. (1 layer)
        (This outputs tensor back into image with 3 channel)
    '''
    def __init__(self, inital_in_channels: int = 3, out_channels: int = 50, output_shape: int = 3):
        super().__init__()
        # First convolutional block uses 3x3 kernel with 50 channels.
        self.conv_block_3x3 = nn.Sequential(
            nn.Conv2d(in_channels=inital_in_channels, # Inital input shape, should be 3 channels for RGB.
                      out_channels=out_channels, # Out channels shape, set to 50.
                      kernel_size=3, # 3x3 kernel for conv block.
                      stride=1, # default, kernel window slides 1 pixel when scanning over image.
                      padding="same"), # "same" (output has same shape as input)
                      # padding=1), # padding of 1 to maintain same shape of 224x224
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, # Takes previous Conv2d out_channels as input.
                      out_channels=out_channels, # Keep 50 for output channels.
                      kernel_size=3,
                      stride=1,
                      padding="same"),
                      # padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, # 3rd conv2d layer
                      out_channels=out_channels,
                      kernel_size=3,
                      stride=1,
                      padding="same"),
                      # padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, # 4th conv2d later
                      out_channels=out_channels,
                      kernel_size=3,
                      stride=1,
                      padding="same"),
                      # padding=1),
            nn.ReLU(),
        )
        # Second convolutional block uses 4x4 kernel with 50 channels.
        self.conv_block_4x4 = nn.Sequential(
            nn.Conv2d(in_channels=inital_in_channels, # Inital input shape, should be 3 channels for RGB.
                      out_channels=out_channels, # Out channels shape, set to 50.
                      kernel_size=4, # 4x4 kernel for conv block.
                      stride=1,
                    #   padding="same"), # UserWarning: Using padding='same' with even kernel lengths and odd dilation may require a zero-padded copy of the input be created
                      padding=1), # alternate padding of 1 and 2 to maintain same shape of 224x224
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, # Takes previous Conv2d out_channels as input.
                      out_channels=out_channels, # Keep 50 for output channels.
                      kernel_size=4,
                      stride=1,
                    #   padding="same"),
                      padding=2), # alternate padding of 1 and 2 to maintain same shape of 224x224
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, # 3rd conv2d layer
                      out_channels=out_channels,
                      kernel_size=4,
                      stride=1,
                    #   padding="same"),
                      padding=1), # alternate padding of 1 and 2 to maintain same shape of 224x224
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, # 4th conv2d later
                      out_channels=out_channels,
                      kernel_size=4,
                      stride=1,
                    #   padding="same"),
                      padding=2), # alternate padding of 1 and 2 to maintain same shape of 224x224
            nn.ReLU(),
        )
        # Third convolutional block uses 5x5 kernel with 50 channels.
        self.conv_block_5x5 = nn.Sequential(
            nn.Conv2d(in_channels=inital_in_channels, # Inital input shape, should be 3 channels for RGB.
                      out_channels=out_channels, # Out channels shape, set to 50.
                      kernel_size=5, # 5x5 kernel for conv block.
                      stride=1,
                      padding="same"),
                      # padding=2), # padding of 2 to maintain same shape of 224x224
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, # Takes previous Conv2d out_channels as input.
                      out_channels=out_channels, # Keep 50 for output channels.
                      kernel_size=5,
                      stride=1,
                      padding="same"),
                      # padding=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, # 3rd conv2d layer
                      out_channels=out_channels,
                      kernel_size=5,
                      stride=1,
                      padding="same"),
                      # padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, # 4th conv2d later
                      out_channels=out_channels,
                      kernel_size=5,
                      stride=1,
                      padding="same"),
                      # padding=2),
            nn.ReLU(),
        )

        ### After first concatenation ###
        # Conv block takes 150 input channels with 3x3 kernels, outputs 50 channels.
        self.conv_block_3x3_concat = nn.Sequential(
            nn.Conv2d(in_channels=out_channels*3, # Inital input shape, should be 150 channels.
                      out_channels=out_channels, # Out channels shape, set to 50.
                      kernel_size=3, # 3x3 kernel.
                      stride=1,
                      padding="same"),
                      # padding=1), 
            nn.ReLU(),
        )
        # Conv block takes 150 input channels with 4x4 kernels, outputs 50 channels. (2 layers due to 4x4 kernel imbalanced padding)
        self.conv_block_4x4_concat = nn.Sequential(
            nn.Conv2d(in_channels=out_channels*3, # Inital input shape, should be 150 channels.
                      out_channels=out_channels, # Out channels shape, set to 50.
                      kernel_size=4, # 4x4 kernel.
                      stride=1,
                    #   padding="same"),
                      padding=1), 
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, # Takes previous Conv2d out_channels as input.
                      out_channels=out_channels, # Keep 50 for output channels.
                      kernel_size=4,
                      stride=1,
                    #   padding="same"),
                      padding=2),
            nn.ReLU(),
        )
        # Conv block takes 150 input channels with 5x5 kernels, outputs 50 channels.
        self.conv_block_5x5_concat = nn.Sequential(
            nn.Conv2d(in_channels=out_channels*3, # Inital input shape, should be 150 channels.
                      out_channels=out_channels, # Out channels shape, set to 50.
                      kernel_size=5, # 5x5 kernel.
                      stride=1,
                      padding="same"),
                      # padding=1), 
            nn.ReLU(),
        )

        ### After second concatenation ###
        # Conv block takes 150 input channels with 1x1 kernels, outputs 3 channels for RGB image.
        self.conv_block_3x3_final = nn.Sequential(
            nn.Conv2d(in_channels=out_channels*3, # Inital input shape, should be 150 channels.
                      out_channels=output_shape, # Output shape, set to 3.
                      kernel_size=1, # 1x1 kernel.
                      stride=1,
                      padding="same"),
                      # padding=0), 
            nn.ReLU(),
        )
    
    def forward(self, input_tensor: torch.Tensor):
        concat_final = self.forward_helper(input_tensor)
        return concat_final
    
    # Helper function that can be unit tested
    def forward_helper(self, input_tensor: torch.Tensor, is_unittest = False):
        # print(input_tensor.shape)
        x3 = self.conv_block_3x3(input_tensor)
        # print(x3.shape)
        x4 = self.conv_block_4x4(input_tensor)
        # print(x4.shape)
        x5 = self.conv_block_5x5(input_tensor)
        # print(x5.shape)

        # Concat the three output tensors together.
        concat_tensor = torch.cat((x3, x4, x5), 1)
        # print(concat_tensor.shape)
        x3_concat = self.conv_block_3x3_concat(concat_tensor)
        # print(x3_concat.shape)
        x4_concat = self.conv_block_4x4_concat(concat_tensor)
        # print(x4_concat.shape)
        x5_concat = self.conv_block_5x5_concat(concat_tensor)
        # print(x5_concat.shape)
        
        # Concat the three output tensors together again.
        concat_final = torch.cat((x3_concat, x4_concat, x5_concat), 1)
        # print(concat_final.shape)

        # Output tensor into image shape.
        tensor_final = self.conv_block_3x3_final(concat_final)
        # print(tensor_final.shape)

        # Return final for backpropagation. If Unittesting, return all tensors.
        if is_unittest:
            return x3, x4, x5, concat_tensor, x3_concat, x4_concat, x5_concat, concat_final, tensor_final
        else:
            return tensor_final


    def forward_helper_operator_fusion(self, input_tensor: torch.Tensor):
        # operator fusion part 1
        concat_tensor = torch.cat((self.conv_block_3x3(input_tensor), self.conv_block_4x4(input_tensor), self.conv_block_5x5(input_tensor)), 1)
        # operator fusion part 2
        tensor_final = self.conv_block_3x3_final(torch.cat((self.conv_block_3x3_concat(concat_tensor), 
                                                            self.conv_block_4x4_concat(concat_tensor), 
                                                            self.conv_block_5x5_concat(concat_tensor)), 1))
        
        return concat_tensor, tensor_final




class CombinedNetwork_Old(nn.Module):
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







