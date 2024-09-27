import torch
from ignite.metrics import Loss, SSIM
from models import *

from ignite.engine import *
from ignite.handlers import *
from ignite.metrics import *
from ignite.utils import *
from ignite.contrib.metrics.regression import *
from ignite.contrib.metrics import *


class MSE_and_SSIM_and_Detector_loss(nn.Module):
    
    def __init__(self, detector_model, BETA = 0.75):
        super(MSE_and_SSIM_and_Detector_loss, self).__init__()
        self.loss_mse = torch.nn.MSELoss()
        self.default_evaluator = Engine(eval_step)
        self.metric = SSIM(data_range=1.0)
        self.metric.attach(self.default_evaluator, 'ssim')
        self.detector = detector_model
        self.BETA = BETA


    def forward(self, cover, secret, cover_original, secret_original):        
        cover_mse = self.loss_mse(input=cover, target=cover_original)
        secret_mse = self.loss_mse(input=secret, target=secret_original)

        cover_ssim_ev = self.default_evaluator.run([[cover, cover_original]])
        
        # print(f"Cover SSIM: {1 - cover_ssim.metrics['ssim']}")
        # print(f"Secret SSIM: {1 - secret_ssim.metrics['ssim']}")

        # print(f"Cover MSE: {cover_loss}")
        # print(f"Secret MSE: {secret_loss}")

        # Get Detector predictions on original and modified images
        cover_pred = self.detector(cover)
        cover_original_pred = self.detector(cover_original)

        # print("cover_pred")
        # print(cover_pred)
        # print("cover_original")
        # print(cover_original_pred)
        bce_difference = torch.subtract(cover_original_pred,cover_pred)
        # print("bce_difference")
        # print(bce_difference)
        bce_difference[bce_difference < 0] = 0
        avg_bce_diff = torch.mean(bce_difference)
        # print("avg_bce_diff")
        # print(avg_bce_diff.item())
        # print(avg_bce_diff.item()*100)

        cover_ssim = (1 - cover_ssim_ev.metrics['ssim'])
        secret_ssim_ev = self.default_evaluator.run([[secret, secret_original]])
        secret_ssim = (1 - secret_ssim_ev.metrics['ssim'])

        cover_loss = cover_mse + (cover_ssim*3)
        secret_loss = secret_mse + (secret_ssim*3)
        
        combined_loss = cover_loss + (self.BETA * secret_loss) + (self.BETA *avg_bce_diff)
        secret_loss_beta = self.BETA * secret_loss
        
        # print(f"cover_mse{cover_mse.item()}")
        # print(f"cover_ssim{cover_ssim}")
        # print(f"Cover Loss: {cover_loss}")
        # print(f"Secret Loss: {secret_loss}")

        # Logging calculation for consistency when testing different loss ratios.
        cover_loss_log = cover_mse + cover_ssim
        secret_loss_log = secret_mse + secret_ssim
        
        combined_loss_log = cover_loss_log  + self.BETA * secret_loss_log
        secret_loss_beta_log = (self.BETA * secret_loss_log)


        return combined_loss, cover_loss, secret_loss_beta, cover_mse, secret_mse, cover_ssim, secret_ssim, combined_loss_log, cover_loss_log, secret_loss_beta_log, avg_bce_diff


# For default evaluator in MSE_and_SSIM_and_Detector_loss
def eval_step(engine, batch):
    return batch
