import torch
import torch.nn as nn
import torch.nn.functional as F


class Reconstruction_Loss(nn.Module):

    def __init__(self):

        super(Reconstruction_Loss, self).__init__()

    def forward(self, real_images, decode_images):

        loss = torch.mean(
                    torch.sum(
                        torch.square(real_images - decode_images) , dim = (1,2,3)
                        )
                    )

        return loss


# KL term in VAE
class VAE_ELBO_Loss(nn.Module):
    
    def __init__(self , Lambda):

        super(VAE_ELBO_Loss, self).__init__()
        self.Lambda = Lambda

    def forward(self, mean , log_var_square):

        loss =  torch.mean(
                    torch.sum(
                        torch.square(mean) + torch.exp(log_var_square) - (1 + log_var_square) , dim = 1
                        )
                    )
        return self.Lambda * loss

