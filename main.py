import torch
import os
import argparse

from trainer import VAE_trainer


def parse_args():

    parser = argparse.ArgumentParser(description='WGAN_GP trainer')

    parser.add_argument("-l" , "--lr" , type = float , default=1e-5)
    parser.add_argument('--batch_size' , type=int , default=64)
    parser.add_argument('--epochs' , type=int , default=200)
    parser.add_argument('--optimizer' , type=str , default='adam')
    parser.add_argument('--Lambda' , type=float , default=1., help = "the coeffient of the KL term in Loss function")
    parser.add_argument("--normalize" , action="store_true" , default=True , help = "True if want to normalize all data.")
    parser.add_argument("--resume" , type=str , default = None , help = "if you want to continue training, setup the .pth file name.")
    parser.add_argument("--log" , action="store_true" , default=True , help = "True if you want to use Tensorboard for the logging.")

    args = parser.parse_args()

    return args


if __name__ == "__main__":

    args = parse_args()
    # print(args.batch_size)
    # print(args.normalize)


    trainer = VAE_trainer(args)

    trainer.load_data()  # prepare dataset and dataLoader
    trainer.setup()      # define our model, loss function, and optimizer
    trainer.train()      # define training pipeline and execute the training process
    trainer.save()       # save model after training

    print("\nDone.\n")
