import torch

from torch.utils.data import Dataset
import torchvision

import random
import numpy as np




class train_dataset(Dataset):

    def __init__(self):
        # we use MNIST as our example
        mnist = torchvision.datasets.MNIST(root = "./data" , 
                                                    train = True,
                                                    transform=None,
                                                    download=True)

        train_x = mnist.data.numpy() / 255.0    # 60000,28,28

        # self.mean , self.std = train_x.mean() , train_x.std()
        self.mean , self.std = 0.5 , 0.5
    
        self.data = torch.from_numpy(
                        np.expand_dims(train_x , axis = 1)).float()

        self.n_samples = self.data.shape[0]    

        self.transforms = None

    def get_statistics(self):
        return self.mean , self.std


    def set_transforms(self , transforms = None):    
        if transforms:
            self.transforms = transforms
    


    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):

        if self.transforms:
            inputs = self.transforms(self.data[index])
        else:
            assert False ,"transforms of dataset ERROR, please check the codes."
            # inputs = self.data[index]

        return inputs



    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples





# simply normalize
class normalize():

    def __init__(self, mean , std):
        
        self.mean = mean
        self.std = std

    def __call__(self, inputs):

        return (inputs - self.mean) / (self.std + 1e-8)






if __name__ == "__main__":


    a = torchvision.datasets.MNIST(root = "./data" , 
                                    train = True,
                                    transform=None,
                                    download=True)

    b = torchvision.datasets.MNIST(root = "./data" , 
                                    train = False,
                                    transform=None)


    ax = a.data.numpy()
    ay = a.targets.numpy()

    print(type(ax) , type(ay))     
    print(ax.shape)
    print(ay.shape)


    # print(ay[5])   # not yet one-hot coding, so it is just a single number
    # print(ax[5])   # not yet normalizer, so the range is from 0 to 255


    # D = train_dataset()
    # print(len(D))