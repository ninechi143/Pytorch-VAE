import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


from VAE_dataset import train_dataset , normalize
from VAE_model import Encoder , Decoder
from VAE_loss import Reconstruction_Loss , VAE_ELBO_Loss

from time import perf_counter

class VAE_trainer():

    def __init__(self,args):

        self.lr = args.lr
        self.batch_size = args.batch_size
        self.epochs = args.epochs
        self.optim = args.optimizer
        self.Lambda = args.Lambda
        self.normalize = args.normalize
        self.resume = args.resume
        self.use_log = args.log

        self.start_epoch = 0

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"[!] torch version: {torch.__version__}")
        print(f"[!] computation device: {self.device}")


    def load_data(self):

        print("[!] Data Loading...")


        self.train_dataset = train_dataset()
        self.data_statistics = self.train_dataset.get_statistics()


        transforms_list = [torchvision.transforms.Resize(64)]
        if self.normalize: 
            transforms_list += [normalize(self.data_statistics[0] , self.data_statistics[1])]

        self.train_dataset.set_transforms(
                            torchvision.transforms.Compose(transforms_list)
                            )
        

        self.train_loader = DataLoader(dataset = self.train_dataset,
                                       batch_size = self.batch_size,
                                       shuffle = True,
                                       num_workers = 1)

        print("[!] Data Loading Done.")


    def setup(self):

        print("[!] Setup...")

        self.data_for_log = torch.randn(64 , 128).to(self.device)
        self.log_writer = SummaryWriter('logs') if self.use_log else None
        
        # define our model, loss function, and optimizer
        self.Encoder = Encoder().to(self.device)
        self.Decoder = Decoder().to(self.device)

        parameter_list = list(self.Encoder.parameters()) + list(self.Decoder.parameters())

        if self.optim.lower() == "adam":
            self.optimizer = torch.optim.Adam(parameter_list, lr=self.lr)
        else:
            self.optimizer = torch.optim.RMSprop(parameter_list, lr = self.lr)


        self.reconstruction_loss = Reconstruction_Loss().to(self.device)
        self.elbo_loss = VAE_ELBO_Loss(self.Lambda).to(self.device)


        # load checkpoint file to resume training
        if self.resume:
            print(f"[!] Resume training from the file : {self.resume}")
            checkpoint = torch.load(self.resume)
            self.Encoder.load_state_dict(checkpoint['model_state'][0])
            self.Decoder.load_state_dict(checkpoint["model_state"][1])
            try:
                self.start_epoch = checkpoint['epoch']
            except:
                pass

        print("[!] Setup Done.")


    def train(self):

        print("[!] Model training...")
        avg_time = 0
        n_total_steps = len(self.train_loader)

        for epoch in range(self.epochs):

            st = perf_counter()
            total_loss = 0
            running_loss = 0

            self.Encoder.train()
            self.Decoder.train()
            for i , data in enumerate(self.train_loader):

                # access data and noise
                real_images = data.to(self.device)
                noise = torch.randn(real_images.shape[0] , 128).to(self.device)

                # feedforward
                mean , log_var_square = self.Encoder(real_images)
                # reparameterization trick
                latent_code = torch.multiply(torch.sqrt(torch.exp(log_var_square)) , noise) + mean
                reconstruction = self.Decoder(latent_code)

                # compute loss
                reconstruct_loss = self.reconstruction_loss(real_images , reconstruction)
                elbo_loss = self.elbo_loss(mean , log_var_square)
                loss = reconstruct_loss + elbo_loss

                # updates
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item() / n_total_steps
                running_loss += loss.item() / n_total_steps

                # tensorboard: track training process
                if (i+1) % 50 == 0:
                    with torch.no_grad():
                        self.Decoder.eval()

                        image_log = self.Decoder(self.data_for_log)
##                        image_log = (image_log * self.data_statistics[1] + self.data_statistics[0]) if self.normalize else image_log
                        image_grid = torchvision.utils.make_grid(image_log , nrow = 8 , normalize = True) # if normalize = True, then we don't need to manually normalize the image to value 0~1
                                                                                                          # otherwise, we need to manually normalize it by using image_log = image_log * 0.5 + 0.5

                        print(f"[!] Epoch : [{epoch+1}], step : [{i+1} / {n_total_steps}], Running Loss: {running_loss:.6f}")
                        running_loss = 0
                        self.log_writer.add_image("VAE Generated Image" , image_grid , epoch * n_total_steps + i + 1)
                        self.Decoder.train()

            print("-------------------------------------------")
            avg_time = avg_time + (perf_counter() - st - avg_time) / (epoch+1)
            print(f"[!] Epoch : [{epoch+1}/{self.epochs}] done. Average Training Time: {avg_time:.3f}, "
                  f"Loss: {total_loss:.6f}\n") 
            if self.use_log:
                self.log_writer.add_scalar('training loss', total_loss, epoch)

        if self.use_log:
            self.log_writer.close()

        print("[!] Training Done.\n")

    
    def save(self):

        print("[!] Model saving...")
        checkpoint = {"model_state": [self.Encoder.state_dict() , self.Decoder.state_dict()]}
        torch.save(checkpoint , "checkpoint.pth")
        print("[!] Saving Done.")
