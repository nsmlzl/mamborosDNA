import argparse
import os

import torch
from torch import optim, nn, utils, Tensor
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import lightning as L


# define the LightningModule
class LitAutoEncoder(L.LightningModule):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = nn.functional.mse_loss(x_hat, x)
        # Logging to TensorBoard (if installed) by default
        #self.log("train_loss", loss)
        print(f"device {self.device} ({torch.cuda.get_device_name()}) global_rank {self.global_rank}: loss {loss.item()}")
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


def main():
    # define any number of nn.Modules (or use your current ones)
    encoder = nn.Sequential(nn.Linear(28 * 28, 64), nn.ReLU(), nn.Linear(64, 3))
    decoder = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 28 * 28))

    # init the autoencoder
    autoencoder = LitAutoEncoder(encoder, decoder)

    # setup data
    dataset = MNIST(os.getcwd(), download=True, transform=ToTensor())
    train_loader = utils.data.DataLoader(dataset)

    # train the model
    trainer = L.Trainer(limit_train_batches=3, max_epochs=2,
                        accelerator="gpu", num_nodes=2, devices=8, strategy="ddp")
    trainer.fit(model=autoencoder, train_dataloaders=train_loader)


if __name__ == "__main__":
    num_gpus_per_node = torch.cuda.device_count()
    print("num_gpus_per_node = " + str(num_gpus_per_node), flush=True)

    main()
