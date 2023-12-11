import glob
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision
import lightning
from tqdm import tqdm
from PIL import Image

from modules.spherical_encoder_decoder import EncoderDecoder


class EmbeddingLoss(nn.Module):
    """
    """
    def __init__(self, embedding_dim: int, delta=1):
        """
        """
        super().__init__()
        self.huber_loss = nn.HuberLoss(delta=delta, reduction='mean')
        self.thetaw = torch.sin(torch.linspace(0, np.pi, embedding_dim)).view(1, 1, -1, 1)
        self.thetaw = nn.Parameter(self.thetaw, requires_grad=False)

    def forward(self, pred, target, use_thetaw=True):
        if use_thetaw:
            pred   = self.thetaw * pred
            target = self.thetaw * target
        #return self.huber_loss(pred, target)
        return nn.functional.mse_loss(pred, target)
    

class EncoderDecoderDataset(torch.utils.data.Dataset):
    """
    """
    def __init__(self, path_images: Path | str, path_features: Path | str, nimages: int = 64, partition='train'):
        """
        """
        npartition = int(0.75 * nimages)
        self.n = npartition if partition == 'train' else (nimages - npartition)

        transforms_image = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.RandomHorizontalFlip(p=0.5),
            torchvision.transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        ])
        self.nimages = nimages
        self.filenames_images_unpartitioned = sorted(glob.glob(f'{path_images  }/*.png'))
        self.filenames_features             = sorted(glob.glob(f'{path_features}/*.npy'))[:10]
        self.filenames_images               = []
        for i in range(len(self.filenames_features)):
            index = i * nimages
            self.filenames_images.extend(
                self.filenames_images_unpartitioned[index:index + npartition] if partition == 'train' else \
                self.filenames_images_unpartitioned[index + npartition:index + nimages]
            )
        self.images   = []
        self.features = []
        for i, filename in tqdm(enumerate(self.filenames_features)):
            self.features.append(torch.from_numpy(np.load(filename)))
            for j in range(self.n):
                self.images.append(transforms_image(
                    Image.open(self.filenames_images[i * self.n + j]).convert('L')
                ))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return self.images[index], self.features[index // self.n]


class LitEncoderDecoder(lightning.LightningModule):
    """
    """
    def __init__(self):
        """
        """
        super().__init__()
        self.encoder_decoder = EncoderDecoder()
        self.loss = EmbeddingLoss(embedding_dim=32)

    def training_step(self, batch, batch_index):
        x, y = batch
        pred = self.encoder_decoder(x)
        loss = self.loss(pred, y, use_thetaw=False)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_index):
        x, y = batch
        loss = self.loss(self.encoder_decoder(x), y, use_thetaw=False)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        return {
            'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'val_loss'
        }
    

from mayavi import mlab
from renderer import generate_equiangular_rays
from PIL import Image

def plot_mayavi(f, n: int):
    directions = generate_equiangular_rays(n)
    mlab.mesh(
        directions[:, :, 0], 
        directions[:, :, 1], 
        directions[:, :, 2], scalars=f, colormap='coolwarm'
    )
    mlab.show()


def train():
    """
    """
    autoencoder = LitEncoderDecoder()
    dataset = EncoderDecoderDataset('/home/gtangg12/data/scnn/images', '/home/gtangg12/data/scnn/scnn_features')
    ntrain = int(0.8 * len(dataset))
    train_dataset, validation_dataset = torch.utils.data.random_split(dataset, [
        ntrain, len(dataset) - ntrain
    ])
    train_loader, validation_loader = \
        torch.utils.data.DataLoader(train_dataset     , batch_size=16, shuffle=True),\
        torch.utils.data.DataLoader(validation_dataset, batch_size=16, shuffle=False)
    '''
    for i, (x, y) in enumerate(dataset):
        print(x.shape, y.shape)
        x *= 255
        Image.fromarray(x.squeeze().numpy()).show()
        plot_mayavi(y[0].squeeze().numpy(), 32)
        print(i)
        if i > 50:
            break
    exit()
    '''
    trainer = lightning.Trainer(
        devices=1, accelerator='gpu', max_epochs=256, default_root_dir='/home/gtangg12/mit/6.860/outputs/autoencoder'
    )
    trainer.fit(model=autoencoder, train_dataloaders=train_loader, val_dataloaders=validation_loader)


def predict(partition='train'):
    """
    """
    autoencoder = LitEncoderDecoder.load_from_checkpoint('/home/gtangg12/mit/6.860/outputs/autoencoder/lightning_logs/version_0/checkpoints/epoch=31-step=768.ckpt').to('cuda')
    autoencoder.eval()
    dataset = EncoderDecoderDataset('/home/gtangg12/data/scnn/images', '/home/gtangg12/data/scnn/scnn_features', partition=partition)
    loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False)

    print('Predicting features using trained autoencoder...')
    
    os.makedirs(f'/home/gtangg12/data/scnn/scnn_features_autoencoder/{partition}', exist_ok=True)
    for i, batch in tqdm(enumerate(loader)):
        x, _ = batch
        x = x.to('cuda')
        with torch.no_grad():
            features = autoencoder.encoder_decoder(x)
        for j, feature in enumerate(features):
            index = i * 16 + j
            fname = Path(dataset.filenames_images[index]).name.strip('.png')
            np.save(f'/home/gtangg12/data/scnn/scnn_features_autoencoder/{partition}/{fname}.npy', feature.cpu().numpy())

    print('Done predicting features!')


if __name__ == '__main__':
    train()
    #predict('train'); predict('test')