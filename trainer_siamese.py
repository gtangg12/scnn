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

from modules.spherical_encoder_decoder import ResidualEncoder
from trainer_encoder_classifier import EncoderClassifierDataset


class SiameseNetwork(nn.Module):
    """
    """
    def __init__(self):
        super().__init__()
        self.resnet = torchvision.models.resnet18(weights=None)

        # over-write the first conv layer to be able to read MNIST images
        # as resnet18 reads (3,x,x) where 3 is RGB channels
        # whereas MNIST has (1,x,x) where 1 is a gray-scale channel
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.fc_in_features = self.resnet.fc.in_features
        
        # remove the last layer of resnet18 (linear layer which is before avgpool layer)
        self.resnet = torch.nn.Sequential(*(list(self.resnet.children())[:-1]))

        # add linear layers to compare between the features of the two images
        self.fc = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 64),
        )

        self.sigmoid = nn.Sigmoid()

        # initialize the weights
        self.resnet.apply(self.init_weights)
        self.fc.apply(self.init_weights)

        self.net = nn.Sequential(
            self.resnet,
            nn.Flatten(),
            self.fc,
            self.sigmoid,
        )
        
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def forward(self, anchor, positive, negative):
        a = self.net(anchor)
        p = self.net(positive)
        n = self.net(negative)
        return a, p, n

    def get_embedding(self, x):
        return self.resnet(x)
    

class TripletLoss(nn.Module):
    """
    """
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin
        self.pairwise_distance = nn.CosineSimilarity(dim=-1)

    def forward(self, anchor, positive, negative):
        positive_distance = self.pairwise_distance(anchor, positive)
        negative_distance = self.pairwise_distance(anchor, negative)
        losses = nn.functional.relu(positive_distance - negative_distance + self.margin)
        return losses.mean()
    

class TripletDataset(EncoderClassifierDataset):
    """
    """
    def __init__(self, path_images: Path | str, path_features: Path | str, nimages: int = 64, partition='train'):
        """
        """
        npartition = 64
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
        self.images = []
        self.labels = []
        for i, _ in tqdm(enumerate(self.filenames_features)):
            for j in range(self.n):
                self.labels.append(i)
                self.images.append(transforms_image(
                    Image.open(self.filenames_images[i * self.n + j]).convert('L')
                ))
        self.images = torch.stack (self.images)
        self.labels = torch.tensor(self.labels)
        self.label_to_indices = {label.item(): torch.where(self.labels == label)[0] for label in torch.unique(self.labels)}


    def __getitem__(self, index):
        anchor = self.images[index]
        anchor_label = self.labels[index].item()
        
        positive_index = index
        while positive_index == index:
            positive_index = np.random.choice(self.label_to_indices[anchor_label])
        positive = self.images[positive_index]

        negative_label = np.random.choice(list(set(self.labels) - set([anchor_label])))
        negative_index = np.random.choice(self.label_to_indices[negative_label])
        negative = self.images[negative_index]

        return anchor, positive, negative
    

class LitSiamese(lightning.LightningModule):
    """
    """
    def __init__(self):
        """
        """
        super().__init__()
        self.siamese = SiameseNetwork()
        self.loss = TripletLoss()

    def training_step(self, batch, batch_index):
        a, p, n = batch
        pred = self.siamese(a, p, n)
        loss = self.loss(*pred)
        self.log('train_loss', loss.item())
        return loss
    
    def validation_step(self, batch, batch_index):
        a, p, n = batch
        pred = self.siamese(a, p, n)
        loss = self.loss(*pred)
        self.log('val_loss', loss.item())
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=2e-4, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        return {
            'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'val_loss'
        }


def train():
    """
    """
    dataset = TripletDataset('/home/gtangg12/data/scnn/images', '/home/gtangg12/data/scnn/scnn_features')
    ntrain = int(0.8 * len(dataset))
    train_dataset, validation_dataset = torch.utils.data.random_split(dataset, [
        ntrain, len(dataset) - ntrain
    ])
    train_loader, validation_loader = \
        torch.utils.data.DataLoader(train_dataset     , batch_size=16, shuffle=True),\
        torch.utils.data.DataLoader(validation_dataset, batch_size=16, shuffle=False)

    siamese = LitSiamese()

    trainer = lightning.Trainer(
        devices=1, accelerator='gpu', max_epochs=64, default_root_dir='/home/gtangg12/mit/6.860/outputs/siamese'
    )
    trainer.fit(model=siamese, train_dataloaders=train_loader, val_dataloaders=validation_loader)


def predict(path: Path | str, path_reference: Path | str):
    """
    """
    image_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])
    images = [Image.open(filename) for filename in glob.glob(f'{path}/*.png')]
    images = list(map(image_transform, images))
    images = torch.stack(images)
    siamese = LitSiamese.load_from_checkpoint(
        '/home/gtangg12/mit/6.860/outputs/scnn/lightning_logs/version_2/checkpoints/epoch=31-step=11744.ckpt',
    ).to('cuda')
    siamese.eval()
    with torch.no_grad():
        embeddings = siamese.siamese.get_embedding(images.to('cuda'))
    
    images_reference = [Image.open(filename) for filename in glob.glob(f'{path_reference}/*.png')]
    images_reference = list(map(image_transform, images_reference))
    images_reference = torch.stack(images_reference)
    bs = 32
    embeddings_reference = []
    for i in range(len(images) // bs):
        embeddings_reference.append(siamese.siamese.get_embedding(images_reference[i * bs:(i + 1) * bs].to('cuda')))
    embeddings_reference = torch.cat(embeddings_reference)

    scores = torch.einsum('bd,cd->bc', embeddings, embeddings_reference)
    indices = torch.argmax(scores, dim=1)
    for i, index in enumerate(indices):
        print(index)
        images[i].save(f'{path}/predicted/{i:04d}.png')
        images_reference[index].save(f'{path}/predicted/{i:04d}_reference.png')


if __name__ == '__main__':
    train()