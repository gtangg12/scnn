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

from modules.spherical_encoder_decoder import EncoderClassifier
    

class EncoderClassifierDataset(torch.utils.data.Dataset):
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
        self.filenames_features             = sorted(glob.glob(f'{path_features}/*.npy'))
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
            self.labels.append(i)
            for j in range(self.n):
                self.images.append(transforms_image(
                    Image.open(self.filenames_images[i * self.n + j]).convert('L')
                ))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return self.images[index], self.labels[index // self.n]


class LitEncoderClassifier(lightning.LightningModule):
    """
    """
    def __init__(self, num_classes: int):
        """
        """
        super().__init__()
        self.encoder_classifier = EncoderClassifier(num_classes)
        self.loss = nn.CrossEntropyLoss()

    def training_step(self, batch, batch_index):
        x, y = batch
        pred = self.encoder_classifier(x)
        loss = self.loss(pred, y)
        print(x.shape, y, pred.argmax(dim=1), loss.item())
        self.log('train_loss', loss.item())
        self.log('train_acc', (pred.argmax(dim=1) == y).float().mean())
        self.log('train_acc_top5', (torch.topk(pred, 5, dim=1).indices == y.unsqueeze(1)).any(dim=1).float().mean())
        return loss
    
    def validation_step(self, batch, batch_index):
        x, y = batch
        pred = self.encoder_classifier(x)
        loss = self.loss(pred, y)
        self.log('val_loss', loss.item())
        self.log('val_acc', (pred.argmax(dim=1) == y).float().mean())
        self.log('val_acc_top5', (torch.topk(pred, 5, dim=1).indices == y.unsqueeze(1)).any(dim=1).float().mean())
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        return {
            'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'val_loss'
        }


def train():
    """
    """
    dataset = EncoderClassifierDataset('/home/gtangg12/data/scnn/images', '/home/gtangg12/data/scnn/scnn_features')
    ntrain = int(0.8 * len(dataset))
    train_dataset, validation_dataset = torch.utils.data.random_split(dataset, [
        ntrain, len(dataset) - ntrain
    ])
    train_loader, validation_loader = \
        torch.utils.data.DataLoader(train_dataset     , batch_size=16, shuffle=True),\
        torch.utils.data.DataLoader(validation_dataset, batch_size=16, shuffle=False)

    classifier = LitEncoderClassifier(num_classes=len(dataset.filenames_features))

    trainer = lightning.Trainer(
        devices=1, accelerator='gpu', max_epochs=64, default_root_dir='/home/gtangg12/mit/6.860/outputs/encoder_classifier'
    )
    trainer.fit(model=classifier, train_dataloaders=train_loader, val_dataloaders=validation_loader)


def predict():
    """
    """
    image_transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])
    '''
    tensor([[1.0000, 0.6274, 0.0904],
        [0.6274, 1.0000, 0.0734],
        [0.0904, 0.0734, 1.0000]], device='cuda:0', grad_fn=<MmBackward0>)

    tensor([[1.0000, 0.1597, 0.4825],
        [0.1597, 1.0000, 0.0734],
        [0.4825, 0.0734, 1.0000]], device='cuda:0', grad_fn=<MmBackward0>)
    '''
    #a = '/home/gtangg12/data/scnn/images/car_0001_0000.png'
    a = '/home/gtangg12/data/scnn/images_test/car_0207_0026.png'
    p = '/home/gtangg12/data/scnn/images/car_0001_0050.png'
    n = '/home/gtangg12/data/scnn/images/car_0005_0005.png'
    a = image_transforms(Image.open(a).convert('L'))
    p = image_transforms(Image.open(p).convert('L'))
    n = image_transforms(Image.open(n).convert('L'))
    inp = torch.stack([a, p, n])

    # cars
    classifier = LitEncoderClassifier.load_from_checkpoint(
        '/home/gtangg12/mit/6.860/outputs/encoder_classifier/lightning_logs/version_1/checkpoints/epoch=63-step=40384.ckpt',
        num_classes=197,
    ).to('cuda')
    classifier.eval()
    '''
    inp = inp.to('cuda')
    embeddings = classifier.encoder_classifier.encoder(inp)
    embeddings = embeddings / torch.norm(embeddings, dim=1, keepdim=True)
    scores = embeddings @ embeddings.T
    print(scores)
    '''
    from trainer_siamese import TripletDataset
    dataset = TripletDataset('/home/gtangg12/data/scnn/images', '/home/gtangg12/data/scnn/scnn_features')
    loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False)
    
    pcnt = 0
    ncnt = 0
    for batch in tqdm(loader):
        a, p, n = batch
        af = classifier.encoder_classifier.encoder(a.to('cuda'))
        pf = classifier.encoder_classifier.encoder(p.to('cuda'))
        nf = classifier.encoder_classifier.encoder(n.to('cuda'))
        af = af / torch.norm(af, dim=1, keepdim=True)
        pf = pf / torch.norm(pf, dim=1, keepdim=True)
        nf = nf / torch.norm(nf, dim=1, keepdim=True)
        pscores = torch.sum(af * pf, dim=1)
        nscores = torch.sum(af * nf, dim=1)
        pcnt += torch.sum(pscores >  nscores).item()
        ncnt += torch.sum(pscores <= nscores).item()
    print(pcnt, ncnt) # 533 107, 0.8328125



if __name__ == '__main__':
    #train()
    predict()