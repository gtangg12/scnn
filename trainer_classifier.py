import glob
import os
from pathlib import Path
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS

import numpy as np
import torch
import torch.nn as nn
import lightning
from tqdm import tqdm

from modules.spherical_classifier import SphericalClassifer
from renderer import ModelNet40_LABELS2INDEX


class ClassifierDataset(torch.utils.data.Dataset):
    """
    """
    def __init__(self, path: Path | str, label_filter: str = None):
        """
        """
        self.filenames = sorted(glob.glob(f'{path}/*.npy'))
        self.filenames = list(filter(lambda x: Path(x).stem.rpartition('_')[0] == label_filter, self.filenames))
        self.x = []
        self.y = []
        for filename in tqdm(self.filenames):
            label = Path(filename).stem.rpartition('_')[0]
            self.x.append(torch.from_numpy(np.load(filename)))
            self.y.append(label)
        self.x = torch.stack(self.x)
        self.y = torch.tensor([ModelNet40_LABELS2INDEX[label] for label in self.y])

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index].unsqueeze(0), self.y[index]


class LitClassifier(lightning.LightningModule):
    """
    """
    def __init__(self, bandwidth: int, num_classes: int):
        """
        """
        super().__init__()
        self.classifier = SphericalClassifer(bandwidth, num_classes)

    def training_step(self, batch, batch_index):
        x, y = batch
        pred = self.classifier(x)
        #print(pred.argmax(dim=1))
        #print(y)
        loss = nn.functional.cross_entropy(pred, y)
        self.log('train_loss', loss.item())
        self.log('train_acc', (pred.argmax(dim=1) == y).float().mean())
        return loss
    
    def validation_step(self, batch, batch_index):
        x, y = batch
        pred = self.classifier(x)
        loss = nn.functional.cross_entropy(pred, y)
        self.log('val_loss', loss.item())
        self.log('val_acc', (pred.argmax(dim=1) == y).float().mean())
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        return {
            'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'val_loss'
        }
    

def train():
    """
    """
    classifier = LitClassifier(bandwidth=32, num_classes=40)
    
    dataset = ClassifierDataset('/home/gtangg12/data/scnn/s2')
    ntrain = int(0.8 * len(dataset))
    train_dataset, validation_dataset = torch.utils.data.random_split(dataset, [
        ntrain, len(dataset) - ntrain
    ])
    train_loader, validation_loader = \
        torch.utils.data.DataLoader(train_dataset     , num_workers=4, batch_size=16, shuffle=True),\
        torch.utils.data.DataLoader(validation_dataset, num_workers=4, batch_size=16, shuffle=False)
    
    trainer = lightning.Trainer(
        devices=1, accelerator='gpu', max_epochs=48, default_root_dir='/home/gtangg12/mit/6.860/outputs/scnn/',
    )
    trainer.fit(model=classifier, train_dataloaders=train_loader, val_dataloaders=validation_loader)


def extract_scnn_features():
    """
    """
    classifier = LitClassifier.load_from_checkpoint(
        '/home/gtangg12/mit/6.860/outputs/scnn/lightning_logs/version_2/checkpoints/epoch=31-step=11744.ckpt',
        bandwidth=32, num_classes=40,
    ).to('cuda')
    classifier.eval()
    dataset = ClassifierDataset('/home/gtangg12/data/scnn/s2/', label_filter='car')
    loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False)

    print('Extracting features...')

    os.makedirs('/home/gtangg12/data/scnn/scnn_features', exist_ok=True)
    for i, batch in tqdm(enumerate(loader)):
        x, _ = batch
        x = x.to('cuda')
        with torch.no_grad():
            features = classifier.classifier(x, return_features=True)
        for j, feature in enumerate(features):
            index = i * 16 + j
            fname = Path(dataset.filenames[index]).name.strip('.npy')
            np.save(f'/home/gtangg12/data/scnn/scnn_features/{fname}.npy', feature.cpu().numpy())

    print('Done extracing features!')


if __name__ == '__main__':
    #train()
    extract_scnn_features()