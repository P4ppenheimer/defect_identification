# general
import os
import time

# data science
from sklearn.model_selection import train_test_split

# torch stuff
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Dataset, sampler
import torchvision.models as models
import pandas as pd
from torchvision.transforms import Resize, Scale, Normalize, RandomHorizontalFlip, ToTensor, Compose, ToPILImage

# custom functions
from helper_functions import *

# device is cuda if gpu is availabe
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SteelDataset(Dataset):
    
    def __init__(self, df, data_folder, mean, std, phase):
        self.df = df
        self.root = data_folder
        self.mean = mean
        self.std = std
        self.phase = phase
        self.transforms = get_transforms(phase, mean, std)
        self.fnames = self.df.index.tolist()

    def __getitem__(self, idx):

        image_id, mask = make_mask(idx, self.df)

        image_path = os.path.join(self.root, "train_images",  image_id)
        
        img = cv2.imread(image_path)
        
        # original transformation with albumenations, but can not install it.
        # augmented = self.transforms(image=img, mask=mask)
        
        # img = augmented['image']
        # mask = augmented['mask'] # 1x256x1600x4
        
        img = self.transforms(img)
        mask = torch.tensor(mask).permute(2,0,1) # 4x256x1600
        
        return img, mask

    def __len__(self):
        return len(self.fnames)


def get_transforms(phase, mean, std):
    list_transforms = []
    if phase == "train":
        list_transforms.extend(
            [
                # from torch
#                 RandomHorizontalFlip(p=0.5)
                
                 # from albuemnations
#                 HorizontalFlip(p=0.5), # only horizontal flip as of now
            ]
        )
    list_transforms.extend(
        [
            ToTensor(),
            Normalize(mean=mean, std=std),   
        ]
    )
    list_trfms = Compose(list_transforms)
    return list_trfms

def provider(
    data_folder,
    df_path,
    phase,
    mean=None,
    std=None,
    batch_size=8,
    num_workers=0,
):
    '''Returns dataloader for the model training'''
    
    df = pd.read_csv(df_path)
    
    # https://www.kaggle.com/amanooo/defect-detection-starter-u-net
    
    df['ClassId'] = df['ClassId'].astype(int)
    
    df = df.pivot(index='ImageId',columns='ClassId',values='EncodedPixels')
    
    df['defects'] = df.count(axis=1)
    
    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df["defects"], random_state=69)
    
    df = train_df if phase == "train" else val_df
    
    image_dataset = SteelDataset(df, data_folder, mean, std, phase)
    
    dataloader = DataLoader(
        image_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=True,   
    )

    return dataloader


class Meter:
    '''A meter to keep track of iou and dice scores throughout an epoch'''
    def __init__(self, phase, epoch):
        self.base_threshold = 0.5 # <<<<<<<<<<< here's the threshold
        self.base_dice_scores = []
        self.dice_neg_scores = []
        self.dice_pos_scores = []
        self.iou_scores = []

    def update(self, targets, outputs):
        probs = torch.sigmoid(outputs)
        dice, dice_neg, dice_pos, _, _ = metric(probs, targets, self.base_threshold)
        self.base_dice_scores.extend(dice.tolist())
        self.dice_pos_scores.extend(dice_pos.tolist())
        self.dice_neg_scores.extend(dice_neg.tolist())
        preds = predict(probs, self.base_threshold)
        iou = compute_iou_batch(preds, targets, classes=[1])
        self.iou_scores.append(iou)

    def get_metrics(self):
        dice = np.nanmean(self.base_dice_scores)
        dice_neg = np.nanmean(self.dice_neg_scores)
        dice_pos = np.nanmean(self.dice_pos_scores)
        dices = [dice, dice_neg, dice_pos]
        iou = np.nanmean(self.iou_scores)
        return dices, iou

def epoch_log(phase, epoch, epoch_loss, meter, start):
    '''logging the metrics at the end of an epoch'''
    dices, iou = meter.get_metrics()
    dice, dice_neg, dice_pos = dices
    print("Loss: %0.4f | IoU: %0.4f | dice: %0.4f | dice_neg: %0.4f | dice_pos: %0.4f" % (epoch_loss, iou, dice, dice_neg, dice_pos))
    return dice, iou

def compute_ious(pred, label, classes, ignore_index=255, only_present=True):
    '''computes iou for one ground truth mask and predicted mask'''
    pred[label == ignore_index] = 0
    ious = []
    for c in classes:
        label_c = label == c
        if only_present and np.sum(label_c) == 0:
            ious.append(np.nan)
            continue
        pred_c = pred == c
        intersection = np.logical_and(pred_c, label_c).sum()
        union = np.logical_or(pred_c, label_c).sum()
        if union != 0:
            ious.append(intersection / union)
    return ious if ious else [1]

def compute_iou_batch(outputs, labels, classes=None):
    '''computes mean iou for a batch of ground truth masks and predicted masks'''
    ious = []
    preds = np.copy(outputs) # copy is imp
    labels = np.array(labels) # tensor to np
    for pred, label in zip(preds, labels):
        ious.append(np.nanmean(compute_ious(pred, label, classes)))
    iou = np.nanmean(ious)
    return iou
    

class Trainer(object):

    '''This class takes care of training and validation of our model'''

    def __init__(self, model, state = None, batch_size = None):

        if state is not None: 
            model.load_state_dict(state['state_dict'])
        
        model.to(device)

        self.num_workers = 0
        if batch_size is None:
            self.batch_size = {"train": 16, "val": 16}
        else:
            self.batch_size = batch_size
            
        self.accumulation_steps = 32 // self.batch_size['train']
        self.lr = 5e-4

        # for testing
        self.num_epochs = 20

        self.best_loss = float("inf")
        self.phases = ["train", "val"]
        # self.device = torch.device("cuda:0")
        self.device = device
        # torch.set_default_tensor_type("torch.cuda.FloatTensor")

        self.net = model

        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode="min", patience=3, verbose=True)
        self.net = self.net.to(self.device)
        cudnn.benchmark = True

        self.dataloaders = {
            phase: provider(
                data_folder=data_folder,
                df_path=train_df_path,
                phase=phase,
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                batch_size=self.batch_size[phase],
                num_workers=self.num_workers,
            )
            for phase in self.phases
        }

        if state is None:
            self.losses = {phase: [] for phase in self.phases}
            self.iou_scores = {phase: [] for phase in self.phases}
            self.dice_scores = {phase: [] for phase in self.phases}
            self.start_epoch = 0
        else:
            self.losses = state['losses']
            self.iou_scores = state['iou_scores']
            self.dice_scores = state['dice_scores']
            self.start_epoch = state['epoch'] 

        print('\n### Trainer Initialized with ###')
        print(f'Pretrained model: {"yes" if state is not None else "no"}')
        print(f'Max Epochs: {self.num_epochs}')
        print(f'Current Epoch: {self.start_epoch}')
        print(f'Batch Size: {self.batch_size}')
        print('################################\n')
        
    def forward(self, images, targets):

        images = images.to(self.device)
        masks = targets.to(self.device)
        outputs = self.net(images)
        loss = self.criterion(outputs, masks)

        return loss, outputs

    def iterate(self, epoch, phase):

        meter = Meter(phase, epoch)
        start = time.strftime("%H:%M:%S")

        print(f"Starting epoch: {epoch} | phase: {phase} | ⏰: {start}")

        batch_size = self.batch_size[phase]
        self.net.train(phase == "train")
        
        dataloader = self.dataloaders[phase]
        running_loss = 0.0
        total_batches = len(dataloader)

        self.optimizer.zero_grad()

        for itr, batch in enumerate(dataloader): # replace `dataloader` with `tk0` for tqdm
            images, targets = batch
            loss, outputs = self.forward(images, targets)
            loss = loss / self.accumulation_steps
            if phase == "train":
                loss.backward()
                if (itr + 1 ) % self.accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            running_loss += loss.item()
            outputs = outputs.detach().cpu()
            meter.update(targets, outputs)

        epoch_loss = (running_loss * self.accumulation_steps) / total_batches
        dice, iou = epoch_log(phase, epoch, epoch_loss, meter, start)
        self.losses[phase].append(epoch_loss)
        self.dice_scores[phase].append(dice)
        self.iou_scores[phase].append(iou)
        torch.cuda.empty_cache()

        return epoch_loss

    def start(self):

        
        for epoch in range(self.start_epoch, self.num_epochs):

            self.iterate(epoch, "train")

            state = {
                "epoch": epoch,
                "best_loss": self.best_loss,
                "state_dict": self.net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                'losses' : self.losses,
                'iou_scores' : self.iou_scores,
                'dice_scores' : self.dice_scores
            }

            with torch.no_grad():
                val_loss = self.iterate(epoch, "val")
                self.scheduler.step(val_loss)
                
            if val_loss < self.best_loss:

                print("******** New optimal found, saving state ********")
                print(f'******** With name model_{ts}.pth5 ********')

                state["best_loss"] = self.best_loss = val_loss
                
                torch.save(state, unet_model_path)

            print()
            