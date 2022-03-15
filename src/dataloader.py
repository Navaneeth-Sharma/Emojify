import torch
import cv2
import pandas as pd
from torch.utils.data import Dataset, DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"

trn_df = pd.read_csv('data/emotionTrainData.csv')


class EmotionDataset(Dataset):
    def __init__(self, df, transform=None, IMAGE_SIZE=48):
        self.df = df
        self.label_dict = self.labelize(df)
        self.IMAGE_SIZE = IMAGE_SIZE
    
    def __len__(self):
        return len(self.df)    
    
    def __getitem__(self, idx):
        f = self.df.iloc[idx].squeeze()
        file = f.file
        label = self.label_dict[f.label]

        img = cv2.imread(file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img, label

    def preprocess_image(self, img):
        im = cv2.resize(img, (self.IMAGE_SIZE, self.IMAGE_SIZE))
        im = torch.tensor(im)
        im = im/255.
        return im[None]

    def collate_fn(self, batch):
        '''
        preprocess images, emotion labels
        '''
        imgs, labels = [], []
        for im, label in batch:

            im = self.preprocess_image(im)
            imgs.append(im)
            
            labels.append(float(label))
        
        labels = torch.tensor(labels).to(device).float()
        imgs = torch.cat(imgs).to(device)
        imgs = imgs.view(-1,1,48,48)
        
        return imgs, labels

    def labelize(self, df):
        label_dict = {}
        for i,name in enumerate(df.label.unique()):
            label_dict[name] = i
        return label_dict


trn = EmotionDataset(trn_df)
TRAIN_LOADER = DataLoader(trn, batch_size=32, shuffle=True, \
                          drop_last=True, collate_fn=trn.collate_fn)
