import os
import glob
import pandas as pd
from PIL import Image


class FaceRecognitionDataset(Dataset):
  def __init__(self, csvfile="train.csv", augmented=False):
    if augmented:
      self.transforms = [transforms.Resize((128,128)),
                         transforms.RandomHorizontalFlip(p=0.5),
                         transforms.ToTensor(),
                         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])]
    else:
      self.transforms = [transforms.Resize((128,128)),
                         transforms.ToTensor(),
                         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])]
    self.transforms = transforms.Compose(self.transforms)

    self.df = pd.read_csv(csvfile)
    self.labels = self.df.id.unique().tolist()
    self.labels.sort()
    self.label2id = {label: id for id, label in enumerate(self.labels)}

  def __len__(self):
    return len(self.df)

  def __getitem__(self, index):
    item = self.df.iloc[index]

    return {
        "file": item['image'],
        "image": self.transforms(Image.open(item['image']).convert("RGB")),
        "id": self.label2id[item.id],
        "label": item.id
    }

  def get_image(self, index):
    image = Image.open(self.df.iloc[index].path)
    return self.transforms(image)

  @property
  def num_classes(self):
    return len(self.labels)

  
class FaceEvalDataset(Dataset):
  def __init__(self, csv_name):
    self.transforms = transforms.Compose(
        [transforms.Resize((128,128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])])
    self.df = pd.read_csv(csv_name)
      
  def __len__(self):
    return len(self.df)

  def __getitem__(self, index):
    item = self.df.iloc[index]
    
    return {
      "image1": self.transforms(Image.open(item["image1"]).convert("RGB")),
      "image2": self.transforms(Image.open(item["image2"]).convert("RGB")),
      "id1": item["id1"],
      "id2": item["id2"],
      "label": item["label"]
    }
    