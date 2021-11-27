import os
import shutil

import os
from glob import glob
import pandas as pd
import random
from collections import defaultdict
from PIL import Image
from torch.utils.data import Dataset, DataLoader

def get_all_images(dir):
  types = ["jpeg", "jpg", "png"]
  files = []
  for t in types:
    path = os.path.join(dir, "**", "*." + t)
    files.extend(glob(path))
      
  return files


def casia(dir):
  files = get_all_images(dir)
  users = defaultdict(set)
  rows = []

  for file in files:
    user = file.split("/")[-2]
    users[user].add(file)
    rows.append({
        "image": file,
        "id": user
    })

  df = pd.DataFrame(rows)
  positives = []
  negatives = []

  for user, files in users.items():
    if len(files) <= 1:
      continue
    
    samples = random.sample(files, 2)
    positives.append({
        "image1": samples[0],
        "image2": samples[1],
        "id1": user,
        "id2": user,
        "label": 1
    })
  
  user_ids = list(users.keys())
  for i in range(0, len(user_ids), 2):
    if i == len(user_ids) - 1:
      continue

    id1, id2 = user_ids[i], user_ids[i + 1]
    files1, files2 = users[id1], users[id2]

    if len(files1) < 2 or len(files2) < 2:
      break
    
    samples1, samples2 = random.sample(files1, 2), random.sample(files2, 2)
    for j in range(2):
      negatives.append({
          "image1": samples1[j],
          "image2": samples2[j],
          "id1": id1,
          "id2": id2,
          "label": -1
      })
  
  test_set = pd.DataFrame(positives + negatives)
  return df, test_set

# trainset, testset = casia("train/")
# trainset.to_csv("train.csv", index=False)
# testset.to_csv("train_eval.csv", index=False)

for file in glob("dataset/validation/**/*.png", recursive=True):
  tokens = file.split("/")
  filename = tokens[-1]
  id = tokens[-3]

  dst = f"mfeval/{id}/{filename}"
  os.makedirs(os.path.abspath(os.path.dirname(dst)), exist_ok=True)
  shutil.copyfile(file, dst)