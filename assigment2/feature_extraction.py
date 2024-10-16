import os
from tqdm import tqdm
from typing import List
import numpy as np

import torch

import torchvision.io as io
from torchvision.models.resnet import ResNet

from common import *

@torch.no_grad
def loadAndExtractFeaturesAndLabels(resnet:ResNet, rootDir: str, loadLabels: bool = False):
  featuresList: List = []
  labelsList: List = []

  for root, subdirs, files in tqdm(os.walk(rootDir), desc = "Extracing Features"):
    for file in files:
      fileName, fileExt = os.path.splitext(file)
      if (fileExt != ".png"): continue
      
      img = io.read_image(os.path.join(root, file), io.ImageReadMode.RGB)
      img = prepImageforResnet(img)
      imgFeatures = resnet(img)
      
      featuresList.append(imgFeatures)
      
      if loadLabels:
        fileNum = fileName.split("-")[0]
        with open(os.path.join(root, f"{fileNum}-box.txt"), "r") as labelFile:
          imgLabels = torch.zeros(len(CLASS_NAME_TO_ID_MAP))
          for line in labelFile:
            objName, *boundingBox = line.split()
            imgLabels[CLASS_NAME_TO_ID_MAP[objName]] = 1
            
          labelsList.append(imgLabels.unsqueeze(0))
          
  if loadLabels:
    return torch.cat(featuresList).cpu().numpy(), torch.cat(labelsList).numpy()
  else:
    return torch.cat(featuresList).cpu().numpy()
      
if __name__ == "__main__":
  print("Loading ResNet")
  resNetNoFC = createResNetNoFCLayer()
  
  features, labels = loadAndExtractFeaturesAndLabels(resNetNoFC, TRAIN_DATSET_DIR, loadLabels = True)
  print(f"Loaded the features and labels for {features.shape[0]} images")
  
  np.savez(FEATURES_OUTPUT_FILE, features= features, labels = labels)
  print(f"Saved features and labels to {FEATURES_OUTPUT_FILE}")
  