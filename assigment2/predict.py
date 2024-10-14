import csv
import numpy as np
import os
from tqdm import tqdm
from typing import List, Tuple

import torchvision.io as io
from torchvision.models.resnet import ResNet

from common import *

@torch.no_grad
def predict(predDir: str, model: ResNet):
  
  predList: List = [] 
  for videoId in tqdm(os.listdir(predDir), desc = "Extracing Features"):    
    videoDir = os.path.join(predDir, videoId)
    if not os.path.isdir(videoDir): continue
    
    for imgId in os.listdir(videoDir):
      imgFile = os.path.join(videoDir, imgId)
      
      img = io.read_image(imgFile, io.ImageReadMode.RGB)
      img = prepImageforResnet(img)
      yPred = model(img)
      predLabels = torch.round(yPred).int()
      
      imgName = imgId.split("-")[0]
      predList.append((f"{videoId}_{imgName}", predLabels.squeeze().cpu().numpy()))

  return predList

def savePredictions(outFile: str, predictions: List[Tuple[str, np.ndarray]]):
  with open(outFile, "w") as file:
    file.write("img_id,class_0,class_1,class_2,class_3,class_4,class_5,class_6,class_7,class_8,class_9,class_10,class_11,class_12,class_13,class_14,class_15,class_16,class_17,class_18,class_19,class_20\n")
    
    for name, labels in predictions:
      file.write(f"{name},")
      for idx, label in enumerate(labels):
        file.write(str(label))
        if (idx < labels.shape[0] - 1):
          file.write(",")
      file.write("\n")
          
if __name__ == "__main__":
  print("Loading Data")
  weights = np.load(MODEL_WEIGHTS_FILE)
  
  resnet: ResNet = createResNetCustomFCLayer(torch.tensor(weights["intercepts"], dtype = torch.float32).squeeze(), torch.tensor(weights["coefs"], dtype = torch.float32).squeeze())
  predictions = predict(".\\data\\ycb_dataset\\test_data", resnet)
  savePredictions(".\\data\\predictions.csv", predictions)