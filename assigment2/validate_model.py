import numpy as np

import torch

from common import *

def validateModel(model: MultiLabelPredictor, features: torch.Tensor, labels: torch.Tensor):
  predY = model(features.to(DEVICE)).cpu()
  predLabels = torch.round(predY)
  
  print(f"{torch.sum(torch.abs(labels - predLabels)).int()} / {labels.shape[0] * labels.shape[1]} incorrect matches")
  
  totalF1Score = 0
  for labelIdx in range(labels.shape[1]):
    tn = 0.0
    fn = 0.0
    fp = 0.0
    tp = 0.0
    for sampleIdx in range(labels.shape[0]):
      if (predLabels[sampleIdx][labelIdx] == 0) and (labels[sampleIdx][labelIdx] == 0):
        tn += 1
        
      elif (predLabels[sampleIdx][labelIdx] == 0) and (labels[sampleIdx][labelIdx] == 1):
        fn += 1
        
      elif (predLabels[sampleIdx][labelIdx] == 1) and (labels[sampleIdx][labelIdx] == 0):
        fp += 1
        
      elif (predLabels[sampleIdx][labelIdx] == 1) and (labels[sampleIdx][labelIdx] == 1):
        tp += 1
        
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * ((precision * recall) / (precision + recall))
    totalF1Score += f1
    
  return totalF1Score / labels.shape[1]

if __name__ == "__main__":
  print("Loading Data")
  weights = np.load(MODEL_WEIGHTS_FILE)
  data = np.load(TEST_DATA_FILE)
  
  logClassifier = MultiLabelPredictor(torch.tensor(weights["intercepts"], dtype = torch.float32).squeeze(), torch.tensor(weights["coefs"], dtype = torch.float32).squeeze()).to(DEVICE)
  macroF1Metric = validateModel(logClassifier, torch.tensor(data["features"]), torch.tensor(data["labels"]))
  print(f"Model has macro F1 metric of {macroF1Metric:.2f} on validation data.")