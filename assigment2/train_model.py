import numpy as np
from sklearn.linear_model import LogisticRegression
from typing import List, Tuple
from tqdm import tqdm

from common import *


def trainModel(features: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
  interceptList: List = []
  coefList: List = []
  for classLabels in tqdm(labels.transpose(), desc="Training Models"):
    model = LogisticRegression(solver="liblinear")
    model.fit(features, classLabels)
    interceptList.append(model.intercept_)
    coefList.append(model.coef_)
    
  return np.stack(interceptList), np.stack(coefList)

if __name__ == "__main__":
  print("Loading Data")
  data = np.load(TRAIN_DATA_FILE)
  
  print("Splitting Data")
  intercepts, coefs = trainModel(data["features"], data["labels"])
  np.savez(MODEL_WEIGHTS_FILE, intercepts= intercepts, coefs = coefs)
  print(f"Saved model weights to {MODEL_WEIGHTS_FILE}")