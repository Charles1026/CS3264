import os
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Tuple

from common import *

def splitData(features: np.ndarray, labels: np.ndarray, trainRatio: float = 0.8) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  return train_test_split(features, labels, train_size = trainRatio)

if __name__ == "__main__":
  print("Loading Data")
  data = np.load(FEATURES_OUTPUT_FILE)
  
  print("Splitting Data")
  X_train, X_test, y_train, y_test = splitData(data["features"], data["labels"], 0.9)
  
  np.savez(TRAIN_DATA_FILE, features= X_train, labels = y_train)
  np.savez(TEST_DATA_FILE, features= X_test, labels = y_test)
  print(f"Saved {X_train.shape[0]} train datapoints to {TRAIN_DATA_FILE} and {X_test.shape[0]} test datapoints to {TEST_DATA_FILE}")