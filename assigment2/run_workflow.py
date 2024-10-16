from feature_extraction import loadAndExtractFeaturesAndLabels
from split_features import splitData
from train_model import trainModel
from validate_model import validateModel
from predict import predict, savePredictions
from common import *

if __name__ == "__main__":
  # Feature Extraction
  print("Loading ResNet")
  resNetNoFC = createResNetNoFCLayer()
  
  features, labels = loadAndExtractFeaturesAndLabels(resNetNoFC, TRAIN_DATSET_DIR, loadLabels = True)
  print(f"Extracted the features and labels for {features.shape[0]} images")
  
  # Split Data
  print("Splitting Data")
  X_train, X_test, y_train, y_test = splitData(features, labels, 0.9)
  
  # Train Model
  intercepts, coefs = trainModel(X_train, y_train)
  
  # Validate Model
  logClassifier = MultiLabelPredictor(torch.tensor(intercepts, dtype = torch.float32).squeeze(), torch.tensor(coefs, dtype = torch.float32).squeeze()).to(DEVICE)
  macroF1Metric = validateModel(logClassifier, torch.tensor(X_test), torch.tensor(y_test))
  print(f"Model has macro F1 metric of {macroF1Metric} on validation data.")
    
  # 
  resnet = createResNetCustomFCLayer(torch.tensor(intercepts, dtype = torch.float32).squeeze(), torch.tensor(coefs, dtype = torch.float32).squeeze())
  predictions = predict(TEST_DATSET_DIR, resnet)
  savePredictions(TEST_OUTPUT_FILE, predictions)
  