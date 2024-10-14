import os

import torch
import torch.nn as nn
from torchvision.models.resnet import resnet18, ResNet18_Weights
import torchvision.transforms as tx

CLASS_NAME_TO_ID_MAP = {
  "002_master_chef_can": 0,
  "003_cracker_box": 1,
  "004_sugar_box": 2,
  "005_tomato_soup_can": 3,
  "006_mustard_bottle": 4,
  "007_tuna_fish_can": 5,
  "008_pudding_box": 6,
  "009_gelatin_box": 7,
  "010_potted_meat_can": 8,
  "011_banana": 9,
  "019_pitcher_base": 10,
  "021_bleach_cleanser": 11,
  "024_bowl": 12,
  "025_mug": 13,
  "035_power_drill": 14,
  "036_wood_block": 15,
  "037_scissors": 16,
  "040_large_marker": 17,
  "051_large_clamp": 18,
  "052_extra_large_clamp": 19,
  "061_foam_brick": 20
}

DATA_DIR = ".\\data"
FEATURES_OUTPUT_FILE = os.path.join(DATA_DIR, "features", "combined_features.npz")
TRAIN_DATA_FILE = os.path.join(DATA_DIR, "features", "train_features.npz")
TEST_DATA_FILE = os.path.join(DATA_DIR, "features", "test_features.npz")
MODEL_WEIGHTS_FILE = os.path.join(DATA_DIR, "features", "model_weights.npz")


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESNET_MEAN = torch.tensor([0.485, 0.456, 0.406])
RESNET_STD = torch.tensor([0.229, 0.224, 0.225])
RESNET_RESIZER = tx.Resize((224, 224))

def createResNetNoFCLayer():
  resnet = resnet18(weights = ResNet18_Weights.DEFAULT)
  resnet.fc = nn.Identity()

  resnet = resnet.to(DEVICE)
  return resnet.eval()

class MultiLabelPredictor(nn.Module):
  def __init__(self) -> None:
    super().__init__()
    self.fc = nn.Linear(512, len(CLASS_NAME_TO_ID_MAP), bias=True)
    self.sigmoid = torch.nn.Sigmoid()
    
  def __init__(self, intercepts: torch.Tensor, coefs: torch.Tensor) -> None:
    super().__init__()
    self.fc = nn.Linear(512, len(CLASS_NAME_TO_ID_MAP), bias=True)
    self.sigmoid = torch.nn.Sigmoid()
    self.fc.bias.data = intercepts
    self.fc.weight.data = coefs
    
  def forward(self, x):
    x = self.fc(x)
    return self.sigmoid(x)

def createResNetCustomFCLayer(intercepts: torch.Tensor, coefs: torch.Tensor):
  predictor = MultiLabelPredictor(intercepts, coefs)
  
  resnet = resnet18(weights = ResNet18_Weights.DEFAULT)
  resnet.fc = predictor

  resnet = resnet.to(DEVICE)
  return resnet.eval()

#TODO: Test if resize helps with feature extraction 
def prepImageforResnet(img: torch.Tensor):
  img = img.float() / 255
  img = (img - RESNET_MEAN[:, None, None]) / RESNET_STD[:, None, None]
  img= img.unsqueeze(0)
  return img.to(DEVICE)
  