from Models.OneDNet import OneDNet
from Models.OneDNetEnsembleInception import OneDNetEnsemble
import torch

model = OneDNet.load_from_checkpoint(checkpoint_path="./model.ckpt")
model.eval()
torch.save(model, "./model.pt")
