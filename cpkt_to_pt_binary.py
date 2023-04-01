from Models.OneDNet import OneDNet
from Models.OneDNetEnsembleInception import OneDNetEnsemble
import torch


model = OneDNetEnsemble.load_from_checkpoint(checkpoint_path="./left.ckpt")
model.eval()
torch.save(model, "./left.pt")

model = OneDNetEnsemble.load_from_checkpoint(checkpoint_path="./right.ckpt")
model.eval()
torch.save(model, "./right.pt")

model = OneDNetEnsemble.load_from_checkpoint(checkpoint_path="./forward.ckpt")
model.eval()
torch.save(model, "./forward.pt")

model = OneDNetEnsemble.load_from_checkpoint(checkpoint_path="./noise.ckpt")
model.eval()
torch.save(model, "./noise.pt")
