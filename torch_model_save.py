from Models.OneDNet import OneDNet
from Models.OneDNetEnsembleInception import OneDNetEnsemble
import torch

inception = True

if inception:
    model = OneDNetEnsemble.load_from_checkpoint(checkpoint_path="./left.ckpt")
else:
    model = OneDNet.load_from_checkpoint(checkpoint_path="./left.ckpt", included_classes=[0, 1, 2],
                                         channel_count=3)

model.eval()
torch.save(model, "./left.pt")

if inception:
    model = OneDNetEnsemble.load_from_checkpoint(checkpoint_path="./right.ckpt")
else:
    model = OneDNet.load_from_checkpoint(checkpoint_path="./right.ckpt", included_classes=[0, 1, 2],
                                         channel_count=3)

model.eval()
torch.save(model, "./right.pt")

if inception:
    model = OneDNetEnsemble.load_from_checkpoint(checkpoint_path="./forward.ckpt")
else:
    model = OneDNet.load_from_checkpoint(checkpoint_path="./forward.ckpt", included_classes=[0, 1, 2],
                                         channel_count=3)

model.eval()
torch.save(model, "./forward.pt")

if inception:
    model = OneDNetEnsemble.load_from_checkpoint(checkpoint_path="./noise.ckpt")
else:
    model = OneDNet.load_from_checkpoint(checkpoint_path="./noise.ckpt", included_classes=[0, 1, 2],
                                         channel_count=3)

model.eval()
torch.save(model, "./noise.pt")


