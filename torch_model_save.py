from Models.OneDNet import OneDNet
from Models.OneDNetInception import OneDNetInception
import torch

inception = True

if inception:
    model = OneDNetInception.load_from_checkpoint(checkpoint_path="./model_for_test.ckpt", included_classes=[0, 1, 2],
                                         channel_count=3)
else:
    model = OneDNet.load_from_checkpoint(checkpoint_path="./model_for_test.ckpt", included_classes=[0, 1, 2],
                                         channel_count=3)

model.eval()
torch.save(model, "./model.pt")
