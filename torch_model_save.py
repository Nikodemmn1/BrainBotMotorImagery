from Models.OneDNet import OneDNet
import torch

model = OneDNet.load_from_checkpoint(checkpoint_path="./lightning_logs/version_51/checkpoints/epoch=84-step=60180.ckpt", included_classes=[0, 1, 2],
                                     channel_count=3)
model.eval()
torch.save(model, "model.pt")

