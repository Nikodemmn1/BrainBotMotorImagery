from Models.OneDNet import OneDNet
import torch

model = OneDNet.load_from_checkpoint(checkpoint_path="./Calibration/lightning_logs/version_2/checkpoints/last.ckpt", included_classes=[0, 1, 2],
                                     channel_count=3)
model.eval()
torch.save(model, "model_kuba_calibrated.pt")

