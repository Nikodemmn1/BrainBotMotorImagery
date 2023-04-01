import torch
import cv2

#class Midas:
#    def __init__(self):
#        midas, transform, device = self.load_model_midas()
#        self.midas = midas
#        self.transform = transform
#        self.device = device
#
#    def load_model_midas(self):
#        print('load_midas')
#        # model_type = "DPT_Large"     # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
#        # model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
#        model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)
#
#        midas = torch.hub.load("intel-isl/MiDaS", model_type)
#        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
#        midas.to(device)
#        midas.eval()
#        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
#
#        if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
#            transform = midas_transforms.dpt_transform
#        else:
#            transform = midas_transforms.small_transform
#        return midas, transform, device
#
#    def predict(self, img):
#        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#        input_batch = self.transform(img).to(self.device)
#        with torch.no_grad():
#            prediction = self.midas(input_batch)
#
#            prediction = torch.nn.functional.interpolate(
#                prediction.unsqueeze(1),
#                size=img.shape[:2],
#                mode="bicubic",
#                align_corners=False,
#            ).squeeze()
#
#        return prediction.cpu().numpy()
#
#midas = Midas()

print('load_midas')
# model_type = "DPT_Large"     # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
# model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)

midas = torch.hub.load("intel-isl/MiDaS", model_type)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform
