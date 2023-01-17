from torchviz import make_dot
from Models.OneDNet import OneDNet, VisualiseModel
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from Dataset.dataset import *
import torch
import torchvision
from torchview import draw_graph
import matplotlib.pyplot as plt
from torchsummary import summary

included_classes = [0, 1, 2]
included_channels = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
full_dataset = EEGDataset("./DataBDF/Out/Out_train.npy",
                              "./DataBDF/Out/Out_val.npy",
                              "./DataBDF/Out/Out_test.npy",
                              included_classes, included_channels)
train_dataset, val_dataset, test_dataset = full_dataset.get_subsets()
train_data = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
val_data = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)
test_data = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)
model = OneDNet(len(included_channels), included_classes, train_dataset.indices,
                val_dataset.indices, test_dataset.indices)
model.cuda()
summary(model, (1, 11, 107))
# x = next(iter(train_data))[0]
# model_graph = draw_graph(model, input_size=(64,1,14,107), expand_nested=True)
# #model_graph.visual_graph.graph_attr={'rankdir':'LR'}
# #model_graph.visual_graph.rotate.rotate=90
# model_graph.visual_graph.view()
#plt.show()
# make_dot(y.mean(), params=dict(model.named_parameters())).render("CNN", format="png")