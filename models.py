# models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import tProdLayerDCTBatch

# We use tensor t-factorization to implement low-rank parameterized tensor neural networks
class facTNNdct3Layers(nn.Module):
    def __init__(self, num_channel,dim_input,dim_latent,rank_w,num_classes=2):
        super(facTNNdct3Layers, self).__init__()
        self.layer1 = tProdLayerDCTBatch(rank_w, dim_input, num_channel)
        self.layer2 = tProdLayerDCTBatch(dim_latent, rank_w, num_channel)
        self.layer3 = tProdLayerDCTBatch(rank_w, dim_latent, num_channel)
        self.layer4 = tProdLayerDCTBatch(dim_latent, rank_w, num_channel)
        self.layer5 = tProdLayerDCTBatch(rank_w, dim_latent, num_channel)
        self.layer6 = tProdLayerDCTBatch(dim_latent, rank_w, num_channel)
        if num_classes == 2:
            self.layer7 = nn.Linear(dim_latent * num_channel, 1)
        else:
            self.layer7 = nn.Linear(dim_latent * num_channel, num_classes)

    def forward(self, x):
        x = x.view(x.shape[0], x.shape[2],1,x.shape[3]) # Adjusting shape to [batch_size, d, 1, c]
        x = self.layer1(x)
        x = nn.ReLU()(self.layer2(x))
        x = self.layer3(x)
        x = nn.ReLU()(self.layer4(x))
        x = self.layer5(x)
        x = nn.ReLU()(self.layer6(x))
        x = x.view(x.size(0), -1)
        x = self.layer7(x)
        x = torch.sigmoid(x)
        return x
    
class TNNdct3Layers(nn.Module):
    def __init__(self, num_channel,dim_input,dim_latent,num_classes=2):
        super(TNNdct3Layers, self).__init__()
        self.layer1 = tProdLayerDCTBatch(dim_latent, dim_input, num_channel)
        self.layer2 = tProdLayerDCTBatch(dim_latent, dim_latent, num_channel)
        self.layer3 = tProdLayerDCTBatch(dim_latent, dim_latent, num_channel)
        if num_classes == 2:
            self.layer4 = nn.Linear(dim_latent * num_channel, 1)
        else:
            self.layer4 = nn.Linear(dim_latent * num_channel, num_classes)

    def forward(self, x):
        # Using the TLinearLayerBatch for computations
        # x should be of shape [batch_size, 1, d, c]
        x = x.view(x.shape[0], x.shape[2],1,x.shape[3]) # Adjusting shape to [batch_size, d, 1, c]
        x = nn.ReLU()(self.layer1(x))
        x = nn.ReLU()(self.layer2(x))
        x = nn.ReLU()(self.layer3(x))
        # Flatten the tensor before the fully connected layer
        x = x.view(x.size(0), -1)
        x = self.layer4(x)
        x = torch.sigmoid(x)
        return x    
# # ---------------------------------------------------------------------------------------------------------------
