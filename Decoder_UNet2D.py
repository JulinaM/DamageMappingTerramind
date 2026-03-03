import torch
import torch.nn as nn
from terratorch.models import necks

class UNet2D(nn.Module):
    def __init__(self, num_classes = 3):
        super().__init__()

        """Set up for the base version of Terramind, accepting outputs from transformer heads [3,5,7,9,11] from heads 0-11
            This uses the right side of U Net where we expand the compressed images. 
            The last transformer output (11) starts at the bottom of the U Net, while the other layers are used as skip connections
            
            Outputs logits, for likelihood of each pixel belonging to one of the classes

        Args:
            TerraMind differenced embeddings in train.py or test.py
        """

        
        # -------------- Neck Portion -------------- #
        self.select_indices = necks.SelectIndices(
            channel_list=[768, 768, 768, 768, 768],
            indices=[3, 5, 7, 9, 11])

        self.reshape_tokens = necks.ReshapeTokensToImage(
            channel_list=[768, 768, 768, 768, 768],
            remove_cls_token=False)

        # Project the representations to fit as the skip layers which will connect to 
        self.projections = nn.ModuleList([
            nn.Conv2d(768, 64, kernel_size=1), #index 3
            nn.Conv2d(768, 128, kernel_size=1), #index 5
            nn.Conv2d(768, 256, kernel_size=1), #index 7
            nn.Conv2d(768, 512, kernel_size=1), #index 9
            nn.Conv2d(768, 1024, kernel_size=1)]) #index 11

        # Per-level upsampling
        self.upsamplers = nn.ModuleList([
            nn.Upsample(scale_factor=16, mode="bilinear", align_corners=False),  #index 3
            nn.Upsample(scale_factor=8, mode="bilinear", align_corners=False),  #index 5
            nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False),  #index 7
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),  #index 9
            nn.Identity()]) #index 11
        
        
        # -------------- U Net Portion -------------- #
        self.relu = nn.ReLU()
        
        # block 4 up
        self.conv2d_0 = nn.Conv2d(1024, 1024, kernel_size=3, padding = 1)
        self.up_conv0 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2)

        # block 3 up
        self.conv2d_1a = nn.Conv2d(1024, 512, kernel_size=3, padding = 1)
        self.conv2d_1b = nn.Conv2d(512, 512, kernel_size=3, padding = 1)
        self.up_conv1 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2)        

        # block 2 up
        self.conv2d_2a = nn.Conv2d(512, 256, kernel_size=3, padding = 1)
        self.conv2d_2b = nn.Conv2d(256, 256, kernel_size=3, padding = 1)
        self.up_conv2 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2)

        # block 1 up
        self.conv2d_3a = nn.Conv2d(256, 128, kernel_size=3, padding =1 )
        self.conv2d_3b = nn.Conv2d(128, 128, kernel_size=3, padding =1)
        self.up_conv3 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2)

        # block 0 up
        self.conv2d_4a = nn.Conv2d(128, 64, kernel_size=3, padding =1)
        self.conv2d_4b = nn.Conv2d(64, 64, kernel_size=3, padding =1)
        self.conv2d_4c = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, embeddings):
        # -------------- Neck Portion -------------- #
        features = self.select_indices(embeddings)      # list of 4 tensors
        features = self.reshape_tokens(features)           # reshape each to (B, C, H, W)

        feature_list = []
        for f, proj, up in zip(features, self.projections, self.upsamplers):
            f = proj(f) 
            f = up(f)
            feature_list.append(f)
        index3, index5, index7, index9, index11 = feature_list 

        # -------------- U Net Portion -------------- #
        # block 5
        x = self.relu(self.conv2d_0(index11))
        x = self.up_conv0(x)

        # block 4
        x = torch.cat([x, index9], 1)
        x = self.relu(self.conv2d_1a(x))
        x = self.relu(self.conv2d_1b(x))
        x = self.up_conv1(x)

        # block 3 
        x = torch.cat([x, index7], 1)
        x = self.relu(self.conv2d_2a(x))
        x = self.relu(self.conv2d_2b(x))
        x = self.up_conv2(x)

        # block 2
        x = torch.cat([x, index5], 1)
        x = self.relu(self.conv2d_3a(x))
        x = self.relu(self.conv2d_3b(x))
        x = self.up_conv3(x)

        # block 1 
        x = torch.cat([x, index3], 1)
        x = self.relu(self.conv2d_4a(x))
        x = self.relu(self.conv2d_4b(x))
        logits = self.conv2d_4c(x)
    
        return logits