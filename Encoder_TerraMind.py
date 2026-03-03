import torch.nn as nn
from terratorch.registry import BACKBONE_REGISTRY

class TerraMindEncoder(nn.Module):
    """
    Creates encoder which feeds embeddings into the decoder.
    All params will be overwritten according to config files
    
    Args:
        version (str): which version of terramind to use
        pretrained (bool): True = pretrained on TerraMesh, False = no pretraining
        modalities (list): list of modalities. Needs to match with train/val data loader ['S2L2A', 'S1GRD', 'LULC', ', 'S1RTC', 'DEM', 'NDVI']

        In train.py or test.py the input will be either the before modalities or after. Which are later differenced before being put into the decoder
        In train.py or test.py the input will also want number of classes defined. In our occaision we will use 3. 0=background, 1=non-damaged ag land, 2= damaged ag-land
    """
    def __init__(self,
                 version: str = "terramind_v1_base",
                 pretrained: bool = True,
                 modalities: list[str] = ['S2L2A']
                 ):
        super().__init__()
        self.model = BACKBONE_REGISTRY.build(
            version, 
            pretrained = pretrained,
            modalities = modalities)


    def forward(self, x):
        embeddings = self.model(x)        
        return embeddings


