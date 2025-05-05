import torch
import joblib
import pickle
from models.vae_resnet152 import ResNet_VAE

class Sensor():
    def __init__(self, name, file_location, sensor_embed_dim, device):
        CNN_fc_hidden1, CNN_fc_hidden2 = 1024, 1024
        CNN_embed_dim = sensor_embed_dim # 512, 256
        dropout_p = 0.2       # dropout probability
        
        self.name = name
        self.dataset = name.split('_')[0]
        self.master_view_dist = None
        self.vae = ResNet_VAE(fc_hidden1=CNN_fc_hidden1, fc_hidden2=CNN_fc_hidden2, drop_p=dropout_p, CNN_embed_dim=CNN_embed_dim).to(device)
        state_dict = torch.load(file_location, map_location="cuda:"+ str(device), weights_only=False)
        self.vae.load_state_dict(state_dict)
        self.vae.eval()
    
    def __str__(self):
        return f"{self.name}"
    
    def __repr__(self):
        return f"{self.name}, vae: {self.file_location}"