import torch
from torch import nn
import positionalEncoding

seqLen = 50
embedSize = 768
batchSize = 10
outputSize = seqLen * embedSize
hidden_size = 300  # dimension of hidden layer size
noiseDim = 1024  # dimension of noise vector
device = 'cuda'

encoder_layer = nn.TransformerEncoderLayer(d_model=768, nhead=8).to(device)

class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()
        self.gen = nn.TransformerEncoder(encoder_layer, num_layers=6).to(device)
        
    # given a noise tensor return a generated "BERT" embedding
    def forward(self, noise):
        return self.gen(noise).to(device)


def get_noise(z_dim):
    t = torch.empty(z_dim)
    nn.init.uniform_(t, a=0.0, b=1.0)
    return t.to(device)
