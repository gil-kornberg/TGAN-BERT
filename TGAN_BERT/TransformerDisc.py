import torch
from torch import nn
import positionalEncoding


seqLen = 50
embedSize = 768
batchSize = 10
inputSize = seqLen * embedSize
hidden_size = 300
outputSize = seqLen * embedSize
# shrink again to hidden size

device = 'cuda'
encoder_layer = nn.TransformerEncoderLayer(d_model=embedSize, nhead=8)
positional_encoding_layer = positionalEncoding.PositionalEncoding(d_model=768, max_len=seqLen).to(device)


class Discriminator(nn.Module):
    def __init__(self, outputDim=embedSize):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            positional_encoding_layer,
            nn.TransformerEncoder(encoder_layer, num_layers=4),
            nn.Linear(outputDim, 1),
        )


    def forward(self, BERTembed):
        # print('BERTembed shape: ', BERTembed.shape)
        return self.disc(BERTembed)

