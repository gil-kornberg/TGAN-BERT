import torch
from torch import nn

seqLen = 50
embedSize = 768
batchSize = 10
inputSize = seqLen * embedSize
hidden_size = 300
outputSize = seqLen * embedSize
# shrink again to hidden size


def make_disc_block(inputDim=inputSize, outputDim=hidden_size):
    return nn.Sequential(
        nn.utils.spectral_norm(nn.Linear(in_features=inputDim, out_features=outputDim)),
        nn.LeakyReLU(negative_slope=0.2, inplace=False),
    )


class Discriminator(nn.Module):
    def __init__(self, inputDim=inputSize,
                 outputDim=hidden_size):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            make_disc_block(inputDim, outputDim),
            make_disc_block(outputDim, outputDim),
            make_disc_block(outputDim, outputDim),
            nn.Linear(outputDim, 1)
        )


    def forward(self, BERTembed):
        # print('BERTembed shape: ', BERTembed.shape)
        return self.disc(BERTembed)

